from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from openai import OpenAI
from tqdm import tqdm

from entity_aug import (
    _build_vllm_client,
    _encode_image_to_data_url,
    _extract_response_text,
    _get_stage_settings,
    _get_thread_client,
    _load_json,
    _normalize_base_url,
    _normalize_text,
    _resolve_path,
    _save_json,
)
from util import (
    PROMPT2_mention_image_1n_image,
    PROMPT2_mention_image_no_image,
    PROMPT2_mention_image_text_1sent,
)


def _load_mention_args(args) -> tuple[Path, Path, Path]:
    mention_data_path = _resolve_path(args.mention.mention_data_dir)
    mention_image_dir = _resolve_path(args.mention.mention_image_dir)
    mention_output_mllm_path = _resolve_path(args.mention.mention_output_dir_mllm)
    return mention_data_path, mention_image_dir, mention_output_mllm_path


def _merge_mentions(base_mentions: list[dict[str, Any]], existing_path: Path) -> list[dict[str, Any]]:
    if not existing_path.exists():
        return [dict(item) for item in base_mentions]

    existing_mentions = _load_json(existing_path)
    existing_map = {item["id"]: item for item in existing_mentions if "id" in item}

    merged = []
    for item in base_mentions:
        record = dict(item)
        old_record = existing_map.get(item["id"])
        if old_record:
            for key, value in old_record.items():
                if value is not None and value != "":
                    record[key] = value
        merged.append(record)
    return merged


def _resolve_mention_image_path(mention_image_dir: Path, img_path: str) -> Path | None:
    if not img_path:
        return None

    image_path = mention_image_dir / img_path
    if image_path.exists() and image_path.stat().st_size > 0:
        return image_path
    return None


def _fallback_desc_mention(mention: dict[str, Any]) -> str:
    mention_name = _normalize_text(mention.get("mentions", "")) or "mention"
    sentence = _normalize_text(mention.get("sentence", ""))

    if sentence:
        return f"The {mention_name} refer to the entity mentioned in the sentence: {sentence}."
    return f"The {mention_name} refer to the target entity in the mention dataset."


def _generate_desc_image_with_mllm(
    client: OpenAI,
    model_name: str,
    mention: dict[str, Any],
    image_path: Path | None,
) -> str:
    mention_name = mention.get("mentions", "")
    sentence = mention.get("sentence", "")

    if image_path is None:
        prompt_text = PROMPT2_mention_image_no_image.format(
            mentions=mention_name,
            sentence=sentence,
        )
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "you are a helpful assistant!"},
                {"role": "user", "content": prompt_text},
            ],
            max_tokens=256,
            temperature=0.2,
            top_p=0.9,
        )
        return _extract_response_text(response)

    prompt_text = PROMPT2_mention_image_1n_image.format(
        mentions=mention_name,
        sentence=sentence,
    )
    image_data_url = _encode_image_to_data_url(image_path)
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": "you are a helpful assistant!"},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt_text},
                    {"type": "image_url", "image_url": {"url": image_data_url}},
                ],
            },
        ],
        max_tokens=256,
        temperature=0.2,
        top_p=0.9,
    )
    return _extract_response_text(response)


def _generate_with_llm(client: OpenAI, model_name: str, prompt_text: str) -> str:
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": "you are a helpful assistant!"},
            {"role": "user", "content": prompt_text},
        ],
        max_tokens=256,
        temperature=0.6,
        top_p=0.9,
    )
    return _extract_response_text(response)


def _run_single_mention_mllm(
    mention: dict[str, Any],
    mention_image_dir: Path,
    base_url: str,
    api_key: str,
    model_name: str,
) -> str:
    image_path = _resolve_mention_image_path(mention_image_dir, mention.get("imgPath", ""))
    if image_path is None:
        return _fallback_desc_mention(mention)

    try:
        client = _get_thread_client(base_url, api_key)
        return _generate_desc_image_with_mllm(client, model_name, mention, image_path)
    except Exception:
        return _fallback_desc_mention(mention)


def _run_single_mention_llm(
    mention: dict[str, Any],
    base_url: str,
    api_key: str,
    model_name: str,
) -> str:
    prompt_text = PROMPT2_mention_image_text_1sent.format(
        mentions=mention.get("mentions", ""),
        sentence=mention.get("sentence", ""),
        desc_image=mention.get("desc_image", ""),
    )
    try:
        client = _get_thread_client(base_url, api_key)
        return _generate_with_llm(client, model_name, prompt_text)
    except Exception:
        return _normalize_text(mention.get("desc_image", ""))


def mention_aug_onegpu_mllm(args) -> list[dict[str, Any]]:
    mention_data_path, mention_image_dir, mention_output_mllm_path = _load_mention_args(args)
    base_mentions = _load_json(mention_data_path)
    mentions = _merge_mentions(base_mentions, mention_output_mllm_path)

    settings = _get_stage_settings(args, "mllm")
    base_url = _normalize_base_url(f"http://127.0.0.1:{settings['base_port']}")
    client = None

    for idx, mention in enumerate(tqdm(mentions, desc="Mention MLLM")):
        if _normalize_text(mention.get("desc_image", "")):
            continue

        image_path = _resolve_mention_image_path(mention_image_dir, mention.get("imgPath", ""))
        try:
            if image_path is None:
                mention["desc_image"] = _fallback_desc_mention(mention)
            else:
                if client is None:
                    client = _build_vllm_client(base_url, settings["api_key"])
                mention["desc_image"] = _generate_desc_image_with_mllm(
                    client, settings["model"], mention, image_path
                )
        except Exception:
            mention["desc_image"] = _fallback_desc_mention(mention)

        if idx % settings["save_every"] == 0:
            _save_json(mention_output_mllm_path, mentions)

    _save_json(mention_output_mllm_path, mentions)
    return mentions


def mention_aug_onegpu_llm(args) -> list[dict[str, Any]]:
    mention_output_mllm_path = _resolve_path(args.mention.mention_output_dir_mllm)
    mention_output_llm_path = _resolve_path(args.mention.mention_output_dir_mllm_llm)

    base_mentions = _load_json(mention_output_mllm_path)
    mentions = _merge_mentions(base_mentions, mention_output_llm_path)

    settings = _get_stage_settings(args, "llm")
    base_url = _normalize_base_url(f"http://127.0.0.1:{settings['base_port']}")
    client = None

    for idx, mention in enumerate(tqdm(mentions, desc="Mention LLM")):
        mention.pop("desc_image_summary", None)
        if not _normalize_text(mention.get("desc_image", "")):
            mention["desc_image"] = _fallback_desc_mention(mention)

        if _normalize_text(mention.get("desc_image_1sentence", "")):
            continue

        try:
            if client is None:
                client = _build_vllm_client(base_url, settings["api_key"])
            prompt_text = PROMPT2_mention_image_text_1sent.format(
                mentions=mention.get("mentions", ""),
                sentence=mention.get("sentence", ""),
                desc_image=mention.get("desc_image", ""),
            )
            mention["desc_image_1sentence"] = _generate_with_llm(
                client, settings["model"], prompt_text
            )
        except Exception:
            mention["desc_image_1sentence"] = _normalize_text(mention.get("desc_image", ""))

        if idx % settings["save_every"] == 0:
            _save_json(mention_output_llm_path, mentions)

    _save_json(mention_output_llm_path, mentions)
    return mentions


def mention_aug_parallel_mllm(args, endpoints: list[str]) -> list[dict[str, Any]]:
    mention_data_path, mention_image_dir, mention_output_mllm_path = _load_mention_args(args)
    base_mentions = _load_json(mention_data_path)
    mentions = _merge_mentions(base_mentions, mention_output_mllm_path)

    settings = _get_stage_settings(args, "mllm")
    future_to_index: dict[Any, int] = {}
    pending_count = 0
    skipped_count = 0
    image_backed_count = 0
    no_image_count = 0

    for mention in mentions:
        if _normalize_text(mention.get("desc_image", "")):
            skipped_count += 1
            continue
        pending_count += 1
        if _resolve_mention_image_path(mention_image_dir, mention.get("imgPath", "")) is None:
            no_image_count += 1
        else:
            image_backed_count += 1

    print(
        f"[MLLM] Mention augmentation started: {pending_count} pending, "
        f"{skipped_count} skipped, {image_backed_count} with image, "
        f"{no_image_count} without image, {len(endpoints)} endpoints, "
        f"{settings['workers']} workers."
    )

    with ThreadPoolExecutor(max_workers=settings["workers"]) as executor:
        for idx, mention in enumerate(mentions):
            if _normalize_text(mention.get("desc_image", "")):
                continue

            endpoint = endpoints[idx % len(endpoints)]
            future = executor.submit(
                _run_single_mention_mllm,
                dict(mention),
                mention_image_dir,
                endpoint,
                settings["api_key"],
                settings["model"],
            )
            future_to_index[future] = idx

        completed = 0
        for future in tqdm(as_completed(future_to_index), total=len(future_to_index), desc="Mention MLLM"):
            idx = future_to_index[future]
            mentions[idx]["desc_image"] = future.result()
            completed += 1
            if completed % settings["save_every"] == 0:
                _save_json(mention_output_mllm_path, mentions)
                print(f"[MLLM] Completed {completed} items. Progress has been saved.")

    _save_json(mention_output_mllm_path, mentions)
    print("[MLLM] Mention augmentation completed. All results have been saved.")
    return mentions


def mention_aug_parallel_llm(args, endpoints: list[str]) -> list[dict[str, Any]]:
    mention_output_mllm_path = _resolve_path(args.mention.mention_output_dir_mllm)
    mention_output_llm_path = _resolve_path(args.mention.mention_output_dir_mllm_llm)

    base_mentions = _load_json(mention_output_mllm_path)
    mentions = _merge_mentions(base_mentions, mention_output_llm_path)

    settings = _get_stage_settings(args, "llm")
    future_to_index: dict[Any, int] = {}
    pending_count = 0
    skipped_count = 0

    for mention in mentions:
        has_desc_image = _normalize_text(mention.get("desc_image", ""))
        has_one_sentence = _normalize_text(mention.get("desc_image_1sentence", ""))
        if not has_desc_image:
            pending_count += 1
        elif has_one_sentence:
            skipped_count += 1
        else:
            pending_count += 1

    print(
        f"[LLM] Mention information refinement started: {pending_count} pending, "
        f"{skipped_count} skipped, {len(endpoints)} endpoints, "
        f"{settings['workers']} workers."
    )

    with ThreadPoolExecutor(max_workers=settings["workers"]) as executor:
        for idx, mention in enumerate(mentions):
            mention.pop("desc_image_summary", None)
            if not _normalize_text(mention.get("desc_image", "")):
                mention["desc_image"] = _fallback_desc_mention(mention)

            if _normalize_text(mention.get("desc_image_1sentence", "")):
                continue

            endpoint = endpoints[idx % len(endpoints)]
            future = executor.submit(
                _run_single_mention_llm,
                dict(mention),
                endpoint,
                settings["api_key"],
                settings["model"],
            )
            future_to_index[future] = idx

        completed = 0
        for future in tqdm(as_completed(future_to_index), total=len(future_to_index), desc="Mention LLM"):
            idx = future_to_index[future]
            mentions[idx]["desc_image_1sentence"] = future.result()
            completed += 1
            if completed % settings["save_every"] == 0:
                _save_json(mention_output_llm_path, mentions)
                print(f"[LLM] Completed {completed} items. Progress has been saved.")

    _save_json(mention_output_llm_path, mentions)
    print("[LLM] Mention information refinement completed. All results have been saved.")
    return mentions


mention_aug_multigpu_mllm = mention_aug_parallel_mllm
mention_aug_multigpu_llm = mention_aug_parallel_llm


def mention_aug_multigpu(args, mllm_endpoints: list[str], llm_endpoints: list[str]) -> list[dict[str, Any]]:
    mention_aug_parallel_mllm(args, mllm_endpoints)
    return mention_aug_parallel_llm(args, llm_endpoints)


def mention_aug_onegpu(args) -> list[dict[str, Any]]:
    mention_aug_onegpu_mllm(args)
    return mention_aug_onegpu_llm(args)
