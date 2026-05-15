from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from tqdm import tqdm

from entity_aug import (
    _build_vllm_client,
    _extract_response_text,
    _get_stage_settings,
    _get_thread_client,
    _load_json,
    _normalize_base_url,
    _normalize_text,
    _resolve_path,
    _save_json,
)
from util import PROMPT3_ranker


def _load_decoding_args(args):
    entity_data_path = _resolve_path(args.infer_kc.entity_desc_llm_mllm_dir)
    mention_input_path = _resolve_path(args.decoding.mention_entity_diff_dir)
    result_output_path = _resolve_path(args.decoding.result)
    return entity_data_path, mention_input_path, result_output_path


def _get_decoding_candidate_limit(args) -> int:
    return int(getattr(args.decoding, "m", 10))


def _get_decoding_runtime(args) -> dict[str, Any]:
    decoding_args = getattr(args, "decoding", None)
    provider = str(getattr(decoding_args, "provider", "vllm")).lower()

    if provider == "gpt":
        base_url = _normalize_base_url(
            str(getattr(decoding_args, "base_url", "https://api.openai.com/v1"))
        )
        api_key = str(getattr(decoding_args, "api_key", ""))
        model = str(getattr(decoding_args, "model", "gpt-4o-mini"))
        workers = int(getattr(decoding_args, "workers", 8))
        save_every = int(getattr(getattr(args, "vllm", None), "save_every", 500))
        return {
            "provider": "gpt",
            "base_url": base_url,
            "api_key": api_key,
            "model": model,
            "workers": max(workers, 1),
            "save_every": save_every,
        }

    settings = _get_stage_settings(args, "llm")
    return {
        "provider": "vllm",
        "base_url": _normalize_base_url(f"http://127.0.0.1:{settings['base_port']}"),
        "api_key": settings["api_key"],
        "model": settings["model"],
        "workers": settings["workers"],
        "save_every": settings["save_every"],
    }


def _merge_results(base_mentions: list[dict[str, Any]], existing_path) -> list[dict[str, Any]]:
    if not existing_path.exists():
        return [dict(item) for item in base_mentions]

    existing_results = _load_json(existing_path)
    existing_map = {item["id"]: item for item in existing_results if "id" in item}

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


def _build_entity_table_diff(
    mention_info: dict[str, Any],
    qid_entity_dict: dict[str, dict[str, Any]],
    candidate_limit: int,
) -> str:
    lines: list[str] = []
    knowledge_contrast = mention_info.get("knowledge_contrast", [])

    for index, qid in enumerate(mention_info.get("ndtop10", [])[:candidate_limit], start=1):
        entity = qid_entity_dict.get(qid)
        if entity is None:
            continue

        entity_name = _normalize_text(entity.get("entity_name", ""))
        entity_desc = _normalize_text(entity.get("desc_image_1sentence", "")) or _normalize_text(
            entity.get("desc_image", "")
        )
        contrast = ""
        if index - 1 < len(knowledge_contrast):
            contrast = _normalize_text(knowledge_contrast[index - 1])

        line = f"{index}. {entity_name}"
        if entity_desc:
            line += f": {entity_desc}"
        if contrast:
            line += f" {contrast}"
        lines.append(line)

    return "\n".join(lines)


def _default_rank_output(candidate_count: int) -> str:
    if candidate_count==10:
        return "2 4 3 6 1 5 7 8 9 10"
    return " ".join(str(i) for i in range(1, candidate_count + 1))


def _parse_rank_output(response_text: str, candidate_count: int) -> str | None:
    numbers = re.findall(r"\d+", response_text)
    seen: set[int] = set()
    ordered: list[int] = []

    for token in numbers:
        value = int(token)
        if 1 <= value <= candidate_count and value not in seen:
            seen.add(value)
            ordered.append(value)

    if len(ordered) != candidate_count:
        return None

    return " ".join(str(value) for value in ordered)


def _build_result_item(
    mention_info: dict[str, Any],
    ranked_output: str,
) -> dict[str, Any]:
    answer = mention_info.get("answer")
    nd_candidates = mention_info.get("ndtop10", [])
    true_rank = nd_candidates.index(answer) + 1 if answer in nd_candidates else 0

    return {
        "id": mention_info.get("id"),
        "mentions": mention_info.get("mentions"),
        "entities": mention_info.get("entities"),
        "answer": answer,
        "true": true_rank,
        "top1_30": ranked_output,
    }


def _run_single_decoding(
    mention_info: dict[str, Any],
    qid_entity_dict: dict[str, dict[str, Any]],
    base_url: str,
    api_key: str,
    model_name: str,
    candidate_limit: int,
) -> tuple[dict[str, Any], bool]:
    entity_table_info = _build_entity_table_diff(mention_info, qid_entity_dict, candidate_limit)
    available_candidates = min(candidate_limit, len(mention_info.get("ndtop10", [])))
    default_output = _default_rank_output(available_candidates)

    mention_category = ""
    answer_entity = qid_entity_dict.get(mention_info.get("answer", ""))
    if answer_entity is not None:
        mention_category = _normalize_text(answer_entity.get("instance", ""))

    prompt_text = PROMPT3_ranker.format(
        Entity_table_info=entity_table_info,
        mention_name=mention_info.get("mentions", ""),
        mention_context=mention_info.get("sentence", ""),
        mention_des=mention_info.get("desc_image_1sentence", ""),
        mention_cate=mention_category,
    )
    prompt_text = prompt_text.replace(
        "Rank all 30 entities in the Entity Table based on their relevance to the given mention.",
        f"Rank all {available_candidates} entities in the Entity Table based on their relevance to the given mention.",
    )
    prompt_text = prompt_text.replace(
        "\"1 8 11 12 21 22 13 5 26 23 24 15 30 14 29 3 28 20 27 9 19 25 2 6 18 7 17 16 10 4\"",
        f"\"{_default_rank_output(available_candidates)}\"",
    )

    try:
        client = _get_thread_client(base_url, api_key)
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "you are a helpful assistant!"},
                {"role": "user", "content": prompt_text},
            ],
            max_tokens=128,
            temperature=0.3,
            top_p=0.9,
        )
        response_text = _extract_response_text(response)
        parsed = _parse_rank_output(response_text, available_candidates)
        if parsed is not None:
            return _build_result_item(mention_info, parsed), False
    except Exception:
        pass

    return _build_result_item(mention_info, default_output), True


def infer_decoding_onegpu(args) -> list[dict[str, Any]]:
    entity_data_path, mention_input_path, result_output_path = _load_decoding_args(args)
    entity_data = _load_json(entity_data_path)
    qid_entity_dict = {item["qid"]: item for item in entity_data}
    base_mentions = _load_json(mention_input_path)
    results = _merge_results(base_mentions, result_output_path)

    runtime = _get_decoding_runtime(args)
    candidate_limit = _get_decoding_candidate_limit(args)

    completed = 0
    success_count = 0
    fallback_count = 0
    skipped_count = 0

    for idx, mention in enumerate(tqdm(results, desc="Decoding")):
        if _normalize_text(mention.get("top1_30", "")):
            skipped_count += 1
            continue

        result_item, used_fallback = _run_single_decoding(
            mention,
            qid_entity_dict,
            runtime["base_url"],
            runtime["api_key"],
            runtime["model"],
            candidate_limit,
        )
        results[idx] = result_item
        if used_fallback:
            fallback_count += 1
        else:
            success_count += 1
        completed += 1
        if completed % runtime["save_every"] == 0:
            _save_json(result_output_path, results)

    _save_json(result_output_path, results)
    print(
        f"[LLM] Decoding completed: {success_count} succeeded, "
        f"{fallback_count} fell back, {skipped_count} skipped."
    )
    return results


def infer_decoding_multigpu(args, endpoints: list[str] | None = None) -> list[dict[str, Any]]:
    entity_data_path, mention_input_path, result_output_path = _load_decoding_args(args)
    entity_data = _load_json(entity_data_path)
    qid_entity_dict = {item["qid"]: item for item in entity_data}
    base_mentions = _load_json(mention_input_path)
    results = _merge_results(base_mentions, result_output_path)

    runtime = _get_decoding_runtime(args)
    if not endpoints:
        endpoints = [runtime["base_url"]]
    candidate_limit = _get_decoding_candidate_limit(args)

    future_to_index: dict[Any, int] = {}
    pending_count = 0
    skipped_count = 0
    for item in results:
        if _normalize_text(item.get("top1_30", "")):
            skipped_count += 1
        else:
            pending_count += 1

    print(
        f"[LLM] Final decoding started: {pending_count} pending, {skipped_count} skipped, "
        f"{len(endpoints)} endpoints, {runtime['workers']} workers."
    )

    with ThreadPoolExecutor(max_workers=runtime["workers"]) as executor:
        for idx, mention in enumerate(results):
            if _normalize_text(mention.get("top1_30", "")):
                continue

            endpoint = endpoints[idx % len(endpoints)]
            future = executor.submit(
                _run_single_decoding,
                dict(mention),
                qid_entity_dict,
                endpoint,
                runtime["api_key"],
                runtime["model"],
                candidate_limit,
            )
            future_to_index[future] = idx

        completed = 0
        success_count = 0
        fallback_count = 0
        for future in tqdm(as_completed(future_to_index), total=len(future_to_index), desc="Decoding"):
            idx = future_to_index[future]
            result_item, used_fallback = future.result()
            results[idx] = result_item
            if used_fallback:
                fallback_count += 1
            else:
                success_count += 1
            completed += 1
            if completed % runtime["save_every"] == 0:
                _save_json(result_output_path, results)
                print(f"[LLM] Completed {completed} items. Progress has been saved.")

    _save_json(result_output_path, results)
    print(
        f"[LLM] Decoding completed. {success_count} succeeded, "
        f"{fallback_count} fell back, {skipped_count} skipped."
    )
    return results
