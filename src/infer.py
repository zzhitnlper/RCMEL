from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from tqdm import tqdm

from entity_aug import (
    _extract_response_text,
    _get_stage_settings,
    _get_thread_client,
    _load_json,
    _normalize_text,
    _resolve_path,
    _save_json,
)
from util import PROMPT3_knowledge_contrast_1to30


def _load_infer_args(args):
    mention_rank_input_path = _resolve_path(args.infer_kc.mention_rank_input_dir)
    entity_data_path = _resolve_path(args.infer_kc.entity_desc_llm_mllm_dir)
    mention_output_path = _resolve_path(args.infer_kc.mention_infer_kc)
    return mention_rank_input_path, entity_data_path, mention_output_path


def _get_candidate_limit(args) -> int:
    infer_args = getattr(args, "infer_kc", None)
    if infer_args is not None and hasattr(infer_args, "candidate_k"):
        return int(infer_args.candidate_k)
    return int(getattr(args.decoding, "m", 10))


def _merge_mentions(base_mentions: list[dict[str, Any]], existing_path) -> list[dict[str, Any]]:
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


def _build_entity_table_info(
    mention_info: dict[str, Any],
    qid_entity_dict: dict[str, dict[str, Any]],
    candidate_limit: int,
) -> str:
    lines: list[str] = []
    for index, qid in enumerate(mention_info.get("ndtop10", [])[:candidate_limit], start=1):
        entity = qid_entity_dict.get(qid)
        if entity is None:
            continue
        entity_name = _normalize_text(entity.get("entity_name", ""))
        entity_desc = _normalize_text(entity.get("desc_image_1sentence", "")) or _normalize_text(
            entity.get("desc_image", "")
        )
        if entity_desc:
            lines.append(f"{index}. {entity_name}: {entity_desc}")
        else:
            lines.append(f"{index}. {entity_name}")
    return "\n".join(lines)


def _fallback_knowledge_contrast(
    mention_info: dict[str, Any],
    qid_entity_dict: dict[str, dict[str, Any]],
    candidate_limit: int,
) -> list[str]:
    results = []
    for qid in mention_info.get("ndtop10", [])[:candidate_limit]:
        entity = qid_entity_dict.get(qid, {})
        entity_name = _normalize_text(entity.get("entity_name", "This entity")) or "This entity"
        entity_desc = _normalize_text(entity.get("desc_image_1sentence", "")) or _normalize_text(
            entity.get("desc_image", "")
        )
        if entity_desc:
            results.append(f"Unlike the other entities, {entity_name} is {entity_desc}")
        else:
            results.append(f"Unlike the other entities, {entity_name} is a distinct candidate entity.")
    return results


def _parse_single_knowledge_contrast_response(response_text: str) -> str:
    text = _normalize_text(response_text)
    if not text:
        return ""
    if ". " in text:
        prefix, content = text.split(". ", 1)
        if prefix.isdigit():
            return _normalize_text(content)
    return text


def _run_single_infer_kc(
    mention_info: dict[str, Any],
    qid_entity_dict: dict[str, dict[str, Any]],
    base_url: str,
    api_key: str,
    model_name: str,
    candidate_limit: int,
) -> tuple[list[str], bool]:
    entity_table_info = _build_entity_table_info(mention_info, qid_entity_dict, candidate_limit)
    if not entity_table_info:
        return [], True

    try:
        client = _get_thread_client(base_url, api_key)
        generated_results: list[str] = []
        for entity_num, qid in enumerate(mention_info.get("ndtop10", [])[:candidate_limit], start=1):
            entity = qid_entity_dict.get(qid)
            if entity is None:
                generated_results.append("")
                continue

            prompt_text = PROMPT3_knowledge_contrast_1to30.format(
                Entity_Num=entity_num,
                Entity_Name=_normalize_text(entity.get("entity_name", "")),
                Entity_table_info=entity_table_info,
            )
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": "you are a helpful assistant!"},
                    {"role": "user", "content": prompt_text},
                ],
                max_tokens=256,
                temperature=0.3,
                top_p=0.9,
            )
            response_text = _extract_response_text(response)
            parsed = _parse_single_knowledge_contrast_response(response_text)
            if not parsed:
                break
            generated_results.append(parsed)

        if len(generated_results) >= candidate_limit and all(generated_results[:candidate_limit]):
            return generated_results[:candidate_limit], False
    except Exception:
        pass

    return _fallback_knowledge_contrast(mention_info, qid_entity_dict, candidate_limit), True


def infer_kc_onegpu(args) -> list[dict[str, Any]]:
    mention_rank_input_path, entity_data_path, mention_output_path = _load_infer_args(args)
    base_mentions = _load_json(mention_rank_input_path)
    mentions = _merge_mentions(base_mentions, mention_output_path)
    entity_data = _load_json(entity_data_path)
    qid_entity_dict = {item["qid"]: item for item in entity_data}

    settings = _get_stage_settings(args, "llm")
    base_url = f"http://127.0.0.1:{settings['base_port']}/v1"
    candidate_limit = _get_candidate_limit(args)

    completed = 0
    success_count = 0
    fallback_count = 0
    skipped_count = 0
    for mention in tqdm(mentions, desc="Infer KC"):
        if len(mention.get("knowledge_contrast", [])) > candidate_limit:
            mention["knowledge_contrast"] = mention["knowledge_contrast"][:candidate_limit]
        if len(mention.get("knowledge_contrast", [])) >= candidate_limit:
            skipped_count += 1
            continue

        knowledge_contrast, used_fallback = _run_single_infer_kc(
            mention,
            qid_entity_dict,
            base_url,
            settings["api_key"],
            settings["model"],
            candidate_limit,
        )
        mention["knowledge_contrast"] = knowledge_contrast
        if used_fallback:
            fallback_count += 1
        else:
            success_count += 1
        completed += 1
        if completed % settings["save_every"] == 0:
            _save_json(mention_output_path, mentions)

    _save_json(mention_output_path, mentions)
    print(
        f"[LLM] Knowledge contrast generation completed: "
        f"{success_count} succeeded, {fallback_count} fell back, {skipped_count} skipped."
    )
    return mentions


def infer_kc_multigpu(args, endpoints: list[str] | None = None) -> list[dict[str, Any]]:
    mention_rank_input_path, entity_data_path, mention_output_path = _load_infer_args(args)
    base_mentions = _load_json(mention_rank_input_path)
    mentions = _merge_mentions(base_mentions, mention_output_path)
    entity_data = _load_json(entity_data_path)
    qid_entity_dict = {item["qid"]: item for item in entity_data}

    settings = _get_stage_settings(args, "llm")
    if not endpoints:
        endpoints = [f"http://127.0.0.1:{settings['base_port']}/v1"]
    candidate_limit = _get_candidate_limit(args)

    future_to_index: dict[Any, int] = {}
    pending_count = 0
    skipped_count = 0
    for mention in mentions:
        if len(mention.get("knowledge_contrast", [])) > candidate_limit:
            mention["knowledge_contrast"] = mention["knowledge_contrast"][:candidate_limit]
        if len(mention.get("knowledge_contrast", [])) >= candidate_limit:
            skipped_count += 1
        else:
            pending_count += 1

    print(
        f"[LLM] Knowledge contrast generation started: {pending_count} pending, "
        f"{skipped_count} skipped, {len(endpoints)} endpoints, "
        f"{settings['workers']} workers."
    )

    with ThreadPoolExecutor(max_workers=settings["workers"]) as executor:
        for idx, mention in enumerate(mentions):
            if len(mention.get("knowledge_contrast", [])) >= candidate_limit:
                continue

            endpoint = endpoints[idx % len(endpoints)]
            future = executor.submit(
                _run_single_infer_kc,
                dict(mention),
                qid_entity_dict,
                endpoint,
                settings["api_key"],
                settings["model"],
                candidate_limit,
            )
            future_to_index[future] = idx

        completed = 0
        success_count = 0
        fallback_count = 0
        for future in tqdm(as_completed(future_to_index), total=len(future_to_index), desc="Infer KC"):
            idx = future_to_index[future]
            knowledge_contrast, used_fallback = future.result()
            mentions[idx]["knowledge_contrast"] = knowledge_contrast
            if used_fallback:
                fallback_count += 1
            else:
                success_count += 1
            completed += 1
            if completed % settings["save_every"] == 0:
                _save_json(mention_output_path, mentions)
                print(f"[LLM] Completed {completed} items. Progress has been saved.")

    _save_json(mention_output_path, mentions)
    print(
        f"[LLM] Knowledge contrast generation completed. "
        f"{success_count} succeeded, {fallback_count} fell back, {skipped_count} skipped."
    )
    return mentions
