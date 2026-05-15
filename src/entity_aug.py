from __future__ import annotations

import base64
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
from typing import Any
from urllib.error import URLError
from urllib.request import urlopen

from openai import OpenAI
from PIL import ImageFile
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from util import PROMPT1_entity_image_1n_image, PROMPT1_entity_image_text_1sent

ImageFile.LOAD_TRUNCATED_IMAGES = True
_THREAD_LOCAL = threading.local()


def _resolve_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _normalize_text(text: str) -> str:
    return " ".join(text.replace("\n", " ").split()).strip()


def _clean_generated_text(text: str) -> str:
    text = text.strip()
    for marker in ("[/INST]", "ASSISTANT:", "Assistant:"):
        if marker in text:
            text = text.split(marker, 1)[-1]
    return _normalize_text(text)


def _choose_article(text: str) -> str:
    if not text:
        return "a"
    return "an" if text[0].lower() in {"a", "e", "i", "o", "u"} else "a"


def _fallback_desc_image(entity: dict[str, Any]) -> str:
    entity_name = entity.get("entity_name", "entity").strip() or "entity"
    desc = _normalize_text(entity.get("desc", ""))
    instance = _normalize_text(entity.get("instance", ""))
    attr = _normalize_text(entity.get("attr", ""))

    if desc:
        if desc.lower().startswith(("a ", "an ", "the ")):
            text = f"The {entity_name} refer to {desc}."
        else:
            text = f"The {entity_name} refer to {_choose_article(desc)} {desc}."
    elif instance:
        text = f"The {entity_name} refer to {_choose_article(instance)} {instance}."
    elif attr:
        text = f"The {entity_name} refer to an entity with attributes: {attr}."
    else:
        text = f"The {entity_name} refer to an entity in the knowledge base."

    if desc and instance and instance.lower() not in desc.lower():
        text += f" It is related to {instance}."
    if attr and attr.lower() not in text.lower():
        text += f" Relevant attributes include {attr}."
    return _normalize_text(text)


def _merge_entities(base_entities: list[dict[str, Any]], existing_path: Path) -> list[dict[str, Any]]:
    if not existing_path.exists():
        return [dict(item) for item in base_entities]

    existing_entities = _load_json(existing_path)
    existing_map = {item["qid"]: item for item in existing_entities if "qid" in item}

    merged = []
    for item in base_entities:
        record = dict(item)
        old_record = existing_map.get(item["qid"])
        if old_record:
            for key, value in old_record.items():
                if value is not None and value != "":
                    record[key] = value
        merged.append(record)
    return merged


def _iter_existing_image_paths(image_dir: Path, image_names: list[str]) -> list[Path]:
    paths = []
    for image_name in image_names:
        image_path = image_dir / image_name
        if image_path.exists() and image_path.stat().st_size > 0:
            paths.append(image_path)
    return paths


def _normalize_base_url(base_url: str) -> str:
    base_url = base_url.rstrip("/")
    if not base_url.endswith("/v1"):
        base_url = f"{base_url}/v1"
    return base_url


def _get_stage_settings(args, stage_name: str) -> dict[str, Any]:
    vllm_args = getattr(args, "vllm", None)
    if vllm_args is None:
        raise ValueError("Missing `vllm` section in the config.")

    stage_args = getattr(vllm_args, stage_name, None)
    if stage_args is None:
        raise ValueError(f"Missing `vllm.{stage_name}` section in the config.")

    shared_gpus = getattr(vllm_args, "gpus", None)
    stage_gpus = getattr(stage_args, "gpus", None)
    gpu_values = stage_gpus if stage_gpus is not None else shared_gpus
    if gpu_values is None:
        raise ValueError(
            f"Missing GPU configuration for `vllm.{stage_name}`. "
            "Set `vllm.gpus` or `vllm.{stage_name}.gpus` in the config."
        )

    gpus = [int(gpu_id) for gpu_id in list(gpu_values)]
    base_port = int(stage_args.base_port)
    max_model_len = int(stage_args.max_model_len)
    shared_tensor_parallel_size = getattr(vllm_args, "tensor_parallel_size", 1)
    shared_workers = getattr(vllm_args, "workers", len(gpus))
    tensor_parallel_size = int(
        getattr(stage_args, "tensor_parallel_size", shared_tensor_parallel_size)
    )
    workers = int(getattr(stage_args, "workers", shared_workers))

    return {
        "api_key": str(getattr(vllm_args, "api_key", "EMPTY")),
        "startup_timeout": int(getattr(vllm_args, "startup_timeout", 600)),
        "save_every": int(getattr(vllm_args, "save_every", 500)),
        "model": str(stage_args.model),
        "gpus": gpus,
        "base_port": base_port,
        "max_model_len": max_model_len,
        "tensor_parallel_size": tensor_parallel_size,
        "workers": max(workers, 1),
    }


def _build_vllm_client(base_url: str, api_key: str) -> OpenAI:
    return OpenAI(base_url=base_url, api_key=api_key)


def _get_thread_client(base_url: str, api_key: str) -> OpenAI:
    cache_key = (base_url, api_key)
    client_cache = getattr(_THREAD_LOCAL, "client_cache", None)
    if client_cache is None:
        client_cache = {}
        _THREAD_LOCAL.client_cache = client_cache

    client = client_cache.get(cache_key)
    if client is None:
        client = _build_vllm_client(base_url, api_key)
        client_cache[cache_key] = client
    return client


def _encode_image_to_data_url(image_path: Path) -> str:
    suffix = image_path.suffix.lower()
    mime_type = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".webp": "image/webp",
    }.get(suffix, "application/octet-stream")

    image_bytes = image_path.read_bytes()
    encoded = base64.b64encode(image_bytes).decode("utf-8")
    return f"data:{mime_type};base64,{encoded}"


def _extract_response_text(response) -> str:
    content = response.choices[0].message.content
    if isinstance(content, str):
        return _clean_generated_text(content)

    if isinstance(content, list):
        text_parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                text_parts.append(item.get("text", ""))
        return _clean_generated_text(" ".join(text_parts))

    return ""


def _generate_desc_image_with_mllm(
    client: OpenAI,
    model_name: str,
    entity: dict[str, Any],
    image_path: Path,
) -> str:
    prompt_body = PROMPT1_entity_image_1n_image.format(
        entity_name=entity.get("entity_name", ""),
        attr=entity.get("attr", ""),
        instance=entity.get("instance", ""),
        desc=entity.get("desc", ""),
    )
    image_data_url = _encode_image_to_data_url(image_path)
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": "you are a helpful assistant!"},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt_body},
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


def _load_entity_args(args) -> tuple[Path, Path, Path]:
    entity_data_path = _resolve_path(args.ent.entity_data_dir)
    entity_image_dir = _resolve_path(args.ent.entity_image_dir)
    entity_output_mllm_path = _resolve_path(args.ent.entity_output_dir_mllm)
    return entity_data_path, entity_image_dir, entity_output_mllm_path


def _wait_for_vllm_ready(base_url: str, timeout_seconds: int) -> None:
    models_url = f"{base_url.rstrip('/')}/models"
    start_time = time.time()

    while time.time() - start_time < timeout_seconds:
        try:
            with urlopen(models_url, timeout=10) as response:
                if response.status == 200:
                    return
        except URLError:
            pass
        except Exception:
            pass
        time.sleep(2)

    raise TimeoutError(f"Timed out while waiting for vLLM service: {models_url}")


def start_vllm_stage(args, stage_name: str) -> tuple[list[dict[str, Any]], list[str]]:
    settings = _get_stage_settings(args, stage_name)
    log_dir = REPO_ROOT / "logs" / "vllm"
    log_dir.mkdir(parents=True, exist_ok=True)

    server_handles: list[dict[str, Any]] = []
    endpoints: list[str] = []

    try:
        print(f"[{stage_name.upper()}] Starting vLLM services...")
        for index, gpu_id in enumerate(settings["gpus"]):
            port = settings["base_port"] + index
            base_url = _normalize_base_url(f"http://127.0.0.1:{port}")
            log_path = log_dir / f"{stage_name}_{port}.log"
            log_file = log_path.open("w", encoding="utf-8")

            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

            command = [
                "vllm",
                "serve",
                settings["model"],
                "--port",
                str(port),
                "--max-model-len",
                str(settings["max_model_len"]),
                "--tensor-parallel-size",
                str(settings["tensor_parallel_size"]),
            ]

            # print(
            #     f"[{stage_name.upper()}] Launching GPU {gpu_id} -> port {port}, "
            #     f"model: {settings['model']}"
            # )
            # print(f"[{stage_name.upper()}] Command: {' '.join(command)}")

            creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
            process = subprocess.Popen(
                command,
                cwd=str(REPO_ROOT),
                env=env,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                creationflags=creationflags,
            )

            server_handles.append(
                {
                    "process": process,
                    "log_file": log_file,
                    "base_url": base_url,
                    "port": port,
                    "gpu_id": gpu_id,
                }
            )
            endpoints.append(base_url)

        for handle in server_handles:
            # print(
            #     f"[{stage_name.upper()}] Waiting for service to be ready: "
            #     f"GPU {handle['gpu_id']} / port {handle['port']}"
            # )
            _wait_for_vllm_ready(handle["base_url"], settings["startup_timeout"])
            # print(
            #     f"[{stage_name.upper()}] Service is ready: "
            #     f"GPU {handle['gpu_id']} / port {handle['port']}"
            # )

        print(f"[{stage_name.upper()}] All vLLM services are ready.")
        return server_handles, endpoints
    except Exception:
        stop_vllm_servers(server_handles)
        raise


def stop_vllm_servers(server_handles: list[dict[str, Any]]) -> None:
    if server_handles:
        print("[VLLM] Shutting down current stage services...")
    for handle in server_handles:
        process = handle.get("process")
        if process is not None and process.poll() is None:
            # print(f"[VLLM] Stopping port {handle['port']} (GPU {handle['gpu_id']})")
            process.terminate()

    deadline = time.time() + 20
    for handle in server_handles:
        process = handle.get("process")
        if process is None:
            continue
        while process.poll() is None and time.time() < deadline:
            time.sleep(0.5)
        if process.poll() is None:
            process.kill()

    for handle in server_handles:
        log_file = handle.get("log_file")
        if log_file is not None and not log_file.closed:
            log_file.close()
    if server_handles:
        print("[VLLM] Current stage services have been stopped.")


def _run_single_entity_mllm(
    entity: dict[str, Any],
    entity_image_dir: Path,
    base_url: str,
    api_key: str,
    model_name: str,
) -> str:
    image_paths = _iter_existing_image_paths(entity_image_dir, entity.get("image_list", []))
    if not image_paths:
        return _fallback_desc_image(entity)

    try:
        client = _get_thread_client(base_url, api_key)
        return _generate_desc_image_with_mllm(client, model_name, entity, image_paths[0])
    except Exception:
        return _fallback_desc_image(entity)


def _run_single_entity_llm(
    entity: dict[str, Any],
    base_url: str,
    api_key: str,
    model_name: str,
) -> str:
    common_kwargs = {
        "entity_name": entity.get("entity_name", ""),
        "attr": entity.get("attr", ""),
        "instance": entity.get("instance", ""),
        "desc": entity.get("desc", ""),
        "desc_image": entity.get("desc_image", ""),
    }

    try:
        client = _get_thread_client(base_url, api_key)
        prompt_text = PROMPT1_entity_image_text_1sent.format(**common_kwargs)
        return _generate_with_llm(client, model_name, prompt_text)
    except Exception:
        return _normalize_text(entity.get("desc_image", ""))


def entity_aug_onegpu_mllm(args) -> list[dict[str, Any]]:
    entity_data_path, entity_image_dir, entity_output_mllm_path = _load_entity_args(args)
    base_entities = _load_json(entity_data_path)
    entities = _merge_entities(base_entities, entity_output_mllm_path)

    settings = _get_stage_settings(args, "mllm")
    base_url = _normalize_base_url(f"http://127.0.0.1:{settings['base_port']}")
    client = None

    for idx, entity in enumerate(tqdm(entities, desc="Entity MLLM")):
        if _normalize_text(entity.get("desc_image", "")):
            continue

        image_paths = _iter_existing_image_paths(entity_image_dir, entity.get("image_list", []))
        try:
            if image_paths:
                if client is None:
                    client = _build_vllm_client(base_url, settings["api_key"])
                entity["desc_image"] = _generate_desc_image_with_mllm(
                    client, settings["model"], entity, image_paths[0]
                )
            else:
                entity["desc_image"] = _fallback_desc_image(entity)
        except Exception:
            entity["desc_image"] = _fallback_desc_image(entity)

        if idx % settings["save_every"] == 0:
            _save_json(entity_output_mllm_path, entities)

    _save_json(entity_output_mllm_path, entities)
    return entities


def entity_aug_onegpu_llm(args) -> list[dict[str, Any]]:
    entity_output_mllm_path = _resolve_path(args.ent.entity_output_dir_mllm)
    entity_output_llm_path = _resolve_path(args.ent.entity_output_dir_mllm_llm)

    base_entities = _load_json(entity_output_mllm_path)
    entities = _merge_entities(base_entities, entity_output_llm_path)

    settings = _get_stage_settings(args, "llm")
    base_url = _normalize_base_url(f"http://127.0.0.1:{settings['base_port']}")
    client = None

    for idx, entity in enumerate(tqdm(entities, desc="Entity LLM")):
        entity.pop("desc_image_summary", None)
        if not _normalize_text(entity.get("desc_image", "")):
            entity["desc_image"] = _fallback_desc_image(entity)

        if _normalize_text(entity.get("desc_image_1sentence", "")):
            continue

        common_kwargs = {
            "entity_name": entity.get("entity_name", ""),
            "attr": entity.get("attr", ""),
            "instance": entity.get("instance", ""),
            "desc": entity.get("desc", ""),
            "desc_image": entity.get("desc_image", ""),
        }

        try:
            if client is None:
                client = _build_vllm_client(base_url, settings["api_key"])
            prompt_text = PROMPT1_entity_image_text_1sent.format(**common_kwargs)
            entity["desc_image_1sentence"] = _generate_with_llm(
                client, settings["model"], prompt_text
            )
        except Exception:
            entity["desc_image_1sentence"] = _normalize_text(entity.get("desc_image", ""))

        if idx % settings["save_every"] == 0:
            _save_json(entity_output_llm_path, entities)

    _save_json(entity_output_llm_path, entities)
    return entities


def entity_aug_parallel_mllm(args, endpoints: list[str]) -> list[dict[str, Any]]:
    entity_data_path, entity_image_dir, entity_output_mllm_path = _load_entity_args(args)
    base_entities = _load_json(entity_data_path)
    entities = _merge_entities(base_entities, entity_output_mllm_path)

    settings = _get_stage_settings(args, "mllm")
    future_to_index: dict[Any, int] = {}
    pending_count = 0
    skipped_count = 0

    for entity in entities:
        if _normalize_text(entity.get("desc_image", "")):
            skipped_count += 1
        else:
            pending_count += 1

    print(
        f"[MLLM] Entity augmentation started: {pending_count} pending, "
        f"{skipped_count} skipped, {len(endpoints)} endpoints, "
        f"{settings['workers']} workers."
    )

    with ThreadPoolExecutor(max_workers=settings["workers"]) as executor:
        for idx, entity in enumerate(entities):
            if _normalize_text(entity.get("desc_image", "")):
                continue

            endpoint = endpoints[idx % len(endpoints)]
            future = executor.submit(
                _run_single_entity_mllm,
                dict(entity),
                entity_image_dir,
                endpoint,
                settings["api_key"],
                settings["model"],
            )
            future_to_index[future] = idx

        completed = 0
        for future in tqdm(as_completed(future_to_index), total=len(future_to_index), desc="Entity MLLM"):
            idx = future_to_index[future]
            entities[idx]["desc_image"] = future.result()
            completed += 1
            if completed % settings["save_every"] == 0:
                _save_json(entity_output_mllm_path, entities)
                print(f"[MLLM] Completed {completed} items. Progress has been saved.")

    _save_json(entity_output_mllm_path, entities)
    print("[MLLM] Entity augmentation completed. All results have been saved.")
    return entities


def entity_aug_parallel_llm(args, endpoints: list[str]) -> list[dict[str, Any]]:
    entity_output_mllm_path = _resolve_path(args.ent.entity_output_dir_mllm)
    entity_output_llm_path = _resolve_path(args.ent.entity_output_dir_mllm_llm)

    base_entities = _load_json(entity_output_mllm_path)
    entities = _merge_entities(base_entities, entity_output_llm_path)

    settings = _get_stage_settings(args, "llm")
    future_to_index: dict[Any, int] = {}
    pending_count = 0
    skipped_count = 0

    for entity in entities:
        has_desc_image = _normalize_text(entity.get("desc_image", ""))
        has_one_sentence = _normalize_text(entity.get("desc_image_1sentence", ""))
        if not has_desc_image:
            pending_count += 1
        elif has_one_sentence:
            skipped_count += 1
        else:
            pending_count += 1

    print(
        f"[LLM] Entity information refinement started: {pending_count} pending, "
        f"{skipped_count} skipped, {len(endpoints)} endpoints, "
        f"{settings['workers']} workers."
    )

    with ThreadPoolExecutor(max_workers=settings["workers"]) as executor:
        for idx, entity in enumerate(entities):
            entity.pop("desc_image_summary", None)
            if not _normalize_text(entity.get("desc_image", "")):
                entity["desc_image"] = _fallback_desc_image(entity)

            if _normalize_text(entity.get("desc_image_1sentence", "")):
                continue

            endpoint = endpoints[idx % len(endpoints)]
            future = executor.submit(
                _run_single_entity_llm,
                dict(entity),
                endpoint,
                settings["api_key"],
                settings["model"],
            )
            future_to_index[future] = idx

        completed = 0
        for future in tqdm(as_completed(future_to_index), total=len(future_to_index), desc="Entity LLM"):
            idx = future_to_index[future]
            entities[idx]["desc_image_1sentence"] = future.result()
            completed += 1
            if completed % settings["save_every"] == 0:
                _save_json(entity_output_llm_path, entities)
                print(f"[LLM] Completed {completed} items. Progress has been saved.")

    _save_json(entity_output_llm_path, entities)
    print("[LLM] Entity information refinement completed. All results have been saved.")
    return entities
