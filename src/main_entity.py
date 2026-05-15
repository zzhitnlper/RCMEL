import sys
from pathlib import Path

from omegaconf import OmegaConf

from entity_aug import (
    entity_aug_parallel_llm,
    entity_aug_parallel_mllm,
    start_vllm_stage,
    stop_vllm_servers,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def setup_parser(config_path):
    return OmegaConf.load(str(config_path))


def resolve_config_path(argv: list[str]) -> Path:
    if len(argv) <= 1:
        return REPO_ROOT / "config" / "wikidiverse.yaml"

    raw_arg = argv[1]
    raw_path = Path(raw_arg)

    if raw_path.suffix in {".yaml", ".yml"}:
        if raw_path.is_absolute():
            return raw_path
        return REPO_ROOT / raw_path

    return REPO_ROOT / "config" / f"{raw_arg}.yaml"


if __name__ == "__main__":
    config_path = resolve_config_path(sys.argv)
    args = setup_parser(config_path)

    print("[Main] Entity augmentation pipeline started.")
    print("[Main] Stage 1/2: Starting MLLM services...")
    mllm_handles, mllm_endpoints = start_vllm_stage(args, "mllm")
    try:
        print("[Main] MLLM services are ready. Starting entity augmentation.")
        entity_aug_parallel_mllm(args, mllm_endpoints)
    finally:
        stop_vllm_servers(mllm_handles)

    print("[Main] MLLM entity augmentation completed.")
    print("[Main] Stage 2/2: Starting LLM services...")

    llm_handles, llm_endpoints = start_vllm_stage(args, "llm")
    try:
        print("[Main] LLM services are ready. Starting entity information refinement.")
        entity_aug_parallel_llm(args, llm_endpoints)
    finally:
        stop_vllm_servers(llm_handles)

    print("[Main] LLM information refinement completed.")
    print("[Main] Entity augmentation pipeline finished.")
