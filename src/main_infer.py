import sys
from pathlib import Path

from omegaconf import OmegaConf

from entity_aug import start_vllm_stage, stop_vllm_servers
from infer import infer_kc_multigpu, infer_kc_onegpu

REPO_ROOT = Path(__file__).resolve().parents[1]


def setup_parser(config_path: Path):
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

    print("[Main] Knowledge contrast pipeline started.")
    print("[Main] Stage 1/1: Starting LLM services...")
    llm_handles, llm_endpoints = start_vllm_stage(args, "llm")
    try:
        print("[Main] LLM services are ready. Starting knowledge contrast generation.")
        if getattr(args.infer_kc, "use_multigpu", True):
            infer_kc_multigpu(args, llm_endpoints)
        else:
            infer_kc_onegpu(args)
    finally:
        stop_vllm_servers(llm_handles)

    print("[Main] Knowledge contrast pipeline finished.")
