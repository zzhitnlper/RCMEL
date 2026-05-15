import sys
from pathlib import Path

from omegaconf import OmegaConf

from decoding import infer_decoding_multigpu, infer_decoding_onegpu
from entity_aug import start_vllm_stage, stop_vllm_servers

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
    provider = str(getattr(args.decoding, "provider", "vllm")).lower()

    print("[Main] Final decoding pipeline started.")
    if provider == "gpt":
        print("[Main] GPT decoding mode enabled. Skipping local vLLM startup.")
        if getattr(args.decoding, "use_multigpu", True):
            infer_decoding_multigpu(args)
        else:
            infer_decoding_onegpu(args)
    else:
        print("[Main] Stage 1/1: Starting LLM services...")
        llm_handles, llm_endpoints = start_vllm_stage(args, "llm")
        try:
            print("[Main] LLM services are ready. Starting final candidate ranking.")
            if getattr(args.decoding, "use_multigpu", True):
                infer_decoding_multigpu(args, llm_endpoints)
            else:
                infer_decoding_onegpu(args)
        finally:
            stop_vllm_servers(llm_handles)

    print("[Main] Final decoding pipeline finished.")
