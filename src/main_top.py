import sys
from pathlib import Path

from omegaconf import OmegaConf

from embedding import run_emb, run_top, run_top_multigpu

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


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

    print("[Main] Retrieval pipeline started.")

    # print("[Main] Stage 1/2: Building embeddings...")
    # run_emb(args)
    # print("[Main] Embedding stage completed.")

    print("[Main] Stage 2/2: Running top-k retrieval...")
    if getattr(args.top, "use_multigpu", True):
        run_top_multigpu(args)
    else:
        run_top(args)

    print("[Main] Top-k retrieval completed.")
    print("[Main] Retrieval pipeline finished.")
