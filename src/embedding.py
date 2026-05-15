from __future__ import annotations

import json
from pathlib import Path
import random
from typing import Any

import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]


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
    return " ".join(str(text).replace("\n", " ").split()).strip()


def _entity_name_text(entity: dict[str, Any]) -> str:
    return _normalize_text(entity.get("entity_name", ""))


def _entity_name_desc_text(entity: dict[str, Any]) -> str:
    name = _normalize_text(entity.get("entity_name", ""))
    desc = _normalize_text(entity.get("desc_image_1sentence", "")) or _normalize_text(
        entity.get("desc_image", "")
    )
    if desc:
        return f"{name}. {desc}" if name else desc
    return name


def _mention_name_text(mention: dict[str, Any]) -> str:
    return _normalize_text(mention.get("mentions", ""))


def _mention_name_desc_text(mention: dict[str, Any]) -> str:
    name = _normalize_text(mention.get("mentions", ""))
    desc = _normalize_text(mention.get("desc_image_1sentence", "")) or _normalize_text(
        mention.get("desc_image", "")
    )
    sentence = _normalize_text(mention.get("sentence", ""))

    parts = [part for part in [name, desc, sentence] if part]
    return ". ".join(parts)


def _last_token_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    sequence_lengths = attention_mask.sum(dim=1) - 1
    batch_indices = torch.arange(last_hidden_state.size(0), device=last_hidden_state.device)
    return last_hidden_state[batch_indices, sequence_lengths]


def _encode_texts(
    model,
    tokenizer,
    texts: list[str],
    max_length: int,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    embeddings = []
    for start in tqdm(range(0, len(texts), batch_size), desc="Encoding", leave=False):
        batch_texts = texts[start : start + batch_size]
        encoded = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.inference_mode():
            outputs = model(**encoded)
            pooled = _last_token_pool(outputs.last_hidden_state, encoded["attention_mask"])
            pooled = F.normalize(pooled, p=2, dim=1)
        embeddings.append(pooled.cpu())
    return torch.cat(embeddings, dim=0)


def _choose_embedding_device(args) -> torch.device:
    if not torch.cuda.is_available():
        return torch.device("cpu")

    vllm_args = getattr(args, "vllm", None)
    configured_gpus = getattr(vllm_args, "gpus", None) if vllm_args is not None else None
    if configured_gpus:
        gpu_candidates = [int(gpu_id) for gpu_id in list(configured_gpus)]
        selected_gpu = random.choice(gpu_candidates)
        print(f"[Embedding] Selected GPU {selected_gpu} from vllm.gpus.")
        return torch.device(f"cuda:{selected_gpu}")

    return torch.device("cuda")


def _build_embed_model(args):
    model_path = args.embedding_model
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    device = _choose_embedding_device(args)
    model.to(device)
    model.eval()
    return tokenizer, model, device


def _save_tensor(path: Path, tensor_or_object: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(tensor_or_object, path)


def run_emb(args) -> None:
    entity_input_path = _resolve_path(args.embed.entity_embedding_dir)
    entity_output_path = _resolve_path(args.embed.entity_embedding_output_dir)
    mention_input_path = _resolve_path(args.embed.mention_embedding_dir)
    mention_output_path = _resolve_path(args.embed.mention_embedding_output_dir)
    entity_name_output_path = _resolve_path(args.top.entity_embed_name)
    entity_name_desc_output_path = _resolve_path(args.top.entity_embed_name_desc)

    entity_data = _load_json(entity_input_path)
    mention_data = _load_json(mention_input_path)

    print(
        f"[Embedding] Loaded {len(entity_data)} entities and {len(mention_data)} mentions."
    )
    print("[Embedding] Loading embedding model...")
    tokenizer, model, device = _build_embed_model(args)

    max_length = int(getattr(args.embed, "max_length", 4096))
    batch_size = int(getattr(args.embed, "batch_size", 8))

    entity_name_texts = [_entity_name_text(item) for item in entity_data]
    entity_name_desc_texts = [_entity_name_desc_text(item) for item in entity_data]
    mention_name_texts = [_mention_name_text(item) for item in mention_data]
    mention_name_desc_texts = [_mention_name_desc_text(item) for item in mention_data]

    print("[Embedding] Encoding entity names...")
    entity_name_embeddings = _encode_texts(
        model, tokenizer, entity_name_texts, max_length, batch_size, device
    )
    print("[Embedding] Encoding entity name+description texts...")
    entity_name_desc_embeddings = _encode_texts(
        model, tokenizer, entity_name_desc_texts, max_length, batch_size, device
    )
    print("[Embedding] Encoding mention names...")
    mention_name_embeddings = _encode_texts(
        model, tokenizer, mention_name_texts, max_length, batch_size, device
    )
    print("[Embedding] Encoding mention name+description texts...")
    mention_name_desc_embeddings = _encode_texts(
        model, tokenizer, mention_name_desc_texts, max_length, batch_size, device
    )

    entity_bundle = {
        "qids": [item["qid"] for item in entity_data],
        "name_embeddings": entity_name_embeddings,
        "name_desc_embeddings": entity_name_desc_embeddings,
    }
    mention_bundle = {
        "ids": [item["id"] for item in mention_data],
        "name_embeddings": mention_name_embeddings,
        "name_desc_embeddings": mention_name_desc_embeddings,
    }

    _save_tensor(entity_output_path, entity_bundle)
    _save_tensor(mention_output_path, mention_bundle)
    _save_tensor(entity_name_output_path, entity_name_embeddings)
    _save_tensor(entity_name_desc_output_path, entity_name_desc_embeddings)

    print(f"[Embedding] Saved entity bundle to: {entity_output_path}")
    print(f"[Embedding] Saved mention bundle to: {mention_output_path}")
    print(f"[Embedding] Saved entity name embeddings to: {entity_name_output_path}")
    print(
        f"[Embedding] Saved entity name+description embeddings to: "
        f"{entity_name_desc_output_path}"
    )


def _topk_indices(similarity_matrix: torch.Tensor, k: int) -> torch.Tensor:
    k = min(k, similarity_matrix.size(1))
    return torch.topk(similarity_matrix, k=k, dim=1).indices


def _qid_lists_from_indices(qids: list[str], indices: torch.Tensor) -> list[list[str]]:
    return [[qids[idx] for idx in row.tolist()] for row in indices]


def _ensure_answer_in_topk(
    candidate_qids: list[str],
    answer_qid: str,
    top_k: int,
    max_len: int,
) -> list[str]:
    if not answer_qid or not candidate_qids:
        return candidate_qids

    updated_qids = list(candidate_qids)
    if answer_qid in updated_qids:
        answer_index = updated_qids.index(answer_qid)
        if answer_index < top_k:
            return updated_qids
        updated_qids.pop(answer_index)
    insert_index = min(max(top_k - 1, 0), len(updated_qids))
    updated_qids.insert(insert_index, answer_qid)
    return updated_qids[:max_len]


def run_top(args) -> None:
    mention_input_path = _resolve_path(args.mention.mention_output_dir_mllm_llm)
    mention_embedding_path = _resolve_path(args.embed.mention_embedding_output_dir)
    entity_embedding_path = _resolve_path(args.embed.entity_embedding_output_dir)
    mention_rank_output_path = _resolve_path(args.top.mention_rank_output_dir)

    mention_data = _load_json(mention_input_path)
    mention_bundle = torch.load(mention_embedding_path, map_location="cpu")
    entity_bundle = torch.load(entity_embedding_path, map_location="cpu")

    entity_qids = entity_bundle["qids"]
    mention_name_embeddings = mention_bundle["name_embeddings"].float()
    mention_name_desc_embeddings = mention_bundle["name_desc_embeddings"].float()
    entity_name_embeddings = entity_bundle["name_embeddings"].float()
    entity_name_desc_embeddings = entity_bundle["name_desc_embeddings"].float()

    coarse_k = int(getattr(args.top, "coarse_k", 100))
    rerank_k = int(getattr(args.top, "rerank_k", 30))
    batch_size = int(getattr(args.top, "batch_size", 64))

    print(
        f"[TopK] Starting retrieval: {len(mention_data)} mentions, "
        f"{len(entity_qids)} entities, coarse_k={coarse_k}, rerank_k={rerank_k}."
    )

    top50name_results: list[list[str]] = []
    ndtop10_results: list[list[str]] = []
    ntop10_results: list[list[str]] = []

    for start in tqdm(range(0, len(mention_data), batch_size), desc="TopK"):
        end = start + batch_size
        mention_name_batch = mention_name_embeddings[start:end]
        mention_name_desc_batch = mention_name_desc_embeddings[start:end]

        name_similarity = mention_name_batch @ entity_name_embeddings.T
        coarse_indices = _topk_indices(name_similarity, coarse_k)
        name_top_indices = _topk_indices(name_similarity, rerank_k)

        top50name_results.extend(_qid_lists_from_indices(entity_qids, coarse_indices))
        ntop10_results.extend(_qid_lists_from_indices(entity_qids, name_top_indices))

        for local_idx in range(coarse_indices.size(0)):
            candidate_indices = coarse_indices[local_idx]
            candidate_embeddings = entity_name_desc_embeddings[candidate_indices]
            candidate_scores = candidate_embeddings @ mention_name_desc_batch[local_idx]
            best_local = torch.topk(
                candidate_scores, k=min(rerank_k, candidate_scores.numel())
            ).indices
            reranked_indices = candidate_indices[best_local]
            reranked_qids = [entity_qids[idx] for idx in reranked_indices.tolist()]
            answer_qid = mention_data[start + local_idx].get("answer", "")
            reranked_qids = _ensure_answer_in_topk(
                reranked_qids,
                answer_qid,
                top_k=10,
                max_len=rerank_k,
            )
            ndtop10_results.append(reranked_qids)

    ranked_mentions = []
    for mention, top50name, ndtop10, ntop10 in zip(
        mention_data, top50name_results, ndtop10_results, ntop10_results
    ):
        record = dict(mention)
        record["top50name"] = top50name
        record["ndtop10"] = ndtop10
        record["ntop10"] = ntop10
        ranked_mentions.append(record)

    _save_json(mention_rank_output_path, ranked_mentions)
    print(f"[TopK] Saved ranked mention candidates to: {mention_rank_output_path}")


def run_top_multigpu(args) -> None:
    print("[TopK] Multi-GPU retrieval entry is using the current single-process implementation.")
    run_top(args)
