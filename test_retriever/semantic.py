#Score = cosine similarity càng gần 1 càng tốt 

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional, Set

import numpy as np

# ==== Prompt builder for guarded QA ====
from rag_hybrid.guard_prompt import render_guard_string, format_sources

try:
    from sentence_transformers import SentenceTransformer
except Exception as e:
    SentenceTransformer = None
    _st_error = e

# ------------------------- utilities -------------------------

def _read_json(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        # allow {"items": [...]} style
        if "items" in data and isinstance(data["items"], list):
            return data["items"]
        # or {"data": [...]} style
        if "data" in data and isinstance(data["data"], list):
            return data["data"]
        # fallback to single-item list
        return [data]
    return data


def _best_text(d: Dict[str, Any]) -> str:
    # Prefer explicit combined search_text if available
    for key in ("search_text", "text", "content"):
        val = d.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()
    # Some datasets have a title/tittle field
    title = d.get("title") or d.get("tittle")
    q = d.get("question")
    a = d.get("answer") or d.get("answer_text")
    # If we have Q/A, combine
    if isinstance(q, str) and isinstance(a, str):
        base = (q.strip() + "\n" + a.strip()).strip()
        if isinstance(title, str) and title.strip():
            return (title.strip() + "\n" + base).strip()
        return base
    # Else, fallbacks
    if isinstance(q, str) and q.strip():
        if isinstance(title, str) and title.strip():
            return (title.strip() + "\n" + q.strip()).strip()
        return q.strip()
    if isinstance(title, str) and title.strip():
        return title.strip()
    return ""


def _meta(d: Dict[str, Any]) -> Dict[str, Any]:
    keep = {k: d.get(k) for k in [
        "_id", "id", "doc_id", "subject", "subject_id", "topic", "grade", "lesson", "lesson_id",
        "source", "file", "chunk_id", "title", "tittle"
    ] if k in d}
    return keep


def _normalize(v: np.ndarray) -> np.ndarray:
    # supports (n,d) or (d,)
    if v.ndim == 1:
        nrm = np.linalg.norm(v)
        return v / nrm if nrm > 0 else v
    nrm = np.linalg.norm(v, axis=1, keepdims=True)
    nrm[nrm == 0.0] = 1.0
    return v / nrm

# ------------------------- index loader -------------------------

def load_corpus(data_root: Path, channels: List[str]) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    vectors: List[np.ndarray] = []
    records: List[Dict[str, Any]] = []

    for ch in channels:
        dir_path = data_root / ch
        if not dir_path.exists():
            print(f"[WARN] Channel folder not found: {dir_path}")
            continue
        for p in sorted(dir_path.glob("*.json")):
            items = _read_json(p)
            for it in items:
                # Case A: flat records with 'embedding' or 'embed'
                if isinstance(it, dict) and ("embedding" in it or "embed" in it):
                    emb = it.get("embedding") if it.get("embedding") is not None else it.get("embed")
                    try:
                        vec = np.array(emb, dtype=np.float32)
                    except Exception:
                        continue
                    rec = {
                        "text": _best_text(it),
                        "meta": _meta(it),
                        "file": p.name,
                        "channel": ch,
                    }
                    vectors.append(vec)
                    records.append(rec)
                    continue

                # Case B: lesson-style doc with 'chunks': [ { text, embed } ]
                if isinstance(it, dict) and isinstance(it.get("chunks"), list):
                    parent_meta = _meta(it)
                    for idx_c, chobj in enumerate(it["chunks"], 1):
                        emb2 = chobj.get("embedding") if chobj.get("embedding") is not None else chobj.get("embed")
                        if emb2 is None:
                            continue
                        try:
                            vec2 = np.array(emb2, dtype=np.float32)
                        except Exception:
                            continue
                        rec2 = {
                            "text": _best_text({**it, **chobj}) or chobj.get("text", ""),
                            "meta": {**parent_meta, **_meta(chobj), "chunk_index": idx_c},
                            "file": p.name,
                            "channel": ch,
                        }
                        vectors.append(vec2)
                        records.append(rec2)
                    continue

                # Unknown shape: skip
                continue
        print(f"[OK] Loaded {len(vectors)} vectors so far from channel '{ch}'.")

    if not vectors:
        raise RuntimeError("No vectors found. Make sure your JSON has an 'embedding' or 'embed' field.")

    mat = np.vstack(vectors)  # (N, D)
    return mat, records

# ------------------------- model -------------------------

def load_query_encoder(model_id: str, device: str = "cpu") -> Any:
    if SentenceTransformer is None:
        raise RuntimeError(
            "sentence-transformers not available. Install it: pip install sentence-transformers\n"
            f"Original import error: {_st_error}"
        )
    print(f"[Model] Loading query encoder: {model_id} (device={device})")
    model = SentenceTransformer(model_id, device=device)
    return model

# ------------------------- search -------------------------

def search(query: str, encoder: Any, index: np.ndarray, records: List[Dict[str, Any]], topk: int = 8) -> List[Dict[str, Any]]:
    q = encoder.encode([query], batch_size=1, normalize_embeddings=True)
    # Ensure corpus is normalized for cosine via dot-product
    idx = index
    if not hasattr(search, "_normed"):
        search._normed = True
        idx[:] = _normalize(idx)
    sims = np.dot(idx, q[0].astype(np.float32))  # (N,)
    topk = int(min(topk, len(records)))
    top_idx = np.argpartition(-sims, topk - 1)[:topk]
    top_idx = top_idx[np.argsort(-sims[top_idx])]

    out: List[Dict[str, Any]] = []
    for i in top_idx:
        r = records[i]
        out.append({
            "score": float(sims[i]),
            "text": r["text"],
            "file": r["file"],
            "channel": r["channel"],
            "meta": r["meta"],
        })
    return out


def max_score(results: List[Dict[str, Any]]) -> float:
    return max((r.get("score", 0.0) for r in results), default=0.0)


def mean_score(results: List[Dict[str, Any]]) -> float:
    if not results:
        return 0.0
    return float(np.mean([r.get("score", 0.0) for r in results]))

# ------------------------- pretty print -------------------------

def print_results(results: List[Dict[str, Any]]):
    for rank, r in enumerate(results, 1):
        head = f"#{rank:02d}  score={r['score']:.4f}  [{r['channel']}/{r['file']}]"
        print(head)
        m = r.get("meta") or {}
        if m:
            metas = [f"{k}={m[k]}" for k in ["subject","topic","grade","lesson","chunk_id","id","_id","lesson_id","title","tittle"] if k in m]
            if metas:
                print("     meta:", ", ".join(map(str, metas)))
        txt = r.get("text") or ""
        if txt:
            lines = txt.strip().splitlines()
            preview = lines[0] if lines else ""
            if len(preview) > 180:
                preview = preview[:180] + "…"
            print("     text:", preview)
        print()

# ------------------------- Evaluation -------------------------

def _load_eval_items(path: Path) -> List[Dict[str, Any]]:
    """Load eval set from JSONL or JSON.
    Schema per item: {"query": str, "relevants": [str, ...], optional "answer": str, optional "unanswerable": bool}
    The IDs should correspond to your document/ chunk IDs (e.g., meta._id).
    """
    text = path.read_text(encoding="utf-8").strip()
    items: List[Dict[str, Any]] = []
    if not text:
        return items
    if "\n" in text and text.lstrip().startswith("{") is False:
        # JSONL
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
        return items
    obj = json.loads(text)
    if isinstance(obj, list):
        return obj
    return [obj]


def _result_ids(r: Dict[str, Any]) -> Set[str]:
    m = r.get("meta", {}) or {}
    out: Set[str] = set()
    for k in ("_id", "id", "doc_id", "lesson_id", "chunk_id"):
        v = m.get(k)
        if isinstance(v, (str, int)):
            out.add(str(v))
    # also allow file+chunk index key
    if "file" in r and "meta" in r and isinstance(r["meta"].get("chunk_index"), int):
        out.add(f"{r['file']}::{r['meta']['chunk_index']}")
    return out


def _ranklist(results: List[Dict[str, Any]], relevant_ids: Set[str]) -> List[int]:
    ranks = []
    for i, r in enumerate(results, 1):
        ids = _result_ids(r)
        if ids & relevant_ids:
            ranks.append(i)
    return ranks


def _recall_at_k(ranks: List[int], num_rel: int, k: int) -> float:
    hits = sum(1 for r in ranks if r <= k)
    return hits / max(1, num_rel)


def _precision_at_k(ranks: List[int], k: int) -> float:
    hits = sum(1 for r in ranks if r <= k)
    return hits / max(1, k)


def _mrr_at_k(ranks: List[int], k: int) -> float:
    best = min([r for r in ranks if r <= k], default=None)
    return 0.0 if best is None else 1.0 / float(best)


def _ndcg_at_k(ranks: List[int], num_rel: int, k: int) -> float:
    # binary gains
    dcg = 0.0
    for r in ranks:
        if r <= k:
            dcg += 1.0 / np.log2(r + 1.0)
    ideal = 0.0
    for i in range(1, min(num_rel, k) + 1):
        ideal += 1.0 / np.log2(i + 1.0)
    return 0.0 if ideal == 0.0 else dcg / ideal


def _embed_texts(encoder: Any, texts: List[str]) -> np.ndarray:
    vecs = encoder.encode(texts, batch_size=8, normalize_embeddings=True)
    return np.array(vecs, dtype=np.float32)


def evaluate(encoder: Any, index: np.ndarray, records: List[Dict[str, Any]], eval_items: List[Dict[str, Any]], topk: int, acc_threshold: float = 0.60, score_threshold: float = 0.35) -> Dict[str, Any]:
    agg = {"queries": 0, "recall@k": 0.0, "precision@k": 0.0, "mrr@k": 0.0, "ndcg@k": 0.0,
           "answerable_acc": 0.0, "answer_sim": 0.0}
    for it in eval_items:
        q = it.get("query", "").strip()
        rels = set(str(x) for x in (it.get("relevants") or []))
        gold = (it.get("answer") or "").strip()
        unans = bool(it.get("unanswerable", False))
        if not q:
            continue
        res = search(q, encoder, index, records, topk=topk)
        ranks = _ranklist(res, rels)
        agg["queries"] += 1
        agg["recall@k"] += _recall_at_k(ranks, len(rels), topk)
        agg["precision@k"] += _precision_at_k(ranks, topk)
        agg["mrr@k"] += _mrr_at_k(ranks, topk)
        agg["ndcg@k"] += _ndcg_at_k(ranks, len(rels), topk)

        # Answerability accuracy (detect nonsense queries)
        mx = max_score(res)
        is_answerable_pred = mx >= score_threshold
        is_answerable_gold = not unans
        agg["answerable_acc"] += 1.0 if (is_answerable_pred == is_answerable_gold) else 0.0

        # Optional semantic answer accuracy if gold is provided
        if gold and res:
            # naive generated answer = top-1 text (truncate)
            gen = (res[0].get("text") or "")[:512]
            emb = _embed_texts(encoder, [gold, gen])
            sim = float(np.dot(emb[0], emb[1]))  # cosine since normalized
            agg["answer_sim"] += sim
        else:
            agg["answer_sim"] += 0.0
    n = max(1, agg["queries"])
    for k in ["recall@k", "precision@k", "mrr@k", "ndcg@k", "answerable_acc", "answer_sim"]:
        agg[k] = agg[k] / n
    return agg

# ------------------------- CLI / REPL -------------------------

def main():
    ap = argparse.ArgumentParser(description="Semantic retrieval tester for pre-embedded corpora (+prompt +eval +confidence)")
    ap.add_argument("--data-root", type=str, default="embed_data", help="Root folder containing channels (dialogue, lesson)")
    ap.add_argument("--channels", type=str, nargs="+", default=["dialogue","lesson"], help="Which subfolders to load")
    ap.add_argument("--model-id", type=str, default=os.getenv("EMB_MODEL", "AITeamVN/Vietnamese_Embedding"), help="HF model id for query encoding (should match corpus embeddings)")
    ap.add_argument("--device", type=str, default=os.getenv("DEVICE","cpu"), choices=["cpu"], help="Encoder device")
    ap.add_argument("--topk", type=int, default=4, help="Top-k results for display")
    ap.add_argument("--once", type=str, default=None, help="Run a single query then exit")
    # Confidence gating
    ap.add_argument("--threshold", type=float, default=0.35, help="Min max-score to consider answerable")
    # Evaluation
    ap.add_argument("--eval-file", type=str, default=None, help="Path to eval queries (JSONL or JSON)")
    ap.add_argument("--eval-topk", type=int, default=10, help="k for eval metrics (Recall/Precision/MRR/nDCG)")
    ap.add_argument("--eval-out", type=str, default=None, help="Optional path to save eval summary JSON")
    # Answer accuracy eval (optional)
    ap.add_argument("--acc-threshold", type=float, default=0.60, help="Cosine threshold to count generated answer as correct vs gold (reported as answer_sim only)")
    args = ap.parse_args()

    data_root = Path(args.data_root)
    index, records = load_corpus(data_root, args.channels)
    print(f"[Index] Corpus size: {index.shape[0]} vectors, dim={index.shape[1]}")

    encoder = load_query_encoder(args.model_id, device=args.device)

    # Evaluation mode
    if args.eval_file:
        eval_path = Path(args.eval_file)
        items = _load_eval_items(eval_path)
        if not items:
            print(f"[EVAL] Empty or invalid eval file: {eval_path}")
        else:
            summary = evaluate(encoder, index, records, items, topk=args.eval_topk, acc_threshold=args.acc_threshold, score_threshold=args.threshold)
            print("\n[EVAL] Summary (k=%d):" % args.eval_topk)
            for k, v in summary.items():
                print(f" - {k}: {v:.4f}" if isinstance(v, float) else f" - {k}: {v}")
            if args.eval_out:
                Path(args.eval_out).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
                print(f"[EVAL] Saved to {args.eval_out}")

    if args.once:
        res = search(args.once, encoder, index, records, topk=args.topk)
        mx, mn = max_score(res), mean_score(res)
        print_results(res)
        print(f"[Confidence] max={mx:.4f}  mean@topk={mn:.4f}  threshold={args.threshold:.2f}")
        if mx < args.threshold:
            print("\nKẾT LUẬN: Chưa đủ căn cứ để kết luận từ KB")
            return
        print("Nguồn:")
        print(format_sources(res))
        print("\n=== Prompt vào LLM (copy/paste sang PhoGPT) ===")
        print(render_guard_string(args.once, res))
        return

    # REPL mode
    print("\nEnter a query (blank to quit). Examples:")
    print(" - 'Biển P.130 là gì?'")

    while True:
        try:
            q = input("Q> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break
        if not q:
            break
        res = search(q, encoder, index, records, topk=args.topk)
        mx, mn = max_score(res), mean_score(res)
        print_results(res)
        print(f"[Confidence] max={mx:.4f}  mean@topk={mn:.4f}  threshold={args.threshold:.2f}")
        if mx < args.threshold:
            print("KẾT LUẬN: Chưa đủ căn cứ để kết luận từ KB\n")
            continue
        print("Nguồn:")
        print(format_sources(res))
        print("\n=== Prompt vào LLM ===")
        print(render_guard_string(q, res))


if __name__ == "__main__":
    main()
