

import argparse, json, sys
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
from sentence_transformers import SentenceTransformer


def load_json_any(path: Path) -> List[Dict[str, Any]]:
    """Đọc file .json hoặc .jsonl"""
    if path.suffix.lower() == ".jsonl":
        return [json.loads(line) for line in path.open("r", encoding="utf-8") if line.strip()]
    data = json.load(path.open("r", encoding="utf-8"))
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        if isinstance(data.get("items"), list):
            return data["items"]
        if isinstance(data.get("data"), list):
            return data["data"]
        return [data]
    return []


def build_qa_text(item: Dict[str, Any]) -> str:
    q = (item.get("question") or "").strip()
    a = (item.get("answer") or "").strip()
    return f"Q: {q}\nA: {a}" if q and a else q or a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input", type=str, help="Đường dẫn file JSON/JSONL (có question/answer)")
    ap.add_argument("--model-id", type=str, default="AITeamVN/Vietnamese_Embedding")
    ap.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    args = ap.parse_args()

    path = Path(args.input)
    rows = load_json_any(path)
    if not rows:
        print("❌ File trống hoặc không hợp lệ."); sys.exit(1)

    questions, qas, metas = [], [], []
    for it in rows:
        q, a = (it.get("question") or "").strip(), (it.get("answer") or "").strip()
        if not q or not a: 
            continue
        questions.append(q)
        qas.append(build_qa_text(it))
        metas.append(it)

    if not questions:
        print("❌ Không có mẫu hợp lệ."); sys.exit(1)

    model = SentenceTransformer(args.model_id, device=args.device)
    emb_q  = model.encode(questions, normalize_embeddings=True, show_progress_bar=True)
    emb_qa = model.encode(qas,        normalize_embeddings=True, show_progress_bar=True)

    sims = np.sum(emb_q * emb_qa, axis=1)  # dot = cosine vì đã normalize
    sims = np.clip(sims, -1.0, 1.0)

    max_sim = float(np.max(sims))
    min_sim = float(np.min(sims))
    mean_sim = float(np.mean(sims))

    worst_idx = int(np.argmin(sims))
    worst_q   = metas[worst_idx].get("question")

    print("====== KẾT QUẢ ======")
    print(f"Score cao nhất : {max_sim:.4f}")
    print(f"Score thấp nhất: {min_sim:.4f}")
    print(f"Score trung bình: {mean_sim:.4f}")
    print("\nCâu hỏi có score thấp nhất:")
    print(f"  {worst_q}")


if __name__ == "__main__":
    main()
