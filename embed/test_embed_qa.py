# chạy code python test_embed_qa.py data/dialogue/CTSC.json

import json
import sys
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer

# ====== Load model Vi-bge ======
MODEL_ID = "AITeamVN/Vietnamese_Embedding"
model = SentenceTransformer(MODEL_ID)

def embed_texts(texts):
    """Sinh embedding từ danh sách text"""
    return model.encode(
        texts,
        normalize_embeddings=True,  # cosine-friendly
        batch_size=64,
        show_progress_bar=False
    ).astype(np.float32)

def build_qa_text(item: dict) -> str:
    q = (item.get("question") or "").strip()
    a = (item.get("answer") or "").strip()
    if q and a:
        return f"Q: {q}\nA: {a}"
    return q or a  # nếu thiếu một trong hai thì dùng phần còn lại

def main():
    # ====== Nhận đường dẫn file qua terminal ======
    if len(sys.argv) < 2:
        print("❌ Bạn chưa nhập đường dẫn file JSON.")
        print("👉 Cách dùng: python test_em_qa.py <path_toi_file_json>")
        sys.exit(1)

    input_path = Path(sys.argv[1])
    if not input_path.exists():
        print(f"❌ Không tìm thấy file: {input_path}")
        sys.exit(1)

    # ====== Load dữ liệu ======
    try:
        with open(input_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"❌ Lỗi đọc JSON ở {input_path}: {e}")
        sys.exit(1)

    if not isinstance(data, list):
        print("❌ File JSON phải là một mảng các bản ghi QA.")
        sys.exit(1)

    # ====== Gom text để embed theo batch ======
    idx_refs = []   # lưu index của item hợp lệ
    texts = []
    for i, item in enumerate(data):
        text = build_qa_text(item)
        if text:
            idx_refs.append(i)
            texts.append(text)

    if not texts:
        print("⚠️ Không tìm thấy 'question'/'answer' hợp lệ để embed.")
    else:
        # ====== Tạo embeddings theo batch ======
        embs = embed_texts(texts)

        # ====== Gán embed về lại từng item ======
        for i, emb in zip(idx_refs, embs):
            data[i]["embed"] = emb.tolist()

        # Với item không có text → set None
        for i, item in enumerate(data):
            if "embed" not in item:
                item["embed"] = None

    # ====== Ghi ra JSON (giữ nguyên cấu trúc) ======
    out_dir = Path("embed_data/dialogue/")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{input_path.stem}_with_emb.json"

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    ok = sum(1 for it in data if it.get("embed") is not None)
    print(f"✅ Done! Đã tạo file {out_path} — {ok}/{len(data)} bản ghi có embedding.")

if __name__ == "__main__":
    main()
