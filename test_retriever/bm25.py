# bm25_rrf_cli.py
import json, orjson, difflib, re, sys, hashlib
from pathlib import Path
from typing import List, Dict, Tuple
from collections import Counter
from langchain_community.retrievers import BM25Retriever
from langchain.schema import Document

DATA_FILES = [
    ("CTSC", Path("data/bm25/full_text_CTSC.json")),
    ("KTLX", Path("data/bm25/full_text_KTLX.json")),
    ("DDNLX", Path("data/bm25/full_text_DDNLX.json")),
]

TOPK_PER_SRC = 10   # lấy rộng ở mỗi nguồn để RRF hiệu quả
TOPN_PRINT   = 5    # số kết quả in ra cuối

_word_re = re.compile(r"\w+", flags=re.UNICODE)

def load_records(p: Path):
    raw = p.read_bytes()
    try: return orjson.loads(raw)
    except Exception: return json.loads(raw)

def normalize_text(s: str) -> str:
    # Chuẩn hóa nhẹ cho tiếng Việt (giữ dấu, chỉ làm sạch khoảng trắng/ký tự)
    s = (s or "").replace("\xa0", " ").strip()
    s = re.sub(r"\s+", " ", s)
    return s

def tokenize(s: str) -> List[str]:
    # token đơn giản, đủ dùng cho freq_score
    return [w.lower() for w in _word_re.findall(s.lower())]

def build_docs(rows, source_name) -> List[Document]:
    docs = []
    for r in rows:
        q = normalize_text(r.get("question") or "")
        a = normalize_text(r.get("answer") or "")
        search_text = normalize_text(r.get("search_text") or "")
        # Nếu có Q/A thì lắp dạng 2 dòng; nếu không có thì dùng search_text
        if q and a:
            content = f"Q: {q}\nA: {a}"
        else:
            content = search_text or (q or a)
        if not content:
            continue
        _id = r.get("_id")
        if not _id:
            # fallback id ổn định
            _id = hashlib.md5((source_name + "|" + content).encode("utf-8")).hexdigest()
        docs.append(Document(
            page_content=content,
            metadata={
                "_id": _id,
                "lesson": r.get("lesson"),
                "question": q if q else None,
                "source": source_name
            }
        ))
    return docs

def is_repeated_char(s: str) -> bool:
    return len(s) > 2 and s.isalpha() and all(c == s[0] for c in s)

def seq_sim(a: str, b: str) -> float:
    # 0..1
    return difflib.SequenceMatcher(None, a.lower().strip(), b.lower().strip()).ratio()

def freq_score(query: str, content: str) -> int:
    q_tokens = tokenize(query)
    c_tokens = tokenize(content)
    c_freq = Counter(c_tokens)
    return sum(c_freq.get(t, 0) for t in q_tokens)

def rrf_fuse(lists: List[List[Document]], k: int = 60) -> List[Tuple[Document, float]]:
    """
    lists: danh sách các rank list (mỗi item là list Document)
    k: hằng số RRF (60 mặc định OK)
    Trả về list (doc, score) đã fuse, sắp xếp giảm dần theo score
    """
    score_map: Dict[str, float] = {}
    doc_map: Dict[str, Document] = {}

    for ranked in lists:
        for rank, d in enumerate(ranked, start=1):
            _id = d.metadata.get("_id")
            if not _id:
                # dùng hash nội dung nếu thiếu
                _id = hashlib.md5(d.page_content.encode("utf-8")).hexdigest()
                d.metadata["_id"] = _id
            doc_map[_id] = d
            score_map[_id] = score_map.get(_id, 0.0) + 1.0 / (k + rank)

    fused = [(doc_map[_id], s) for _id, s in score_map.items()]
    fused.sort(key=lambda x: x[1], reverse=True)
    return fused

def main():
    # 1) Tạo 3 retriever riêng
    per_source_docs: Dict[str, List[Document]] = {}
    for name, path in DATA_FILES:
        if not path.exists():
            print(f"⚠️  File không tồn tại: {path}")
            continue
        rows = load_records(path)
        docs = build_docs(rows, name)
        per_source_docs[name] = docs
        print(f"✅ Nạp {len(docs):,} docs cho nguồn {name}")

    retrievers: Dict[str, BM25Retriever] = {}
    for src, docs in per_source_docs.items():
        if docs:
            r = BM25Retriever.from_documents(docs)
            r.k = TOPK_PER_SRC
            retrievers[src] = r

    if not retrievers:
        print("❌ Không có retriever nào được tạo. Kiểm tra đường dẫn DATA_FILES.")
        sys.exit(1)

    # 2) CLI loop
    while True:
        query = input("\nNhập câu hỏi (hoặc 'q' để thoát): ").strip()
        if query.lower() == 'q':
            print("Tạm biệt!")
            break
        if is_repeated_char(query):
            print("Không tìm thấy câu trả lời phù hợp hoặc không hiểu câu hỏi.")
            continue

        # 3) Lấy top-k ở mỗi nguồn + RRF
        ranked_lists = []
        for src, rtv in retrievers.items():
            try:
                ranked_lists.append(rtv.invoke(query))
            except Exception as e:
                print(f"⚠️ Lỗi truy vấn {src}: {e}")

        fused = rrf_fuse(ranked_lists, k=60)

        # 4) Hậu xếp hạng nhẹ (re-rank nội bộ):
        #    - Ưu tiên độ giống với câu hỏi gốc (nếu doc có field question)
        #    - Ưu tiên tần suất token xuất hiện trong content
        def post_score(doc: Document) -> float:
            ques = doc.metadata.get("question") or ""
            content = doc.page_content
            s1 = seq_sim(query, ques) if ques else 0.0
            s2 = freq_score(query, content)  # số nguyên
            return 0.65 * s1 + 0.35 * (min(s2, 8) / 8.0)  # kẹp freq để không “nổ”

        # Lấy một rổ rộng từ RRF rồi re-rank nhẹ
        pool = [d for d, _ in fused[:30]]
        pool_scored = [(d, post_score(d)) for d in pool]
        pool_scored.sort(key=lambda x: x[1], reverse=True)

        # 5) In kết quả
        print(f"\nTop kết quả cho: {query}")
        shown = 0
        seen = set()
        for d, s in pool_scored:
            _id = d.metadata.get("_id")
            if _id in seen:
                continue
            seen.add(_id)
            src = d.metadata.get("source")
            les = d.metadata.get("lesson")
            ques = d.metadata.get("question") or "(không có question)"
            preview = d.page_content.replace("\n", " ")
            print(f"- [{src}] lesson={les} | id={_id[:8]} | sim={s:.3f}")
            print(f"  Q: {ques}")
            print(f"  {preview[:160]}…\n")
            shown += 1
            if shown >= TOPN_PRINT:
                break

if __name__ == "__main__":
    main()