#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, os, re, unicodedata, shutil, textwrap, json
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from datetime import datetime

from load_data import load_corpus
from model_semantic import SemanticSearcher
from bm25 import BM25
from model_llm import build_llm, generate_answer
# removed: from chat_memory import build_memory_chain


# ================= Pretty printing =================
def _term_width(default: int = 100) -> int:
    try: return max(60, shutil.get_terminal_size((default, 20)).columns)
    except Exception: return default

def _box(title: str, body: str, width: int = None) -> str:
    width = width or _term_width()
    line = "═" * (width - 2)
    top, bot = f"╔{line}╗", f"╚{line}╝"
    head, sep = f"║ {title.strip()}".ljust(width - 1) + "║", "╟" + ("─" * (width - 2)) + "╢"
    rows = []
    for paragraph in (body or "").split("\n"):
        if not paragraph.strip():
            rows.append("║".ljust(width - 1) + "║"); continue
        for w in (textwrap.wrap(paragraph, width=width - 4) or [""]):
            rows.append(("║ " + w).ljust(width - 1) + "║")
    return "\n".join([top, head, sep] + rows + [bot])

def _clean_answer(s: str) -> str:
    s = (s or "").strip()
    
    # Loại bỏ các ký tự encoding lỗi ngay từ đầu
    import re
    s = re.sub(r'[^\x00-\x7F\u00C0-\u024F\u1E00-\u1EFF]', '', s)  # Chỉ giữ ASCII + Vietnamese
    
    # Loại bỏ "cộng cộng" và patterns tương tự
    s = re.sub(r'\bcộng\s+cộng\b', '', s, flags=re.IGNORECASE)
    s = re.sub(r'\b(\w+)\s+\1\b', r'\1', s)  # Loại bỏ từ lặp: "cộng cộng" -> "cộng"
    
    # Loại bỏ ký tự replacement và lạ
    s = re.sub(r'[�□◊▪▫•‰…]', '', s)  # Ký tự replacement
    s = re.sub(r'[^\w\s\u00C0-\u024F\u1E00-\u1EFF.,!?():;"\'-]', '', s)
    
    # Loại bỏ ký tự lặp bất thường
    s = re.sub(r'(.)\1{3,}', r'\1', s)  # aa aa aa -> a
    
    # Loại bỏ nhiều newlines liên tiếp
    while "\n\n\n" in s: s = s.replace("\n\n\n", "\n\n")
    
    # Làm sạch khoảng trắng
    s = re.sub(r'\s+', ' ', s).strip()
    
    # Loại bỏ các từ ngữ không có nghĩa
    meaningless_patterns = [
        r'\b[a-zA-Z]{1,2}\b',  # Từ quá ngắn
        r'\b\d{5,}\b',         # Số quá dài
        r'\b[^\w\s]{2,}\b'     # Ký tự đặc biệt liên tiếp
    ]
    for pattern in meaningless_patterns:
        s = re.sub(pattern, '', s)
    
    s = re.sub(r'\s+', ' ', s).strip()
    
    # Nếu quá ngắn hoặc chỉ chứa ký tự lặp
    if len(s) < 5 or len(set(s.replace(' ', ''))) < 3:
        return "Thông tin chưa đủ để trả lời."
    
    return s

def _preview_line(record: Dict[str, Any]) -> str:
    qa = record.get("qa") or {}; ch = record.get("channel")
    if ch in ("dialogue","qa600"):
        parts = []
        if qa.get("question"): parts.append("Q: " + qa["question"])
        if qa.get("answer"):   parts.append("A: " + qa["answer"])
        txt = " | ".join(parts)
    else:
        txt = (record.get("text") or "").strip().splitlines()
        txt = txt[0] if txt else ""
    return (txt[:180] + "…") if len(txt) > 180 else txt

def _print_hits(title: str, records: List[Dict[str, Any]], hits: List[Dict[str, Any]]):
    lines = []
    for i, h in enumerate(hits, 1):
        r = records[h["rid"]]
        src = h.get("src", ""); src_tag = f" [{src}]" if src else ""
        score_show = h.get("score", h.get("score_raw", 0.0))
        lines.append(f"#{i:02d}  score={score_show:.5f}{src_tag}  [{r['channel']}/{r['file']}]")
        lines.append("     " + _preview_line(r))
    print(_box(title, "\n".join(lines) if lines else "(không có)"))

# ================= Cleans & helpers =================
_word = re.compile(r"\w+", flags=re.UNICODE)
def _tok(s: str) -> List[str]:
    if not s: return []
    return [w.lower() for w in _word.findall(s)]

def clean_text(s: str) -> str:
    if not s: return ""
    s = unicodedata.normalize("NFC", s).lower()
    s = re.sub(r"[^a-z0-9\u00c0-\u024f\u1e00-\u1eff\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _extract_q_and_a(r: Dict[str, Any]) -> Tuple[Optional[str], Optional[str]]:
    qa = r.get("qa") or {}
    q = qa.get("question") or r.get("question")
    a = qa.get("answer")   or r.get("answer")
    q = q.strip() if isinstance(q, str) and q.strip() else None
    a = a.strip() if isinstance(a, str) and a.strip() else None
    return q, a

def _extract_text(r: Dict[str, Any]) -> str:
    for key in ["text","content","search_text","body","chunk_text","paragraph"]:
        v = r.get(key)
        if isinstance(v, str) and v.strip():
            return v.strip()
    q, a = _extract_q_and_a(r)
    if q or a: return ("Q: " + (q or "")) + (("\nA: " + a) if a else "")
    return ""

# ================= Signals (log) =================
_PAT = [
    r"\bđáp án\s*[:\-]", r"\btrả lời\s*[:\-]", r"\bkết luận\s*[:\-]",
    r"\bchọn đáp án\b", r"\bđúng là\b", r"\blà\s*[:\-]\s", r"\bđịnh nghĩa\s*[:\-]"
]
def _has_direct_answer(r: Dict[str, Any]) -> bool:
    qa = r.get("qa") or {}; ans = qa.get("answer") or r.get("answer")
    if isinstance(ans, str) and ans.strip(): return True
    txt = _extract_text(r)
    if not isinstance(txt, str) or not txt: return False
    for p in _PAT:
        if re.search(p, txt, flags=re.IGNORECASE): return True
    return False

def _context_relevance_score(question: str, context: str) -> float:
    """Đánh giá mức độ liên quan giữa câu hỏi và context"""
    if not context or not question:
        return 0.0
    
    question_words = set(clean_text(question).split())
    context_words = set(clean_text(context).split())
    
    if not question_words:
        return 0.0
    
    # Tính overlap ratio
    overlap = len(question_words & context_words)
    return overlap / len(question_words)

def _evidence_score(records: List[Dict[str, Any]], hits: List[Dict[str, Any]]) -> float:
    if not hits: return 0.0
    return sum(1 for h in hits if _has_direct_answer(records[h["rid"]])) / max(1, len(hits))

# ================= Fusion / context helpers =================
def rrf_fusion(sem_hits, bm_hits, k: int = 60):
    rank_sem = {h["rid"]: i for i, h in enumerate(sorted(sem_hits, key=lambda x: -x["score"]), start=1)}
    rank_bm  = {h["rid"]: i for i, h in enumerate(sorted(bm_hits, key=lambda x: -x["score"]), start=1)}
    rids = set(list(rank_sem.keys()) + list(rank_bm.keys()))
    fused = []
    for rid in rids:
        r1 = rank_sem.get(rid, 10**9); r2 = rank_bm.get(rid, 10**9)
        fused.append({"rid": rid, "score": 1.0/(k+r1) + 1.0/(k+r2), "src": "rrf"})
    fused.sort(key=lambda x: -x["score"])
    return fused

def dedup_by_source(records, hits, max_per_source: int = 1):
    seen, kept = {}, []
    for h in hits:
        r = records[h["rid"]]; key = f"{r.get('channel','?')}::{r.get('file','?')}"
        if seen.get(key, 0) < max_per_source:
            kept.append(h); seen[key] = seen.get(key, 0) + 1
    return kept

_DEF_PAT = re.compile(r"\b(là gì|định nghĩa|ý nghĩa|ký hiệu|đáp án)\b", re.IGNORECASE)
def sort_by_definition_first(records, hits):
    def score(h): return 1 if _DEF_PAT.search((records[h["rid"]].get("text") or "").lower()) else 0
    return sorted(hits, key=score, reverse=True)

def join_adjacent_chunks(records, hits, window: int = 1, max_chars: int = 900) -> List[str]:
    out, taken = [], set()
    for h in hits:
        rid = h["rid"]
        if rid in taken: continue
        r = records[rid]; file_id = r.get("file")
        pieces = [r.get("text") or ""]
        for off in range(1, window+1):
            for rrid in [rid-off, rid+off]:
                if 0 <= rrid < len(records) and rrid not in taken:
                    rr = records[rrid]
                    if rr.get("file") == file_id:
                        pieces.append(rr.get("text") or ""); taken.add(rrid)
        merged = " ".join([p.strip() for p in pieces if p and p.strip()])
        out.append(merged[:max_chars] + ("…" if len(merged) > max_chars else ""))
        taken.add(rid)
    return out

def _build_context_block_from_hits(ctx_hits: List[str], topn_ctx: int, max_chars_per_item: int, max_total_chars: int) -> str:
    blocks, total = [], 0
    for idx, text in enumerate(ctx_hits[:topn_ctx], 1):
        head = f"[{idx}]"
        body = (text or "").replace("\r"," ").replace("\t"," ").strip()
        if len(body) > max_chars_per_item: body = body[:max_chars_per_item] + "…"
        block = head + "\n" + body
        if total + len(block) > max_total_chars: break
        blocks.append(block); total += len(block)
    return "\n\n".join(blocks)

def _build_bm25_hints_block(records: List[Dict[str,Any]], bm25_hits: List[Dict[str,Any]], max_chars_per_item: int = 160) -> str:
    lines = []
    for i, h in enumerate(bm25_hits, 1):
        r = records[h["rid"]]
        q, a = _extract_q_and_a(r)
        txt = (q or a or _extract_text(r)).replace("\n", " ")
        if len(txt) > max_chars_per_item: txt = txt[:max_chars_per_item] + "…"
        lines.append(f"[B{i}] {txt}")
    return "\n".join(lines)

def _build_bm25_only_block(records: List[Dict[str, Any]], bm25_hits: List[Dict[str, Any]], topn: int = 3, max_chars: int = 700) -> str:
    blocks = []
    for i, h in enumerate(bm25_hits[:topn], 1):
        r = records[h["rid"]]
        q, a = _extract_q_and_a(r)
        txt = ("Q: " + (q or "") + ("\nA: " + a if a else "")).strip() if (q or a) else (_extract_text(r) or "").strip()
        if len(txt) > max_chars: txt = txt[:max_chars] + "…"
        blocks.append(f"[B{i}]\n{txt}")
    return "\n\n".join(blocks)

def _print_top3_bm25_QA(records: List[Dict[str, Any]], bm25_hits: List[Dict[str, Any]]):
    lines = []
    for i, h in enumerate(bm25_hits[:3], 1):
        r = records[h["rid"]]; q, a = _extract_q_and_a(r)
        if q or a:
            prev_q = (q or "").replace("\n", " ")
            prev_a = (a or "").replace("\n", " ")
            if len(prev_q) > 160: prev_q = prev_q[:160] + "…"
            if len(prev_a) > 160: prev_a = prev_a[:160] + "…"
            lines.append(f"#{i:02d} [{r['channel']}/{r['file']}]")
            lines.append(f"  Q: {prev_q}")
            lines.append(f"  A: {prev_a}")
        else:
            txt = (_extract_text(r) or "").replace("\n", " ")
            if len(txt) > 180: txt = txt[:180] + "…"
            lines.append(f"#{i:02d} [{r['channel']}/{r['file']}]")
            lines.append(f"  {txt}")
    body = "\n".join(lines) if lines else "(không có)"
    print(_box("3 GỢI Ý TỪ BM25 (Q&A THAM KHẢO)", body))

# ================= Main =================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--embed-root", type=str, default="embed_data")
    ap.add_argument("--channels", nargs="+", default=["dialogue","lesson"])

    # qa600.json (BM25-only)
    ap.add_argument("--qa600-path", type=str, default="data/qa600.json")
    ap.add_argument("--qa600-topn", type=int, default=3)
    ap.add_argument("--qa600-thr", type=float, default=0.90)

    # Semantic (embed)
    ap.add_argument("--model-id", type=str, default=os.getenv("EMB_MODEL","AITeamVN/Vietnamese_Embedding"))
    ap.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda", "auto"])

    # Retrieval - Giảm để tăng tốc độ
    ap.add_argument("--topk", type=int, default=2)           # Giảm xuống 2
    ap.add_argument("--bm25-topn", type=int, default=2)      # Giảm xuống 2 
    ap.add_argument("--bm25-k1", type=float, default=1.5)
    ap.add_argument("--bm25-b", type=float, default=0.75)
    ap.add_argument("--sem-limit", type=int, default=3)      # Giảm xuống 3

    # LLM - Tối ưu cho tốc độ (greedy decode)
    ap.add_argument("--llm-off", action="store_true")
    ap.add_argument("--llm-model-id", type=str, default=os.getenv("LLM_MODEL_ID", r"D:\AI.LLM-khanh-no_rrf\models\PhoGPT-4B"))  # Use original model
    ap.add_argument("--llm-temp", type=float, default=float(os.getenv("LLM_TEMP","0.0")))  # 0.0 cho greedy pure
    ap.add_argument("--llm-max-new", type=int, default=32, help="Max tokens sinh ra (cân bằng tốc độ vs chất lượng)")  # Tăng lên 32
    ap.add_argument("--use-onnx", action="store_true", help="Sử dụng ONNX Runtime để tăng tốc")  # ONNX option

    # Context - Tối ưu cho tốc độ và tránh garbled text
    ap.add_argument("--ctx-topn", type=int, default=1)       # Giảm xuống 1 để tránh nhiễu
    ap.add_argument("--ctx-max-chars", type=int, default=300) # Giảm xuống 300 chars
    ap.add_argument("--ctx-join-window", type=int, default=0) # Tắt join để nhanh hơn

    # Thresholds
    ap.add_argument("--thr-warn", type=float, default=0.5)

    # Query gate
    ap.add_argument("--min-query-chars", type=int, default=4)
    ap.add_argument("--min-keyword-len", type=int, default=2)

    # Debug
    ap.add_argument("--no-print-prompt", action="store_true")
    ap.add_argument("--max-preview-len", type=int, default=180)

    # One shot
    ap.add_argument("--once", type=str, default=None)

    args = ap.parse_args()

    # 1) Load embedded corpus (semantic)
    index, records = load_corpus(root=args.embed_root, channels=tuple(args.channels))
    sem = SemanticSearcher(index, model_id=args.model_id, device=args.device)

    # 2) BM25 over ALL embedded records
    bm25_docs_all = []
    for r in records:
        q, a = _extract_q_and_a(r)
        bm25_docs_all.append("\n".join([x for x in [q, a] if x]) if (q or a) else _extract_text(r))
    bm25_all = BM25(bm25_docs_all, k1=args.bm25_k1, b=args.bm25_b)

    # 3) Load qa600.json (BM25-only)
    qa_records, qa_docs = [], []
    if args.qa600_path and os.path.isfile(args.qa600_path):
        try:
            with open(args.qa600_path, "r", encoding="utf-8") as f:
                items = json.load(f)
            if isinstance(items, dict): items = [items]
            for it in items:
                q = (it.get("question") or "").strip()
                a = (it.get("answer")   or it.get("answer_text") or "").strip()
                if not (q or a): continue
                qa_records.append({
                    "text": (q + ("\n" + a if a else "")).strip(),
                    "qa": {"question": q or None, "answer": a or None},
                    "meta": {"order": it.get("order"), "importance": it.get("importance", False)},
                    "file": os.path.basename(args.qa600_path),
                    "channel": "qa600",
                })
                qa_docs.append("\n".join([x for x in [q, a] if x]))
            print(f"[OK] Loaded {len(qa_records)} QA from {args.qa600_path} (BM25-first).")
        except Exception as e:
            print(f"[WARN] Cannot read qa600 at {args.qa600_path}: {e}")
    else:
        print(f"[WARN] qa600 not found at: {args.qa600_path}")
    bm25_qa = BM25(qa_docs, k1=args.bm25_k1, b=args.bm25_b) if qa_docs else None

    # 4) LLM (Transformers hoặc ONNX Runtime)
    llm = None
    if not args.llm_off:
        try:
            # Kiểm tra environment variable cho ONNX
            use_onnx = args.use_onnx or os.getenv("USE_ONNX", "0") == "1"
            
            llm = build_llm(
                args.llm_model_id, 
                None, 
                args.llm_temp, 
                device="cpu",
                use_onnx=use_onnx,
                max_new_tokens=args.llm_max_new
            )
        except Exception as e:
            print(_box("LLM", f"Không khởi tạo được model tại {args.llm_model_id}. Lỗi: {e}"))

    def run_query(q: str):
        q_proc = clean_text(q)
        toks = [t for t in _tok(q_proc) if len(t) >= args.min_keyword_len]
        if not ((len(q_proc) >= args.min_query_chars) and len(toks) >= 1):
            print(_box("RETRIEVAL", f"Truy vấn quá ngắn/không rõ ràng.\nSau khi làm sạch: “{q_proc}”")); print(); return

        setup = "\n".join([
            f"Truy vấn (đã làm sạch): {q_proc}",
            f"QA600: path={args.qa600_path}, thr={args.qa600_thr:.2f}, topn={args.qa600_topn}",
            f"Semantic topk={args.topk} | BM25(all) topn={args.bm25_topn}",
        ])
        print(_box("THIẾT LẬP", setup))

        # ---------- 1) BM25 trên QA600 trước ----------
        if bm25_qa is not None:
            qa_hits = bm25_qa.search(q_proc, topk=max(1, args.qa600_topn), min_keyword_len=args.min_keyword_len)
            _print_hits("KẾT QUẢ BM25 (qa600)", qa_records, qa_hits)
            if qa_hits and qa_hits[0]["score"] >= args.qa600_thr:
                top = qa_hits[0]; rec = qa_records[top["rid"]]
                ans = (rec.get("qa") or {}).get("answer") or _extract_text(rec) or ""
                head = f"[FAST QA600] score_top1={top['score']:.3f} ≥ {args.qa600_thr:.2f}\nNguồn: [{rec['channel']}/{rec['file']}]"
                print(_box("TRẢ LỜI (CHẮC CHẮN)", head + "\n\n" + _clean_answer(ans))); print(); return

        # ---------- 2) Semantic + BM25(all) ----------
        se_hits = sem.search(q_proc, topk=args.topk)
        bm_hits = bm25_all.search(q_proc, topk=args.bm25_topn, min_keyword_len=args.min_keyword_len)
        mx = max([h["score"] for h in se_hits], default=0.0)
        ev = _evidence_score(records, se_hits)

        print(_box("TÍN HIỆU TIN CẬY", f"SEM_max={mx:.4f} | Evidence≈{ev:.2f}"))
        _print_hits("KẾT QUẢ SEMANTIC", records, se_hits)
        _print_hits("KẾT QUẢ BM25 ", records, bm_hits)

        if args.llm_off or llm is None:
            print(_box("LLM", "LLM đang tắt hoặc chưa khởi tạo.")); print(); return

        # --- semantic >= 0.5 => chắc chắn ---
        if mx >= 0.5:
            fused = rrf_fusion(se_hits, bm_hits, k=60)
            fused = sort_by_definition_first(records, fused)
            fused = dedup_by_source(records, fused, max_per_source=1)
            ctx_ids = fused[:args.sem_limit]
            if args.ctx_join_window > 0:
                blocks = join_adjacent_chunks(records, ctx_ids, window=args.ctx_join_window, max_chars=900)
            else:
                blocks = [ (records[h["rid"]].get("text") or "")[:900] for h in ctx_ids ]
            context_text = _build_context_block_from_hits(blocks, args.ctx_topn, 700, args.ctx_max_chars)

            # chèn BM25-hints vào context để "bm25 prompt"
            hints = _build_bm25_hints_block(records, bm_hits, max_chars_per_item=160)
            if hints.strip():
                context_text = context_text + "\n\n[BÍ KÍP BM25]\n" + hints

            if not args.no_print_prompt:
                print(_box("CONTEXT (SEMANTIC)", context_text or "(rỗng)"))

            try:
                answer = generate_answer(llm, q, context_text, max_new_tokens=args.llm_max_new, temperature=args.llm_temp)
            except Exception as e:
                print(_box("TRẢ LỜI", f"[LLM ERROR] {e}")); return

            warn = "" if mx >= args.thr_warn else "(Lưu ý: căn cứ chưa mạnh/độ tự tin chưa cao.)\n"
            print(_box("TRẢ LỜI (CHẮC CHẮN)", warn + (_clean_answer(answer) or "Thông tin chưa đủ."))); print(); return

        # --- semantic < 0.5 ---
        if not bm_hits:
            print(_box("KẾT LUẬN", "Không tìm thấy căn cứ đủ mạnh (semantic < 0.5) và không có gợi ý từ BM25.")); print(); return

        # Fallback: dùng BM25 tham khảo + in 3 Q/A
        _print_top3_bm25_QA(records, bm_hits)
        bm25_block = _build_bm25_only_block(records, bm_hits, topn=3, max_chars=700)
        
        # THÊM: Kiểm tra context có liên quan không
        relevance = _context_relevance_score(q, bm25_block)
        print(f"🎯 Context relevance: {relevance:.2f}")
        
        if relevance < 0.2:  # Context quá không liên quan
            print("❌ Context không liên quan, trả lời từ kiến thức chung...")
            try:
                from model_llm import build_prompt_direct
                direct_prompt = build_prompt_direct(q)
                answer = llm.generate(direct_prompt, stop=None)
                answer = _clean_answer(answer)
                if answer and len(answer) > 10:
                    msg = "(Trả lời từ kiến thức chung - vui lòng kiểm chứng.)\n" + answer
                    print(_box("TRẢ LỜI (KIẾN THỨC CHUNG)", msg)); print(); return
            except Exception as e:
                print(f"❌ Lỗi kiến thức chung: {e}")

        if not args.no_print_prompt:
            print(_box("CONTEXT (BM25-ONLY)", bm25_block or "(rỗng)"))

        try:
            answer = generate_answer(llm, q, bm25_block, max_new_tokens=args.llm_max_new, temperature=max(0.1, args.llm_temp*0.9))
            
            # THÊM: Nếu answer không có ý nghĩa, thử trả lời trực tiếp từ kiến thức
            if not answer or answer.strip() in ["Thông tin không đủ rõ ràng để trả lời.", "Thông tin chưa đủ để trả lời chính xác.", "Xin lỗi, tôi không thể trả lời câu hỏi này."]:
                print("🔄 Context không phù hợp, thử trả lời từ kiến thức chung...")
                from model_llm import build_prompt_direct
                direct_prompt = build_prompt_direct(q)
                answer = llm.generate(direct_prompt, stop=None)  # Không dùng STOP_TOKENS để tự nhiên hơn
                answer = _clean_answer(answer)
                if answer and len(answer) > 10:  # Nếu có câu trả lời tốt
                    msg = "(Trả lời từ kiến thức chung - vui lòng kiểm chứng.)\n" + answer
                    print(_box("TRẢ LỜI (KIẾN THỨC CHUNG)", msg)); print(); return
            
        except Exception as e:
            print(_box("TRẢ LỜI (THAM KHẢO)", f"[LLM ERROR] {e}")); return

        msg = "(Tham khảo từ BM25 — vui lòng kiểm chứng.)\n" + (_clean_answer(answer) or "Thông tin chưa đủ để kết luận chính xác.")
        print(_box("TRẢ LỜI (THAM KHẢO)", msg)); print()

    # Run once / REPL
    if args.once:
        run_query(args.once); return

    print(_box("HƯỚNG DẪN", "Nhập câu hỏi (Enter để thoát). Ví dụ: 'Biển P.130 là gì?'"))
    while True:
        try:
            q = input("Q> ").strip()
        except (EOFError, KeyboardInterrupt):
            print(); break
        if not q: break
        run_query(q)

if __name__ == "__main__":
    main()
   