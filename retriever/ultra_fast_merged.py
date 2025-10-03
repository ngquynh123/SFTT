#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ultra Fast Mode với Merged Model
Tốc độ tối đa với model đã merge
"""

import sys, subprocess

print("🚀🚀🚀 SIÊU NHANH MODE VỚI MERGED MODEL 🚀🚀🚀")
print("=" * 60)
print("🎯 Cấu hình tối ưu:")
print("   • Model: phogpt4b-ft-merged-manual")
print("   • Max tokens: 20 (cực ngắn)")
print("   • Temperature: 0.0 (greedy 100%)")
print("   • Context: 200 chars, 1 item")
print("   • Hits: 1 semantic + 1 BM25")
print("   • Device: CPU optimized")
print("⚡ Mục tiêu: Câu trả lời trong < 5 giây")
print("=" * 60)

# Args tối ưu tốc độ
args = [
    sys.executable, "main.py",
    "--device", "cpu",
    "--llm-max-new", "20",          # Cực ngắn
    "--llm-temp", "0.0",            # Greedy hoàn toàn
    "--topk", "1",                  # Chỉ 1 semantic hit
    "--bm25-topn", "1",             # Chỉ 1 BM25 hit
    "--sem-limit", "1",             # Chỉ 1 semantic result
    "--ctx-topn", "1",              # Chỉ 1 context item
    "--ctx-max-chars", "200",       # Context cực ngắn
    "--ctx-join-window", "0",       # Không join
    "--qa600-topn", "1",            # Chỉ 1 QA
    "--qa600-thr", "0.85",          # Threshold cao
    "--thr-warn", "0.2",            # Warning thấp
    "--min-query-chars", "2",       # Query ngắn OK
    "--min-keyword-len", "2",       # Keyword ngắn OK
    "--once"
] + sys.argv[1:]

try:
    result = subprocess.run(args, check=True)
    print(f"\n✅ Hoàn thành với exit code: {result.returncode}")
except subprocess.CalledProcessError as e:
    print(f"\n❌ Lỗi chạy script: {e}")
    sys.exit(e.returncode)
except KeyboardInterrupt:
    print(f"\n⚠️  Script bị dừng bởi người dùng")
    sys.exit(1)