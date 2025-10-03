#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
INSTANT MODE - Câu trả lời tức thì
Tối ưu cho tốc độ cực đại
"""

import sys, subprocess, os, time

print("🚀🚀🚀 INSTANT MODE - SIÊU TỐC ĐỘ 🚀🚀🚀")
print("=" * 70)
print("⚡ Tối ưu cực đại:")
print("   🔥 Max tokens: 16 (cực ngắn)")
print("   🎯 Temperature: 0.0 (greedy 100%)")
print("   📏 Context: 150 chars (tối thiểu)")
print("   🎪 Hits: 1 semantic ONLY")
print("   💨 No BM25, no fusion, no join")
print("   🚫 Skip tất cả optimization không cần thiết")
print("⏱️  Mục tiêu: < 3 giây")
print("=" * 70)

start_time = time.time()

# Cấu hình SIÊU NHANH
args = [
    sys.executable, "main.py",
    
    # Model tối ưu
    "--device", "cpu",
    "--llm-max-new", "16",           # CỰC NGẮN
    "--llm-temp", "0.0",             # Greedy pure
    
    # Retrieval tối thiểu
    "--topk", "1",                   # CHỈ 1 hit
    "--bm25-topn", "0",              # TẮT BM25 
    "--sem-limit", "1",              # CHỈ 1 result
    
    # Context cực ngắn  
    "--ctx-topn", "1",               # CHỈ 1 context
    "--ctx-max-chars", "150",        # CỰC NGẮN
    "--ctx-join-window", "0",        # TẮT join
    
    # QA600 nhanh
    "--qa600-topn", "1",
    "--qa600-thr", "0.95",           # Threshold CAO để nhanh
    
    # Thresholds loose
    "--thr-warn", "0.1",             # Rất thấp
    "--min-query-chars", "2",        # Tối thiểu
    "--min-keyword-len", "2",
    
    # Tắt debug
    "--no-print-prompt",
]

# Thêm question
if len(sys.argv) > 1:
    args.extend(["--once", " ".join(sys.argv[1:])])
else:
    args.extend(["--once", "test"])

print(f"⚡ Command: {' '.join(args[2:])}")
print("🏃 Đang chạy...")

try:
    result = subprocess.run(args, check=True, capture_output=False)
    
    end_time = time.time()
    duration = end_time - start_time
    
    print(f"\n✅ Hoàn thành trong {duration:.2f} giây")
    
    if duration > 5:
        print("⚠️  Chậm hơn mong đợi - có thể do:")
        print("   • Model chưa cached")
        print("   • Embeddings chưa sẵn sàng") 
        print("   • System đang bận")
        
except subprocess.CalledProcessError as e:
    print(f"\n❌ Lỗi: {e}")
    sys.exit(1)
except KeyboardInterrupt:
    print(f"\n⏹️  Đã dừng")
    sys.exit(1)