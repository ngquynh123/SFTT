#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script chạy SIÊU NHANH V2 - Tối ưu cực đại cho tốc độ
Phiên bản cải tiến với cấu hình tốt nhất cho câu trả lời tức thì
"""

import subprocess
import sys
import os

def run_ultra_fast_v2():
    """Chạy main.py với cấu hình SIÊU NHANH V2"""
    
    # Đường dẫn tới main.py
    main_script = os.path.join(os.path.dirname(__file__), "main.py")
    
    cmd = [
        sys.executable, main_script,
        
        # Device setting - CPU ổn định
        "--device", "cpu",
        
        # LLM settings - CỰC NHANH
        "--llm-max-new", "16",            # Cực ngắn 16 tokens
        "--llm-temp", "0.0",              # Greedy hoàn toàn (không random)
        
        # Retrieval settings - Tối thiểu tuyệt đối
        "--topk", "1",                    # Chỉ 1 semantic hit
        "--bm25-topn", "1",               # Chỉ 1 BM25 hit  
        "--sem-limit", "1",               # Chỉ 1 semantic result
        
        # Context settings - Cực ngắn
        "--ctx-topn", "1",                # Chỉ 1 context item
        "--ctx-max-chars", "200",         # Context cực ngắn 200 chars
        "--ctx-join-window", "0",         # Không join chunks
        
        # QA600 settings - Nhanh nhất
        "--qa600-topn", "1",              # Chỉ 1 QA item
        "--qa600-thr", "0.7",             # Threshold thấp để nhanh
        
        # Thresholds - Loose để nhanh
        "--thr-warn", "0.2",              # Warning threshold rất thấp
        
        # Query processing - Minimal
        "--min-query-chars", "2",         # Chỉ 2 ký tự
        "--min-keyword-len", "1",         # Keywords 1 ký tự
        
        # Debug - Tắt print để nhanh hơn
        "--no-print-prompt",              # Không in prompt ra
    ]
    
    # Thêm query nếu có argument
    if len(sys.argv) > 1:
        cmd.extend(["--once", " ".join(sys.argv[1:])])
    
    # Hiển thị thông tin cấu hình
    print("⚡⚡⚡ KHỞI ĐỘNG CHẾ ĐỘ SIÊU NHANH V2 ⚡⚡⚡")
    print("=" * 65)
    print("🚀 Cấu hình tối ưu cực đại cho tốc độ:")
    print("   💨 Max tokens: 16 (cực ngắn)")
    print("   🎯 Temperature: 0.0 (greedy 100%)")
    print("   📦 Context: 200 chars, 1 item")
    print("   🔍 Hits: 1 semantic + 1 BM25")
    print("   💻 Device: CPU only")
    print("   🚫 Không in prompt (tăng tốc)")
    print("   ⚡ Không join context chunks")
    print("🎯 Mục tiêu: Câu trả lời trong < 5 giây")
    print("⚠️  Lưu ý: Câu trả lời rất ngắn, phù hợp câu hỏi đơn giản")
    print("=" * 65)
    print()
    
    # Chạy script
    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\n⏹️  Đã dừng bằng Ctrl+C")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Lỗi chạy script: {e}")
        return 1
    except Exception as e:
        print(f"\n❌ Lỗi: {e}")
        return 1
    
    return 0

def show_usage():
    """Hiển thị hướng dẫn sử dụng"""
    print("🚀 ULTRA FAST MODE V2 - Siêu tốc độ")
    print("Cách sử dụng:")
    print("  python ultra_fast_v2.py                    # Chế độ chat")
    print("  python ultra_fast_v2.py \"Biển P.130 là gì?\"  # Chạy 1 lần")
    print()
    print("Đặc điểm V2:")
    print("  🚀 Tốc độ cực đại (< 5 giây)")
    print("  💨 Câu trả lời cực ngắn (16 tokens)")
    print("  📦 Context tối thiểu (200 chars)")
    print("  🎯 Phù hợp cho câu hỏi đơn giản, tra cứu nhanh")
    print("  🚫 Không hiển thị prompt để tăng tốc")
    print()
    print("So sánh với các chế độ khác:")
    print("  • ultra_fast.py: 24 tokens, 400 chars context")
    print("  • ultra_fast_v2.py: 16 tokens, 200 chars context (nhanh nhất)")
    print("  • fast_mode.py: 48 tokens, 800 chars context (cân bằng)")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] in ["-h", "--help", "help"]:
        show_usage()
    else:
        exit(run_ultra_fast_v2())