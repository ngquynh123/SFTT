#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script chạy SIÊU NHANH với cấu hình tối ưu cực đại
Dành cho những lúc cần câu trả lời ngay lập tức
"""

import subprocess
import sys
import os

def run_ultra_fast():
    """Chạy main.py với cấu hình SIÊU NHANH"""
    
    # Đường dẫn tới main.py
    main_script = os.path.join(os.path.dirname(__file__), "main.py")
    
    cmd = [
        sys.executable, main_script,
        
        # Device setting
        "--device", "cpu",                # Force CPU stable mode
        
        # LLM settings - SIÊU NHANH
        "--llm-max-new", "24",            # Cực ngắn cho tốc độ tối đa
        "--llm-temp", "0.0",              # Greedy decode hoàn toàn
        
        # Retrieval settings - tối thiểu
        "--topk", "1",                    # Chỉ 1 semantic hit
        "--bm25-topn", "1",               # Chỉ 1 BM25 hit  
        "--sem-limit", "2",               # Giới hạn tối thiểu
        
        # Context settings - cực ngắn
        "--ctx-topn", "2",                # 2 context items
        "--ctx-max-chars", "400",         # Context cực ngắn
        "--ctx-join-window", "0",         # Không join chunks
        
        # QA600 settings - nhanh nhất
        "--qa600-topn", "1",              # Chỉ 1 QA item
        "--qa600-thr", "0.8",             # Threshold thấp để nhanh
        
        # Thresholds - loose để nhanh
        "--thr-warn", "0.3",              # Warning threshold thấp
        
        # Query processing - minimal
        "--min-query-chars", "3",         # Ít ký tự hơn
        "--min-keyword-len", "2",         # Keywords ngắn hơn
    ]
    
    # Thêm query nếu có argument
    if len(sys.argv) > 1:
        cmd.extend(["--once", " ".join(sys.argv[1:])])
    
    # Hiển thị thông tin cấu hình
    print("⚡⚡⚡ KHỞI ĐỘNG CHỂ ĐỘ SIÊU NHANH ⚡⚡⚡")
    print("=" * 60)
    print("🚀 Cấu hình cực đại cho tốc độ:")
    print("   • Max tokens: 24 (cực ngắn)")
    print("   • Temperature: 0.0 (greedy hoàn toàn)")
    print("   • Context: 400 chars, 2 items")
    print("   • Hits: 1 semantic + 1 BM25")
    print("   • Device: CPU only")
    print("   • Không có tối ưu hóa context")
    print("🎯 Mục tiêu: Câu trả lời trong < 10 giây")
    print("⚠️  Lưu ý: Câu trả lời có thể rất ngắn")
    print("=" * 60)
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
    print("🚀 ULTRA FAST MODE")
    print("Cách sử dụng:")
    print("  python ultra_fast.py                    # Chế độ chat")
    print("  python ultra_fast.py \"Biển P.130 là gì?\"  # Chạy 1 lần")
    print()
    print("Đặc điểm:")
    print("  • Tốc độ tối đa")
    print("  • Câu trả lời cực ngắn (24 tokens)")
    print("  • Context tối thiểu")
    print("  • Phù hợp cho câu hỏi đơn giản")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] in ["-h", "--help", "help"]:
        show_usage()
    else:
        exit(run_ultra_fast())