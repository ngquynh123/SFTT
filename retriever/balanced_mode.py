#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script chạy với tốc độ VỪA PHẢI - Cân bằng giữa tốc độ và chất lượng
Đảm bảo câu trả lời đúng và đầy đủ, không quá nhanh
"""

import subprocess
import sys
import os

def run_balanced_mode():
    """Chạy main.py với cấu hình cân bằng tốc độ và chất lượng"""
    
    # Đường dẫn tới main.py
    main_script = os.path.join(os.path.dirname(__file__), "main.py")
    
    cmd = [
        sys.executable, main_script,
        
        # Device setting - CPU ổn định
        "--device", "cpu",
        
        # LLM settings - CÂN BẰNG cho kết quả tốt
        "--llm-max-new", "64",            # Đủ dài cho câu trả lời đầy đủ
        "--llm-temp", "0.05",             # Hơi cao hơn để linh hoạt
        
        # Retrieval settings - Đủ để có kết quả tốt
        "--topk", "3",                    # 3 semantic hits (đủ context)
        "--bm25-topn", "3",               # 3 BM25 hits
        "--sem-limit", "4",               # 4 semantic results
        
        # Context settings - Đủ thông tin
        "--ctx-topn", "3",                # 3 context items
        "--ctx-max-chars", "900",         # 900 chars - đủ chi tiết
        "--ctx-join-window", "1",         # Join 1 chunk để có context liền mạch
        
        # QA600 settings - Threshold hợp lý
        "--qa600-topn", "3",              # 3 QA items
        "--qa600-thr", "0.88",            # Threshold cao để đảm bảo chính xác
        
        # Thresholds - Hợp lý cho độ tin cậy
        "--thr-warn", "0.5",              # Warning threshold tiêu chuẩn
        
        # Query processing - Tiêu chuẩn
        "--min-query-chars", "4",         # 4 ký tự tối thiểu
        "--min-keyword-len", "2",         # Keywords 2 ký tự
    ]
    
    # Thêm query nếu có argument
    if len(sys.argv) > 1:
        cmd.extend(["--once", " ".join(sys.argv[1:])])
    
    # Hiển thị thông tin cấu hình
    print("⚖️  KHỞI ĐỘNG CHẾ ĐỘ CÂN BẰNG ⚖️")
    print("=" * 55)
    print("🎯 Cấu hình cân bằng tốc độ và chất lượng:")
    print("   📝 Max tokens: 64 (đủ cho câu trả lời đầy đủ)")
    print("   🌡️  Temperature: 0.05 (hơi linh hoạt)")
    print("   📦 Context: 900 chars, 3 items + join chunks")
    print("   🔍 Hits: 3 semantic + 3 BM25")
    print("   💻 Device: CPU stable")
    print("   🎚️  Threshold: 0.88 (cao cho độ chính xác)")
    print("🎯 Mục tiêu: Câu trả lời chính xác trong ~8-12 giây")
    print("✅ Ưu điểm: Kết quả đúng, đầy đủ, tốc độ hợp lý")
    print("=" * 55)
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
    print("⚖️  BALANCED MODE - Chế độ cân bằng")
    print("Cách sử dụng:")
    print("  python balanced_mode.py                     # Chế độ chat")
    print("  python balanced_mode.py \"Biển P.130 là gì?\"  # Chạy 1 lần")
    print()
    print("Đặc điểm chế độ cân bằng:")
    print("  ⚖️  Cân bằng tốc độ và chất lượng")
    print("  📝 Câu trả lời đầy đủ (64 tokens)")
    print("  📦 Context đủ chi tiết (900 chars)")
    print("  🎯 Độ chính xác cao (threshold 0.88)")
    print("  ⏱️  Thời gian: ~8-12 giây")
    print()
    print("So sánh các chế độ:")
    print("  • ultra_fast_v2.py: Siêu nhanh (~3-5s) nhưng câu trả lời ngắn")
    print("  • balanced_mode.py: Cân bằng (~8-12s) - kết quả đúng và đầy đủ")
    print("  • main.py: Chất lượng cao (~15-20s) - chi tiết nhất")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] in ["-h", "--help", "help"]:
        show_usage()
    else:
        exit(run_balanced_mode())