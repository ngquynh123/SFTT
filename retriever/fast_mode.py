#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Chạy hệ thống với cài đặt tối ưu tốc độ
"""
import subprocess
import sys
import os

def run_fast():
    """Chạy main.py với cấu hình tối ưu cho tốc độ"""
    
    # Đường dẫn tới main.py
    main_script = os.path.join(os.path.dirname(__file__), "main.py")
    
    cmd = [
        sys.executable, main_script,
        
        # Device setting
        "--device", "cpu",            # Force CPU stable mode
        
        # LLM settings - tối ưu hợp lý cho tốc độ
        "--llm-max-new", "56",        # Tăng lên 56 cho câu trả lời đầy đủ hơn
        "--llm-temp", "0.03",         # Tăng nhẹ để linh hoạt hơn
        
        # Retrieval settings - giảm số lượng
        "--topk", "2",                # Chỉ 2 semantic hits
        "--bm25-topn", "2",           # Chỉ 2 BM25 hits  
        "--sem-limit", "3",           # Giới hạn semantic results
        
        # Context settings - tối ưu
        "--ctx-topn", "3",            # 3 context items
        "--ctx-max-chars", "800",     # Context vừa phải  
        "--ctx-join-window", "0",     # Không join chunks để tăng tốc
        
        # QA600 settings
        "--qa600-topn", "2",          # Ít QA items hơn
        "--qa600-thr", "0.85",        # Threshold để nhanh hơn
    ]
    
    # Hiển thị thông tin cấu hình
    print("🚀 KHỞI ĐỘNG HỆ THỐNG VỚI CẤU HÌNH TỐI ƯU TỐC ĐỘ")
    print("=" * 60)
    print("🔧 Device: CPU mode (ổn định, không lỗi CUDA)")
    print("⚡ Tối ưu hóa:")
    print("   • Max new tokens: 48 (đủ cho câu trả lời ngắn)")
    print("   • Temperature: 0.01 (gần greedy decode)")
    print("   • Context: 800 chars, 3 items")
    print("   • Retrieval: 2 semantic + 2 BM25 hits")
    print("   • Không join chunks (tăng tốc)")
    print("📌 Lưu ý: Câu trả lời sẽ ngắn gọn nhưng nhanh hơn đáng kể")
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

if __name__ == "__main__":
    exit(run_fast())