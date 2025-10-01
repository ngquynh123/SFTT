#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script chạy nhanh hơn - settings đã tối ưu
"""

import subprocess
import sys
import os

def run_faster():
    """Chạy main.py với settings nhanh hơn"""
    
    main_script = os.path.join(os.path.dirname(__file__), "main.py")
    
    cmd = [sys.executable, main_script]
    
    # Thêm query nếu có
    if len(sys.argv) > 1:
        cmd.extend(["--once", " ".join(sys.argv[1:])])
    
    print("🚀 CHẠY VỚI SETTINGS NHANH HƠN")
    print("Settings tối ưu:")
    print("  • Max tokens: 32 (thay vì 48)")
    print("  • Temperature: 0.0 (pure greedy)")
    print("  • Context: 600 chars, 2 items")
    print("  • Hits: 2 semantic + 2 BM25")
    print("  • Pure greedy decode (no sampling)")
    print("  • Không join chunks")
    print("  • Loại bỏ all warnings")
    print()
    
    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\n⏹️  Đã dừng")
    except Exception as e:
        print(f"\n❌ Lỗi: {e}")

if __name__ == "__main__":
    run_faster()