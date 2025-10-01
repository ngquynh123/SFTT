#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quick Fix Script - Khắc phục lỗi và test nhanh
"""

import os
import sys

def main():
    print("🔧 QUICK FIX - Khắc phục lỗi hệ thống")
    print("=" * 50)
    
    # 1. Kiểm tra model path
    model_path = r"D:\AI.LLM-khanh-no_rrf\models\phogpt4b-ft-merged"
    print(f"📂 Kiểm tra model path: {model_path}")
    
    if os.path.exists(model_path):
        print("✅ Model path tồn tại")
        files = os.listdir(model_path)
        print(f"📁 Files trong model: {len(files)} files")
        
        # Kiểm tra các file quan trọng
        important_files = ['config.json', 'pytorch_model.bin', 'tokenizer.json']
        for file in important_files:
            if any(f.startswith(file.split('.')[0]) for f in files):
                print(f"✅ {file} OK")
            else:
                print(f"⚠️  {file} missing")
    else:
        print("❌ Model path không tồn tại")
    
    print()
    
    # 2. Kiểm tra embed data
    embed_path = "embed_data"
    print(f"📂 Kiểm tra embed data: {embed_path}")
    
    if os.path.exists(embed_path):
        print("✅ Embed data folder tồn tại")
        for channel in ['dialogue', 'lesson']:
            channel_path = os.path.join(embed_path, channel)
            if os.path.exists(channel_path):
                files = len(os.listdir(channel_path))
                print(f"✅ {channel}: {files} files")
            else:
                print(f"❌ {channel}: missing")
    else:
        print("❌ Embed data không tồn tại")
    
    print()
    
    # 3. Đưa ra giải pháp
    print("💡 GIẢI PHÁP:")
    print()
    
    if not os.path.exists(model_path):
        print("🔴 VẤN ĐỀ: Model không tồn tại")
        print("➡️  Giải pháp: Kiểm tra lại đường dẫn model")
    elif not os.path.exists(embed_path):
        print("🔴 VẤN ĐỀ: Embed data không tồn tại") 
        print("➡️  Giải pháp: Tạo embed data hoặc dùng script test-only")
    else:
        print("🟢 CÓ THỂ CHẠY:")
        print()
        print("1️⃣  Test chỉ LLM (không cần embed):")
        print("   python test_llm_speed.py")
        print()
        print("2️⃣  Tạo embed data:")
        print("   cd ../embed")
        print("   python create_all_embeddings.py")
        print()
        print("3️⃣  Chạy script chính:")
        print("   python balanced_mode.py")
    
    print()
    print("🚀 SCRIPTS SẴN SÀNG:")
    scripts = [
        "balanced_mode.py - Cân bằng tốc độ và chất lượng",
        "fast_mode.py - Nhanh hơn", 
        "ultra_fast_v2.py - Siêu nhanh",
        "test_llm_speed.py - Test model only"
    ]
    
    for script in scripts:
        script_file = script.split(' - ')[0]
        if os.path.exists(script_file):
            print(f"✅ {script}")
        else:
            print(f"❌ {script}")

if __name__ == "__main__":
    main()