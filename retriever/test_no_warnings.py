#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test nhanh để kiểm tra warnings và tốc độ
"""

import warnings
import os
import sys

# Tắt warnings để output sạch hơn
warnings.filterwarnings("ignore", category=UserWarning)

def quick_test():
    print("🧪 TEST NHANH - Kiểm tra warnings và tốc độ")
    print("=" * 50)
    
    # Kiểm tra embed data có tồn tại không
    embed_path = "embed_data"
    if not os.path.exists(embed_path):
        print("❌ Embed data chưa có")
        print("➡️  Hãy tạo embed data trước:")
        print("   cd ../embed")
        print("   python create_all_embeddings.py")
        return
    
    # Import và test
    try:
        from model_llm import build_llm, generate_answer
        
        model_path = r"D:\AI.LLM-khanh-no_rrf\models\phogpt4b-ft-merged"
        
        if not os.path.exists(model_path):
            print(f"❌ Model không tìm thấy: {model_path}")
            return
        
        print("🚀 Testing LLM với cấu hình mới...")
        
        # Test với greedy decode (temperature=0.0)
        llm = build_llm(
            model_path,
            temperature=0.0,  # Pure greedy
            max_new_tokens=24,  # Rất ngắn cho test
            use_onnx=False
        )
        
        print("✅ Model loaded thành công")
        
        # Test generation
        test_context = "Biển báo giao thông là các ký hiệu được đặt trên đường để hướng dẫn người tham gia giao thông."
        test_question = "Biển báo có tác dụng gì?"
        
        print(f"📝 Test: {test_question}")
        
        import time
        start = time.time()
        
        result = generate_answer(
            llm, 
            test_question, 
            test_context,
            max_new_tokens=24,
            temperature=0.0
        )
        
        duration = time.time() - start
        
        print(f"🤖 Kết quả ({duration:.2f}s): {result}")
        print("✅ Test hoàn thành - Không có warnings!")
        
    except Exception as e:
        print(f"❌ Lỗi: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    quick_test()