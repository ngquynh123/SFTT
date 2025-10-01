#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test output sạch - kiểm tra không có ký tự lạ
"""

import os
import sys
import warnings
warnings.filterwarnings("ignore")

def test_clean_output():
    print("🧪 TEST OUTPUT SẠCH")
    print("=" * 40)
    
    try:
        from model_llm import build_llm, generate_answer
        
        model_path = r"D:\AI.LLM-khanh-no_rrf\models\phogpt4b-ft-merged"
        
        if not os.path.exists(model_path):
            print(f"❌ Model không tìm thấy")
            return
        
        print("🚀 Loading model...")
        llm = build_llm(
            model_path,
            temperature=0.0,
            max_new_tokens=20,  # Ngắn để test
            use_onnx=False
        )
        
        # Test cases
        test_cases = [
            {
                "question": "Biển báo P.130 là gì?",
                "context": "Biển báo P.130 là biển cấm ô tô."
            },
            {
                "question": "Luật giao thông quy định gì?", 
                "context": "Luật giao thông quy định về an toàn đường bộ."
            },
            {
                "question": "Phạt nguội là gì?",
                "context": "Phạt nguội là hình thức xử phạt vi phạm giao thông qua camera."
            }
        ]
        
        for i, test in enumerate(test_cases, 1):
            print(f"\n📝 Test {i}: {test['question']}")
            
            result = generate_answer(
                llm,
                test['question'],
                test['context'],
                max_new_tokens=20,
                temperature=0.0
            )
            
            print(f"🤖 Output: '{result}'")
            
            # Kiểm tra output sạch
            issues = []
            if "� cộng" in result:
                issues.append("Có ký tự lạ")
            if len(result) < 5:
                issues.append("Quá ngắn") 
            if len(set(result.replace(' ', ''))) < 3:
                issues.append("Lặp ký tự")
            
            if issues:
                print(f"⚠️  Issues: {', '.join(issues)}")
            else:
                print("✅ Output sạch!")
        
        print(f"\n🎯 Test hoàn thành!")
        
    except Exception as e:
        print(f"❌ Lỗi: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_clean_output()