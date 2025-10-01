#!/usr/bin/env python3
"""
Test script để thử model gốc PhoGPT chưa fine-tune
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

from model_llm import TransformersLLM

def test_original_model():
    print("🧪 Testing model gốc PhoGPT...")
    
    # Thử với model gốc (nếu có)
    original_model_paths = [
        r"D:\AI.LLM-khanh-no_rrf\models\PhoGPT-4B",  # Model gốc
        "vinai/PhoGPT-4B-Chat"  # Hoặc download từ HuggingFace
    ]
    
    for model_path in original_model_paths:
        if os.path.exists(model_path):
            print(f"✅ Found original model: {model_path}")
            
            try:
                llm = TransformersLLM(
                    model_path=model_path,
                    device="cpu",
                    temperature=0.0,
                    max_new_tokens=32
                )
                
                # Test simple question
                response = llm.generate("Biển P.130 là gì?")
                print(f"📋 Original model response: {response}")
                
                return True
                
            except Exception as e:
                print(f"❌ Error with {model_path}: {e}")
                continue
    
    print("❌ Không tìm thấy model gốc để test")
    return False

if __name__ == "__main__":
    test_original_model()