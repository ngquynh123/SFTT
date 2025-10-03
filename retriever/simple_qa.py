#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple QA without embeddings - Chỉ dùng model LLM trực tiếp
"""

import sys, os
sys.path.append(os.path.dirname(__file__))

from model_llm import build_llm

def simple_qa(question: str):
    """QA đơn giản không cần embeddings"""
    
    print("🚀 SIMPLE QA MODE (GPU Test)")
    print("=" * 50)
    
    # Kiểm tra GPU availability
    try:
        import torch
        if torch.cuda.is_available():
            print(f"🔥 GPU Available: {torch.cuda.get_device_name(0)}")
            print(f"💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory // 1024**3} GB")
            device = "cuda"
        else:
            print("💻 No GPU available, using CPU")
            device = "cpu"
    except:
        device = "cpu"
    
    # Load model trực tiếp với GPU
    try:
        llm = build_llm(
            model_path_or_id=r"D:\AI.LLM-khanh-no_rrf\models\phogpt4b-ft-merged-manual",
            device=device,
            temperature=0.0,
            max_new_tokens=24
        )
        print("✅ Model loaded")
        
        # Tạo prompt đơn giản hơn
        prompt = f"Câu hỏi: {question}\nCâu trả lời:"
        
        print(f"🤔 Question: {question}")
        print("⏳ Generating...")
        
        # Generate answer
        answer = llm.generate(prompt, stop=["Câu hỏi:", "Q:", "A:", "\n\n"])
        
        print(f"💡 Answer: {answer}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        
        # Fallback to original model
        try:
            print("🔄 Trying original model...")
            llm = build_llm(
                model_path_or_id=r"D:\AI.LLM-khanh-no_rrf\models\PhoGPT-4B",
                device=device, 
                temperature=0.0,
                max_new_tokens=24
            )
            
            prompt = f"Câu hỏi: {question}\nCâu trả lời:"
            answer = llm.generate(prompt, stop=["Câu hỏi:", "\n\n"])
            print(f"💡 Answer: {answer}")
            
        except Exception as e2:
            print(f"❌ Both models failed: {e2}")
            return "Không thể trả lời."

if __name__ == "__main__":
    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
    else:
        question = input("Q> ")
    
    simple_qa(question)