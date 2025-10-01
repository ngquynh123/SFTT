#!/usr/bin/env python3
"""
Test script để debug model loading
"""
import sys
import os

# Thêm path hiện tại
sys.path.append(os.path.dirname(__file__))

def test_model_loading():
    print("🔍 Testing model loading...")
    
    model_path = r"D:\AI.LLM-khanh-no_rrf\models\phogpt4b-ft-merged"
    
    # Kiểm tra files tồn tại
    print(f"📁 Model path exists: {os.path.exists(model_path)}")
    
    if os.path.exists(model_path):
        files = os.listdir(model_path)
        print(f"📄 Files in model dir: {len(files)} files")
        
        # Kiểm tra các file quan trọng
        important_files = ["config.json", "tokenizer.json", "modeling_mpt.py"]
        for file in important_files:
            exists = file in files
            print(f"  ✅ {file}: {exists}")
    
    # Test import transformers
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        import torch
        print("✅ Transformers import OK")
        
        # Test tokenizer loading
        print("🔄 Testing tokenizer loading...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            local_files_only=True,
            use_fast=True
        )
        print("✅ Tokenizer loaded successfully")
        
        # Test model loading với minimal config
        print("🔄 Testing model loading...")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            local_files_only=True,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True
        )
        print("✅ Model loaded successfully")
        
        # Test generation
        print("🔄 Testing generation...")
        inputs = tokenizer("Xin chào", return_tensors="pt")
        with torch.no_grad():
            outputs = model.generate(
                inputs["input_ids"],
                max_new_tokens=5,
                do_sample=False
            )
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"✅ Generation test: '{response}'")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_model_loading()