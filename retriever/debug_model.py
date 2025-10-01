#!/usr/bin/env python3
"""
Minimal test script để isolate model generation issues
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

def test_pure_generation():
    print("🧪 Testing pure model generation without any context...")
    
    try:
        from model_llm import TransformersLLM
        
        model_path = r"D:\AI.LLM-khanh-no_rrf\models\phogpt4b-ft-merged"
        
        # Create LLM with minimal settings
        llm = TransformersLLM(
            model_path=model_path,
            device="cpu",
            temperature=0.0,  # Pure greedy
            max_new_tokens=16  # Very short
        )
        
        # Test with very simple prompts
        test_prompts = [
            "Xin chào",
            "Hôm nay là thứ mấy?",
            "Đèn giao thông có màu gì?",
            "1 + 1 = ?",
            "Hỏi: Đèn đỏ có nghĩa gì?\nTrả lời:"
        ]
        
        for i, prompt in enumerate(test_prompts, 1):
            print(f"\n--- Test {i}: '{prompt}' ---")
            try:
                # Generate without ANY stop tokens
                response = llm.generate(prompt, stop=None)
                print(f"✅ Raw output: '{response}'")
                
                # Check for garbled patterns
                garbled_patterns = ['cộng', 'Pont', 'anov', 'opsis', 'endron']
                is_garbled = any(pattern in response for pattern in garbled_patterns)
                
                if is_garbled:
                    print(f"❌ GARBLED DETECTED! Contains: {[p for p in garbled_patterns if p in response]}")
                else:
                    print(f"✅ Clean output")
                    
            except Exception as e:
                print(f"❌ Generation error: {e}")
        
        # Test tokenizer directly
        print("\n--- Tokenizer Test ---")
        test_text = "Đèn giao thông có ba màu: đỏ, vàng, xanh."
        tokens = llm.tokenizer.encode(test_text)
        decoded = llm.tokenizer.decode(tokens)
        print(f"Original: {test_text}")
        print(f"Tokens: {tokens[:10]}...")  # First 10 tokens
        print(f"Decoded: {decoded}")
        
        if decoded != test_text:
            print("❌ Tokenizer encode/decode mismatch!")
        else:
            print("✅ Tokenizer working correctly")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

def test_model_config():
    """Check model configuration files"""
    print("\n🔍 Checking model configuration...")
    
    model_path = r"D:\AI.LLM-khanh-no_rrf\models\phogpt4b-ft-merged"
    
    config_files = {
        "config.json": "Model configuration",
        "tokenizer_config.json": "Tokenizer configuration", 
        "generation_config.json": "Generation configuration"
    }
    
    for filename, desc in config_files.items():
        filepath = os.path.join(model_path, filename)
        if os.path.exists(filepath):
            try:
                import json
                with open(filepath, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                print(f"✅ {desc}: {filename}")
                
                # Print key settings
                if filename == "config.json":
                    print(f"  - Model type: {config.get('model_type', 'Unknown')}")
                    print(f"  - Vocab size: {config.get('vocab_size', 'Unknown')}")
                elif filename == "generation_config.json":
                    print(f"  - Max length: {config.get('max_length', 'Unknown')}")
                    print(f"  - Pad token ID: {config.get('pad_token_id', 'Unknown')}")
                    
            except Exception as e:
                print(f"❌ Error reading {filename}: {e}")
        else:
            print(f"❌ Missing: {filename}")

if __name__ == "__main__":
    test_model_config()
    test_pure_generation()