#!/usr/bin/env python3
"""
Simple tokenizer test
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

def test_tokenizer():
    print("🔍 Testing tokenizer in isolation...")
    
    try:
        from model_llm import TransformersLLM
        
        model_path = r"D:\AI.LLM-khanh-no_rrf\models\phogpt4b-ft-merged"
        
        # Just test tokenizer loading and basic functionality
        llm = TransformersLLM(
            model_path=model_path,
            device="cpu",
            temperature=0.0,
            max_new_tokens=8
        )
        
        # Test simple generation
        simple_prompt = "Đèn đỏ có nghĩa là"
        print(f"\n🧪 Testing with: '{simple_prompt}'")
        
        result = llm.generate(simple_prompt, stop=None)
        print(f"Final result: '{result}'")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_tokenizer()