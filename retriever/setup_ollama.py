#!/usr/bin/env python3
"""
Setup và test Ollama với PhoGPT model
"""

import requests
import time
import json

def check_ollama():
    """Kiểm tra xem Ollama có đang chạy không"""
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json().get("models", [])
            print(f"✅ Ollama running with {len(models)} models")
            for model in models:
                print(f"  - {model['name']}")
            return True
        else:
            print(f"❌ Ollama response error: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ Ollama not running. Please start Ollama first.")
        print("💡 Download from: https://ollama.ai")
        return False
    except Exception as e:
        print(f"❌ Error checking Ollama: {e}")
        return False

def pull_phogpt_model():
    """Pull PhoGPT model từ Ollama"""
    model_name = "mrjacktung/phogpt-4b-chat-gguf:latest"
    print(f"🔄 Pulling model: {model_name}")
    
    try:
        response = requests.post(
            "http://localhost:11434/api/pull",
            json={"name": model_name},
            stream=True,
            timeout=300
        )
        
        for line in response.iter_lines():
            if line:
                data = json.loads(line)
                status = data.get("status", "")
                if "downloading" in status:
                    print(f"📥 {status}")
                elif data.get("status") == "success":
                    print("✅ Model pulled successfully")
                    return True
                    
    except Exception as e:
        print(f"❌ Error pulling model: {e}")
        return False

def test_phogpt():
    """Test PhoGPT với câu hỏi đơn giản"""
    model_name = "mrjacktung/phogpt-4b-chat-gguf:latest"
    
    test_questions = [
        "Đèn giao thông có bao nhiêu màu?",
        "Biển P.130 là gì?",
        "Xin chào"
    ]
    
    for question in test_questions:
        print(f"\n🧪 Testing: '{question}'")
        
        try:
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": model_name,
                    "prompt": question,
                    "stream": False,
                    "options": {
                        "temperature": 0.0,
                        "num_predict": 32
                    }
                },
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                answer = data.get("response", "").strip()
                print(f"✅ Answer: '{answer}'")
                
                # Check for garbled text
                garbled_patterns = ['cộng', 'Pont', 'anov', 'opsis']
                is_garbled = any(pattern in answer for pattern in garbled_patterns)
                
                if is_garbled:
                    print(f"❌ GARBLED TEXT DETECTED!")
                else:
                    print(f"✅ Clean answer")
            else:
                print(f"❌ Error: {response.status_code}")
                
        except Exception as e:
            print(f"❌ Error: {e}")

def main():
    print("🚀 Ollama PhoGPT Setup & Test")
    print("=" * 50)
    
    # 1. Check if Ollama is running
    if not check_ollama():
        return
    
    # 2. Check if PhoGPT model exists, if not pull it
    model_name = "mrjacktung/phogpt-4b-chat-gguf:latest"
    try:
        response = requests.get("http://localhost:11434/api/tags")
        models = response.json().get("models", [])
        model_exists = any(model["name"] == model_name for model in models)
        
        if not model_exists:
            print(f"📥 Model {model_name} not found, pulling...")
            if not pull_phogpt_model():
                return
        else:
            print(f"✅ Model {model_name} already exists")
            
    except Exception as e:
        print(f"❌ Error checking models: {e}")
        return
    
    # 3. Test the model
    print("\n🧪 Testing PhoGPT model...")
    test_phogpt()
    
    print("\n🎉 Setup complete! You can now use Ollama with PhoGPT.")
    print("💡 Run: python main.py \"Đèn giao thông có bao nhiêu màu?\"")

if __name__ == "__main__":
    main()