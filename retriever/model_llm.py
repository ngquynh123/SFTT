#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Model LLM wrapper with optimized settings for speed
Hỗ trợ ONNX Runtime (chưa bật), fallback sang Transformers hoặc Ollama
"""

from typing import List, Optional
import os, re, requests

# Try to import transformers
try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

# ONNX hiện chưa dùng
ONNX_AVAILABLE = False

# =========================
# Base LLM class
# =========================
class BaseLLM:
    def generate(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        raise NotImplementedError

# =========================
# Transformers LLM
# =========================
class TransformersLLM(BaseLLM):
    def __init__(self, model_path: str, device: str = "cpu",
                 temperature: float = 0.01, max_new_tokens: int = 32):
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers package is not available")

        self.model_path = model_path
        self.device = "cuda" if device == "cuda" and torch.cuda.is_available() else "cpu"
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens

        print(f"⚡ Loading model {model_path} on {self.device}, max_new={max_new_tokens}, temp={temperature}")

        try:
            # THÊM: Clear HuggingFace cache để tránh cache conflicts
            import importlib
            import shutil
            if hasattr(importlib, 'invalidate_caches'):
                importlib.invalidate_caches()
            
            # Clear transformers cache nếu gặp vấn đề
            try:
                import transformers
                cache_dir = os.path.expanduser("~/.cache/huggingface/modules/transformers_modules")
                model_cache = os.path.join(cache_dir, os.path.basename(model_path))
                if os.path.exists(model_cache):
                    print(f"🧹 Clearing cache: {model_cache}")
                    shutil.rmtree(model_cache, ignore_errors=True)
            except Exception as cache_err:
                print(f"⚠️ Cache clear warning: {cache_err}")
            
            # Load tokenizer với config an toàn và debug info
            print(f"🔄 Loading tokenizer from {model_path}...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=True,   # PhoGPT (MPT) cần True
                local_files_only=True,    # Chỉ dùng file local
                use_fast=True,
                padding_side="left",
                cache_dir=None           # Không dùng cache
            )
            
            # Detailed tokenizer info
            print(f"📋 Tokenizer info:")
            print(f"  - Type: {type(self.tokenizer).__name__}")
            print(f"  - Vocab size: {self.tokenizer.vocab_size}")
            print(f"  - Model max length: {getattr(self.tokenizer, 'model_max_length', 'Unknown')}")
            
            # Set pad token properly
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                print(f"  - Set pad_token to eos_token: {self.tokenizer.eos_token}")
            else:
                print(f"  - Existing pad_token: {self.tokenizer.pad_token}")
                
            # Test tokenizer with Vietnamese text
            test_text = "Đèn giao thông có ba màu"
            test_tokens = self.tokenizer.encode(test_text, add_special_tokens=False)
            test_decoded = self.tokenizer.decode(test_tokens)
            print(f"  - Test encode/decode:")
            print(f"    Original: '{test_text}'")
            print(f"    Tokens: {test_tokens[:10]}{'...' if len(test_tokens) > 10 else ''}")
            print(f"    Decoded: '{test_decoded}'")
            
            if test_decoded.strip() != test_text.strip():
                print(f"  ⚠️ Tokenizer encode/decode mismatch!")
            else:
                print(f"  ✅ Tokenizer working correctly")
                
            print("✅ Tokenizer loaded successfully")

            # dtype phù hợp
            dtype = torch.float32
            if self.device == "cuda":
                if torch.cuda.is_bf16_supported():
                    dtype = torch.bfloat16
                else:
                    dtype = torch.float16

            # Load model với config an toàn và error handling
            print(f"🔄 Loading model from {model_path}...")
            try:
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    trust_remote_code=True,
                    local_files_only=True,    # Chỉ dùng file local
                    torch_dtype=dtype,
                    low_cpu_mem_usage=True,
                    use_cache=True,
                    device_map="auto" if self.device == "cuda" else None,
                    revision=None,           # Không dùng specific revision
                    ignore_mismatched_sizes=True,  # Ignore size mismatch
                    cache_dir=None           # Không dùng cache
                )
                print("✅ Model loaded successfully")
            except (UnicodeDecodeError, FileNotFoundError) as e:
                print(f"⚠️ Loading error, trying with fallback: {e}")
                # Fallback: thử without trust_remote_code
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    trust_remote_code=False,  # Disable custom code
                    local_files_only=True,
                    torch_dtype=dtype,
                    low_cpu_mem_usage=True,
                    use_cache=True,
                    cache_dir=None
                )
                print("✅ Model loaded with fallback method")
            if self.device == "cuda":
                self.model.to("cuda")

            self.model.eval()
            torch.set_grad_enabled(False)

        except Exception as e:
            raise RuntimeError(f"❌ Failed to load model from {model_path}: {e}")

    def generate(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        try:
            # Debug: Print input prompt
            print(f"🔤 Input prompt: '{prompt[:100]}{'...' if len(prompt) > 100 else ''}'")
            
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=768,
                padding=False
            )
            
            # Debug: Print tokenized input
            input_tokens = inputs["input_ids"][0].tolist()
            print(f"🔢 Input tokens: {input_tokens[:10]}{'...' if len(input_tokens) > 10 else ''} (total: {len(input_tokens)})")
            
            if self.device == "cuda":
                inputs = {k: v.to("cuda") for k, v in inputs.items()}

            with torch.inference_mode():
                outputs = self.model.generate(
                    inputs["input_ids"],
                    attention_mask=inputs.get("attention_mask"),
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    num_beams=1,
                    repetition_penalty=1.15,
                    no_repeat_ngram_size=3,
                    use_cache=True,
                    return_dict_in_generate=False
                )

            seq = outputs[0] if outputs.dim() == 2 else outputs
            new_ids = seq[inputs["input_ids"].shape[1]:]
            
            # Debug: Print raw generated tokens
            print(f"🎯 Generated tokens: {new_ids.tolist()}")
            
            response = self.tokenizer.decode(new_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True).strip()
            
            # Debug: Print raw response before cleaning
            print(f"📝 Raw response: '{response}'")

            # Clean up text with more aggressive patterns
            original_response = response
            
            # Remove "cộng" patterns first
            response = re.sub(r'\bcộng\s+cộng\b', '', response, flags=re.IGNORECASE)
            response = re.sub(r'\bcộng\s*[^\w\s]\s*', '', response, flags=re.IGNORECASE)  # "cộng X"
            response = re.sub(r'\bcộng(?!\s+(tác|đồng|với|hòa))\b', '', response, flags=re.IGNORECASE)  # Remove standalone "cộng" except valid words
            
            # Remove garbled patterns
            response = re.sub(r'\b(\w+)\s+\1\b', r'\1', response)  # Word repetition
            response = re.sub(r'[�□◊▪▫•‰…\ufffd]', '', response)  # Bad chars
            response = re.sub(r'\b[A-Z][a-z]{1,3}\s+[A-Z][a-z]{1,3}\b', '', response)  # "Pont Nhan"
            response = re.sub(r'\b\w{1,2}\s+\w{1,2}\s+\w{1,2}\b', '', response)  # Short word sequences
            
            # Remove weird patterns
            response = re.sub(r'[^\w\s\u00C0-\u024F\u1E00-\u1EFF.,!?():;"\'-]', '', response)  # Keep only valid chars
            response = re.sub(r'\s+', ' ', response).strip()
            
            # Final validation
            if len(response) < 3 or len(set(response.replace(' ', ''))) < 3:
                response = "Xin lỗi, tôi không thể trả lời câu hỏi này."
            
            if response != original_response:
                print(f"🧹 Cleaned response: '{response}'")

            if stop:
                for st in stop:
                    idx = response.find(st)
                    if idx != -1:
                        response = response[:idx].strip()
                        break

            return response or "Xin lỗi, tôi không thể trả lời câu hỏi này."
        except Exception as e:
            return f"[ERROR] {e}"

# =========================
# Ollama client
# =========================
class OllamaGenerateLLM(BaseLLM):
    def __init__(self,
                 model_id: str = None,
                 host: str = None,
                 temperature: float = 0.05,
                 top_p: float = 0.95,
                 max_new_tokens: int = 64,
                 timeout_connect: int = 10,
                 timeout_read: int = 120,
                 keep_alive: str = "5m"):
        self.model_id = model_id or os.getenv("LLM_MODEL_ID", "mrjacktung/phogpt-4b-chat-gguf:latest")
        self.host = (host or os.getenv("LLM_OLLAMA_HOST", "http://localhost:11434")).rstrip("/")
        self.temperature = float(temperature)
        self.top_p = float(top_p)
        self.max_new_tokens = int(max_new_tokens)
        self.timeout_connect = int(timeout_connect)
        self.timeout_read = int(timeout_read)
        self.keep_alive = keep_alive

    def set_params(self, temperature: float = None, max_new_tokens: int = None):
        if temperature is not None:
            self.temperature = float(temperature)
        if max_new_tokens is not None:
            self.max_new_tokens = int(max_new_tokens)

    def generate(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        url = f"{self.host}/api/generate"
        payload = {
            "model": self.model_id,
            "prompt": prompt,
            "stream": False,
            "keep_alive": self.keep_alive,
            "options": {
                "temperature": self.temperature,
                "top_p": self.top_p,
                "num_predict": self.max_new_tokens,
            },
        }
        if stop:
            payload["stop"] = stop
        try:
            r = requests.post(url, json=payload,
                              timeout=(self.timeout_connect, self.timeout_read))
            r.raise_for_status()
            data = r.json()
            return (data.get("response") or "").strip()
        except Exception as e:
            return f"[Ollama ERROR] {e}"

# =========================
# Builder
# =========================
def build_llm(model_path_or_id, host=None, temperature=0.05, device="cpu", use_onnx=True, **kwargs):
    if model_path_or_id and (os.path.isdir(model_path_or_id) or re.match(r"^[A-Za-z]:\\", model_path_or_id) or model_path_or_id.startswith(('/', '\\'))):
        print("📦 Using Transformers (PhoGPT/MPT).")
        return TransformersLLM(
            model_path=model_path_or_id,
            device=device,
            temperature=temperature,
            max_new_tokens=kwargs.get("max_new_tokens", 48)
        )
    else:
        return OllamaGenerateLLM(
            model_id=model_path_or_id,
            host=host,
            temperature=temperature,
            max_new_tokens=kwargs.get("max_new_tokens", 48)
        )

# =========================
# STOP tokens & sanitize
# =========================
STOP_TOKENS = [
    "<sources>", "</sources>",
    "<bm25_hints>", "</bm25_hints>",
    "<bm25_only>", "</bm25_only>",
    "<question>", "</question>",
    "<sample_answer>", "</sample_answer>",
    "<theory>", "</theory>",
    "<given_answer>", "</given_answer>",
    "� cộng", "�", "===", "---",
    "cộng cộng", "cộng  cộng",
    "Nhan Pont", "Kaz tre", "Lip Chú",
    "Traceback", "Error", "ERROR",
    "▪", "▫", "•", "□", "◊",
    "...", "…",
    "</s>", "<s>", "[INST]", "[/INST]",
]

_TAGS_RE = re.compile(r"</?(sources|bm25_hints|bm25_only|question|sample_answer|theory|given_answer)>", re.IGNORECASE)
_BM25_LINE_RE = re.compile(r"^\s*\[B\d+\].*$", flags=re.MULTILINE)

def sanitize(text: str) -> str:
    s = text if isinstance(text, str) else (str(text) if text is not None else "")
    s = _BM25_LINE_RE.sub("", s)
    s = _TAGS_RE.sub("", s)
    s = re.sub(r'[�□◊▪▫•‰…\ufffd]', '', s)
    s = re.sub(r'\bcộng\s+cộng\b', '', s, flags=re.IGNORECASE)
    s = re.sub(r'\b(\w+)\s+\1\b', r'\1', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s

# =========================
# Prompt templates
# =========================
def build_prompt_ref_with_sample(user_question: str, theory_block: str, sample_answer: Optional[str] = None) -> str:
    sample = (sample_answer or "").strip()
    prompt = f"Thông tin: {theory_block.strip()}\n\n"
    if sample:
        prompt += f"Mẫu: {sample}\n\n"
    prompt += f"Hỏi: {user_question.strip()}\nTrả lời ngắn gọn, chính xác:"
    return prompt

def build_prompt_with_context(user_question: str, theory_block: str) -> str:
    return f"Thông tin: {theory_block.strip()}\n\nHỏi: {user_question.strip()}\nTrả lời ngắn:"

def build_prompt_with_sample(user_question: str, sample_answer: str) -> str:
    return f"Mẫu: {sample_answer.strip()}\n\nHỏi: {user_question.strip()}\nTrả lời tương tự:"

def build_prompt_direct(user_question: str) -> str:
    return f"Hỏi: {user_question.strip()}\nTrả lời ngắn:"

# =========================
# Generate wrapper
# =========================
def generate_answer(llm, question: str, context: str, max_new_tokens: int = 64, temperature: Optional[float] = None) -> str:
    try:
        if hasattr(llm, 'set_params'):
            llm.set_params(
                temperature=temperature if temperature is not None else 0.05,
                max_new_tokens=max_new_tokens
            )
        prompt = build_prompt_ref_with_sample(question, context)
        return llm.generate(prompt, stop=STOP_TOKENS)
    except Exception as e:
        return f"[Generation ERROR] {e}"
