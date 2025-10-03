#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Model LLM wrapper with optimized settings for speed & stability
- Ưu tiên model merged nếu có; fallback sang base
- Hỗ trợ Transformers (PhoGPT/MPT) và Ollama
- Fallback khi thiếu file local (tokenizer/model)
"""

from typing import List, Optional
import os, re, requests

# Try to import transformers/torch (optional at import-time)
try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch
    TRANSFORMERS_AVAILABLE = True
except Exception:
    TRANSFORMERS_AVAILABLE = False

# ONNX: chưa dùng (placeholder)
ONNX_AVAILABLE = False

# =========================
# Base LLM class
# =========================
class BaseLLM:
    def generate(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        raise NotImplementedError

# =========================
# Helpers
# =========================
def _is_local_path(p: str) -> bool:
    if not p:
        return False
    # Windows drive (C:\...), absolute Unix (/ or \), or existing directory
    return os.path.isdir(p) or bool(re.match(r"^[A-Za-z]:\\", p)) or p.startswith(('/', '\\'))

def _enable_cuda_fastmath():
    try:
        if torch.cuda.is_available():
            # Cho phép TF32 (trên Ampere+)
            try:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
            except Exception:
                pass
            # PyTorch >= 2.0
            try:
                torch.set_float32_matmul_precision("high")
            except Exception:
                pass
    except Exception:
        pass

def _pick_dtype(device: str):
    if device == "cuda" and torch.cuda.is_available():
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.float32

def _safe_eos_id(tokenizer) -> Optional[int | List[int]]:
    try:
        # Có model.generation_config.eos_token_id có thể là int hoặc list
        return getattr(tokenizer, "eos_token_id", None) or getattr(tokenizer, "eos_token", None)
    except Exception:
        return None

# =========================
# Transformers LLM
# =========================
class TransformersLLM(BaseLLM):
    def __init__(
        self,
        model_path: str,
        device: str = "cpu",
        temperature: float = 0.01,
        max_new_tokens: int = 32,
        local_files_only: bool = True,
        clear_cache_first: bool = False,
        allow_ignore_mismatch: bool = False,
    ):
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers/torch is not available.")

        # Chọn device
        self.device = "cuda" if (device == "cuda" and torch.cuda.is_available()) else "cpu"
        self.temperature = float(temperature)
        self.max_new_tokens = int(max_new_tokens)
        self.local_files_only = bool(local_files_only)
        self.clear_cache_first = bool(clear_cache_first)
        self.allow_ignore_mismatch = bool(allow_ignore_mismatch)

        # LATEST MERGE: Sử dụng model merged mới nhất
        latest_model_path = r"D:\AI.LLM-khanh-no_rrf\models\phogpt4b-ft-merged"

        print("=" * 60)
        print("🔍 MODEL SELECTION")
        print("=" * 60)

        # Chỉ sử dụng latest merged model
        if os.path.exists(latest_model_path) and os.path.exists(os.path.join(latest_model_path, "config.json")):
            self.model_path = latest_model_path
            print(f"🎯 USING: LATEST MERGED MODEL")
            print(f"📂 Path: {self.model_path}")
            print(f"⭐ Benefits: Most recent merge, optimized stability, anti-garbled")
        else:
            raise FileNotFoundError(f"❌ Latest merged model not found at: {latest_model_path}")
            
        print("=" * 60)
        print(f"⚡ Loading model {self.model_path} on {self.device}, max_new={self.max_new_tokens}, temp={self.temperature}")

        # Tuỳ chọn dọn cache (thực sự chỉ nên dùng khi gặp xung đột)
        if self.clear_cache_first:
            try:
                import shutil
                cache_dir = os.path.expanduser("~/.cache/huggingface")
                print(f"🧹 Clearing HuggingFace cache (selected folders)...")
                # Không xóa toàn bộ; chỉ modules/transformers_modules để tránh đụng model repo cache
                modules_dir = os.path.join(cache_dir, "modules", "transformers_modules")
                if os.path.exists(modules_dir):
                    shutil.rmtree(modules_dir, ignore_errors=True)
                    print(f"🧹 Removed: {modules_dir}")
            except Exception as cache_err:
                print(f"⚠️ Cache clear warning: {cache_err}")

        # CUDA fastmat
        _enable_cuda_fastmath()

        # Chọn dtype
        dtype = _pick_dtype(self.device)

        # Load tokenizer (chỉ từ stable merged model)
        print(f"🔄 Loading tokenizer from {self.model_path}...")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                trust_remote_code=False,  # Stable model không cần trust_remote_code
                local_files_only=self.local_files_only,
                use_fast=True,
                padding_side="left",
                cache_dir=None
            )
        except Exception as e_tok:
            if self.local_files_only:
                print(f"⚠️ Tokenizer local load failed: {e_tok}. Retrying with local_files_only=False...")
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.model_path,
                    trust_remote_code=False,  # Vẫn giữ False cho stable
                    local_files_only=False,
                    use_fast=True,
                    padding_side="left",
                )
            else:
                raise

        # Tokenizer info
        print(f"📋 Tokenizer info:")
        try:
            print(f"  - Type: {type(self.tokenizer).__name__}")
            print(f"  - Vocab size: {getattr(self.tokenizer, 'vocab_size', 'Unknown')}")
            print(f"  - Model max length: {getattr(self.tokenizer, 'model_max_length', 'Unknown')}")
        except Exception:
            pass

        # Set pad token
        if getattr(self.tokenizer, "pad_token", None) is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            print(f"  - Set pad_token to eos_token: {self.tokenizer.eos_token}")
        else:
            print(f"  - Existing pad_token: {self.tokenizer.pad_token}")

        # Quick VN encode/decode self-test
        try:
            test_text = "Đèn giao thông có ba màu"
            test_tokens = self.tokenizer.encode(test_text, add_special_tokens=False)
            test_decoded = self.tokenizer.decode(test_tokens)
            print(f"  - Test encode/decode: OK" if test_decoded.strip() == test_text.strip() else "  ⚠️ Tokenizer encode/decode mismatch!")
        except Exception:
            print("  ⚠️ Tokenizer self-test skipped")

        print("✅ Tokenizer loaded")

        # Load model (latest merged model với settings tối ưu)
        print(f"🔄 Loading model from {self.model_path}...")
        
        # Latest model không cần flash_attn_triton.py
        print(f"🔍 Using latest merged model - no flash attention needed")
        
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                trust_remote_code=False,      # Latest model không cần trust remote code
                local_files_only=self.local_files_only,
                torch_dtype=dtype,
                low_cpu_mem_usage=True,
                use_cache=True,
                device_map="auto" if self.device == "cuda" else None,
                revision=None,
                ignore_mismatched_sizes=self.allow_ignore_mismatch,
                cache_dir=None
            )
            print(f"✅ Latest merged model loaded successfully")
        except Exception as e:
            print(f"❌ Error loading latest model: {e}")
            raise RuntimeError(f"Failed to load latest merged model: {e}")

        # Không ép .to("cuda") nếu đã dùng device_map="auto"
        if self.device == "cuda" and getattr(self.model, "hf_device_map", None) is None:
            # Model chưa được map tự động (ví dụ device="cpu") → có thể move
            try:
                self.model.to("cuda")
            except Exception:
                pass

        self.model.eval()
        try:
            torch.set_grad_enabled(False)
        except Exception:
            pass

        # Thông tin model
        print("=" * 60)
        print("📊 MODEL INFORMATION")
        print("=" * 60)
        print(f"🏷️  Model Name: {os.path.basename(self.model_path)}")
        print(f"📂 Model Path: {self.model_path}")
        print(f"🔧 Model Type: {type(self.model).__name__}")
        print(f"💾 Device: {self.device}")
        print(f"🔢 Data Type: {dtype}")
        print(f"⭐ Status: LATEST MERGED MODEL (newest version, anti-garbled)")
        print("=" * 60)

        # Lưu EOS/Pad ID an toàn
        self._eos_token_id = getattr(self.tokenizer, "eos_token_id", None)
        self._pad_token_id = getattr(self.tokenizer, "pad_token_id", None)

    def _tokenize_prompt(self, prompt: str):
        # Tính toán max input an toàn: model_max_length - max_new_tokens - margin
        model_max = int(getattr(self.tokenizer, "model_max_length", 2048) or 2048)
        margin = 16
        max_input_len = max(32, model_max - self.max_new_tokens - margin)

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=max_input_len,
            padding=False
        )
        if self.device == "cuda":
            inputs = {k: v.to("cuda") for k, v in inputs.items()}
        return inputs

    def generate(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        try:
            # Debug input
            print(f"🔤 Input prompt: '{prompt[:200]}{'...' if len(prompt) > 200 else ''}'")

            inputs = self._tokenize_prompt(prompt)
            input_len = inputs["input_ids"].shape[1]
            print(f"🔢 Input tokens: {input_len}")

            # Chọn eos/pad id
            eos_token_id = self._eos_token_id
            pad_token_id = self._pad_token_id

            gen_kwargs = dict(
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
                num_beams=1,
                repetition_penalty=1.15,
                no_repeat_ngram_size=3,
                use_cache=True,
                return_dict_in_generate=False,
            )

            with torch.inference_mode():
                outputs = self.model.generate(
                    inputs["input_ids"],
                    attention_mask=inputs.get("attention_mask"),
                    **gen_kwargs
                )

            seq = outputs[0] if getattr(outputs, "dim", lambda: 2)() == 2 else outputs
            new_ids = seq[input_len:]
            print(f"🎯 Generated token count: {len(new_ids)}")

            response = self.tokenizer.decode(
                new_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            ).strip()

            # Clean-up với kiểm soát
            response = sanitize(response)

            # Stop tokens (nếu có)
            if stop:
                for st in stop:
                    idx = response.find(st)
                    if idx != -1:
                        response = response[:idx].strip()
                        break

            # Kiểm tra tối thiểu
            if len(response) < 3 or len(set(response.replace(' ', ''))) < 3:
                return "Xin lỗi, tôi không thể trả lời câu hỏi này."

            return response
        except Exception as e:
            return f"[ERROR] {e}"

# =========================
# Ollama client
# =========================
class OllamaGenerateLLM(BaseLLM):
    def __init__(
        self,
        model_id: str = None,
        host: str = None,
        temperature: float = 0.05,
        top_p: float = 0.95,
        max_new_tokens: int = 64,
        timeout_connect: int = 10,
        timeout_read: int = 120,
        keep_alive: str = "5m"
    ):
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
            print(f"🛠️ Ollama call: model={self.model_id}, host={self.host}")
            r = requests.post(url, json=payload, timeout=(self.timeout_connect, self.timeout_read))
            r.raise_for_status()
            data = r.json()
            return (data.get("response") or "").strip()
        except Exception as e:
            return f"[Ollama ERROR] {e}"

# =========================
# Builder
# =========================
def build_llm(
    model_path_or_id,
    host=None,
    temperature=0.05,
    device="cpu",
    use_onnx=True,
    **kwargs
):
    # Cảnh báo ONNX chưa bật
    if use_onnx and not ONNX_AVAILABLE:
        print("ℹ️ ONNX path requested but not enabled yet. Using Transformers/Ollama.")

    max_new_tokens = int(kwargs.get("max_new_tokens", 48))
    local_files_only = bool(kwargs.get("local_files_only", True))
    clear_cache_first = bool(kwargs.get("clear_cache_first", False))
    allow_ignore_mismatch = bool(kwargs.get("allow_ignore_mismatch", False))

    if model_path_or_id and _is_local_path(model_path_or_id):
        print("📦 Using Transformers (PhoGPT/MPT).")
        return TransformersLLM(
            model_path=model_path_or_id,
            device=device,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            local_files_only=local_files_only,
            clear_cache_first=clear_cache_first,
            allow_ignore_mismatch=allow_ignore_mismatch,
        )
    else:
        print("🌐 Using Ollama backend.")
        return OllamaGenerateLLM(
            model_id=model_path_or_id,
            host=host,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
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
    "Traceback", "Error", "ERROR",
    "▪", "▫", "•", "□", "◊",
    "...", "…",
    "</s>", "<s>", "[INST]", "[/INST]",
]

_TAGS_RE = re.compile(
    r"</?(sources|bm25_hints|bm25_only|question|sample_answer|theory|given_answer)>",
    re.IGNORECASE
)
_BM25_LINE_RE = re.compile(r"^\s*\[B\d+\].*$", flags=re.MULTILINE)

def sanitize(text: str) -> str:
    s = text if isinstance(text, str) else (str(text) if text is not None else "")
    # Loại tag & hint debug
    s = _BM25_LINE_RE.sub("", s)
    s = _TAGS_RE.sub("", s)

    # Ký tự lỗi thường gặp
    s = re.sub(r'[�□◊▪▫•‰…\ufffd]', '', s)

    # Gộp từ lặp liền nhau
    s = re.sub(r'\b(\w+)\s+\1\b', r'\1', s, flags=re.IGNORECASE)

    # Một số rác in/tiền xử lý (giữ an toàn VN)
    s = s.replace("cộng  cộng", " ").replace("cộng cộng", " ")
    # Không xoá “cộng” đơn lẻ để tránh ăn nhầm từ hợp lệ (Cộng hoà, cộng tác, cộng đồng, ...)

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
def generate_answer(
    llm: BaseLLM,
    question: str,
    context: str,
    max_new_tokens: int = 64,
    temperature: Optional[float] = None
) -> str:
    try:
        prompt = build_prompt_ref_with_sample(question, context)
        return llm.generate(prompt, stop=STOP_TOKENS)
    except Exception as e:
        return f"[Generation ERROR] {e}"