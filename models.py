from __future__ import annotations

import json
import os
import re
import time
import urllib.request
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union


@dataclass
class ModelResponse:
    text: str
    prompt_tokens: Optional[int]
    completion_tokens: Optional[int]
    total_tokens: Optional[int]
    raw: Optional[Dict[str, Any]] = None
    estimated: bool = False


class ModelError(RuntimeError):
    pass


class Model:
    def __init__(self, name: str) -> None:
        self.name = name

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
        stop: Optional[List[str]] = None,
        # [FIX] 新增采样参数
        do_sample: Optional[bool] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
    ) -> ModelResponse:
        raise NotImplementedError


# ============================================================
# Model Series Detection and Max Tokens Configuration
# ============================================================

def _detect_model_series(model_name: str) -> str:
    """
    Detect model series from model name.
    
    Returns:
        "qwen3" for Qwen3 series
        "qwen2.5" for Qwen2.5 series
        "unknown" for other models
    """
    name_lower = model_name.lower()
    
    # Qwen3 series patterns
    qwen3_patterns = [
        r"qwen3",
        r"qwen-3",
        r"qwen_3",
        r"qwq",  # QwQ is Qwen3 based
    ]
    for pattern in qwen3_patterns:
        if re.search(pattern, name_lower):
            return "qwen3"
    
    # Qwen2.5 series patterns
    qwen25_patterns = [
        r"qwen2\.5",
        r"qwen2-5",
        r"qwen2_5",
        r"qwen-2\.5",
        r"qwen-2-5",
    ]
    for pattern in qwen25_patterns:
        if re.search(pattern, name_lower):
            return "qwen2.5"
    
    # Generic Qwen detection (default to qwen2.5 for backward compatibility)
    if "qwen" in name_lower:
        # Check for version indicators
        if "qwen-plus" in name_lower or "qwen-max" in name_lower or "qwen-flash" in name_lower or "qwen-turbo" in name_lower:
            # API models - check for qwen3 specific names
            if "qwen3" in name_lower:
                return "qwen3"
            return "qwen2.5"
        return "qwen2.5"
    
    return "unknown"


def _get_default_max_tokens(model_name: str, user_max_tokens: Optional[int]) -> Optional[int]:
    """Return an effective max_tokens (max_new_tokens) budget.

    We hard-cap generation to 8192 tokens to avoid runaway 'invalid thinking' and
    to make runs comparable across models. If the caller provides a smaller budget,
    we respect it.
    """
    name_lower = str(model_name or "").lower()
    cap = 8192
    allow_override = os.environ.get("ALLOW_MAX_TOKENS_OVERRIDE", "").strip().lower() in {
        "1",
        "true",
        "yes",
    }
    if user_max_tokens is None:
        return cap
    try:
        user_max_tokens_int = int(user_max_tokens)
    except Exception:
        return cap
    if user_max_tokens_int <= 0:
        return cap
    if allow_override:
        return user_max_tokens_int
    return min(user_max_tokens_int, cap)


def _resolve_sampling(
    temperature: float,
    do_sample: Optional[bool],
    default_temperature: float = 0.5,
) -> Tuple[bool, float]:
    """
    Normalize sampling settings to avoid conflicting flags.
    - If do_sample is False, force greedy (temperature=0.0) and ignore top_p/top_k.
    - If do_sample is True and temperature <= 0, use default_temperature.
    - If do_sample is None, enable sampling only when temperature > 0.
    """
    try:
        temp_val = float(temperature)
    except Exception:
        temp_val = 0.0
    if do_sample is False:
        return False, 0.0
    if do_sample is True:
        if temp_val <= 0.0:
            return True, default_temperature
        return True, temp_val
    if temp_val > 0.0:
        return True, temp_val
    return False, 0.0


class LocalHFModel(Model):
    def __init__(self, name: str) -> None:
        super().__init__(name)
        self._tokenizer = None
        self._model = None
        self._model_series = _detect_model_series(name)

    def _ensure_loaded(self) -> Tuple[Any, Any]:
        if self._tokenizer is None or self._model is None:
            try:
                import torch
                from transformers import AutoModelForCausalLM, AutoTokenizer
            except ImportError as exc:
                raise ModelError("transformers is required for local HF models") from exc
            self._tokenizer = AutoTokenizer.from_pretrained(self.name, trust_remote_code=True)
            if self._tokenizer.pad_token is None:
                self._tokenizer.pad_token = self._tokenizer.eos_token
            use_cuda = torch.cuda.is_available()
            if use_cuda and torch.cuda.is_bf16_supported():
                torch_dtype = torch.bfloat16
            elif use_cuda:
                torch_dtype = torch.float16
            else:
                torch_dtype = torch.float32
            model_kwargs: Dict[str, Any] = {
                "trust_remote_code": True,
                "torch_dtype": torch_dtype,
                "low_cpu_mem_usage": use_cuda,
            }
            if use_cuda:
                model_kwargs["device_map"] = "auto"
            self._model = AutoModelForCausalLM.from_pretrained(self.name, **model_kwargs)
            self._model.eval()
        return self._tokenizer, self._model

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
        stop: Optional[List[str]] = None,
        # [FIX] 新增采样参数
        do_sample: Optional[bool] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
    ) -> ModelResponse:
        tokenizer, model = self._ensure_loaded()
        full_prompt = _apply_chat_template_if_available(tokenizer, system_prompt, prompt)
        if not full_prompt:
            full_prompt = _merge_prompts(system_prompt, prompt)
        inputs = tokenizer(full_prompt, return_tensors="pt")
        if hasattr(model, "device"):
            inputs = {key: value.to(model.device) for key, value in inputs.items()}

        # Apply model-series-based max_tokens
        effective_max_tokens = _get_default_max_tokens(self.name, max_tokens) or 512

        generation_args: Dict[str, Any] = {
            "max_new_tokens": effective_max_tokens,
            "pad_token_id": tokenizer.pad_token_id,
        }

        # [FIX] 与 test_fixed.py 保持一致的采样参数
        use_sampling, eff_temp = _resolve_sampling(temperature, do_sample)
        if use_sampling:
            generation_args["do_sample"] = True
            generation_args["temperature"] = eff_temp
            # [FIX] 添加 top_p 和 top_k
            if top_p is not None:
                generation_args["top_p"] = top_p
            elif self._model_series in ("qwen3", "qwen2.5"):
                generation_args["top_p"] = 0.8  # 与 test_fixed.py 一致
            if top_k is not None:
                generation_args["top_k"] = top_k
            elif self._model_series in ("qwen3", "qwen2.5"):
                generation_args["top_k"] = 20  # 与 test_fixed.py 一致
        else:
            generation_args["do_sample"] = False

        stop_sequences = [s for s in (stop or []) if s]
        stop_criteria = []
        prompt_len = len(inputs["input_ids"][0])
        # Optional early stop: if the prompt requests \boxed{}, stop after we see a boxed answer.
        # This reduces runaway reasoning and saves compute.
        if "\\boxed" in full_prompt:
            try:
                from transformers import StoppingCriteria, StoppingCriteriaList  # type: ignore

                boxed_re = re.compile(r"\\\\boxed\s*\{[^\}]{0,32}\}")

                class _StopOnBoxed(StoppingCriteria):
                    def __init__(self, tok, prompt_len: int, window: int = 128):
                        super().__init__()
                        self.tok = tok
                        self.prompt_len = prompt_len
                        self.window = window

                    def __call__(self, input_ids, scores, **kwargs):
                        generated = input_ids[0][self.prompt_len:]
                        if generated.numel() == 0:
                            return False
                        # Decode only a small tail window to keep this cheap.
                        tail = generated[-self.window:]
                        tail_text = self.tok.decode(tail, skip_special_tokens=True)
                        return boxed_re.search(tail_text) is not None

                stop_criteria.append(_StopOnBoxed(tokenizer, prompt_len))
            except Exception:
                # If transformers stopping criteria isn't available, continue without it.
                pass

        if stop_sequences:
            try:
                from transformers import StoppingCriteria, StoppingCriteriaList  # type: ignore

                class _StopOnStrings(StoppingCriteria):
                    def __init__(self, tok, stops: List[str], prompt_len: int, window: int = 256):
                        super().__init__()
                        self.tok = tok
                        self.stops = stops
                        self.prompt_len = prompt_len
                        self.window = window

                    def __call__(self, input_ids, scores, **kwargs):
                        generated = input_ids[0][self.prompt_len:]
                        if generated.numel() == 0:
                            return False
                        tail = generated[-self.window:]
                        tail_text = self.tok.decode(tail, skip_special_tokens=True)
                        return any(stop in tail_text for stop in self.stops)

                stop_criteria.append(_StopOnStrings(tokenizer, stop_sequences, prompt_len))
            except Exception:
                pass

        if stop_criteria:
            try:
                from transformers import StoppingCriteriaList  # type: ignore

                generation_args["stopping_criteria"] = StoppingCriteriaList(stop_criteria)
            except Exception:
                pass
        output = model.generate(**inputs, **generation_args)
        prompt_tokens = len(inputs["input_ids"][0])
        # Decode only the newly generated tokens (avoid prompt-echo in response text).
        gen_ids = output[0][prompt_tokens:]
        text = tokenizer.decode(gen_ids, skip_special_tokens=True)
        if stop_sequences:
            cut_idx = min(
                (text.find(s) for s in stop_sequences if s and s in text),
                default=-1,
            )
            if cut_idx != -1:
                text = text[:cut_idx]
        completion_tokens = len(gen_ids)
        total_tokens = prompt_tokens + completion_tokens
        return ModelResponse(
            text=text,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            raw=None,
            estimated=False,
        )


def _normalize_api_bases(api_base: Union[str, List[str], Tuple[str, ...]]) -> List[str]:
    if isinstance(api_base, (list, tuple)):
        candidates = list(api_base)
    else:
        candidates = [api_base]
    bases: List[str] = []
    for base in candidates:
        base = str(base or "").strip().rstrip("/")
        if base:
            bases.append(base)
    return bases


class OpenAICompatibleModel(Model):
    def __init__(
        self,
        name: str,
        api_base: Union[str, List[str], Tuple[str, ...]],
        api_key: str,
        timeout: int = 180,
        max_retries: int = 3,
    ) -> None:
        super().__init__(name)
        self.api_bases = _normalize_api_bases(api_base)
        self.api_base = self.api_bases[0] if self.api_bases else ""
        self.api_key = api_key
        self.timeout = timeout
        self.max_retries = max_retries
        self._model_series = _detect_model_series(name)
        self._api_base_idx = 0

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
        stop: Optional[List[str]] = None,
        # [FIX] 新增采样参数，与 test_fixed.py 保持一致
        do_sample: Optional[bool] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
    ) -> ModelResponse:
        if not self.api_key:
            raise ModelError(f"API key missing for model {self.name}")
        messages: List[Dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        # Apply model-series-based max_tokens
        effective_max_tokens = _get_default_max_tokens(self.name, max_tokens)

        use_sampling, eff_temp = _resolve_sampling(temperature, do_sample)
        payload: Dict[str, Any] = {
            "model": self.name,
            "messages": messages,
            "temperature": eff_temp,
        }
        if effective_max_tokens is not None:
            payload["max_tokens"] = effective_max_tokens
        if stop:
            payload["stop"] = stop

        # [FIX] 添加采样参数（与 test_fixed.py 一致）
        # 官方文档: "For non-thinking mode, use Temperature=0.5, TopP=0.8, TopK=20"
        # 官方文档: "DO NOT use greedy decoding, as it can lead to performance degradation"
        if use_sampling:
            # 使用采样模式
            if top_p is not None:
                payload["top_p"] = top_p
            elif self._model_series in ("qwen3", "qwen2.5"):
                # [FIX] 默认 top_p=0.8，与 test_fixed.py 一致
                payload["top_p"] = 0.8

            if top_k is not None:
                payload["top_k"] = top_k
            elif self._model_series in ("qwen3", "qwen2.5"):
                # [FIX] 默认 top_k=20，与 test_fixed.py 一致
                payload["top_k"] = 20

        # Only set enable_thinking=False for models that support thinking mode
        model_lower = self.name.lower()
        supports_thinking = any(
            x in model_lower
            for x in [
                "qwen-plus",
                "qwen-flash",
                "qwen-turbo",
                "qwq",
                "qwen3-max-preview",
                "qwen3-max",
                "qwen-max",
            ]
        )
        if supports_thinking:
            payload["enable_thinking"] = False


        last_err: Optional[Exception] = None
        for attempt in range(self.max_retries):
            try:
                base = self.api_bases[(self._api_base_idx + attempt) % len(self.api_bases)]
                url = f"{base}/v1/chat/completions"
                data = _post_json(url, payload, self.api_key, timeout=self.timeout)
                choice = data["choices"][0]["message"]["content"]
                usage = data.get("usage") or {}
                prompt_tokens = usage.get("prompt_tokens")
                completion_tokens = usage.get("completion_tokens")
                total_tokens = usage.get("total_tokens")
                estimated = prompt_tokens is None or completion_tokens is None or total_tokens is None
                if estimated:
                    prompt_tokens = prompt_tokens or _estimate_tokens(system_prompt or "") + _estimate_tokens(prompt)
                    completion_tokens = completion_tokens or _estimate_tokens(choice)
                    total_tokens = total_tokens or (prompt_tokens + completion_tokens)
                self._api_base_idx = (self._api_base_idx + attempt + 1) % len(self.api_bases)
                return ModelResponse(
                    text=choice,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=total_tokens,
                    raw=data,
                    estimated=estimated,
                )
            except Exception as exc:
                last_err = exc
                time.sleep(1.5 * (attempt + 1))
        raise ModelError(f"API call failed after retries: {last_err}")


class GeminiModel(Model):
    def __init__(
        self,
        name: str,
        api_key: str,
        timeout: int = 180,
        max_retries: int = 3,
    ) -> None:
        super().__init__(name)
        self.api_key = api_key
        self.timeout = timeout
        self.max_retries = max_retries

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
        stop: Optional[List[str]] = None,
        # [FIX] 新增采样参数（Gemini API 支持 top_p 和 top_k）
        do_sample: Optional[bool] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
    ) -> ModelResponse:
        if not self.api_key:
            raise ModelError(f"API key missing for model {self.name}")

        full_prompt = _merge_prompts(system_prompt, prompt)
        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self.name}:generateContent?key={self.api_key}"
        )
        use_sampling, eff_temp = _resolve_sampling(temperature, do_sample)
        payload: Dict[str, Any] = {
            "contents": [{"role": "user", "parts": [{"text": full_prompt}]}],
            "generationConfig": {"temperature": eff_temp},
        }
        if max_tokens is not None:
            payload["generationConfig"]["maxOutputTokens"] = max_tokens
        if stop:
            payload["generationConfig"]["stopSequences"] = stop
        # [FIX] 添加采样参数支持
        if use_sampling and top_p is not None:
            payload["generationConfig"]["topP"] = top_p
        if use_sampling and top_k is not None:
            payload["generationConfig"]["topK"] = top_k
        # do_sample 在 Gemini 中由 temperature > 0 隐式控制
        _ = do_sample  # 避免 unused 警告

        last_err: Optional[Exception] = None
        for attempt in range(self.max_retries):
            try:
                data = _post_json(url, payload, None, timeout=self.timeout)
                candidates = data.get("candidates") or []
                if not candidates:
                    raise ModelError("No candidates returned from Gemini")
                parts = candidates[0].get("content", {}).get("parts") or []
                text = parts[0].get("text", "") if parts else ""
                prompt_tokens = _estimate_tokens(full_prompt)
                completion_tokens = _estimate_tokens(text)
                total_tokens = prompt_tokens + completion_tokens
                return ModelResponse(
                    text=text,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=total_tokens,
                    raw=data,
                    estimated=True,
                )
            except Exception as exc:
                last_err = exc
                time.sleep(1.5 * (attempt + 1))
        raise ModelError(f"Gemini API call failed after retries: {last_err}")


_GEMINI_TOKEN_CACHE: Dict[str, Tuple[str, float, str]] = {}


def _get_gemini_credentials_path() -> Optional[str]:
    env_path = os.environ.get("GEMINI_CREDENTIALS") or os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    if env_path and os.path.exists(env_path):
        return env_path
    return None


def _load_gemini_vertex_token(credentials_path: str) -> Optional[Tuple[str, str]]:
    if not credentials_path or not os.path.exists(credentials_path):
        return None
    cached = _GEMINI_TOKEN_CACHE.get(credentials_path)
    if cached:
        token, expiry_ts, project_id = cached
        if expiry_ts - time.time() > 60 and token and project_id:
            return token, project_id
    try:
        from google.oauth2 import service_account
        from google.auth.transport.requests import Request
    except ImportError:
        return None
    try:
        creds = service_account.Credentials.from_service_account_file(
            credentials_path,
            scopes=["https://www.googleapis.com/auth/cloud-platform"],
        )
        creds.refresh(Request())
        project_id = creds.project_id
        if not project_id:
            with open(credentials_path, "r", encoding="utf-8") as f:
                project_id = json.load(f).get("project_id")
        if not project_id:
            return None
        token = creds.token
        expiry = getattr(creds, "expiry", None)
        expiry_ts = time.time() + 300
        if expiry is not None:
            try:
                expiry_ts = expiry.timestamp()
            except Exception:
                pass
        if token:
            _GEMINI_TOKEN_CACHE[credentials_path] = (token, expiry_ts, project_id)
        return token, project_id
    except Exception:
        return None


class VertexGeminiModel(Model):
    def __init__(
        self,
        name: str,
        credentials_path: str,
        location: str = "us-central1",
        timeout: int = 180,
        max_retries: int = 3,
    ) -> None:
        super().__init__(name)
        self.credentials_path = credentials_path
        self.location = location
        self.timeout = timeout
        self.max_retries = max_retries

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
        stop: Optional[List[str]] = None,
        do_sample: Optional[bool] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
    ) -> ModelResponse:
        token_and_project = _load_gemini_vertex_token(self.credentials_path)
        if not token_and_project:
            raise ModelError("Gemini Vertex credentials missing or invalid.")
        access_token, project_id = token_and_project
        full_prompt = _merge_prompts(system_prompt, prompt)

        url = (
            f"https://{self.location}-aiplatform.googleapis.com/v1/projects/{project_id}/"
            f"locations/{self.location}/publishers/google/models/{self.name}:generateContent"
        )
        use_sampling, eff_temp = _resolve_sampling(temperature, do_sample)
        generation_config: Dict[str, Any] = {"temperature": eff_temp}
        if max_tokens is not None:
            generation_config["maxOutputTokens"] = max_tokens
        if stop:
            generation_config["stopSequences"] = stop
        if use_sampling and top_p is not None:
            generation_config["topP"] = top_p
        if use_sampling and top_k is not None:
            # Gemini 2.0 Flash requires topK=64 (fixed). Avoid 400s by forcing it.
            if "gemini-2.0-flash" in self.name and top_k != 64:
                top_k = 64
            generation_config["topK"] = top_k

        payload: Dict[str, Any] = {
            "contents": [{"role": "user", "parts": [{"text": full_prompt}]}],
            "generationConfig": generation_config,
        }
        _ = do_sample  # avoid unused warning

        last_err: Optional[Exception] = None
        for attempt in range(self.max_retries):
            try:
                data = _post_json(url, payload, access_token, timeout=self.timeout)
                candidates = data.get("candidates") or []
                if not candidates:
                    raise ModelError("No candidates returned from Gemini Vertex")
                content = candidates[0].get("content", {}) if isinstance(candidates[0], dict) else {}
                parts = content.get("parts", []) or []
                texts: List[str] = []
                for part in parts:
                    if isinstance(part, dict) and part.get("text"):
                        texts.append(str(part["text"]))
                text = "".join(texts).strip()
                prompt_tokens = _estimate_tokens(full_prompt)
                completion_tokens = _estimate_tokens(text)
                total_tokens = prompt_tokens + completion_tokens
                return ModelResponse(
                    text=text,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=total_tokens,
                    raw=data,
                    estimated=True,
                )
            except Exception as exc:
                last_err = exc
                time.sleep(1.5 * (attempt + 1))
        raise ModelError(f"Gemini Vertex API call failed after retries: {last_err}")


def _resolve_gemini_fallback_model(model_name: str) -> str:
    override = os.environ.get("GEMINI_FALLBACK_MODEL") or os.environ.get("QWEN_FALLBACK_MODEL", "")
    override = override.strip()
    if override:
        lowered = override.lower()
        if lowered.startswith("gemini"):
            override = ""
        elif ":" not in override:
            override = f"qwen:{override}"
    if override:
        return override
    lowered = model_name.lower()
    if "pro" in lowered or "ultra" in lowered:
        return "qwen:qwen-max"
    if "flash" in lowered:
        return "qwen:qwen-flash"
    return "qwen:qwen-flash"


def _get_qwen_api_key() -> str:
    """
    Get Qwen API Key, supports multiple environment variable names.
    Priority: QWEN_API_KEY > DASHSCOPE_API_KEY
    """
    key = os.environ.get("QWEN_API_KEY", "")
    if not key:
        key = os.environ.get("DASHSCOPE_API_KEY", "")
    return key


def _parse_api_bases(env_value: str) -> List[str]:
    if not env_value:
        return []
    parts = re.split(r"[,\s]+", env_value.strip())
    return [part for part in (p.strip() for p in parts) if part]


# Qwen model name mapping (old name -> actual API name)
QWEN_MODEL_ALIASES = {
    "qwen-max": "qwen3-max",
    "qwen-plus": "qwen-plus", 
    "qwen-flash": "qwen-flash",
    "qwen-turbo": "qwen-turbo",
}

# Only these Qwen models are allowed to use API by default.
QWEN_API_ALLOWLIST = {
    "qwen3-max",
    "qwen-plus",
    "qwen-flash",
}


def _resolve_qwen_model_name(name: str) -> str:
    """Resolve model alias to actual API model name"""
    return QWEN_MODEL_ALIASES.get(name, name)

def _is_qwen_api_allowed(name: str) -> bool:
    resolved = _resolve_qwen_model_name(name).lower()
    return resolved in QWEN_API_ALLOWLIST


def _load_qwen_api_model(
    model_name: str,
    effective_timeout: int,
    effective_retries: int,
) -> "OpenAICompatibleModel":
    model_name = _resolve_qwen_model_name(model_name)
    api_key = _get_qwen_api_key()
    api_bases_env = os.environ.get("QWEN_API_BASES", "")
    api_bases = _parse_api_bases(api_bases_env)
    if not api_bases:
        api_bases = [
            os.environ.get(
                "QWEN_API_BASE",
                "https://dashscope.aliyuncs.com/compatible-mode",
            )
        ]
    return OpenAICompatibleModel(
        model_name,
        api_base=api_bases,
        api_key=api_key,
        timeout=effective_timeout,
        max_retries=effective_retries,
    )


def load_model(
    name: str,
    timeout: Optional[int] = None,
    max_retries: Optional[int] = None,
) -> Model:
    """
    Load a model by name.
    
    Args:
        name: Model name (e.g., "qwen:qwen-plus", "hf:Qwen/Qwen2.5-7B")
        timeout: Optional timeout for API calls (default: 180)
        max_retries: Optional max retries for API calls (default: 3)
    
    Returns:
        Model instance
    """
    effective_timeout = timeout if timeout is not None else 180
    effective_retries = max_retries if max_retries is not None else 3
    
    if name.startswith("hf:"):
        return LocalHFModel(name[len("hf:"):])
    
    lowered = name.lower()
    
    if lowered.startswith("gemini:"):
        model_name = name.split(":", 1)[1]
        api_key = os.environ.get("GEMINI_API_KEY", "")
        if api_key:
            return GeminiModel(model_name, api_key=api_key, timeout=effective_timeout, max_retries=effective_retries)
        credentials_path = _get_gemini_credentials_path()
        if credentials_path:
            location = os.environ.get("GEMINI_LOCATION", "us-central1")
            return VertexGeminiModel(
                model_name,
                credentials_path=credentials_path,
                location=location,
                timeout=effective_timeout,
                max_retries=effective_retries,
            )
        return load_model(_resolve_gemini_fallback_model(model_name), timeout, max_retries)
    
    if "gemini" in lowered:
        api_key = os.environ.get("GEMINI_API_KEY", "")
        if api_key:
            return GeminiModel(name, api_key=api_key, timeout=effective_timeout, max_retries=effective_retries)
        credentials_path = _get_gemini_credentials_path()
        if credentials_path:
            location = os.environ.get("GEMINI_LOCATION", "us-central1")
            return VertexGeminiModel(
                name,
                credentials_path=credentials_path,
                location=location,
                timeout=effective_timeout,
                max_retries=effective_retries,
            )
        return load_model(_resolve_gemini_fallback_model(name), timeout, max_retries)
    
    if lowered.startswith("qwen:"):
        model_name = name.split(":", 1)[1]
        if not _is_qwen_api_allowed(model_name):
            raise ModelError(
                f"Qwen API model not allowed: {model_name}. "
                f"Allowed: {', '.join(sorted(QWEN_API_ALLOWLIST))}"
            )
        return _load_qwen_api_model(model_name, effective_timeout, effective_retries)
    
    if lowered.startswith("deepseek:") or "deepseek" in lowered:
        model_name = name.split(":", 1)[1] if lowered.startswith("deepseek:") else name
        api_key = os.environ.get("DEEPSEEK_API_KEY", "")
        api_base = os.environ.get("DEEPSEEK_API_BASE", "https://api.deepseek.com")
        return OpenAICompatibleModel(
            model_name, 
            api_base=api_base, 
            api_key=api_key,
            timeout=effective_timeout,
            max_retries=effective_retries,
        )
    
    if "qwen" in lowered and "/" not in name:
        # If API key is available, default to Qwen API for bare names like "qwen3-max".
        if _get_qwen_api_key() and _is_qwen_api_allowed(name):
            return _load_qwen_api_model(name, effective_timeout, effective_retries)
        return LocalHFModel(f"Qwen/{name}")
    
    return LocalHFModel(name)


def _merge_prompts(system_prompt: Optional[str], prompt: str) -> str:
    if system_prompt:
        return f"SYSTEM:\n{system_prompt}\n\nUSER:\n{prompt}"
    return prompt


def _apply_chat_template_if_available(
    tokenizer: Any,
    system_prompt: Optional[str],
    prompt: str,
) -> Optional[str]:
    """Best-effort chat template rendering for local HF models."""
    def _log_chat_template_warning(reason: str) -> None:
        if getattr(tokenizer, "_chat_template_warned", False):
            return
        try:
            setattr(tokenizer, "_chat_template_warned", True)
        except Exception:
            pass
        name = getattr(tokenizer, "name_or_path", None)
        model_hint = f" (tokenizer={name})" if name else ""
        print(f"Warning: apply_chat_template failed{model_hint}; falling back to plain prompt. {reason}")

    if not hasattr(tokenizer, "apply_chat_template"):
        return None
    messages: List[Dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    try:
        rendered = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    except TypeError:
        # Older tokenizers may not support add_generation_prompt
        try:
            rendered = tokenizer.apply_chat_template(messages, tokenize=False)
        except Exception:
            _log_chat_template_warning("Fallback apply_chat_template failed.")
            return None
    except Exception:
        _log_chat_template_warning("apply_chat_template raised an exception.")
        return None
    if isinstance(rendered, str) and rendered.strip():
        return rendered
    _log_chat_template_warning("apply_chat_template returned empty text.")
    return None


def _estimate_tokens(text: str) -> int:
    text = text or ""
    return max(1, int(len(text) / 4))


def _post_json(url: str, payload: Dict[str, Any], api_key: Optional[str], timeout: int = 60) -> Dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = resp.read().decode("utf-8")
    return json.loads(body)
