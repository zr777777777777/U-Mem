from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Iterator, Optional, Union


@dataclass
class Sample:
    id: str
    question: str
    answer: Optional[Union[str, int, float]]
    meta: Dict[str, Any]


def _normalize_hf_dataset_path(path: str) -> str:
    normalized = path.strip()
    if normalized.startswith(("http://", "https://")) and "huggingface.co/datasets/" in normalized:
        normalized = normalized.split("/datasets/", 1)[1]
        normalized = normalized.split("?", 1)[0].strip("/")
        parts = normalized.split("/")
        if len(parts) >= 2:
            normalized = "/".join(parts[:2])
    return normalized


def _normalize_key(key: str) -> str:
    key = key.strip().lower()
    key = re.sub(r"[\s\-]+", "_", key)
    key = re.sub(r"[^a-z0-9_]", "", key)
    return key


def _is_rubrichub_dataset(path: str) -> bool:
    lowered = _normalize_hf_dataset_path(path).lower()
    return lowered in {"sojul/rubrichub_v1", "sojul/rubrichub"} or "rubrichub_v1" in lowered


def _is_wildif_dataset(path: str) -> bool:
    lowered = _normalize_hf_dataset_path(path).lower()
    return lowered in {"gililior/wild-if-eval", "wild-if-eval", "wild_if_eval", "wildif"}


def _safe_json_loads(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (dict, list)):
        return value
    if not isinstance(value, str):
        return value
    text = value.strip()
    if not text:
        return value
    try:
        return json.loads(text)
    except Exception:
        return value


def _try_parse_list_string(text: str) -> Optional[list]:
    if not text:
        return None
    stripped = str(text).strip()
    if not stripped:
        return None
    try:
        parsed = json.loads(stripped)
        if isinstance(parsed, (list, tuple)):
            return [str(x).strip() for x in parsed if str(x).strip()]
    except Exception:
        pass
    try:
        parsed = ast.literal_eval(stripped)
        if isinstance(parsed, (list, tuple)):
            return [str(x).strip() for x in parsed if str(x).strip()]
    except Exception:
        pass
    if stripped.startswith("[") and stripped.endswith("]"):
        items = re.findall(r'"([^"]+)"', stripped)
        if not items:
            items = re.findall(r"'([^']+)'", stripped)
        if items:
            return [item.strip() for item in items if item.strip()]
    return None


def _coerce_sample(raw: Dict[str, Any], idx: int) -> Sample:
    normalized_keys: Dict[str, str] = {}
    for key in raw.keys():
        norm = _normalize_key(str(key))
        if norm not in normalized_keys:
            normalized_keys[norm] = str(key)

    def _get_value(keys: Iterable[str]) -> Optional[Any]:
        for key in keys:
            if key in raw and raw[key] is not None:
                return raw[key]
            norm = _normalize_key(key)
            orig = normalized_keys.get(norm)
            if orig is not None and raw.get(orig) is not None:
                return raw[orig]
        return None

    question_keys = [
        "instruction",
        "original_instruction",
        "question",
        "query",
        "prompt",
        "problem",
        "problem_text",
        "problem_statement",
        "statement",
        "stem",
        "text",
        "input",
    ]
    answer_keys = [
        "answer",
        "ground_truth",
        "final_answer",
        "solution",
        "result",
        "ans",
    ]
    id_keys = [
        "id",
        "sample_id",
        "problem_id",
        "problemid",
        "question_id",
        "qid",
        "uid",
    ]

    # Support multiple question field names
    question = _get_value(question_keys)
    if question is None:
        raise ValueError("Sample missing question field")
    question = str(question).strip()
    if not question:
        raise ValueError("Sample missing question field")

    # Support multiple ID field names
    sample_id = _get_value(id_keys)
    if sample_id is None:
        sample_id = str(idx)
    
    # Support multiple answer field names
    answer = _get_value(answer_keys)

    # Everything else goes to meta
    excluded_norm = {_normalize_key(k) for k in (question_keys + answer_keys + id_keys)}
    meta = {k: v for k, v in raw.items() if _normalize_key(str(k)) not in excluded_norm}

    return Sample(id=str(sample_id), question=question, answer=answer, meta=meta)


def load_dataset(path: str, max_samples: Optional[int] = None) -> Iterable[Sample]:
    """
    Load dataset from various sources.
    
    Supports:
    - Local JSONL files (.jsonl)
    - Local JSON files (.json)
    - HuggingFace datasets (e.g., "AI-MO/aimo-validation-aime")
    - Aliases: aime25, math500
    - HotpotQA alias supports config/split: hotpotqa[:config][:split]
    """
    path = _normalize_hf_dataset_path(path)
    # Alias for AIME 2025 dataset
    if path.lower() in {"aime25", "aime-25", "aime_25"}:
        return load_aime25_dataset(max_samples=max_samples)
    # Alias for MATH-500 dataset
    if path.lower() in {"math500", "math-500", "math_500"}:
        return load_math500_dataset(max_samples=max_samples)
    # Alias for HotpotQA dataset (support optional config + split)
    # Examples: hotpotqa, hotpotqa:validation, hotpotqa:distractor, hotpotqa:distractor:train
    lowered = path.lower()
    if lowered.startswith(("hotpotqa", "hotpot-qa", "hotpot_qa")):
        parts = path.split(":")
        base = parts[0]
        config_name = None
        split = None
        if len(parts) == 2:
            if parts[1] in {"train", "validation", "test"}:
                split = parts[1]
            else:
                config_name = parts[1]
        elif len(parts) >= 3:
            config_name = parts[1]
            split = parts[2]
        base_lower = base.lower()
        if base_lower in {"hotpotqa", "hotpot-qa", "hotpot_qa"}:
            dataset_name = "hotpot_qa" if base_lower != "hotpot_qa" else base
            return load_hotpotqa_dataset(
                dataset_name=dataset_name,
                config_name=config_name,
                split=split,
                max_samples=max_samples,
            )

    # HelpSteer3 dataset (support optional config + split)
    # Examples: nvidia/HelpSteer3:preference:train, helpsteer3:preference, helpsteer3:train
    if lowered.startswith(("helpsteer3", "nvidia/helpsteer3")):
        parts = path.split(":")
        base = parts[0]
        config_name = None
        split = None
        if len(parts) == 2:
            if parts[1] in {"train", "validation", "test"}:
                split = parts[1]
            else:
                config_name = parts[1]
        elif len(parts) >= 3:
            config_name = parts[1]
            split = parts[2]
        dataset_name = base
        if base.lower() in {"helpsteer3"}:
            dataset_name = "nvidia/HelpSteer3"
        return load_helpsteer3_dataset(
            dataset_name=dataset_name,
            config_name=config_name,
            split=split,
            max_samples=max_samples,
        )

    # HelpSteer2 dataset (support optional split)
    # Examples: nvidia/HelpSteer2:train, helpsteer2:validation
    if lowered.startswith(("helpsteer2", "nvidia/helpsteer2")):
        parts = path.split(":")
        base = parts[0]
        split = None
        if len(parts) == 2 and parts[1] in {"train", "validation", "test"}:
            split = parts[1]
        elif len(parts) >= 3 and parts[2] in {"train", "validation", "test"}:
            split = parts[2]
        dataset_name = base
        if base.lower() in {"helpsteer2"}:
            dataset_name = "nvidia/HelpSteer2"
        return load_helpsteer2_dataset(
            dataset_name=dataset_name,
            split=split,
            max_samples=max_samples,
        )
    if _is_wildif_dataset(path):
        parts = path.split(":")
        base = parts[0]
        split = None
        if len(parts) >= 2 and parts[1] in {"train", "validation", "test"}:
            split = parts[1]
        dataset_name = base
        if base.lower() in {"wild-if-eval", "wild_if_eval", "wildif"}:
            dataset_name = "gililior/wild-if-eval"
        return load_wildif_dataset(
            dataset_name=dataset_name,
            split=split,
            max_samples=max_samples,
        )

    # AdvancedIF dataset (support optional split)
    if lowered.startswith(("advancedif", "facebook/advancedif", "meta-llama/advancedif")):
        parts = path.split(":")
        base = parts[0]
        split = None
        if len(parts) == 2 and parts[1] in {"train", "validation", "test"}:
            split = parts[1]
        elif len(parts) >= 3 and parts[2] in {"train", "validation", "test"}:
            split = parts[2]
        dataset_name = base
        if base.lower() == "advancedif":
            dataset_name = "facebook/AdvancedIF"
        return load_advancedif_dataset(
            dataset_name=dataset_name,
            split=split,
            max_samples=max_samples,
        )

    # BiGGen-Bench dataset (prometheus-eval/BiGGen-Bench)
    if lowered.startswith(("biggen-bench", "biggen_bench", "prometheus-eval/biggen-bench")):
        parts = path.split(":")
        base = parts[0]
        split = None
        if len(parts) == 2 and parts[1] in {"train", "validation", "test"}:
            split = parts[1]
        elif len(parts) >= 3 and parts[2] in {"train", "validation", "test"}:
            split = parts[2]
        dataset_name = base
        if base.lower() in {"biggen-bench", "biggen_bench"}:
            dataset_name = "prometheus-eval/BiGGen-Bench"
        return load_biggen_bench_dataset(
            dataset_name=dataset_name,
            split=split,
            max_samples=max_samples,
        )

    # Check if it's a HuggingFace dataset path
    if not path.endswith((".jsonl", ".json")) and "/" in path:
        return _load_huggingface(path, max_samples)
    
    if path.endswith(".jsonl"):
        return _load_jsonl(path, max_samples)
    if path.endswith(".json"):
        return _load_json(path, max_samples)
    
    raise ValueError(f"Unsupported dataset format: {path}")


def _load_jsonl(path: str, max_samples: Optional[int]) -> Iterator[Sample]:
    with open(path, "r", encoding="utf-8") as handle:
        for idx, line in enumerate(handle):
            if max_samples is not None and idx >= max_samples:
                break
            line = line.strip()
            if not line:
                continue
            raw = json.loads(line)
            yield _coerce_sample(raw, idx)


def _load_json(path: str, max_samples: Optional[int]) -> Iterator[Sample]:
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError("JSON dataset must be a list of samples")
    for idx, raw in enumerate(data):
        if max_samples is not None and idx >= max_samples:
            break
        if not isinstance(raw, dict):
            raise ValueError("JSON dataset samples must be objects")
        yield _coerce_sample(raw, idx)


def _load_huggingface(dataset_name: str, max_samples: Optional[int]) -> Iterator[Sample]:
    """Load dataset from HuggingFace Hub."""
    try:
        from datasets import load_dataset as hf_load_dataset
    except ImportError:
        raise ImportError(
            "HuggingFace datasets library required. "
            "Install with: pip install datasets"
        )
    
    # Parse dataset name and split
    # Format: "owner/dataset" or "owner/dataset:split"
    if ":" in dataset_name:
        name, split = dataset_name.rsplit(":", 1)
    else:
        name = dataset_name
        split = None
    
    def _select_split(dataset_obj: Any) -> Any:
        if isinstance(dataset_obj, dict):
            if "test" in dataset_obj:
                return dataset_obj["test"]
            if "validation" in dataset_obj:
                return dataset_obj["validation"]
            if "train" in dataset_obj:
                return dataset_obj["train"]
            return dataset_obj[list(dataset_obj.keys())[0]]
        return dataset_obj

    # Load dataset
    try:
        if split:
            ds = hf_load_dataset(name, split=split)
        else:
            ds = hf_load_dataset(name)
            ds = _select_split(ds)
    except Exception:
        # RubricHub has mixed column types; fallback to streaming to avoid Arrow casting errors.
        if _is_rubrichub_dataset(name):
            if split:
                ds = hf_load_dataset(name, split=split, streaming=True)
            else:
                ds = hf_load_dataset(name, streaming=True)
                ds = _select_split(ds)
        else:
            raise
    
    # Iterate and convert
    for idx, raw in enumerate(ds):
        if max_samples is not None and idx >= max_samples:
            break
        yield _coerce_sample(dict(raw), idx)


# ============================================================
# HelpSteer3 preference dataset loader (nvidia/HelpSteer3)
# ============================================================

def _format_helpsteer3_context(context: Any) -> str:
    if context is None:
        return ""
    if isinstance(context, str):
        return context.strip()
    if isinstance(context, dict):
        # If a dict is provided, try common fields.
        content = context.get("content") or context.get("text") or context.get("value")
        role = context.get("role") or context.get("speaker")
        if content:
            prefix = str(role).strip().title() + ": " if role else ""
            return f"{prefix}{content}".strip()
        return json.dumps(context, ensure_ascii=False)
    if not isinstance(context, list):
        return str(context).strip()

    lines: list[str] = []
    for entry in context:
        if entry is None:
            continue
        if isinstance(entry, str):
            if entry.strip():
                lines.append(entry.strip())
            continue
        if isinstance(entry, dict):
            role = entry.get("role") or entry.get("speaker") or ""
            content = entry.get("content") or entry.get("text") or entry.get("value") or ""
            if not content:
                continue
            role_label = str(role).strip().title() if role else ""
            if role_label:
                lines.append(f"{role_label}: {content}".strip())
            else:
                lines.append(str(content).strip())
            continue
        # Fallback
        lines.append(str(entry).strip())
    return "\n".join([line for line in lines if line]).strip()


def load_helpsteer3_dataset(
    dataset_name: str = "nvidia/HelpSteer3",
    config_name: Optional[str] = None,
    split: Optional[str] = None,
    max_samples: Optional[int] = None,
) -> Iterator[Sample]:
    """
    Load HelpSteer3 dataset (preference subset by default) from HuggingFace.

    Expected fields (preference config):
    - context: list of {role, content}
    - response1, response2: candidate responses
    - overall_preference: int in [-3, 3]
    - individual_preference: optional dimension-level preferences
    """
    try:
        from datasets import load_dataset as hf_load_dataset
    except ImportError:
        raise ImportError("pip install datasets")

    if config_name is None:
        config_name = "preference"

    if split:
        ds = hf_load_dataset(dataset_name, name=config_name, split=split)
    else:
        ds = hf_load_dataset(dataset_name, name=config_name)
        if isinstance(ds, dict):
            if "train" in ds:
                ds = ds["train"]
            elif "validation" in ds:
                ds = ds["validation"]
            elif "test" in ds:
                ds = ds["test"]
            else:
                ds = ds[list(ds.keys())[0]]

    for idx, raw in enumerate(ds):
        if max_samples is not None and idx >= max_samples:
            break
        raw_dict = dict(raw)
        context_raw = raw_dict.get("context")
        context_text = _format_helpsteer3_context(context_raw)
        response1 = (
            raw_dict.get("response1")
            or raw_dict.get("response_1")
            or raw_dict.get("response_a")
        )
        response2 = (
            raw_dict.get("response2")
            or raw_dict.get("response_2")
            or raw_dict.get("response_b")
        )
        overall_pref = raw_dict.get("overall_preference")

        # If the dataset doesn't have expected fields, skip gracefully.
        if response1 is None or response2 is None:
            continue

        sample_id = (
            raw_dict.get("id")
            or raw_dict.get("sample_id")
            or raw_dict.get("uid")
            or f"helpsteer3_{idx}"
        )

        question = context_text
        if not question:
            # Fallback if context is missing
            question = (
                raw_dict.get("prompt")
                or raw_dict.get("instruction")
                or raw_dict.get("query")
                or ""
            )

        meta = {
            k: v for k, v in raw_dict.items()
            if k not in {
                "id",
                "sample_id",
                "uid",
                "context",
                "response1",
                "response_1",
                "response_a",
                "response2",
                "response_2",
                "response_b",
                "overall_preference",
            }
        }
        meta.update(
            {
                "context": context_text,
                "response1": response1,
                "response2": response2,
                "overall_preference": overall_pref,
                "source": dataset_name,
                "config": config_name,
            }
        )

        yield Sample(id=str(sample_id), question=str(question), answer=overall_pref, meta=meta)


# ============================================================
# HelpSteer2 dataset loader (nvidia/HelpSteer2)
# ============================================================

def load_helpsteer2_dataset(
    dataset_name: str = "nvidia/HelpSteer2",
    split: Optional[str] = None,
    max_samples: Optional[int] = None,
    dedupe_prompts: bool = True,
) -> Iterator[Sample]:
    """
    Load HelpSteer2 dataset from HuggingFace.

    Expected fields:
    - prompt: str
    - response: str
    - helpfulness, correctness, coherence, complexity, verbosity: int64 (1-5)

    Notes:
    - The same prompt may appear multiple times (different responses). By default we
      deduplicate by prompt text to avoid repeated prompts.
    """
    try:
        from datasets import load_dataset as hf_load_dataset
    except ImportError:
        raise ImportError("pip install datasets")

    if split:
        ds = hf_load_dataset(dataset_name, split=split)
    else:
        ds = hf_load_dataset(dataset_name)
        if isinstance(ds, dict):
            if "train" in ds:
                ds = ds["train"]
            elif "validation" in ds:
                ds = ds["validation"]
            elif "test" in ds:
                ds = ds["test"]
            else:
                ds = ds[list(ds.keys())[0]]

    seen_prompts: set[str] = set()
    collected = 0
    for idx, raw in enumerate(ds):
        raw_dict = dict(raw)
        prompt = (
            raw_dict.get("prompt")
            or raw_dict.get("instruction")
            or raw_dict.get("query")
            or raw_dict.get("question")
            or ""
        )
        prompt = str(prompt).strip()
        if not prompt:
            continue
        if dedupe_prompts and prompt in seen_prompts:
            continue
        if max_samples is not None and collected >= max_samples:
            break
        if dedupe_prompts:
            seen_prompts.add(prompt)
        collected += 1

        response = raw_dict.get("response") or raw_dict.get("output")
        sample_id = (
            raw_dict.get("id")
            or raw_dict.get("sample_id")
            or raw_dict.get("uid")
            or f"helpsteer2_{idx}"
        )

        meta = {
            k: v for k, v in raw_dict.items()
            if k
            not in {
                "id",
                "sample_id",
                "uid",
                "prompt",
                "instruction",
                "query",
                "question",
            }
        }
        meta.update(
            {
                "prompt": prompt,
                "response": response,
                "source": dataset_name,
                "split": split or "",
            }
        )

        yield Sample(id=str(sample_id), question=prompt, answer=None, meta=meta)


# ============================================================
# AdvancedIF dataset loader (facebook/AdvancedIF)
# ============================================================

def _format_advancedif_conversation(history: Any) -> str:
    if history is None:
        return ""
    if isinstance(history, str):
        parsed = _safe_json_loads(history)
        if parsed is not history:
            return _format_advancedif_conversation(parsed)
        return history.strip()
    if isinstance(history, dict):
        if "messages" in history:
            return _format_advancedif_conversation(history.get("messages"))
        return json.dumps(history, ensure_ascii=False)
    if not isinstance(history, list):
        return str(history).strip()
    lines: list[str] = []
    for entry in history:
        if entry is None:
            continue
        if isinstance(entry, str):
            if entry.strip():
                lines.append(entry.strip())
            continue
        if isinstance(entry, dict):
            role = entry.get("role") or entry.get("speaker") or ""
            content = entry.get("content") or entry.get("text") or entry.get("value") or ""
            if not content:
                continue
            role_label = str(role).strip().title() if role else ""
            if role_label:
                lines.append(f"{role_label}: {content}".strip())
            else:
                lines.append(str(content).strip())
            continue
        lines.append(str(entry).strip())
    return "\n".join([line for line in lines if line]).strip()


def _extract_lmsys_messages(raw: Dict[str, Any]) -> list[tuple[str, str]]:
    if not isinstance(raw, dict):
        return []
    candidates = ("conversation", "conversations", "messages", "turns", "dialogue", "chat")
    payload = None
    for key in candidates:
        if key in raw:
            payload = raw.get(key)
            break
    if isinstance(payload, str):
        parsed = _safe_json_loads(payload)
        if parsed is not payload:
            payload = parsed
    if isinstance(payload, dict) and "messages" in payload:
        payload = payload.get("messages")
    if not isinstance(payload, list):
        payload = None
    messages: list[tuple[str, str]] = []
    if isinstance(payload, list):
        for entry in payload:
            if entry is None:
                continue
            if isinstance(entry, str):
                text = entry.strip()
                if text:
                    messages.append(("", text))
                continue
            if not isinstance(entry, dict):
                text = str(entry).strip()
                if text:
                    messages.append(("", text))
                continue
            role = (
                entry.get("role")
                or entry.get("from")
                or entry.get("speaker")
                or entry.get("author")
                or ""
            )
            content = (
                entry.get("content")
                or entry.get("text")
                or entry.get("value")
                or entry.get("message")
                or ""
            )
            if content is None:
                content = ""
            content = str(content).strip()
            if content:
                messages.append((str(role).strip().lower(), content))
    if messages:
        return messages
    # Fallback to prompt/response fields if available.
    prompt = raw.get("prompt") or raw.get("instruction") or raw.get("question") or raw.get("input")
    response = raw.get("response") or raw.get("output") or raw.get("answer")
    out: list[tuple[str, str]] = []
    if prompt:
        out.append(("user", str(prompt).strip()))
    if response:
        out.append(("assistant", str(response).strip()))
    return out


def _extract_last_user_message(messages: list[tuple[str, str]]) -> str:
    for role, content in reversed(messages):
        if role in {"user", "human"}:
            return content.strip()
    return messages[-1][1].strip() if messages else ""


def _format_wildif_conversation(messages: list[tuple[str, str]]) -> str:
    lines: list[str] = []
    for role, content in messages:
        if not content:
            continue
        role_label = role.strip().upper() if role else "CONTEXT"
        # Use an arrow to avoid triggering role-tag parsing downstream.
        lines.append(f"{role_label} -> {content}")
    return "\n".join(lines).strip()


def _parse_rubrics(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        out: list[str] = []
        for item in value:
            text = str(item).strip()
            if text:
                out.append(text)
        return out
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        parsed_list = _try_parse_list_string(text)
        if parsed_list is not None:
            return parsed_list
        parsed = _safe_json_loads(text)
        if parsed is not text:
            return _parse_rubrics(parsed)
        if "\n" in text:
            return [line.strip() for line in text.splitlines() if line.strip()]
        return [text]
    if isinstance(value, dict):
        return []
    text = str(value).strip()
    return [text] if text else []


def load_advancedif_dataset(
    dataset_name: str = "facebook/AdvancedIF",
    split: Optional[str] = None,
    max_samples: Optional[int] = None,
) -> Iterator[Sample]:
    """
    Load AdvancedIF dataset from HuggingFace.

    Expected fields:
    - conversation_history: list of {role, content} or JSON string
    - prompt_metadata: JSON string or dict containing rubrics
    """
    try:
        from datasets import load_dataset as hf_load_dataset
    except ImportError:
        raise ImportError("pip install datasets")

    if split:
        ds = hf_load_dataset(dataset_name, split=split)
    else:
        ds = hf_load_dataset(dataset_name)
        if isinstance(ds, dict):
            if "train" in ds:
                ds = ds["train"]
            elif "validation" in ds:
                ds = ds["validation"]
            elif "test" in ds:
                ds = ds["test"]
            else:
                ds = ds[list(ds.keys())[0]]

    for idx, raw in enumerate(ds):
        if max_samples is not None and idx >= max_samples:
            break

        raw_dict = dict(raw)
        history_raw = (
            raw_dict.get("conversation_history")
            or raw_dict.get("prompt")
            or raw_dict.get("instruction")
            or raw_dict.get("query")
            or raw_dict.get("input")
        )
        history_text = _format_advancedif_conversation(history_raw)
        if not history_text:
            continue

        prompt_metadata_raw = raw_dict.get("prompt_metadata")
        prompt_metadata = _safe_json_loads(prompt_metadata_raw)
        rubrics: list[str] = []
        if isinstance(prompt_metadata, dict):
            rubrics = _parse_rubrics(prompt_metadata.get("rubrics") or prompt_metadata.get("rubric"))
        elif prompt_metadata is not None:
            rubrics = _parse_rubrics(prompt_metadata)

        sample_id = (
            raw_dict.get("id")
            or raw_dict.get("sample_id")
            or raw_dict.get("uid")
            or f"advancedif_{idx}"
        )

        meta = {
            k: v for k, v in raw_dict.items()
            if k not in {"id", "sample_id", "uid", "conversation_history", "prompt_metadata"}
        }
        meta.update(
            {
                "conversation_history": history_raw,
                "prompt_metadata": prompt_metadata,
                "rubrics": rubrics,
                "source": dataset_name,
            }
        )

        yield Sample(id=str(sample_id), question=str(history_text), answer=None, meta=meta)


def load_wildif_dataset(
    dataset_name: str = "gililior/wild-if-eval",
    lmsys_dataset: str = "lmsys/lmsys-chat-1m",
    split: Optional[str] = None,
    max_samples: Optional[int] = None,
) -> Iterator[Sample]:
    """
    Load Wild-IF-Eval dataset and merge with LMSYS conversation data.

    Expected fields in wild-if-eval:
    - conversation_id
    - decomposition (list of rubric strings)

    LMSYS dataset is used to fetch the full multi-turn conversation by conversation_id.
    """
    try:
        from datasets import load_dataset as hf_load_dataset
    except ImportError:
        raise ImportError("pip install datasets")

    if split:
        wild_ds = hf_load_dataset(dataset_name, split=split)
    else:
        wild_ds = hf_load_dataset(dataset_name)
        if isinstance(wild_ds, dict):
            if "test" in wild_ds:
                wild_ds = wild_ds["test"]
            elif "validation" in wild_ds:
                wild_ds = wild_ds["validation"]
            elif "train" in wild_ds:
                wild_ds = wild_ds["train"]
            else:
                wild_ds = wild_ds[list(wild_ds.keys())[0]]

    wild_rows: list[dict[str, Any]] = []
    target_ids: set[str] = set()
    for idx, raw in enumerate(wild_ds):
        if max_samples is not None and idx >= max_samples:
            break
        row = dict(raw)
        conv_id = row.get("conversation_id") or row.get("id") or row.get("uid")
        if conv_id is None:
            continue
        conv_id = str(conv_id)
        row["conversation_id"] = conv_id
        wild_rows.append(row)
        target_ids.add(conv_id)

    if not wild_rows:
        return iter(())

    try:
        lmsys_stream = hf_load_dataset(lmsys_dataset, split="train", streaming=True)
    except Exception as exc:
        raise RuntimeError(
            "Failed to load lmsys/lmsys-chat-1m. "
            "If this is a gated dataset, make sure you are authenticated with the HuggingFace Hub."
        ) from exc

    conv_map: dict[str, dict[str, Any]] = {}
    for row in lmsys_stream:
        conv_id = row.get("conversation_id") or row.get("id") or row.get("uid")
        if conv_id is None:
            continue
        conv_id = str(conv_id)
        if conv_id in target_ids and conv_id not in conv_map:
            conv_map[conv_id] = dict(row)
            if len(conv_map) >= len(target_ids):
                break

    for idx, raw in enumerate(wild_rows):
        conv_id = raw.get("conversation_id")
        convo = conv_map.get(conv_id)
        if not convo:
            continue
        messages = _extract_lmsys_messages(convo)
        if not messages:
            continue
        conversation_text = _format_wildif_conversation(messages)
        last_user = _extract_last_user_message(messages)
        if not conversation_text or not last_user:
            continue

        question = (
            "Conversation context (do NOT continue it):\n"
            f"{conversation_text}\n\n"
            "User request:\n"
            f"{last_user}\n\n"
            "Respond only to the last user request."
        ).strip()

        rubrics = _parse_rubrics(raw.get("decomposition"))
        sample_id = raw.get("conversation_id") or raw.get("id") or f"wildif_{idx}"
        meta = {k: v for k, v in raw.items() if k not in {"decomposition"}}
        meta.update(
            {
                "rubrics": rubrics,
                "source": dataset_name,
                "conversation_id": conv_id,
                "lmsys_source": lmsys_dataset,
            }
        )

        yield Sample(id=str(sample_id), question=question, answer=None, meta=meta)


# ============================================================
# BiGGen-Bench dataset loader (prometheus-eval/BiGGen-Bench)
# ============================================================

def _format_biggen_prompt(system_prompt: Any, user_input: Any) -> str:
    sys_text = str(system_prompt).strip() if system_prompt is not None else ""
    user_text = str(user_input).strip() if user_input is not None else ""
    parts: list[str] = []
    if sys_text:
        parts.append("System prompt\n" + sys_text)
    if user_text:
        parts.append("User input\n" + user_text)
    return "\n\n".join(parts).strip()


def _format_biggen_rubric(score_rubric: Any) -> Optional[str]:
    parsed = _safe_json_loads(score_rubric)
    if isinstance(parsed, dict):
        criteria = str(parsed.get("criteria") or parsed.get("criterion") or "").strip()
        lines: list[str] = []
        if criteria:
            lines.append(f"Criteria: {criteria}")
        for idx in range(1, 6):
            key = f"score{idx}_description"
            desc = parsed.get(key)
            if desc:
                lines.append(f"Score {idx}: {str(desc).strip()}")
        if lines:
            return "\n".join(lines).strip()
    if isinstance(parsed, (list, tuple)):
        flat = [str(item).strip() for item in parsed if str(item).strip()]
        if flat:
            return "\n".join(flat).strip()
    text = str(score_rubric).strip() if score_rubric is not None else ""
    return text or None


def load_biggen_bench_dataset(
    dataset_name: str = "prometheus-eval/BiGGen-Bench",
    split: Optional[str] = None,
    max_samples: Optional[int] = None,
) -> Iterator[Sample]:
    """
    Load BiGGen-Bench dataset from HuggingFace.

    Expected fields:
    - system_prompt
    - input
    - score_rubric (dict or JSON string with criteria + score1-5 descriptions)
    """
    try:
        from datasets import load_dataset as hf_load_dataset
    except ImportError:
        raise ImportError("pip install datasets")

    if split:
        ds = hf_load_dataset(dataset_name, split=split)
    else:
        ds = hf_load_dataset(dataset_name)
        if isinstance(ds, dict):
            if "train" in ds:
                ds = ds["train"]
            elif "validation" in ds:
                ds = ds["validation"]
            elif "test" in ds:
                ds = ds["test"]
            else:
                ds = ds[list(ds.keys())[0]]

    for idx, raw in enumerate(ds):
        if max_samples is not None and idx >= max_samples:
            break

        raw_dict = dict(raw)
        system_prompt = (
            raw_dict.get("system_prompt")
            or raw_dict.get("system")
            or raw_dict.get("system_instruction")
        )
        user_input = (
            raw_dict.get("input")
            or raw_dict.get("prompt")
            or raw_dict.get("instruction")
            or raw_dict.get("query")
        )
        question = _format_biggen_prompt(system_prompt, user_input)
        if not question:
            continue

        score_rubric_raw = (
            raw_dict.get("score_rubric")
            or raw_dict.get("rubric")
            or raw_dict.get("criteria")
        )
        formatted_rubric = _format_biggen_rubric(score_rubric_raw)
        rubrics: list[str] = [formatted_rubric] if formatted_rubric else []

        sample_id = (
            raw_dict.get("id")
            or raw_dict.get("sample_id")
            or raw_dict.get("uid")
            or f"biggen_{idx}"
        )

        meta = {
            k: v
            for k, v in raw_dict.items()
            if k not in {"id", "sample_id", "uid"}
        }
        meta.update(
            {
                "system_prompt": system_prompt,
                "input": user_input,
                "score_rubric": score_rubric_raw,
                "rubrics": rubrics,
                "score_scale": "quarters",
                "source": dataset_name,
            }
        )

        yield Sample(id=str(sample_id), question=str(question), answer=None, meta=meta)


# ============================================================
# AIME-specific loader
# ============================================================

def load_aime_dataset(
    dataset_name: str = "AI-MO/aimo-validation-aime",
    max_samples: Optional[int] = None,
) -> Iterator[Sample]:
    """
    Load AIME dataset from HuggingFace.
    
    The AI-MO/aimo-validation-aime dataset has:
    - problem: The problem text
    - answer: The integer answer (0-999)
    - url: Source URL
    """
    try:
        from datasets import load_dataset as hf_load_dataset
    except ImportError:
        raise ImportError("pip install datasets")
    
    ds = hf_load_dataset(dataset_name)
    
    # Get the appropriate split
    if isinstance(ds, dict):
        if "train" in ds:
            ds = ds["train"]
        else:
            ds = ds[list(ds.keys())[0]]
    
    for idx, raw in enumerate(ds):
        if max_samples is not None and idx >= max_samples:
            break
        
        sample = Sample(
            id=f"aime_{idx}",
            question=raw.get("problem", ""),
            answer=raw.get("answer"),
            meta={
                "url": raw.get("url", ""),
                "source": dataset_name,
            }
        )
        yield sample


def load_aime_by_year(
    years: Optional[list] = None,
    max_samples: Optional[int] = None,
) -> Iterator[Sample]:
    """
    Load AIME problems filtered by year.
    
    Args:
        years: List of years to include (e.g., [2022, 2023, 2024])
                If None, load all years
        max_samples: Maximum number of samples to load
    """
    try:
        from datasets import load_dataset as hf_load_dataset
    except ImportError:
        raise ImportError("pip install datasets")
    
    ds = hf_load_dataset("AI-MO/aimo-validation-aime")
    
    if isinstance(ds, dict):
        if "train" in ds:
            ds = ds["train"]
        else:
            ds = ds[list(ds.keys())[0]]
    
    count = 0
    for idx, raw in enumerate(ds):
        if max_samples is not None and count >= max_samples:
            break
        
        # Try to extract year from URL
        url = raw.get("url", "")
        year = None
        import re
        year_match = re.search(r"20\d{2}", url)
        if year_match:
            year = int(year_match.group())
        
        # Filter by year if specified
        if years is not None and year is not None and year not in years:
            continue
        
        sample = Sample(
            id=f"aime_{year}_{idx}" if year else f"aime_{idx}",
            question=raw.get("problem", ""),
            answer=raw.get("answer"),
            meta={
                "url": url,
                "year": year,
                "source": "AI-MO/aimo-validation-aime",
            }
        )
        yield sample
        count += 1


# ============================================================
# AIME 2025 dataset loader (math-ai/aime25)
# ============================================================

def load_aime25_dataset(
    dataset_name: str = "math-ai/aime25",
    split: Optional[str] = None,
    max_samples: Optional[int] = None,
) -> Iterator[Sample]:
    """
    Load AIME 2025 dataset from HuggingFace.
    """
    try:
        from datasets import load_dataset as hf_load_dataset
    except ImportError:
        raise ImportError("pip install datasets")

    if split:
        ds = hf_load_dataset(dataset_name, split=split)
    else:
        ds = hf_load_dataset(dataset_name)
        if isinstance(ds, dict):
            if "test" in ds:
                ds = ds["test"]
            elif "validation" in ds:
                ds = ds["validation"]
            elif "train" in ds:
                ds = ds["train"]
            else:
                ds = ds[list(ds.keys())[0]]

    for idx, raw in enumerate(ds):
        if max_samples is not None and idx >= max_samples:
            break

        question = (
            raw.get("problem")
            or raw.get("question")
            or raw.get("query")
            or raw.get("prompt")
        )
        if not question:
            continue

        sample_id = raw.get("id") or raw.get("sample_id") or raw.get("problem_id") or f"aime25_{idx}"
        answer = raw.get("answer")
        if answer is None:
            answer = raw.get("ground_truth")
        if answer is None:
            answer = raw.get("solution")
        if answer is None:
            answer = raw.get("final_answer")

        meta = {
            k: v for k, v in raw.items()
            if k not in {
                "id",
                "sample_id",
                "problem_id",
                "question",
                "query",
                "prompt",
                "problem",
                "answer",
                "ground_truth",
                "solution",
                "final_answer",
            }
        }
        meta["source"] = dataset_name

        yield Sample(id=str(sample_id), question=str(question), answer=answer, meta=meta)


# ============================================================
# MATH-500 dataset loader (HuggingFaceH4/MATH-500)
# ============================================================

def _extract_latex_command_arg(text: str, command: str) -> Optional[str]:
    if not text:
        return None
    token = f"\\{command}{{"
    start = text.rfind(token)
    if start == -1:
        return None
    idx = start + len(token)
    depth = 1
    while idx < len(text):
        ch = text[idx]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start + len(token):idx]
        idx += 1
    return None


def _extract_math_answer_from_solution(solution: str) -> Optional[str]:
    if not solution:
        return None
    for command in ("boxed", "fbox"):
        extracted = _extract_latex_command_arg(solution, command)
        if extracted:
            return extracted.strip()
    match = re.search(r"(?:final\s+answer|answer)\s*[:\s]+(.+?)(?:\n|$)", solution, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return None


def _clean_math_answer(answer: str) -> str:
    if not answer:
        return answer
    cleaned = answer.strip()
    if cleaned.startswith("$") and cleaned.endswith("$") and len(cleaned) >= 2:
        cleaned = cleaned[1:-1].strip()
    frac_match = re.fullmatch(r"\\frac\{(-?\d+)\}\{(-?\d+)\}", cleaned)
    if frac_match:
        cleaned = f"{frac_match.group(1)}/{frac_match.group(2)}"
    return cleaned


def load_math500_dataset(
    dataset_name: str = "HuggingFaceH4/MATH-500",
    split: Optional[str] = None,
    max_samples: Optional[int] = None,
) -> Iterator[Sample]:
    """
    Load MATH-500 dataset from HuggingFace.
    """
    try:
        from datasets import load_dataset as hf_load_dataset
    except ImportError:
        raise ImportError("pip install datasets")

    if split:
        ds = hf_load_dataset(dataset_name, split=split)
    else:
        ds = hf_load_dataset(dataset_name)
        if isinstance(ds, dict):
            if "test" in ds:
                ds = ds["test"]
            elif "validation" in ds:
                ds = ds["validation"]
            elif "train" in ds:
                ds = ds["train"]
            else:
                ds = ds[list(ds.keys())[0]]

    for idx, raw in enumerate(ds):
        if max_samples is not None and idx >= max_samples:
            break

        raw_dict = dict(raw)
        question = (
            raw_dict.get("problem")
            or raw_dict.get("question")
            or raw_dict.get("query")
            or raw_dict.get("prompt")
        )
        if not question:
            continue

        sample_id = (
            raw_dict.get("id")
            or raw_dict.get("sample_id")
            or raw_dict.get("problem_id")
            or f"math500_{idx}"
        )
        answer = (
            raw_dict.get("answer")
            or raw_dict.get("final_answer")
            or raw_dict.get("ground_truth")
        )
        solution = raw_dict.get("solution") or raw_dict.get("analysis")
        if answer is None and solution:
            answer = _extract_math_answer_from_solution(str(solution))
        if isinstance(answer, str):
            answer = _clean_math_answer(answer)

        meta = {
            k: v for k, v in raw_dict.items()
            if k not in {
                "id",
                "sample_id",
                "problem_id",
                "question",
                "query",
                "prompt",
                "problem",
                "answer",
                "final_answer",
                "ground_truth",
                "solution",
                "analysis",
            }
        }
        if solution is not None:
            meta["solution"] = solution
        meta["source"] = dataset_name

        yield Sample(id=str(sample_id), question=str(question), answer=answer, meta=meta)


# ============================================================
# HotpotQA dataset loader (hotpot_qa)
# ============================================================

def _format_hotpotqa_context(context: Any) -> str:
    if context is None:
        return ""
    if isinstance(context, str):
        return context.strip()
    lines: list[str] = []
    if isinstance(context, dict):
        for title, sentences in context.items():
            if title:
                lines.append(f"Title: {title}")
            if isinstance(sentences, list):
                for sent in sentences:
                    if sent:
                        lines.append(f"- {sent}")
            elif isinstance(sentences, str) and sentences:
                lines.append(f"- {sentences}")
        return "\n".join(lines).strip()
    if not isinstance(context, list):
        return str(context).strip()
    for entry in context:
        title = None
        sentences = None
        if isinstance(entry, dict):
            title = entry.get("title") or entry.get("doc_title")
            sentences = entry.get("sentences") or entry.get("sents")
        elif isinstance(entry, (list, tuple)) and len(entry) >= 2:
            title = entry[0]
            sentences = entry[1]
        if title:
            lines.append(f"Title: {title}")
        if isinstance(sentences, list):
            for sent in sentences:
                if sent:
                    lines.append(f"- {sent}")
        elif isinstance(sentences, str) and sentences:
            lines.append(f"- {sentences}")
    return "\n".join(lines).strip()


def load_hotpotqa_dataset(
    dataset_name: str = "hotpot_qa",
    config_name: Optional[str] = None,
    split: Optional[str] = None,
    max_samples: Optional[int] = None,
) -> Iterator[Sample]:
    """
    Load HotpotQA dataset from HuggingFace.
    """
    try:
        from datasets import load_dataset as hf_load_dataset
    except ImportError:
        raise ImportError("pip install datasets")

    if config_name is None:
        config_name = "distractor"
    if split:
        ds = hf_load_dataset(dataset_name, name=config_name, split=split)
    else:
        ds = hf_load_dataset(dataset_name, name=config_name)
        if isinstance(ds, dict):
            if "train" in ds:
                ds = ds["train"]
            elif "validation" in ds:
                ds = ds["validation"]
            elif "test" in ds:
                ds = ds["test"]
            else:
                ds = ds[list(ds.keys())[0]]

    for idx, raw in enumerate(ds):
        if max_samples is not None and idx >= max_samples:
            break

        raw_dict = dict(raw)
        question = (
            raw_dict.get("question")
            or raw_dict.get("problem")
            or raw_dict.get("prompt")
        )
        if not question:
            continue

        context_text = _format_hotpotqa_context(raw_dict.get("context"))
        if context_text:
            full_question = f"Question: {question}\n\nContext:\n{context_text}"
        else:
            full_question = str(question)

        sample_id = (
            raw_dict.get("_id")
            or raw_dict.get("id")
            or raw_dict.get("sample_id")
            or f"hotpotqa_{idx}"
        )
        answer = (
            raw_dict.get("answer")
            or raw_dict.get("final_answer")
            or raw_dict.get("ground_truth")
        )

        meta = {
            k: v for k, v in raw_dict.items()
            if k not in {
                "_id",
                "id",
                "sample_id",
                "question",
                "problem",
                "prompt",
                "context",
                "answer",
                "final_answer",
                "ground_truth",
            }
        }
        meta["source"] = dataset_name

        yield Sample(id=str(sample_id), question=str(full_question), answer=answer, meta=meta)
