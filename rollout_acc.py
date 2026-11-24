"""
Unified Rollout Module for GRPO

This module focuses on:
1. GRPO Sampling - Collect rollout samples with rewards

Dependencies:
    pip install datasets transformers torch accelerate tqdm

Usage:
    python rollout.py produce --dataset di-zhang-fdu/AIME_1983_2024 --output ./unified_output
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

# Optional: distributed acceleration + progress bars
try:
    from accelerate import Accelerator
except Exception:  # pragma: no cover
    Accelerator = None

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    tqdm = None

if TYPE_CHECKING:
    from models import Model


# ============================================================
# Data Structures
# ============================================================

@dataclass
class RolloutSample:
    """Single rollout sample from model generation"""
    response: str
    reward: float  # e.g., F1 for QA or 1/0 for verifiable tasks
    parsed_answer: Optional[str]
    is_correct: bool
    tokens_used: int = 0
    generation_time: float = 0.0
    rule_score: Optional[float] = None
    judge_score: Optional[float] = None
    hard_fail: bool = False


@dataclass 
class QuestionRollout:
    """All rollouts for a single question"""
    sample_id: str
    question: str
    ground_truth: Any
    student_samples: List[RolloutSample] = field(default_factory=list)
    timestamp: str = ""
    
    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.utcnow().isoformat() + "Z"
    
    def has_correct_student(self) -> bool:
        return any(s.is_correct for s in self.student_samples)
    
    def student_accuracy(self) -> float:
        if not self.student_samples:
            return 0.0
        return sum(1 for s in self.student_samples if s.is_correct) / len(self.student_samples)
    
    def mean_reward(self) -> float:
        if not self.student_samples:
            return 0.0
        return sum(s.reward for s in self.student_samples) / len(self.student_samples)


@dataclass
class UnifiedProductionResult:
    """Result of production for one question"""
    rollout: QuestionRollout
    stats: Dict[str, Any]


# ============================================================
# Answer Extraction and Judging
# ============================================================

def extract_aime_answer(response: str) -> Optional[str]:
    """Extract AIME answer (integer 0-999) from response."""
    if not response:
        return None
    
    # Pattern 1: ANSWER: N (explicit format)
    for match in re.finditer(r"(?im)^\s*ANSWER\s*:\s*(\d+)", response):
        val = match.group(1)
        try:
            if 0 <= int(val) <= 999:
                return val
        except ValueError:
            pass
    
    # Pattern 2: \boxed{N}
    boxed = re.findall(r"\\boxed\{(\d+)\}", response)
    if boxed:
        try:
            val = int(boxed[-1])
            if 0 <= val <= 999:
                return str(val)
        except ValueError:
            pass
    
    # Pattern 3: Final Answer: N
    match = re.search(r"final\s+answer\s*[:\s]+(\d+)", response, re.IGNORECASE)
    if match:
        try:
            val = int(match.group(1))
            if 0 <= val <= 999:
                return str(val)
        except ValueError:
            pass
    
    # Pattern 4: The answer is N
    match = re.search(r"(?:the\s+)?answer\s+is\s*[:\s]*(\d+)", response, re.IGNORECASE)
    if match:
        try:
            val = int(match.group(1))
            if 0 <= val <= 999:
                return str(val)
        except ValueError:
            pass
    return None

def judge_aime_answer(parsed: Optional[str], ground_truth: Any) -> bool:
    """Judge if AIME answer is correct."""
    if parsed is None:
        return False
    
    try:
        pred_int = int(parsed)
    except (ValueError, TypeError):
        return False
    
    try:
        gt_int = int(ground_truth)
    except (ValueError, TypeError):
        match = re.search(r"\d+", str(ground_truth))
        if match:
            gt_int = int(match.group())
        else:
            return False
    
    return pred_int == gt_int


def extract_hotpotqa_answer(response: str, strict: bool = False) -> Optional[str]:
    """Extract HotpotQA-style answer from response."""
    if not response:
        return None
    cleaned = response.strip()
    boxed = re.findall(r"(?:\\boxed|boxed)\{([^}]*)\}", cleaned)
    for candidate in boxed:
        candidate = candidate.strip()
        if candidate:
            return candidate
    candidate = ""
    for line in cleaned.splitlines():
        line = line.strip()
        if line:
            candidate = line
            break
    if not candidate:
        candidate = cleaned
    candidate = re.split(
        r"(?i)\b(based on|because|since|given|according to|the context|from the context)\b",
        candidate,
    )[0].strip()
    candidate = re.split(r"(?i)\bstop\b", candidate)[0].strip()
    candidate = re.split(r"(?i)\bsystem\b", candidate)[0].strip()
    return candidate.strip(" .;:")


def _normalize_qa_answer(text: str) -> str:
    if not text:
        return ""
    text = text.lower()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = re.sub(r"[^0-9a-z\s]", " ", text)
    text = " ".join(text.split())
    return text


def _f1_score(pred: str, gt: str) -> float:
    pred_tokens = pred.split()
    gt_tokens = gt.split()
    if not pred_tokens or not gt_tokens:
        return 1.0 if pred_tokens == gt_tokens else 0.0
    common = Counter(pred_tokens) & Counter(gt_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gt_tokens)
    return 2 * precision * recall / (precision + recall)


def judge_hotpotqa_answer(parsed: Optional[str], ground_truth: Any) -> Tuple[bool, float]:
    """Return (EM, F1) for HotpotQA-style answers."""
    pred_raw = parsed or ""
    gt_raw = "" if ground_truth is None else str(ground_truth)
    pred_norm = _normalize_qa_answer(pred_raw)
    gt_norm = _normalize_qa_answer(gt_raw)
    em = pred_norm == gt_norm
    f1 = _f1_score(pred_norm, gt_norm)
    return em, f1


HOTPOTQA_EM_LAMBDA = 0.2


def _infer_prompt_profile(dataset_path: Optional[str]) -> Optional[str]:
    if not dataset_path:
        return None
    lowered = dataset_path.lower()
    if "advancedif" in lowered or "advanced-if" in lowered or "advanced_if" in lowered:
        return "advancedif"
    if lowered.startswith(("hotpotqa", "hotpot_qa", "hotpot-qa")):
        return "hotpotqa"
    if "hotpotqa" in lowered or "hotpot_qa" in lowered or "hotpot-qa" in lowered:
        return "hotpotqa"
    return None


# ============================================================
# AdvancedIF Scoring Helpers
# ============================================================

ADVANCEDIF_STOP_SEQUENCES = [
    "END OF MESSAGE",
    "End of message",
    "BEGIN WRITING",
    "Content Approved",
    "USER:",
    "User:",
    "ASSISTANT:",
    "Assistant:",
    "SYSTEM:",
    "System:",
    "\nUSER:",
    "\nUser:",
    "\nASSISTANT:",
    "\nAssistant:",
    "\nSYSTEM:",
    "\nSystem:",
]

ADVANCEDIF_MIN_RESPONSE_CHARS = 20

_BULLET_RE = re.compile(r"(?m)^\s*[-*•]\s+")
_NUMBERED_RE = re.compile(r"(?m)^\s*\d+[.)]\s+")
_ROLE_LABEL_RE = re.compile(r"(?m)^\s*(user|assistant|system)\s*:\s*")
_ROLE_INLINE_RE = re.compile(r"(?i)\b(user|assistant|system)\s*:")


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


def _try_parse_list_string(text: str) -> Optional[List[str]]:
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
    if stripped.startswith("[") and stripped.endswith("]"):
        items = re.findall(r'"([^"]+)"', stripped)
        if not items:
            items = re.findall(r"'([^']+)'", stripped)
        if items:
            return [item.strip() for item in items if item.strip()]
    return None


def _normalize_rubrics_value(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        out: List[str] = []
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
            return _normalize_rubrics_value(parsed)
        if "\n" in text:
            return [line.strip() for line in text.splitlines() if line.strip()]
        return [text]
    if isinstance(value, dict):
        return []
    text = str(value).strip()
    return [text] if text else []


def _clean_rubric_text(text: str, max_chars: int = 400) -> str:
    if not text:
        return ""
    cleaned = re.sub(r"\s+", " ", str(text)).strip()
    if len(cleaned) > max_chars:
        cleaned = cleaned[:max_chars].rstrip()
    return cleaned


def _extract_advancedif_rubrics_from_meta(meta: Optional[Dict[str, Any]]) -> List[str]:
    if not isinstance(meta, dict):
        return []
    rubrics = meta.get("rubrics") or meta.get("rubric")
    if rubrics:
        return _normalize_rubrics_value(rubrics)
    prompt_metadata = meta.get("prompt_metadata")
    parsed = _safe_json_loads(prompt_metadata)
    if isinstance(parsed, dict):
        return _normalize_rubrics_value(parsed.get("rubrics") or parsed.get("rubric"))
    if parsed is not None:
        return _normalize_rubrics_value(parsed)
    return []


def _dedupe_rubrics(rubrics: List[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for rubric in rubrics:
        cleaned = _clean_rubric_text(rubric)
        if not cleaned:
            continue
        key = cleaned.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(cleaned)
    return out


def _parse_rubric_score(text: str) -> Optional[float]:
    if text is None:
        return None
    cleaned = str(text).strip()
    if cleaned == "":
        return None
    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, (int, float)):
            return float(parsed)
        if isinstance(parsed, dict):
            for key in ("score", "rating", "grade"):
                if key in parsed:
                    try:
                        return float(parsed[key])
                    except Exception:
                        pass
        if isinstance(parsed, str):
            cleaned = parsed.strip()
    except Exception:
        pass
    match = re.search(r"(0\.5|1(?:\.0)?|0(?:\.0)?)", cleaned)
    if not match:
        return None
    try:
        val = float(match.group(1))
    except Exception:
        return None
    if abs(val - 1.0) < 1e-6:
        return 1.0
    if abs(val - 0.5) < 1e-6:
        return 0.5
    if abs(val) < 1e-6:
        return 0.0
    return None


def _strip_json_fence(text: str) -> str:
    if not text:
        return ""
    match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return text.strip()


def _extract_first_json_blob(text: str) -> Optional[str]:
    if not text:
        return None
    cleaned = _strip_json_fence(text)
    if cleaned.startswith("{") or cleaned.startswith("["):
        return cleaned
    brace_start = cleaned.find("{")
    brace_end = cleaned.rfind("}")
    if brace_start != -1 and brace_end != -1 and brace_end > brace_start:
        return cleaned[brace_start:brace_end + 1].strip()
    bracket_start = cleaned.find("[")
    bracket_end = cleaned.rfind("]")
    if bracket_start != -1 and bracket_end != -1 and bracket_end > bracket_start:
        return cleaned[bracket_start:bracket_end + 1].strip()
    return None


def _parse_advancedif_judge_items(text: str) -> List[Dict[str, Any]]:
    if not text:
        return []
    cleaned = _strip_json_fence(text)
    cleaned_strip = cleaned.strip()

    # Compact scores format: {"scores":[0,0.5,1,...]} or [0,0.5,1,...]
    if cleaned_strip.startswith(("{", "[")):
        blob = _extract_first_json_blob(cleaned_strip) or cleaned_strip
        try:
            parsed = json.loads(blob)
        except Exception:
            parsed = None
        if isinstance(parsed, dict) and isinstance(parsed.get("scores"), list):
            scores = [item for item in parsed.get("scores") if isinstance(item, (int, float))]
            return [{"score": score} for score in scores]
        if isinstance(parsed, list) and all(isinstance(item, (int, float)) for item in parsed):
            return [{"score": score} for score in parsed]

    if cleaned_strip.startswith("["):
        blob = _extract_first_json_blob(cleaned_strip) or cleaned_strip
        try:
            parsed = json.loads(blob)
        except Exception:
            parsed = None
        if isinstance(parsed, list):
            items = [item for item in parsed if isinstance(item, dict)]
            if items:
                return items
        if isinstance(parsed, dict):
            if isinstance(parsed.get("items"), list):
                items = [item for item in parsed.get("items") or [] if isinstance(item, dict)]
                if items:
                    return items

    lines = [line.strip() for line in cleaned.splitlines() if line.strip()]
    jsonl_items: List[Dict[str, Any]] = []
    for line in lines:
        if not line.startswith(("{", "[")):
            continue
        if line.endswith(","):
            line = line[:-1].rstrip()
        try:
            parsed_line = json.loads(line)
        except Exception:
            continue
        if isinstance(parsed_line, dict):
            jsonl_items.append(parsed_line)
        elif isinstance(parsed_line, list):
            for item in parsed_line:
                if isinstance(item, dict):
                    jsonl_items.append(item)
    if jsonl_items:
        return jsonl_items

    blob = _extract_first_json_blob(cleaned_strip)
    if not blob:
        return []
    try:
        parsed = json.loads(blob)
    except Exception:
        return []
    items: List[Dict[str, Any]] = []
    if isinstance(parsed, dict):
        if isinstance(parsed.get("items"), list):
            for item in parsed.get("items") or []:
                if isinstance(item, dict):
                    items.append(item)
            return items
        if all(not isinstance(v, (list, dict)) for v in parsed.values()):
            for key, val in parsed.items():
                items.append({"rubric": key, "score": val})
            return items
        return [parsed]
    if isinstance(parsed, list):
        for item in parsed:
            if isinstance(item, dict):
                items.append(item)
    return items


def _align_advancedif_items(
    rubrics: List[str],
    items: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    if not rubrics:
        return []
    if not items:
        return [{"rubric": rubric, "score": None} for rubric in rubrics]

    def _norm(text: str) -> str:
        return re.sub(r"\s+", " ", text.strip().lower())

    rubric_to_item: Dict[str, Dict[str, Any]] = {}
    for item in items:
        key = _norm(str(item.get("rubric") or ""))
        if key and key not in rubric_to_item:
            rubric_to_item[key] = item

    aligned: List[Dict[str, Any]] = []
    if len(items) == len(rubrics):
        for idx, rubric in enumerate(rubrics):
            item = rubric_to_item.get(_norm(rubric)) or items[idx]
            aligned.append(item)
        return aligned

    for rubric in rubrics:
        item = rubric_to_item.get(_norm(rubric))
        if item is None:
            aligned.append({"rubric": rubric, "score": None})
        else:
            aligned.append(item)
    return aligned


def _truncate_text(text: str, max_chars: int) -> str:
    if not text:
        return ""
    return text if len(text) <= max_chars else text[:max_chars]


def _judge_advancedif_rubrics(
    judge_model: "Model",
    prompt: str,
    response: str,
    rubrics: List[str],
    max_tokens: int = 768,
) -> Tuple[List[Dict[str, Any]], str]:
    if not rubrics:
        return [], ""
    rubric_lines = []
    for idx, item in enumerate(rubrics, start=1):
        rubric_lines.append(f"{idx}. {item}")
    rubric_block = "\n".join(rubric_lines)

    system_prompt = (
        "You are a strict rubric grader.\n"
        "Return ONLY valid JSON with no extra text."
    )
    user_prompt = f"""User prompt:
{_truncate_text(prompt, 2000)}

Model response:
{_truncate_text(response, 2000)}

Rubrics:
{_truncate_text(rubric_block, 2000)}

OUTPUT JSON FORMAT (single object):
{{"scores":[0,0.5,1,...]}}

Rules:
- Output ONLY the JSON object above, no extra text.
- The scores array MUST have exactly {len(rubrics)} numbers, in the same order as the rubrics listed.
- Each score MUST be ONLY 0, 0.5, or 1.
- Do NOT output markdown or code fences.
"""
    try:
        result = judge_model.generate(
            prompt=user_prompt,
            system_prompt=system_prompt,
            temperature=0.0,
            do_sample=False,
            max_tokens=max_tokens,
        )
    except Exception:
        return [], ""
    raw_text = result.text or ""
    return _parse_advancedif_judge_items(raw_text), raw_text


def _rubric_mentions_negative(text: str) -> bool:
    lowered = text.lower()
    neg_markers = ["do not", "don't", "avoid", "no ", "without", "must not", "should not", "not use"]
    return any(marker in lowered for marker in neg_markers)


def _rubrics_require_bullets(rubrics: List[str]) -> bool:
    for rubric in rubrics:
        text = rubric.lower()
        if "bullet" in text and "list" in text and not _rubric_mentions_negative(text):
            return True
    return False


def _rubrics_require_numbered(rubrics: List[str]) -> bool:
    for rubric in rubrics:
        text = rubric.lower()
        if "numbered" in text and "list" in text and not _rubric_mentions_negative(text):
            return True
    return False


def _rubrics_avoid_bullets(rubrics: List[str]) -> bool:
    for rubric in rubrics:
        text = rubric.lower()
        if "bullet" in text and "list" in text and _rubric_mentions_negative(text):
            return True
    return False


def _rubrics_avoid_numbered(rubrics: List[str]) -> bool:
    for rubric in rubrics:
        text = rubric.lower()
        if "numbered" in text and "list" in text and _rubric_mentions_negative(text):
            return True
    return False


def _contains_forbidden_markers(text: str) -> bool:
    if not text:
        return True
    lowered = text.lower()
    forbidden = [
        "end of message",
        "begin writing",
        "content approved",
    ]
    if any(marker in lowered for marker in forbidden):
        return True
    if _ROLE_LABEL_RE.search(text) or _ROLE_INLINE_RE.search(text):
        return True
    return False


def _clean_advancedif_response(text: str) -> str:
    if not text:
        return ""
    # Strip a leading role label if the model starts with one (salvage the content).
    text = re.sub(r"(?i)^\s*(user|assistant|system)\s*:\s*", "", text, count=1)
    pattern = re.compile(
        r"(?i)(\b(user|assistant|system)\s*:|END OF MESSAGE|End of message|BEGIN WRITING|Content Approved)"
    )
    match = pattern.search(text)
    if match:
        text = text[:match.start()]
    return text.strip()


def _evaluate_advancedif_rule_checks(
    response: str,
    rubrics: List[str],
    raw_response: Optional[str] = None,
) -> Tuple[float, bool, Dict[str, Any]]:
    checks: List[Tuple[str, float]] = []
    hard_fail = False

    if response and response.strip():
        checks.append(("non_empty", 1.0))
    else:
        checks.append(("non_empty", 0.0))
        hard_fail = True

    # Forbidden markers should be checked on the uncleaned text (after prompt echo removal),
    # so we still penalize leaked role labels or meta markers even if we later truncate them.
    forbidden_source = raw_response if raw_response is not None else response
    if _contains_forbidden_markers(forbidden_source):
        checks.append(("forbidden_markers", 0.0))
        hard_fail = True
    else:
        checks.append(("forbidden_markers", 1.0))

    avoid_bullets = _rubrics_avoid_bullets(rubrics)
    avoid_numbered = _rubrics_avoid_numbered(rubrics)
    require_bullets = _rubrics_require_bullets(rubrics)
    require_numbered = _rubrics_require_numbered(rubrics)

    has_bullets = bool(_BULLET_RE.search(response or ""))
    has_numbered = bool(_NUMBERED_RE.search(response or ""))

    if avoid_bullets:
        if has_bullets:
            checks.append(("avoid_bullets", 0.0))
            hard_fail = True
        else:
            checks.append(("avoid_bullets", 1.0))
    if avoid_numbered:
        if has_numbered:
            checks.append(("avoid_numbered", 0.0))
            hard_fail = True
        else:
            checks.append(("avoid_numbered", 1.0))
    if require_bullets:
        if has_bullets:
            checks.append(("require_bullets", 1.0))
        else:
            checks.append(("require_bullets", 0.0))
            hard_fail = True
    if require_numbered:
        if has_numbered:
            checks.append(("require_numbered", 1.0))
        else:
            checks.append(("require_numbered", 0.0))
            hard_fail = True

    if checks:
        rule_score = sum(score for _, score in checks) / len(checks)
    else:
        rule_score = 1.0
    return rule_score, hard_fail, {"checks": checks}

def _build_system_prompt(prompt_profile: Optional[str], hotpotqa_short_prompt: bool = False) -> str:
    if prompt_profile == "hotpotqa":
        if hotpotqa_short_prompt:
            return (
                "You are a question-answering assistant. "
                "Answer the question using only the provided context. "
                "Be concise and provide only the final answer. "
                "Do not repeat instructions. Output only your answer in the format: \\boxed{<your answer>}."
            )
        return (
            "You are a question-answering assistant. "
            "Answer the question using only the provided context. "
            "The answer is always contained in the provided context. "
            "Do not respond with 'unknown' or 'cannot determine'. "
            "Be concise and provide only the final answer. "
            "Do not repeat instructions. Output only your answer in the format: \\boxed{<your answer>}."
        )
    if prompt_profile == "advancedif":
        return (
            "You are a helpful assistant. Follow the user's instructions carefully and satisfy all "
            "explicit constraints on content, format, length, tone, and style. "
            "The input may include a multi-turn conversation transcript; respond only to the LAST user request. "
            "Do not continue or simulate the transcript. "
            "Return ONE complete response only. Do NOT include role labels (e.g., System/Assistant/User), "
            "do NOT add meta commentary, revision notes, or self-evaluation, and do NOT repeat or append extra messages. "
            "Do not repeat any section; do not include 'END OF MESSAGE', 'End of message', 'BEGIN WRITING', "
            "'Content Approved', or any transcript markers like 'User:' or 'Assistant:'. "
            "Output one response and stop immediately."
        )
    return (
        "You are a math expert solving AIME competition problems. "
        "AIME answers are always integers from 0 to 999."
    )


def _build_user_prompt(
    question: str,
    prompt_profile: Optional[str],
    hotpotqa_short_prompt: bool = False,
) -> str:
    if prompt_profile == "hotpotqa":
        if hotpotqa_short_prompt:
            return (
                f"{question}\n\n"
                "Please answer using only the provided context. "
                "Put your final answer within \\boxed{}. "
                "After you output the boxed answer, STOP immediately and do not repeat it."
            )
        return (
            f"{question}\n\n"
            "Please answer using only the provided context. "
            "Put your final answer within \\boxed{}. "
            "After you output the boxed answer, STOP immediately and do not repeat it."
        )
    if prompt_profile == "advancedif":
        question_text = str(question or "").strip()

        def _split_advancedif_conversation(text: str) -> Tuple[str, str]:
            if not text:
                return "", ""
            def _parse_json_turns(raw: Any) -> Optional[List[Tuple[str, str]]]:
                parsed = raw
                if isinstance(raw, str):
                    stripped = raw.strip()
                    if not stripped:
                        return None
                    if not stripped.startswith(("{", "[")):
                        return None
                    try:
                        parsed = json.loads(stripped)
                    except Exception:
                        return None
                if isinstance(parsed, dict):
                    if "messages" in parsed:
                        parsed = parsed.get("messages")
                    else:
                        return None
                if not isinstance(parsed, list):
                    return None
                turns: List[Tuple[str, str]] = []
                for entry in parsed:
                    if entry is None:
                        continue
                    if isinstance(entry, dict):
                        role = str(entry.get("role") or entry.get("speaker") or "").strip().lower()
                        content = entry.get("content") or entry.get("text") or entry.get("value") or ""
                        if content:
                            turns.append((role, str(content).strip()))
                    elif isinstance(entry, str):
                        if entry.strip():
                            turns.append(("", entry.strip()))
                return turns or None

            json_turns = _parse_json_turns(text)
            if json_turns:
                turns = json_turns
            else:
                turns = []
            role_pattern = re.compile(r"^(user|assistant|system|human)\s*:\s*", re.IGNORECASE)
            if not turns:
                current_role = ""
                buffer: List[str] = []
                for line in text.splitlines():
                    match = role_pattern.match(line)
                    if match:
                        if buffer:
                            turns.append((current_role, "\n".join(buffer).strip()))
                        current_role = match.group(1).lower()
                        buffer = [line[match.end():].strip()]
                    else:
                        if buffer:
                            buffer.append(line)
                        else:
                            buffer = [line]
                if buffer:
                    turns.append((current_role, "\n".join(buffer).strip()))

            last_user_idx = None
            for idx in range(len(turns) - 1, -1, -1):
                if turns[idx][0] in {"user", "human"}:
                    last_user_idx = idx
                    break
            if last_user_idx is None:
                if turns:
                    last_user_idx = len(turns) - 1
                else:
                    return "", text.strip()
            context_parts = []
            for role, content in turns[:last_user_idx]:
                if not content:
                    continue
                label = role.upper() if role else "CONTEXT"
                context_parts.append(f"{label}: {content}")
            context = "\n".join(context_parts).strip()
            last_user = turns[last_user_idx][1].strip()
            return context, last_user

        context, last_user = _split_advancedif_conversation(question_text)
        if last_user:
            if context:
                return (
                    "Conversation context (do NOT continue it):\n"
                    f"{context}\n\n"
                    "User request:\n"
                    f"{last_user}\n\n"
                    "Respond only to the user request."
                )
            return last_user
        return question_text
    return (
        f"{question}\n\n"
        "Please reason step by step, and put your final answer within \\boxed{}. "
        "Please make your reasoning process concise yet comprehensive. "
        "After you output the boxed answer, STOP immediately and do not repeat it."
    )



# ============================================================
# Output Truncation Helpers
# ============================================================

_BOXED_RE = re.compile(r"(?:\\boxed|boxed)\{[^}]*\}")

def _truncate_after_boxed(s: str) -> str:
    """
    If the model already produced a boxed answer, truncate right after the FIRST \\boxed{...}.
    This prevents runaway outputs like repeated 'Final Answer' blocks.
    """
    m = _BOXED_RE.search(s)
    if not m:
        return s
    return s[:m.end()]


def _truncate_after_answer_line(s: str) -> str:
    """
    For HotpotQA-style outputs, keep only the first non-empty line.
    """
    if not s:
        return s
    for line in s.splitlines():
        line = line.strip()
        if line:
            return line
    return s


def _strip_prompt_echo(generated: str, system_prompt: str, user_prompt: str) -> str:
    """
    Some backends return prompt+completion in response.text (or may echo prompt when completion is empty).
    Remove the prompt prefix if present, so we don't parse numbers from the question/prompt.
    """
    if not generated:
        return generated

    candidates = [
        f"SYSTEM: {system_prompt}\nUSER: {user_prompt}",
        f"SYSTEM: {system_prompt}\n\nUSER: {user_prompt}",
        f"SYSTEM: {system_prompt}\r\nUSER: {user_prompt}",
        f"SYSTEM: {system_prompt}\r\n\r\nUSER: {user_prompt}",
        user_prompt,
    ]
    for pref in candidates:
        if generated.startswith(pref):
            return generated[len(pref):].lstrip()

    # Also handle cases where the backend uses "<|system|>" templates by just removing the user prompt if present later.
    idx = generated.find(user_prompt)
    if idx != -1 and idx < 200:  # only if it's near the start
        return generated[idx + len(user_prompt):].lstrip()

    return generated

# ============================================================
# Student Rollout Generation
# ============================================================

def run_student_rollout(
    model: Model,
    question: str,
    ground_truth: Any,
    rubrics: Optional[List[str]] = None,
    judge_model: Optional[Model] = None,
    num_samples: int = 4,
    temperature: float = 0.9,
    max_tokens: int = 8192,
    top_p: float = 0.8,
    top_k: int = 50,
    prompt_profile: Optional[str] = None,
    hotpotqa_strict_train: bool = False,
    hotpotqa_short_prompt: bool = False,
    advancedif_judge_max_tokens: int = 768,
    advancedif_rule_weight: float = 0.3,
    advancedif_hard_cap: float = 0.2,
    advancedif_debug_out: Optional[List[Dict[str, Any]]] = None,
) -> List[RolloutSample]:
    """
    Generate N rollout samples from student model.
    
    These samples serve GRPO training: all samples with their rewards.
    
    Args:
        model: Student model for generation
        question: Math problem text
        ground_truth: Expected answer
        num_samples: Number of rollout samples to generate
        temperature: Sampling temperature (0.5 recommended for diversity)
        max_tokens: Maximum tokens for generation
        top_p: Top-p sampling parameter
        top_k: Top-k sampling parameter
    
    Returns:
        List of RolloutSample objects
    """
    from models import ModelError
    
    system_prompt = _build_system_prompt(prompt_profile, hotpotqa_short_prompt)
    user_prompt = _build_user_prompt(question, prompt_profile, hotpotqa_short_prompt)
    normalized_rubrics = _dedupe_rubrics(rubrics or [])
    stop_sequences = ADVANCEDIF_STOP_SEQUENCES if prompt_profile == "advancedif" else None

    # Keep Gemini student calls aligned with judge-style config (no stop/top_p/top_k),
    # except temperature=0.5 as requested.
    is_gemini_student = "gemini" in (getattr(model, "name", "") or "").lower()
    student_temperature = 0.5 if is_gemini_student else temperature
    student_top_p = None if is_gemini_student else top_p
    student_top_k = None if is_gemini_student else top_k
    student_stop = None if is_gemini_student else stop_sequences
    student_do_sample = None if is_gemini_student else True

    if prompt_profile == "advancedif" and judge_model is None:
        raise ValueError("AdvancedIF requires a judge_model for rubric scoring.")
    
    samples = []
    
    for i in range(num_samples):
        start_time = time.time()
        try:
            response = model.generate(
                prompt=user_prompt,
                system_prompt=system_prompt,
                temperature=student_temperature,
                max_tokens=max_tokens,
                stop=student_stop,
                do_sample=student_do_sample,
                top_p=student_top_p,
                top_k=student_top_k,
            )
            # Hard stop: if a boxed answer already appeared, keep only up to the first one
            raw_text = response.text or ""
            clean_text = _strip_prompt_echo(raw_text, system_prompt, user_prompt)
            resp_text = clean_text.strip()
            if prompt_profile == "hotpotqa" and not hotpotqa_strict_train:
                resp_text = _truncate_after_answer_line(resp_text)
            elif prompt_profile != "advancedif":
                resp_text = _truncate_after_boxed(resp_text)
            if prompt_profile == "advancedif":
                resp_text = _clean_advancedif_response(resp_text)
                if student_stop is not None and len(resp_text) < ADVANCEDIF_MIN_RESPONSE_CHARS:
                    response = model.generate(
                        prompt=user_prompt,
                        system_prompt=system_prompt,
                        temperature=student_temperature,
                        max_tokens=max_tokens,
                        stop=None,
                        do_sample=student_do_sample,
                        top_p=student_top_p,
                        top_k=student_top_k,
                    )
                    raw_text = response.text or ""
                    clean_text = _strip_prompt_echo(raw_text, system_prompt, user_prompt)
                    resp_text = _clean_advancedif_response(clean_text)
            generation_time = time.time() - start_time
            
            # Parse and judge
            if prompt_profile == "hotpotqa":
                parsed = extract_hotpotqa_answer(resp_text, strict=hotpotqa_strict_train)
                is_correct, f1 = judge_hotpotqa_answer(parsed, ground_truth)
                reward = (1.0 - HOTPOTQA_EM_LAMBDA) * float(f1) + HOTPOTQA_EM_LAMBDA * float(is_correct)
                rule_score = None
                judge_score = None
                hard_fail = False
            elif prompt_profile == "advancedif":
                parsed = None
                rule_score, hard_fail, _ = _evaluate_advancedif_rule_checks(
                    resp_text,
                    normalized_rubrics,
                    raw_response=clean_text,
                )
                judge_score = 0.0
                if normalized_rubrics:
                    adaptive_max_tokens = max(
                        advancedif_judge_max_tokens,
                        32 + 8 * len(normalized_rubrics),
                    )
                    judge_items, judge_raw = _judge_advancedif_rubrics(
                        judge_model=judge_model,
                        prompt=user_prompt,
                        response=resp_text,
                        rubrics=normalized_rubrics,
                        max_tokens=adaptive_max_tokens,
                    )
                    # Retry once with a larger budget if parsing is incomplete.
                    if not judge_items or len(judge_items) < len(normalized_rubrics):
                        retry_max_tokens = max(adaptive_max_tokens * 2, 512)
                        judge_items_retry, judge_raw_retry = _judge_advancedif_rubrics(
                            judge_model=judge_model,
                            prompt=user_prompt,
                            response=resp_text,
                            rubrics=normalized_rubrics,
                            max_tokens=retry_max_tokens,
                        )
                        if judge_items_retry and len(judge_items_retry) >= len(judge_items):
                            judge_items = judge_items_retry
                            judge_raw = judge_raw_retry
                            adaptive_max_tokens = retry_max_tokens
                    if advancedif_debug_out is not None:
                        advancedif_debug_out.append(
                            {
                                "candidate_idx": i,
                                "response": resp_text,
                                "judge_raw": judge_raw,
                                "rubrics": normalized_rubrics,
                                "judge_max_tokens": adaptive_max_tokens,
                            }
                        )
                    judge_items = _align_advancedif_items(normalized_rubrics, judge_items)
                    scores: List[float] = []
                    for item in judge_items:
                        raw_score = item.get("score")
                        score = _parse_rubric_score(str(raw_score) if raw_score is not None else "")
                        if score is None:
                            score = 0.0
                        scores.append(score)
                    if scores:
                        judge_score = sum(scores) / len(scores)
                if normalized_rubrics:
                    reward = (1.0 - advancedif_rule_weight) * judge_score + advancedif_rule_weight * rule_score
                else:
                    reward = rule_score
                reward = max(0.0, min(1.0, reward))
                if hard_fail:
                    reward = min(reward, advancedif_hard_cap)
                is_correct = reward >= 0.5
            else:
                parsed = extract_aime_answer(resp_text)
                is_correct = judge_aime_answer(parsed, ground_truth)
                reward = 1.0 if is_correct else 0.0
                rule_score = None
                judge_score = None
                hard_fail = False
            
            samples.append(RolloutSample(
                response=resp_text,
                reward=reward,
                parsed_answer=parsed,
                is_correct=is_correct,
                tokens_used=response.total_tokens or 0,
                generation_time=generation_time,
                rule_score=rule_score,
                judge_score=judge_score,
                hard_fail=hard_fail,
            ))
            
        except ModelError as e:
            print(f"  [Rollout {i+1}/{num_samples}] Generation failed: {e}")
            continue
        except Exception as e:
            print(f"  [Rollout {i+1}/{num_samples}] Unexpected error: {e}")
            continue
    
    return samples


# ============================================================
# Unified Production Step
# ============================================================

def unified_production_step(
    sample_id: str,
    question: str,
    ground_truth: Any,
    student_model: Model,
    rubrics: Optional[List[str]] = None,
    judge_model: Optional[Model] = None,
    num_student_samples: int = 4,
    temperature: float = 0.5,
    max_tokens: int = 8192,
    prompt_profile: Optional[str] = None,
    hotpotqa_strict_train: bool = False,
    hotpotqa_short_prompt: bool = False,
    advancedif_judge_max_tokens: int = 768,
    advancedif_rule_weight: float = 0.3,
    advancedif_hard_cap: float = 0.2,
    advancedif_debug_out: Optional[List[Dict[str, Any]]] = None,
) -> UnifiedProductionResult:
    """
    Unified production: GRPO samples only.
    
    This is the core function that generates:
    1. N student rollout samples (for GRPO training)
    2. (No memory extraction)
    
    Args:
        sample_id: Unique identifier for this question
        question: Math problem text
        ground_truth: Expected answer
        student_model: Model for student rollouts
        num_student_samples: Number of student rollouts
        temperature: Sampling temperature for student
        max_tokens: Max tokens for generation
        prompt_profile: Prompt profile for formatting
    
    Returns:
        UnifiedProductionResult with rollout data and stats
    """
    start_time = time.time()
    
    # Step 1: Student rollout
    print(f"  [Rollout] Generating {num_student_samples} student samples...")
    student_samples = run_student_rollout(
        model=student_model,
        question=question,
        ground_truth=ground_truth,
        rubrics=rubrics,
        judge_model=judge_model,
        num_samples=num_student_samples,
        temperature=temperature,
                max_tokens=max_tokens,
        prompt_profile=prompt_profile,
        hotpotqa_strict_train=hotpotqa_strict_train,
        hotpotqa_short_prompt=hotpotqa_short_prompt,
        advancedif_judge_max_tokens=advancedif_judge_max_tokens,
        advancedif_rule_weight=advancedif_rule_weight,
        advancedif_hard_cap=advancedif_hard_cap,
        advancedif_debug_out=advancedif_debug_out,
            )
    
    correct_students = [s for s in student_samples if s.is_correct]
    
    # Build rollout object
    rollout = QuestionRollout(
        sample_id=sample_id,
        question=question,
        ground_truth=ground_truth,
        student_samples=student_samples,
    )
    
    # Build stats
    total_time = time.time() - start_time
    stats = {
        "sample_id": sample_id,
        "num_student_samples": len(student_samples),
        "student_correct": len(correct_students),
        "student_accuracy": rollout.student_accuracy(),
        "has_correct": rollout.has_correct_student(),
        "total_time": total_time,
    }
    
    status = "✓" if rollout.has_correct_student() else "✗"
    print(f"  [{status}] Student: {len(correct_students)}/{len(student_samples)} correct, "
          f"Time: {total_time:.1f}s")
    
    return UnifiedProductionResult(
        rollout=rollout,
        stats=stats,
    )


# ============================================================
# Batch Production
# ============================================================


def _load_processed_ids(results_glob: str) -> set:
    """Load already-processed sample_ids from one or more results*.jsonl files."""
    import glob
    processed = set()
    for path in sorted(glob.glob(results_glob)):
        try:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        sid = data.get("sample_id")
                        if sid is not None:
                            processed.add(sid)
                    except Exception:
                        continue
        except FileNotFoundError:
            continue
    return processed


def _load_split_ids(split_path: Optional[str], split_name: str = "train") -> Optional[set]:
    if not split_path:
        return None
    try:
        with open(split_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return None
    if not isinstance(data, dict):
        return None
    split_key = "train_ids" if split_name == "train" else "test_ids"
    ids = data.get(split_key)
    if isinstance(ids, list):
        return {str(x) for x in ids if str(x).strip()}
    # Fallback: if no explicit split, allow full dataset.
    return None


def _merge_shards(output_dir: str, stem: str, num_shards: int) -> str:
    """
    Merge per-rank shard files into a single jsonl.
    Each rank writes: f"{stem}.rank{rank}.jsonl"
    Output: f"{stem}.jsonl"
    """
    final_path = os.path.join(output_dir, f"{stem}.jsonl")
    shard_paths = [os.path.join(output_dir, f"{stem}.rank{r}.jsonl") for r in range(num_shards)]

    with open(final_path, "w", encoding="utf-8") as out:
        for p in shard_paths:
            if not os.path.exists(p):
                continue
            with open(p, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        out.write(line.rstrip("\n") + "\n")
    return final_path


def run_unified_production(
    dataset_path: str,
    output_dir: str,
    student_model_name: str = "qwen:qwen-flash",
    judge_model_name: str = "gemini:gemini-2.5-flash",
    num_student_samples: int = 4,
    temperature: float = 0.9,
    max_tokens: int = 8192,
    max_samples: Optional[int] = None,
    resume: bool = True,
    prompt_profile: Optional[str] = None,
    hotpotqa_strict_train: bool = False,
    hotpotqa_short_prompt: bool = False,
    advancedif_judge_max_tokens: int = 256,
    advancedif_rule_weight: float = 0.3,
    advancedif_hard_cap: float = 0.2,
    advancedif_debug_judge: bool = False,
    advancedif_debug_judge_max_samples: int = 5,
    split_path: Optional[str] = None,
    split_name: str = "train",
) -> Dict[str, Any]:
    """
    Run unified production on entire dataset.

    Added:
      - accelerate.Accelerator: multi-process sharding (one shard per process)
      - tqdm progress bars (only on main/local-main process)

    Single-process outputs:
        - grpo_samples.jsonl
        - results.jsonl
        - stats.json

    Multi-process (`accelerate launch ...`) outputs:
        - grpo_samples.rank{i}.jsonl / results.rank{i}.jsonl
        - main process merges into grpo_samples.jsonl / results.jsonl
        - stats.json is reduced across processes and saved by main process
    """
    from dataset_utils import load_dataset
    from models import load_model

    accelerator = Accelerator() if Accelerator is not None else None
    is_dist = accelerator is not None and getattr(accelerator, "num_processes", 1) > 1
    rank = getattr(accelerator, "process_index", 0) if accelerator is not None else 0
    world = getattr(accelerator, "num_processes", 1) if accelerator is not None else 1
    is_main = getattr(accelerator, "is_main_process", True) if accelerator is not None else True
    is_local_main = getattr(accelerator, "is_local_main_process", True) if accelerator is not None else True

    def log(msg: str) -> None:
        if accelerator is not None:
            accelerator.print(msg)
        else:
            print(msg)

    if prompt_profile is None:
        prompt_profile = _infer_prompt_profile(dataset_path)
    if prompt_profile is None:
        prompt_profile = "aime"

    if is_main:
        log("=" * 60)
        log("Unified Production: GRPO Samples")
        log("=" * 60)
        log(f"Dataset: {dataset_path}")
        log(f"Output: {output_dir}")
        log(f"Prompt profile: {prompt_profile}")
        if prompt_profile == "hotpotqa":
            log(f"HotpotQA strict train: {hotpotqa_strict_train}")
            log(f"HotpotQA short prompt: {hotpotqa_short_prompt}")
        if prompt_profile == "advancedif":
            log(f"Judge model: {judge_model_name}")
            log(f"AdvancedIF judge max tokens: {advancedif_judge_max_tokens}")
            log(f"AdvancedIF rule weight: {advancedif_rule_weight}")
            log(f"AdvancedIF hard cap: {advancedif_hard_cap}")
            if split_path:
                log(f"Using split ids from: {split_path} ({split_name})")
        log(f"Student samples per question: {num_student_samples}")
        if is_dist:
            log(f"Accelerate: enabled (processes={world})")

    # Setup output (main process creates folder)
    if is_main:
        os.makedirs(output_dir, exist_ok=True)
    if accelerator is not None:
        accelerator.wait_for_everyone()

    # Sharded output paths (avoid write contention)
    grpo_path = os.path.join(output_dir, f"grpo_samples.rank{rank}.jsonl" if is_dist else "grpo_samples.jsonl")
    results_path = os.path.join(output_dir, f"results.rank{rank}.jsonl" if is_dist else "results.jsonl")
    stats_path = os.path.join(output_dir, "stats.json")
    debug_path = os.path.join(
        output_dir,
        f"advancedif_judge_debug.rank{rank}.jsonl" if is_dist else "advancedif_judge_debug.jsonl",
    )

    # Load checkpoint if resuming (read merged results + any shards)
    processed_ids = set()
    if resume:
        processed_ids = _load_processed_ids(os.path.join(output_dir, "results*.jsonl"))
        if is_main and processed_ids:
            log(f"Resuming: {len(processed_ids)} samples already processed")

    # Load models (each process loads its own copy)
    if is_main:
        log("\nLoading models...")
    student_model = load_model(student_model_name)
    judge_model = None
    if prompt_profile == "advancedif":
        judge_model = load_model(judge_model_name)

    # Load dataset
    if is_main:
        log("Loading dataset...")
    samples = list(load_dataset(dataset_path, max_samples=max_samples))
    allowed_ids = _load_split_ids(split_path, split_name)
    if allowed_ids:
        samples = [s for s in samples if getattr(s, "id", None) in allowed_ids]
    samples = [s for s in samples if getattr(s, "id", None) not in processed_ids]

    # Split between processes
    if accelerator is not None:
        with accelerator.split_between_processes(samples) as local_samples:
            local_samples = list(local_samples)
    else:
        local_samples = samples

    if is_main:
        log(f"Processing {len(samples)} samples (this process: {len(local_samples)})")
    debug_remaining = advancedif_debug_judge_max_samples if (advancedif_debug_judge and prompt_profile == "advancedif") else 0

    # Aggregate stats (per process)
    total_stats = {
        "total": 0,
        "student_success": 0,
        "total_grpo_samples": 0,
        "avg_student_accuracy": 0.0,
    }

    # Progress bar (main process only)
    use_pbar = (tqdm is not None) and is_local_main
    iterator = tqdm(local_samples, total=len(local_samples), desc=f"Producing (rank {rank})") if use_pbar else local_samples

    if is_main:
        log("\n" + "-" * 60)

    for idx, sample in enumerate(iterator):
        sid = getattr(sample, "id", None)
        if not sid:
            continue

        if not use_pbar and is_local_main:
            log(f"\n[{idx+1}/{len(local_samples)}] Sample: {sid}")

        try:
            rubrics = None
            if prompt_profile == "advancedif":
                rubrics = _dedupe_rubrics(_extract_advancedif_rubrics_from_meta(getattr(sample, "meta", None)))

            debug_holder: Optional[List[Dict[str, Any]]] = None
            if debug_remaining > 0 and prompt_profile == "advancedif":
                debug_holder = []

            result = unified_production_step(
                sample_id=sid,
                question=sample.question,
                ground_truth=sample.answer,
                student_model=student_model,
                rubrics=rubrics,
                judge_model=judge_model,
                num_student_samples=num_student_samples,
                temperature=temperature,
                max_tokens=max_tokens,
                prompt_profile=prompt_profile,
                hotpotqa_strict_train=hotpotqa_strict_train,
                hotpotqa_short_prompt=hotpotqa_short_prompt,
                advancedif_judge_max_tokens=advancedif_judge_max_tokens,
                advancedif_rule_weight=advancedif_rule_weight,
                advancedif_hard_cap=advancedif_hard_cap,
                advancedif_debug_out=debug_holder,
            )

            # Save GRPO samples
            with open(grpo_path, "a", encoding="utf-8") as f:
                for rollout_sample in result.rollout.student_samples:
                    record = {
                        "sample_id": sid,
                        "question": sample.question,
                        "ground_truth": str(sample.answer),
                        "response": rollout_sample.response,
                        "reward": rollout_sample.reward,
                        "parsed_answer": rollout_sample.parsed_answer,
                        "is_correct": rollout_sample.is_correct,
                        "prompt_profile": prompt_profile,
                    }
                    if rollout_sample.rule_score is not None:
                        record["rule_score"] = rollout_sample.rule_score
                    if rollout_sample.judge_score is not None:
                        record["judge_score"] = rollout_sample.judge_score
                    if rollout_sample.hard_fail:
                        record["hard_fail"] = True
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")

            if debug_holder:
                with open(debug_path, "a", encoding="utf-8") as f:
                    for item in debug_holder:
                        record = {
                            "sample_id": sid,
                            "question": sample.question,
                            "judge_raw": item.get("judge_raw", ""),
                            "response": item.get("response", ""),
                            "candidate_idx": item.get("candidate_idx"),
                            "rubrics": item.get("rubrics", []),
                            "judge_max_tokens": item.get("judge_max_tokens"),
                        }
                        f.write(json.dumps(record, ensure_ascii=False) + "\n")
                debug_remaining -= 1

            # Save full result
            with open(results_path, "a", encoding="utf-8") as f:
                full_result = {
                    "sample_id": sid,
                    "question": sample.question,
                    "ground_truth": str(sample.answer),
                    "rollout": {
                        "student_accuracy": result.rollout.student_accuracy(),
                        "has_correct": result.rollout.has_correct_student(),
                        "num_samples": len(result.rollout.student_samples),
                    },
                    "stats": result.stats,
                }
                f.write(json.dumps(full_result, ensure_ascii=False) + "\n")

            # Update stats
            total_stats["total"] += 1
            total_stats["total_grpo_samples"] += len(result.rollout.student_samples)
            total_stats["avg_student_accuracy"] += result.rollout.student_accuracy()

            if result.rollout.has_correct_student():
                total_stats["student_success"] += 1

        except Exception as e:
            if is_local_main:
                log(f"  [ERROR] {e}")
            continue

    # Reduce stats across processes
    if accelerator is not None:
        if hasattr(accelerator, "gather_object"):
            gathered = accelerator.gather_object(total_stats)
        else:
            # Fallback for older accelerate: write stats per-rank and merge on main.
            stats_shard = os.path.join(output_dir, f"stats.rank{rank}.json")
            with open(stats_shard, "w", encoding="utf-8") as f:
                json.dump(total_stats, f)
            accelerator.wait_for_everyone()
            gathered = []
            if is_main:
                for r in range(world):
                    p = os.path.join(output_dir, f"stats.rank{r}.json")
                    if not os.path.exists(p):
                        continue
                    with open(p, "r", encoding="utf-8") as f:
                        gathered.append(json.load(f))
            accelerator.wait_for_everyone()

        if is_main:
            reduced = {
                "total": 0,
                "student_success": 0,
                "total_grpo_samples": 0,
                "avg_student_accuracy": 0.0,
            }
            for d in gathered:
                reduced["total"] += d.get("total", 0)
                reduced["student_success"] += d.get("student_success", 0)
                reduced["total_grpo_samples"] += d.get("total_grpo_samples", 0)
                reduced["avg_student_accuracy"] += d.get("avg_student_accuracy", 0.0)
            total_stats = reduced

    # Finalize + merge shards (main process)
    if is_main:
        if total_stats["total"] > 0:
            total_stats["avg_student_accuracy"] /= total_stats["total"]
            total_stats["overall_accuracy"] = total_stats["student_success"] / total_stats["total"]

        if is_dist:
            grpo_final = _merge_shards(output_dir, "grpo_samples", world)
            results_final = _merge_shards(output_dir, "results", world)
        else:
            grpo_final = os.path.join(output_dir, "grpo_samples.jsonl")
            results_final = os.path.join(output_dir, "results.jsonl")

        with open(stats_path, "w", encoding="utf-8") as f:
            json.dump(total_stats, f, indent=2)

        log("\n" + "=" * 60)
        log("Production Complete!")
        log("=" * 60)
        log(f"Total samples: {total_stats['total']}")
        log(f"Student success: {total_stats['student_success']} ({100*total_stats.get('student_success',0)/max(1,total_stats['total']):.1f}%)")
        log(f"Total GRPO samples: {total_stats['total_grpo_samples']}")
        log(f"\nOutputs:")
        log(f"  - GRPO samples: {grpo_final}")
        log(f"  - Full results: {results_final}")
        log(f"  - Stats: {stats_path}")

    return total_stats


# ============================================================
# Main Entry Point
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Unified GRPO Production")
    subparsers = parser.add_subparsers(dest="command")
    
    # Produce command
    produce = subparsers.add_parser("produce", help="Run unified production")
    produce.add_argument("--dataset", required=True, help="Dataset path or HF name")
    produce.add_argument("--output", required=True, help="Output directory")
    produce.add_argument("--student_model", default="qwen:qwen-flash")
    produce.add_argument("--num_samples", type=int, default=4, help="Rollouts per question")
    produce.add_argument("--temperature", type=float, default=0.9)
    produce.add_argument("--max_tokens", type=int, default=8192, help="Max tokens per generation (prevents runaway outputs)")
    produce.add_argument("--max_samples", type=int, default=None)
    produce.add_argument("--no_resume", action="store_true")
    produce.add_argument("--prompt_profile", choices=["aime", "hotpotqa", "advancedif"], default=None,
                         help="Override prompt profile (default: auto-detect from dataset)")
    produce.add_argument("--hotpotqa_strict_train", action="store_true",
                         help="Use strict HotpotQA answer format for rewards (training)")
    produce.add_argument("--hotpotqa_short_prompt", action="store_true",
                         help="Use short HotpotQA prompt (A/B test)")
    produce.add_argument("--judge_model", default="gemini:gemini-2.5-flash",
                         help="LLM judge model for AdvancedIF rubric scoring")
    produce.add_argument("--gemini_credentials", default=None,
                         help="Path to Gemini service account JSON (sets GEMINI_CREDENTIALS/GOOGLE_APPLICATION_CREDENTIALS)")
    produce.add_argument("--gemini_location", default="us-central1",
                         help="Gemini Vertex location (sets GEMINI_LOCATION)")
    produce.add_argument("--advancedif_judge_max_tokens", type=int, default=256,
                         help="Max tokens for AdvancedIF judge output")
    produce.add_argument("--advancedif_rule_weight", type=float, default=0.3,
                         help="Weight of rule-based score in AdvancedIF reward")
    produce.add_argument("--advancedif_hard_cap", type=float, default=0.2,
                         help="Reward cap when AdvancedIF hard constraints fail")
    produce.add_argument("--advancedif_debug_judge", action="store_true",
                         help="Save raw AdvancedIF judge outputs for a few samples")
    produce.add_argument("--advancedif_debug_judge_max_samples", type=int, default=5,
                         help="How many AdvancedIF samples to capture judge raw output")
    produce.add_argument("--data_split_path", type=str, default=None,
                         help="Path to data_split.json to restrict rollouts to a split")
    produce.add_argument("--data_split_name", choices=["train", "test"], default="train",
                         help="Which split ids to use from data_split.json")
    
    args = parser.parse_args()
    
    if args.command == "produce":
        if args.gemini_credentials:
            os.environ["GEMINI_CREDENTIALS"] = args.gemini_credentials
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = args.gemini_credentials
        if args.gemini_location:
            os.environ["GEMINI_LOCATION"] = args.gemini_location
        run_unified_production(
            dataset_path=args.dataset,
            output_dir=args.output,
            student_model_name=args.student_model,
            num_student_samples=args.num_samples,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            max_samples=args.max_samples,
            resume=not args.no_resume,
            prompt_profile=args.prompt_profile,
            hotpotqa_strict_train=args.hotpotqa_strict_train,
            hotpotqa_short_prompt=args.hotpotqa_short_prompt,
            judge_model_name=args.judge_model,
            advancedif_judge_max_tokens=args.advancedif_judge_max_tokens,
            advancedif_rule_weight=args.advancedif_rule_weight,
            advancedif_hard_cap=args.advancedif_hard_cap,
            advancedif_debug_judge=args.advancedif_debug_judge,
            advancedif_debug_judge_max_samples=args.advancedif_debug_judge_max_samples,
            split_path=args.data_split_path,
            split_name=args.data_split_name,
        )
    else:
        parser.print_help()
        print("\nExamples:")
        print("  # Run unified production (GRPO samples)")
        print("  python rollout.py produce --dataset di-zhang-fdu/AIME_1983_2024 --output ./unified")
        print("")
        print("  # Then train GRPO using collected samples")
        print("  python dpo_baseline.py grpo train --data ./unified/grpo_samples.jsonl --model Qwen/Qwen2.5-7B")


if __name__ == "__main__":
    main()
