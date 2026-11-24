"""
Experiment Runner for Memory-Augmented Reasoning

Supports running ablation studies with different memory strategies:
- Baseline (no memory)
- Student-only memory
- Teacher-only memory  
- Contrastive memory

Usage:
    python experiment.py --phase produce --train_dataset di-zhang-fdu/AIME_1983_2024 --test_dataset aime25 --strategy all
    python experiment.py --phase evaluate --train_dataset di-zhang-fdu/AIME_1983_2024 --test_dataset aime25 --strategy contrastive

If no dataset arguments are provided, defaults to:
    train=di-zhang-fdu/AIME_1983_2024, test=aime25
"""

from __future__ import annotations

import argparse
import json
import os

# Optional accelerate-based sharding (best-effort). If accelerate is unavailable,
# fall back to env vars used by torchrun / accelerate.
def _dist_info() -> tuple[int, int, bool]:
    try:
        from accelerate import Accelerator  # type: ignore
        acc = Accelerator()
        return int(acc.process_index), int(acc.num_processes), bool(acc.is_main_process)
    except Exception:
        rank = int(os.environ.get("RANK") or os.environ.get("LOCAL_RANK") or 0)
        world = int(os.environ.get("WORLD_SIZE") or os.environ.get("LOCAL_WORLD_SIZE") or 1)
        return rank, world, rank == 0

import ast
import random
import re
import time
from dataclasses import dataclass, asdict
from typing import Any, Dict, Iterable, List, Literal, Optional, Set, Tuple

_GEMINI_TOKEN_CACHE: Dict[str, Tuple[str, float, str]] = {}
from datetime import datetime

from dataset_utils import Sample, load_dataset
from models import load_model, Model, ModelError
from memory import (
    MemoryBank,
    MemoryItem,
    extract_memories_from_cascade,
    extract_preference_memories,
    refine_memories,
    _load_embedding_model,
    _embed_texts,
    _build_memory_items,
    _dict_to_memory_item,
    _memory_item_to_dict,
)
from cascade import CascadeConfig, build_default_tools, run_cascade
from judge import create_judge, MathJudge


# ============================================================
# Configuration
# ============================================================

DEFAULT_TRAIN_DATASET = "di-zhang-fdu/AIME_1983_2024"
DEFAULT_TEST_DATASET = "aime25"


def _normalize_dataset_ref(path: str) -> str:
    normalized = path.strip().lower()
    if "huggingface.co/datasets/" in normalized:
        normalized = normalized.split("/datasets/", 1)[1]
    normalized = normalized.split("?", 1)[0].strip("/")
    if ":" in normalized:
        normalized = normalized.split(":", 1)[0]
    return normalized


def _is_hotpotqa_dataset(path: Optional[str]) -> bool:
    if not path:
        return False
    normalized = _normalize_dataset_ref(path)
    dataset_name = normalized.split("/")[-1]
    return dataset_name in {"hotpotqa", "hotpot_qa", "hotpot-qa"}


def _is_math500_dataset(path: Optional[str]) -> bool:
    if not path:
        return False
    normalized = _normalize_dataset_ref(path)
    dataset_name = normalized.split("/")[-1]
    return dataset_name in {"math500", "math-500", "math_500"}


def _is_helpsteer3_dataset(path: Optional[str]) -> bool:
    if not path:
        return False
    normalized = _normalize_dataset_ref(path)
    dataset_name = normalized.split("/")[-1]
    return dataset_name in {"helpsteer3", "helpsteer-3", "helpsteer_3"}


def _is_helpsteer2_dataset(path: Optional[str]) -> bool:
    if not path:
        return False
    normalized = _normalize_dataset_ref(path)
    dataset_name = normalized.split("/")[-1]
    return dataset_name in {"helpsteer2", "helpsteer-2", "helpsteer_2"}


def _is_rubrichub_dataset(path: Optional[str]) -> bool:
    if not path:
        return False
    normalized = _normalize_dataset_ref(path)
    dataset_name = normalized.split("/")[-1]
    return dataset_name in {"rubrichub", "rubrichub_v1", "rubrichub-v1"}


def _is_advancedif_dataset(path: Optional[str]) -> bool:
    if not path:
        return False
    normalized = _normalize_dataset_ref(path)
    dataset_name = normalized.split("/")[-1]
    return dataset_name in {
        "advancedif",
        "advanced-if",
        "advanced_if",
        "ifbench",
        "if-bench",
        "if_bench",
        "wild-if-eval",
        "wild_if_eval",
        "wildif",
        "biggen-bench",
        "biggen_bench",
        "rubrichub",
        "rubrichub_v1",
        "rubrichub-v1",
    }


def _infer_prompt_profile(
    dataset_path: Optional[str],
    train_dataset_path: Optional[str],
    test_dataset_path: Optional[str],
) -> Optional[str]:
    if any(
        _is_hotpotqa_dataset(path)
        for path in (dataset_path, train_dataset_path, test_dataset_path)
    ):
        return "hotpotqa"
    if any(
        _is_advancedif_dataset(path)
        for path in (dataset_path, train_dataset_path, test_dataset_path)
    ):
        return "advancedif"
    if any(
        _is_rubrichub_dataset(path)
        for path in (dataset_path, train_dataset_path, test_dataset_path)
    ):
        return "advancedif"
    if any(
        _is_helpsteer3_dataset(path)
        for path in (dataset_path, train_dataset_path, test_dataset_path)
    ):
        return "helpsteer3"
    if any(
        _is_helpsteer2_dataset(path)
        for path in (dataset_path, train_dataset_path, test_dataset_path)
    ):
        return "helpsteer2"
    return None


def _is_preference_task(task_type: str, prompt_profile: Optional[str]) -> bool:
    return task_type.lower() == "preference" or prompt_profile == "helpsteer3"


def _disable_stop_for_aime(task_type: str) -> bool:
    """AIME generation/extraction should not use stop sequences."""
    return (task_type or "").strip().lower() == "aime"


def _normalize_domain_value(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip().lower()


def _parse_domain_filter(domain_filter: Optional[str]) -> Optional[Set[str]]:
    if not domain_filter:
        return None
    tokens: List[str] = []
    raw = str(domain_filter)
    for part in raw.replace(",", " ").split():
        if part.strip():
            tokens.append(_normalize_domain_value(part))
    allowed = {tok for tok in tokens if tok}
    return allowed or None


def _apply_helpsteer3_domain_filter(
    samples: List[Sample],
    allowed_domains: Set[str],
) -> List[Sample]:
    if not allowed_domains:
        return samples
    filtered: List[Sample] = []
    for sample in samples:
        domain = None
        if isinstance(sample.meta, dict):
            domain = sample.meta.get("domain")
            if domain is None:
                domain = sample.meta.get("domain_name")
        domain_norm = _normalize_domain_value(domain)
        if domain_norm and domain_norm in allowed_domains:
            filtered.append(sample)
    return filtered


def _maybe_filter_helpsteer3_domains(
    train_samples: List[Sample],
    test_samples: List[Sample],
    split_info: Dict[str, Any],
    config: ExperimentConfig,
    train_path: Optional[str],
    test_path: Optional[str],
) -> Tuple[List[Sample], List[Sample], Dict[str, Any]]:
    allowed = _parse_domain_filter(config.helpsteer3_domain)
    if not allowed:
        return train_samples, test_samples, split_info
    is_helpsteer3 = config.prompt_profile == "helpsteer3" or any(
        _is_helpsteer3_dataset(path) for path in (train_path, test_path) if path
    )
    if not is_helpsteer3:
        return train_samples, test_samples, split_info
    before_train = len(train_samples)
    before_test = len(test_samples)
    train_samples = _apply_helpsteer3_domain_filter(train_samples, allowed)
    test_samples = _apply_helpsteer3_domain_filter(test_samples, allowed)
    split_info = split_info.copy()
    split_info["helpsteer3_domain"] = config.helpsteer3_domain
    split_info["helpsteer3_train_before"] = before_train
    split_info["helpsteer3_train_after"] = len(train_samples)
    split_info["helpsteer3_test_before"] = before_test
    split_info["helpsteer3_test_after"] = len(test_samples)
    return train_samples, test_samples, split_info

@dataclass
class ExperimentConfig:
    # Dataset
    dataset_path: str
    max_samples: Optional[int] = None
    train_dataset_path: Optional[str] = None
    test_dataset_path: Optional[str] = None
    split_ratio: float = 0.8
    split_seed: int = 42
    hotpotqa_train_pool_size: int = 2000
    hotpotqa_train_size: int = 400
    hotpotqa_test_size: int = 100
    hotpotqa_train_top_k: int = 4
    hotpotqa_lock_split_path: Optional[str] = None
    math500_train_size: int = 400
    math500_test_size: int = 100
    task_type: Literal["aime", "math", "preference"] = "aime"
    prompt_profile: Optional[str] = None
    strict_hotpotqa_eval: bool = False
    hotpotqa_short_prompt_eval: bool = False
    eval_instruction: Optional[str] = None
    helpsteer3_domain: Optional[str] = None
    data_split_path: Optional[str] = None
    data_split_name: Optional[str] = None
    
    # Models
    student_model: str = "qwen:qwen-flash"
    teacher_model: str = "qwen:qwen-plus"
    expert_model: str = "qwen:qwen-max"
    judge_model: str = "qwen:qwen-flash"
    extractor_model: str = "student"  # use student model for memory extraction
    deepseek_reasoner_model: Optional[str] = "deepseek:deepseek-reasoner"
    api_timeout: Optional[int] = None
    api_retries: Optional[int] = None
    
    # Memory
    memory_strategy: Literal["student", "teacher", "contrastive", "all"] = "contrastive"
    memory_top_k: int = 2
    memory_top_k_global: int = 3
    memory_top_k_local: int = 5
    memory_top_k_preference: int = 5  # HelpSteer3 only: top-k preference_rule to retrieve (no local/global split)
    memory_answer_leak_filter: bool = True
    memory_max_similarity: Optional[float] = 0.9
    memory_min_similarity: Optional[float] = 0.1
    memory_max_items: int = 0
    exclude_self_memory: bool = True
    extraction_reasoning: bool = True
    enable_memory_refinement: bool = False  # 是否启用 memory 精炼
    memory_retrieval: Literal["tf", "embedding"] = "tf"
    embedding_model: str = "Qwen/Qwen3-Embedding-4B"
    embedding_cache_path: Optional[str] = None
    embedding_batch_size: int = 32
    embedding_device: Optional[str] = None
    memory_query_expand: bool = True
    memory_query_expand_max_tokens: int = 128
    memory_extract_mode: Literal["two_agent", "combined"] = "two_agent"
    memory_one_per_dimension: bool = False

    # Streaming memory refinement (production)
    streaming_memory_refinement: bool = False
    streaming_memory_refinement_frequency: int = 1
    streaming_memory_refinement_model: str = "extractor"
    streaming_memory_history_top_k: int = 5

    # SA-CTS memory bandit (retrieval)
    enable_memory_bandit: bool = False
    memory_bandit_lambda: float = 0.3
    memory_bandit_noise_var: float = 1.0
    memory_bandit_explore_epsilon: float = 0.25
    memory_bandit_neighbor_k: int = 5
    memory_bandit_prior_mu: float = 0.0
    memory_bandit_prior_sigma2: float = 1.0
    memory_bandit_state_path: Optional[str] = None
    memory_bandit_seed: Optional[int] = 42

    # AdvancedIF
    advancedif_judge_max_tokens: int = 768
    advancedif_debug_judge: bool = False
    advancedif_debug_judge_max_samples: int = 3
    advancedif_memory_mode: str = "rubric"  # rubric | teacher | temp | rubric+teacher | rubric+temp | teacher+temp | all
    advancedif_compare_max_tokens: int = 1024
    advancedif_temp_low: float = 0.3
    advancedif_temp_high: float = 1.2

    # Train subset selection (similarity to test set)
    train_subset_top_k: int = 0
    train_subset_embedding_model: str = "Qwen/Qwen3-Embedding-4B"
    train_subset_embedding_batch_size: int = 32
    train_subset_embedding_device: Optional[str] = None
    
    # Cascade
    cascade_threshold: float = 0.6
    enable_tools: bool = False
    temperature: float = 0.5
    upgrade_on_failure_only: bool = True
    cascade_max_tokens: int = 8192
    cascade_max_tokens_l3: int = 8192
    
    # Output
    output_dir: str = "./experiments"
    experiment_name: str = "default"
    resume_production: bool = False
    llm_extract_answer: bool = False
    sample_id_filter: Optional[List[str]] = None


def _build_base_system_prompt(
    task_type: str,
    prompt_profile: Optional[str] = None,
    hotpotqa_short_prompt_eval: bool = False,
) -> str:
    """
    [FIX] 与 test_fixed.py 保持一致的 prompt 格式
    使用官方推荐的 \\boxed{} 格式
    """
    if prompt_profile == "hotpotqa":
        if hotpotqa_short_prompt_eval:
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
    if prompt_profile in {"advancedif", "helpsteer2"}:
        return (
            "You are a helpful assistant. Answer only the last user request from the input. "
            "Do not continue or simulate any transcript. Output a single final answer only. "
            "Do not include role labels or any meta commentary. "
            "Do not repeat the prompt, rules, or constraints."
        )
    if prompt_profile == "helpsteer3" or task_type == "preference":
        return (
            "You are a careful preference judge. "
            "Given the conversation context and two candidate responses, choose the better response overall. "
            "Focus on helpfulness, correctness, safety, clarity, and completeness. "
            "Answer with ONLY 'Response 1' or 'Response 2'."
        )
    if task_type == "aime":
        return (
            "You are a math expert solving AIME competition problems. "
            "AIME answers are always integers from 0 to 999. "
        )
    return "You are a math expert. The final answer may be an integer, fraction, or expression."


def _apply_eval_instruction(base_prompt: str, instruction: Optional[str]) -> str:
    if not instruction:
        return base_prompt
    cleaned = instruction.strip()
    if not cleaned:
        return base_prompt
    if base_prompt.endswith((".", "!", "?")):
        return f"{base_prompt} {cleaned}"
    return f"{base_prompt}. {cleaned}"


def _build_base_system_prompt_with_memory(
    task_type: str,
    prompt_profile: Optional[str] = None,
    hotpotqa_short_prompt_eval: bool = False,
) -> str:
    """
    [FIX] 带 memory 的 system prompt
    对于 HotpotQA，简短说明即可，不要与"only use context"冲突
    """
    base = _build_base_system_prompt(task_type, prompt_profile, hotpotqa_short_prompt_eval)
    if prompt_profile == "hotpotqa":
        # 简短说明，不与context-only指令冲突
        memory_note = " You may see reasoning hints below. Use them if relevant."
    elif prompt_profile in {"advancedif", "helpsteer2"}:
        memory_note = ""
    elif prompt_profile == "helpsteer3" or task_type == "preference":
        memory_note = " You may see preference rules below. Use them if relevant."
    else:
        memory_note = (
            " You may be provided with relevant problem-solving strategies from past experience. "
            "Use them if they are helpful for the current problem."
        )
    return base + memory_note


def _build_user_prompt(
    question: str,
    task_type: str,
    prompt_profile: Optional[str] = None,
    hotpotqa_short_prompt_eval: bool = False,
) -> str:
    """
    [FIX] 与 test_fixed.py 保持一致的 user prompt 格式
    使用官方推荐的 \\boxed{} 格式
    """
    if prompt_profile == "hotpotqa":
        if hotpotqa_short_prompt_eval:
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
    if prompt_profile in {"advancedif", "helpsteer2"}:
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
            # Parse simple role-tagged transcripts into turns.
            if not turns:
                role_pattern = re.compile(r"^(user|assistant|system|human)\s*:\s*", re.IGNORECASE)
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
            return last_user
        return question_text
    if task_type == "aime":
        return (
            f"{question}\n\n"
            "Please reason step by step, and put your final answer within \\boxed{}. "
            "Please make your reasoning process concise yet comprehensive. "
            "After you output the boxed answer, STOP immediately and do not repeat it."
        )
    return (
        f"{question}\n\n"
        "Please reason step by step, and put your final answer within \\boxed{}. "
        "Please make your reasoning process concise yet comprehensive. "
        "After you output the boxed answer, STOP immediately and do not repeat it."
    )


def _build_preference_user_prompt(
    context: str,
    response1: str,
    response2: str,
) -> str:
    context = (context or "").strip()
    r1 = (response1 or "").strip()
    r2 = (response2 or "").strip()
    return (
        "Input/Prompt (may be conversation, code, or problem statement):\n"
        f"{context}\n\n"
        "Response 1:\n"
        f"{r1}\n\n"
        "Response 2:\n"
        f"{r2}\n\n"
        "Which response is better overall? "
        "Answer with ONLY 'Response 1' or 'Response 2'."
    )


def _build_judge_system_prompt() -> str:
    return (
        "You are an impartial evaluator. Given the user request and two candidate responses, choose the better response.\n"
        "Decision rule (priority order):\n"
        "1) Follow the user's explicit constraints (format, length, scope, tone). Any violation is heavily penalized.\n"
        "2) Correctness and safety (no incorrect claims or unsafe advice).\n"
        "3) Task usefulness: directly solves what the user asked for.\n"
        "4) Clarity and organization: easy to follow, appropriate structure.\n"
        "5) Conciseness: no unnecessary filler; do not reward generic templates unless they directly help.\n"
        "If both responses are similarly good, choose the one that better matches the user's constraints and avoids unnecessary content.\n"
        "Ties are not allowed. You must choose exactly one.\n"
        "Answer with ONLY: Response 1 or Response 2"
    )


def _build_judge_user_prompt(context: str, response1: str, response2: str) -> str:
    return (
        f"User Request:\n{(context or '').strip()}\n\n"
        f"Response 1:\n{(response1 or '').strip()}\n\n"
        f"Response 2:\n{(response2 or '').strip()}\n\n"
        "Which response is better? Reply with ONLY: Response 1 or Response 2 (no other words)."
    )


def _build_judge_rank_system_prompt() -> str:
    return (
        "You are an impartial evaluator. Given the user request and three candidate responses, "
        "rank them from best to worst.\n"
        "Decision rule (priority order):\n"
        "1) Follow the user's explicit constraints (format, length, scope, tone). Any violation is heavily penalized.\n"
        "2) Correctness and safety (no incorrect claims or unsafe advice).\n"
        "3) Task usefulness: directly solves what the user asked for.\n"
        "4) Clarity and organization: easy to follow, appropriate structure.\n"
        "5) Conciseness: no unnecessary filler; do not reward generic templates unless they directly help.\n"
        "Ties are not allowed. You must choose a strict ordering.\n"
        "Answer with ONLY one of: 1>2>3, 1>3>2, 2>1>3, 2>3>1, 3>1>2, 3>2>1"
    )


def _build_judge_rank_user_prompt(
    context: str,
    response1: str,
    response2: str,
    response3: str,
) -> str:
    return (
        f"User Request:\n{(context or '').strip()}\n\n"
        f"Response 1:\n{(response1 or '').strip()}\n\n"
        f"Response 2:\n{(response2 or '').strip()}\n\n"
        f"Response 3:\n{(response3 or '').strip()}\n\n"
        "Rank the responses best to worst. Reply with ONLY one of: "
        "1>2>3, 1>3>2, 2>1>3, 2>3>1, 3>1>2, 3>2>1."
    )


def _parse_rank_choice(response_text: str) -> Optional[List[int]]:
    if not response_text:
        return None
    cleaned = response_text.strip().lower()
    if not cleaned:
        return None
    digits = re.findall(r"[123]", cleaned)
    order: List[int] = []
    for d in digits:
        val = int(d)
        if val not in order:
            order.append(val)
        if len(order) == 3:
            break
    if len(order) == 3 and set(order) == {1, 2, 3}:
        return order
    return None


def _extract_openai_message_text(message: Any) -> str:
    if message is None:
        return ""
    text = ""
    reasoning = ""
    if isinstance(message, dict):
        text = message.get("content") or message.get("text") or ""
        reasoning = message.get("reasoning_content") or message.get("reasoning") or ""
    else:
        text = getattr(message, "content", "") or ""
        reasoning = getattr(message, "reasoning_content", "") or getattr(message, "reasoning", "") or ""
    if text:
        return str(text)
    if reasoning:
        return str(reasoning)
    return ""


def _parse_judge_choice(response_text: str) -> Optional[int]:
    """
    Return 1 for Response 1, 2 for Response 2, 0 for tie, None if unknown.
    """
    if not response_text:
        return None
    cleaned = response_text.strip().lower()
    if not cleaned:
        return None
    if any(token in cleaned for token in ("tie", "equal", "same", "both")):
        return 0
    patterns_1 = [
        r"\bresponse\s*1\b",
        r"\bresponse\s*one\b",
        r"\boption\s*1\b",
        r"\boption\s*one\b",
        r"\bchoice\s*1\b",
        r"\bchoice\s*one\b",
        r"\bfirst\b",
        r"\bresponse\s*a\b",
        r"\boption\s*a\b",
        r"\bchoice\s*a\b",
    ]
    patterns_2 = [
        r"\bresponse\s*2\b",
        r"\bresponse\s*two\b",
        r"\boption\s*2\b",
        r"\boption\s*two\b",
        r"\bchoice\s*2\b",
        r"\bchoice\s*two\b",
        r"\bsecond\b",
        r"\bresponse\s*b\b",
        r"\boption\s*b\b",
        r"\bchoice\s*b\b",
    ]
    has_1 = any(re.search(pat, cleaned) for pat in patterns_1)
    has_2 = any(re.search(pat, cleaned) for pat in patterns_2)
    if has_1 and not has_2:
        return 1
    if has_2 and not has_1:
        return 2
    # Fallback: parse single-number answers
    has_num_1 = re.search(r"\b1\b", cleaned) is not None
    has_num_2 = re.search(r"\b2\b", cleaned) is not None
    if has_num_1 and not has_num_2:
        return 1
    if has_num_2 and not has_num_1:
        return 2
    return None


def _parse_preference_choice(response_text: str) -> Optional[int]:
    """
    Return -1 for Response 1, +1 for Response 2, 0 for tie, None if unknown.
    """
    if not response_text:
        return None
    cleaned = response_text.strip().lower()
    if not cleaned:
        return None

    head = cleaned.splitlines()[0].strip()


def _student_pick_better_response(
    student_model: Model,
    prompt: str,
    response1: str,
    response2: str,
    max_tokens: int = 12,
    comparison_note: Optional[str] = None,
) -> Optional[int]:
    """
    Ask the student model to choose which response is better.
    Returns 1 if response1 is better, 2 if response2 is better, 0 for tie, None if unknown.
    """
    system_prompt = (
        "You are a strict evaluator. Compare two candidate responses to the same user prompt.\n"
        "Reply with ONLY: Response 1 or Response 2. No other text."
    )
    if comparison_note:
        system_prompt = f"{system_prompt}\n{comparison_note}"
    user_prompt = (
        f"User prompt:\n{_truncate_text(prompt, 2000)}\n\n"
        f"Response 1:\n{_truncate_text(response1, 2000)}\n\n"
        f"Response 2:\n{_truncate_text(response2, 2000)}\n\n"
        "Which response better satisfies the user's instructions? Reply ONLY with: Response 1 or Response 2."
    )
    try:
        result = student_model.generate(
            prompt=user_prompt,
            system_prompt=system_prompt,
            temperature=0.0,
            do_sample=False,
            max_tokens=max_tokens,
            stop=["\n"],
        )
    except Exception:
        return None
    return _parse_judge_choice(result.text or "")


# ============================================================
# NEW: Qwen3-Max LLM-as-a-Judge API for HelpSteer3 Comparison
# ============================================================

def _call_qwen3_max_judge(
    context: str,
    response1: str,
    response2: str,
    api_key: Optional[str] = None,
    base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1",
    model: str = "qwen3-max",  # Changed to qwen3-max
    timeout: int = 60,
    max_retries: int = 3,
) -> Optional[int]:
    """
    Call Qwen3-Max API to judge which response is better.
    
    Returns:
        1 if response1 is better
        2 if response2 is better
        None if error or tie
    """
    try:
        from openai import OpenAI
    except ImportError:
        print("Warning: openai package not installed. Install with: pip install openai")
        return None
    
    api_key = api_key or os.environ.get("DASHSCOPE_API_KEY") or os.environ.get("QWEN_API_KEY")
    if not api_key:
        print("Warning: No API key found. Set DASHSCOPE_API_KEY or QWEN_API_KEY environment variable.")
        return None
    
    client = OpenAI(api_key=api_key, base_url=base_url)
    
    # Force compact output to reduce format errors.
    system_prompt = "Choose the better response. Ties are not allowed. Answer with ONLY: 1 or 2."
    user_prompt = (
        f"User Request:\n{(context or '').strip()}\n\n"
        f"Response 1:\n{(response1 or '').strip()}\n\n"
        f"Response 2:\n{(response2 or '').strip()}\n\n"
        "Reply with ONLY: 1 or 2"
    )
    
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.0,
                max_tokens=4,
                stop=["\n"],
                timeout=timeout,
            )
            
            message = response.choices[0].message if response.choices else None
            result_text = _extract_openai_message_text(message)
            parsed = _parse_judge_choice(result_text)
            if parsed == 0:
                print("  Qwen judge returned tie (disallowed). Retrying...")
                if attempt < max_retries - 1:
                    continue
                return None
            if parsed is None:
                print(f"  Qwen judge parse failed: {result_text[:200]!r}")
            return parsed
                
        except Exception as e:
            print(f"  API call failed (attempt {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # Exponential backoff
            continue
    
    return None


def _call_deepseek_judge(
    context: str,
    response1: str,
    response2: str,
    api_key: Optional[str] = None,
    base_url: str = "https://api.deepseek.com/v1",
    model: str = "deepseek-reasoner",
    timeout: int = 60,
    max_retries: int = 3,
) -> Optional[int]:
    """
    Call DeepSeek API to judge which response is better.

    Returns:
        1 if response1 is better
        2 if response2 is better
        0 if tie
        None if error or unknown
    """
    try:
        from openai import OpenAI
    except ImportError:
        print("Warning: openai package not installed. Install with: pip install openai")
        return None

    api_key = api_key or os.environ.get("DEEPSEEK_API_KEY") or os.environ.get("DEEPSEEK_KEY")
    if not api_key:
        print("Warning: No API key found. Set DEEPSEEK_API_KEY environment variable.")
        return None

    client = OpenAI(api_key=api_key, base_url=base_url)
    system_prompt = _build_judge_system_prompt()
    user_prompt = _build_judge_user_prompt(context, response1, response2)

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.0,
                max_tokens=20,
                timeout=timeout,
            )
            message = response.choices[0].message if response.choices else None
            result_text = _extract_openai_message_text(message)
            parsed = _parse_judge_choice(result_text)
            if parsed == 0:
                print("  DeepSeek judge returned tie (disallowed). Retrying...")
                if attempt < max_retries - 1:
                    continue
                return None
            if parsed is None:
                print(f"  DeepSeek judge parse failed: {result_text[:200]!r}")
            return parsed
        except Exception as e:
            print(f"  DeepSeek API call failed (attempt {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            continue

    return None


def _call_openai_rank_judge(
    context: str,
    response1: str,
    response2: str,
    response3: str,
    api_key: Optional[str],
    base_url: str,
    model: str,
    timeout: int = 60,
    max_retries: int = 3,
    provider_name: str = "OpenAI-compatible",
) -> Optional[List[int]]:
    try:
        from openai import OpenAI
    except ImportError:
        print("Warning: openai package not installed. Install with: pip install openai")
        return None

    if not api_key:
        print(f"Warning: No API key found for {provider_name}.")
        return None

    client = OpenAI(api_key=api_key, base_url=base_url)
    system_prompt = _build_judge_rank_system_prompt()
    user_prompt = _build_judge_rank_user_prompt(context, response1, response2, response3)

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.0,
                max_tokens=8,
                stop=["\n"],
                timeout=timeout,
            )
            message = response.choices[0].message if response.choices else None
            result_text = _extract_openai_message_text(message)
            parsed = _parse_rank_choice(result_text)
            if parsed is None:
                print(f"  {provider_name} rank parse failed: {result_text[:200]!r}")
                if attempt < max_retries - 1:
                    continue
            return parsed
        except Exception as e:
            print(f"  {provider_name} rank API call failed (attempt {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            continue

    return None


def _call_qwen3_max_rank_judge(
    context: str,
    response1: str,
    response2: str,
    response3: str,
    api_key: Optional[str] = None,
    base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1",
    model: str = "qwen-max",
    timeout: int = 60,
    max_retries: int = 3,
) -> Optional[List[int]]:
    api_key = api_key or os.environ.get("DASHSCOPE_API_KEY") or os.environ.get("QWEN_API_KEY")
    return _call_openai_rank_judge(
        context=context,
        response1=response1,
        response2=response2,
        response3=response3,
        api_key=api_key,
        base_url=base_url,
        model=model,
        timeout=timeout,
        max_retries=max_retries,
        provider_name="Qwen rank judge",
    )


def _call_deepseek_rank_judge(
    context: str,
    response1: str,
    response2: str,
    response3: str,
    api_key: Optional[str] = None,
    base_url: str = "https://api.deepseek.com/v1",
    model: str = "deepseek-reasoner",
    timeout: int = 60,
    max_retries: int = 3,
) -> Optional[List[int]]:
    api_key = api_key or os.environ.get("DEEPSEEK_API_KEY") or os.environ.get("DEEPSEEK_KEY")
    return _call_openai_rank_judge(
        context=context,
        response1=response1,
        response2=response2,
        response3=response3,
        api_key=api_key,
        base_url=base_url,
        model=model,
        timeout=timeout,
        max_retries=max_retries,
        provider_name="DeepSeek rank judge",
    )


def _load_gemini_credentials(credentials_path: str) -> Optional[Tuple[str, str]]:
    """
    Load GCP access token and project_id from a service account JSON.
    Returns (access_token, project_id) or None on failure.
    """
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
        print("Warning: google-auth not installed. Install with: pip install google-auth")
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
            print("Warning: project_id missing in Gemini credentials.")
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
    except Exception as exc:
        print(f"Warning: Failed to load Gemini credentials: {exc}")
        return None


def _gemini_thinking_budget(model: str) -> Optional[int]:
    name = (model or "").lower()
    if "gemini-2.5" not in name:
        return None
    if "pro" in name:
        return 128
    if "flash" in name:
        return 0
    return None


def _gemini_max_output_tokens(model: str, use_schema: bool) -> int:
    name = (model or "").lower()
    if "gemini-2.5-pro" in name:
        return 256
    if "gemini-2.5-flash" in name:
        return 32 if use_schema else 64
    return 64


def _gemini_supports_system_instruction(model: str) -> bool:
    name = (model or "").lower()
    return "gemini-2.0-flash" in name


def _call_gemini_judge(
    context: str,
    response1: str,
    response2: str,
    credentials_path: Optional[str],
    location: str = "us-central1",
    model: str = "gemini-2.5-pro",
    timeout: int = 60,
    max_retries: int = 3,
) -> Optional[int]:
    """
    Call Gemini (Vertex AI) to judge which response is better.

    Returns:
        1 if response1 is better
        2 if response2 is better
        0 if tie
        None if error or unknown
    """
    if not credentials_path:
        print("Warning: Gemini credentials not provided.")
        return None

    token_and_project = _load_gemini_credentials(credentials_path)
    if not token_and_project:
        return None
    access_token, project_id = token_and_project

    url = (
        f"https://{location}-aiplatform.googleapis.com/v1/projects/{project_id}/"
        f"locations/{location}/publishers/google/models/{model}:generateContent"
    )
    system_prompt = _build_judge_system_prompt()
    user_prompt = _build_judge_user_prompt(context, response1, response2)
    compact_system_prompt = "Choose the better response. Ties are not allowed. Answer with ONLY: 1 or 2."
    compact_user_prompt = (
        f"User Request:\n{(context or '').strip()}\n\n"
        f"Response 1:\n{(response1 or '').strip()}\n\n"
        f"Response 2:\n{(response2 or '').strip()}\n\n"
        "Reply with ONLY: 1 or 2"
    )
    thinking_budget = _gemini_thinking_budget(model)
    supports_system = _gemini_supports_system_instruction(model)

    def _build_payload(use_schema: bool, compact: bool) -> Dict[str, Any]:
        active_system = compact_system_prompt if compact else system_prompt
        active_user = compact_user_prompt if compact else user_prompt
        if not supports_system:
            active_user = f"{active_system}\n\n{active_user}".strip()
        generation_config: Dict[str, Any] = {
            "temperature": 0.0,
            "maxOutputTokens": _gemini_max_output_tokens(model, use_schema),
            "stopSequences": ["\n"],
        }
        if thinking_budget is not None:
            generation_config["thinkingConfig"] = {"thinkingBudget": thinking_budget}
        if use_schema:
            generation_config["responseMimeType"] = "text/x.enum"
            generation_config["responseSchema"] = {
                "type": "STRING",
                "enum": ["1", "2"],
            }
        else:
            generation_config["responseMimeType"] = "text/plain"
        payload: Dict[str, Any] = {
            "contents": [{"role": "user", "parts": [{"text": active_user}]}],
            "generationConfig": generation_config,
        }
        if supports_system:
            payload["systemInstruction"] = {"parts": [{"text": active_system}]}
        return payload

    use_schema = True
    use_compact = True
    for attempt in range(max_retries):
        try:
            payload = _build_payload(use_schema, use_compact)
            try:
                import requests  # type: ignore
                resp = requests.post(
                    url,
                    headers={"Authorization": f"Bearer {access_token}"},
                    json=payload,
                    timeout=timeout,
                )
                status = resp.status_code
                data = resp.json() if resp.content else {}
            except Exception:
                import urllib.request
                import urllib.error

                req = urllib.request.Request(
                    url,
                    data=json.dumps(payload).encode("utf-8"),
                    headers={
                        "Authorization": f"Bearer {access_token}",
                        "Content-Type": "application/json",
                    },
                    method="POST",
                )
                with urllib.request.urlopen(req, timeout=timeout) as response:
                    status = response.status
                    body = response.read()
                    data = json.loads(body.decode("utf-8")) if body else {}

            if status != 200:
                print(f"  Gemini API error (status {status}): {data}")
                if status == 400 and isinstance(data, dict):
                    message = str(data.get("error", {}).get("message", "")).lower()
                    if "thinking_budget" in message:
                        if thinking_budget == 0:
                            thinking_budget = 1
                            continue
                        if thinking_budget is not None:
                            thinking_budget = None
                            continue
                if status == 429:
                    backoff = 10 * (2 ** attempt)
                    print(f"  Gemini rate limited; sleeping {backoff}s")
                    time.sleep(backoff)
                    continue
                if use_schema:
                    use_schema = False
                    continue
                if use_compact:
                    raise RuntimeError(f"Gemini API status {status}")
                use_compact = True
                continue

            candidates = data.get("candidates") or []
            if not candidates:
                print(f"  Gemini response missing candidates: {data}")
                use_schema = False
                use_compact = True
                continue
            content = candidates[0].get("content", {}) if isinstance(candidates[0], dict) else {}
            parts = content.get("parts", [])
            if not parts:
                print(f"  Gemini response missing parts: {data}")
                use_schema = False
                use_compact = True
                continue
            texts: List[str] = []
            for part in parts:
                if isinstance(part, dict):
                    text_val = part.get("text")
                    if text_val:
                        texts.append(str(text_val))
            result_text = "".join(texts).strip()
            parsed: Optional[int] = None
            if use_schema:
                # text/x.enum returns a plain enum string (not JSON)
                parsed = _parse_judge_choice(result_text)
            if parsed is None:
                parsed = _parse_judge_choice(result_text)
            if parsed == 0:
                print("  Gemini judge returned tie (disallowed). Retrying...")
                if attempt < max_retries - 1:
                    continue
                return None
            if parsed is None:
                finish_reason = candidates[0].get("finishReason")
                safety = candidates[0].get("safetyRatings")
                print(
                    f"  Gemini judge parse failed: {result_text[:200]!r} | "
                    f"finish={finish_reason} safety={safety}"
                )
                if finish_reason == "MAX_TOKENS" or result_text.strip().lower() in {"response", "here"}:
                    use_schema = False
                    use_compact = True
                    continue
            return parsed
        except Exception as e:
            print(f"  Gemini API call failed (attempt {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            continue

    return None


def _call_gemini_rank_judge(
    context: str,
    response1: str,
    response2: str,
    response3: str,
    credentials_path: Optional[str],
    location: str = "us-central1",
    model: str = "gemini-2.5-pro",
    timeout: int = 60,
    max_retries: int = 3,
) -> Optional[List[int]]:
    if not credentials_path:
        print("Warning: Gemini credentials not provided.")
        return None

    token_and_project = _load_gemini_credentials(credentials_path)
    if not token_and_project:
        return None
    access_token, project_id = token_and_project

    url = (
        f"https://{location}-aiplatform.googleapis.com/v1/projects/{project_id}/"
        f"locations/{location}/publishers/google/models/{model}:generateContent"
    )
    system_prompt = _build_judge_rank_system_prompt()
    user_prompt = _build_judge_rank_user_prompt(context, response1, response2, response3)
    thinking_budget = _gemini_thinking_budget(model)
    supports_system = _gemini_supports_system_instruction(model)
    enum_values = ["1>2>3", "1>3>2", "2>1>3", "2>3>1", "3>1>2", "3>2>1"]

    def _build_payload(use_schema: bool) -> Dict[str, Any]:
        active_user = user_prompt
        if not supports_system:
            active_user = f"{system_prompt}\n\n{active_user}".strip()
        generation_config: Dict[str, Any] = {
            "temperature": 0.0,
            "maxOutputTokens": 16 if use_schema else 32,
            "stopSequences": ["\n"],
        }
        if thinking_budget is not None:
            generation_config["thinkingConfig"] = {"thinkingBudget": thinking_budget}
        if use_schema:
            generation_config["responseMimeType"] = "text/x.enum"
            generation_config["responseSchema"] = {
                "type": "STRING",
                "enum": enum_values,
            }
        else:
            generation_config["responseMimeType"] = "text/plain"
        payload: Dict[str, Any] = {
            "contents": [{"role": "user", "parts": [{"text": active_user}]}],
            "generationConfig": generation_config,
        }
        if supports_system:
            payload["systemInstruction"] = {"parts": [{"text": system_prompt}]}
        return payload

    use_schema = True
    for attempt in range(max_retries):
        try:
            payload = _build_payload(use_schema)
            try:
                import requests  # type: ignore
                resp = requests.post(
                    url,
                    headers={"Authorization": f"Bearer {access_token}"},
                    json=payload,
                    timeout=timeout,
                )
                status = resp.status_code
                data = resp.json() if resp.content else {}
            except Exception:
                import urllib.request
                import urllib.error

                req = urllib.request.Request(
                    url,
                    data=json.dumps(payload).encode("utf-8"),
                    headers={
                        "Authorization": f"Bearer {access_token}",
                        "Content-Type": "application/json",
                    },
                    method="POST",
                )
                with urllib.request.urlopen(req, timeout=timeout) as response:
                    status = response.status
                    body = response.read()
                    data = json.loads(body.decode("utf-8")) if body else {}

            if status != 200:
                print(f"  Gemini rank API error (status {status}): {data}")
                if status == 400 and isinstance(data, dict):
                    message = str(data.get("error", {}).get("message", "")).lower()
                    if "thinking_budget" in message:
                        if thinking_budget == 0:
                            thinking_budget = 1
                            continue
                        if thinking_budget is not None:
                            thinking_budget = None
                            continue
                if status == 429:
                    backoff = 10 * (2 ** attempt)
                    print(f"  Gemini rate limited; sleeping {backoff}s")
                    time.sleep(backoff)
                    continue
                if use_schema:
                    use_schema = False
                    continue
                if attempt < max_retries - 1:
                    continue
                return None

            candidates = data.get("candidates") or []
            if not candidates:
                print(f"  Gemini rank response missing candidates: {data}")
                use_schema = False
                continue
            content = candidates[0].get("content", {}) if isinstance(candidates[0], dict) else {}
            parts = content.get("parts", [])
            if not parts:
                print(f"  Gemini rank response missing parts: {data}")
                use_schema = False
                continue
            texts: List[str] = []
            for part in parts:
                if isinstance(part, dict):
                    text_val = part.get("text")
                    if text_val:
                        texts.append(str(text_val))
            result_text = "".join(texts).strip()
            parsed = _parse_rank_choice(result_text)
            if parsed is None:
                finish_reason = candidates[0].get("finishReason")
                safety = candidates[0].get("safetyRatings")
                print(
                    f"  Gemini rank parse failed: {result_text[:200]!r} | "
                    f"finish={finish_reason} safety={safety}"
                )
                if use_schema:
                    use_schema = False
                    continue
            return parsed
        except Exception as e:
            print(f"  Gemini rank API call failed (attempt {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            continue

    return None


def _call_qwen3_max_judge_bidirectional(
    context: str,
    response_a: str,
    response_b: str,
    label_a: str = "a",
    label_b: str = "b",
    api_key: Optional[str] = None,
    base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1",
    model: str = "qwen-max",
    timeout: int = 60,
    max_retries: int = 3,
) -> Tuple[Optional[str], str]:
    """ 
    Args:
        context: The question/context
        response_a: First response (e.g., from base model)
        response_b: Second response (e.g., from RL model)
        label_a: Label for response_a (e.g., "base", "rl", "memory")
        label_b: Label for response_b
        
    Returns:
        Tuple of (winner_label, consistency):
        - winner_label: label_a, label_b, or None (if inconsistent/error)
        - consistency: "consistent", "inconsistent", or "error"
    """
    # Round 1: A as Response 1, B as Response 2
    result1 = _call_qwen3_max_judge(
        context=context,
        response1=response_a,
        response2=response_b,
        api_key=api_key,
        base_url=base_url,
        model=model,
        timeout=timeout,
        max_retries=max_retries,
    )
    
    # Round 2: B as Response 1, A as Response 2
    result2 = _call_qwen3_max_judge(
        context=context,
        response1=response_b,
        response2=response_a,
        api_key=api_key,
        base_url=base_url,
        model=model,
        timeout=timeout,
        max_retries=max_retries,
    )
    
    if result1 is None or result2 is None or result1 == 0 or result2 == 0:
        return None, "error"
    
    # Check consistency
    # Round 1: 1 means A wins, 2 means B wins
    # Round 2: 1 means B wins, 2 means A wins (positions swapped)
    
    round1_winner = label_a if result1 == 1 else label_b  # who won in round 1
    round2_winner = label_a if result2 == 2 else label_b  # who won in round 2 (note: swapped)
    
    if round1_winner == round2_winner:
        return round1_winner, "consistent"
    else:
        # Position bias detected - results are inconsistent
        return None, "inconsistent"


def _parse_judge_models(judge_models: Optional[str]) -> List[str]:
    if not judge_models:
        return ["qwen3-max"]
    models: List[str] = []
    for part in str(judge_models).replace(",", " ").split():
        item = part.strip().lower()
        if not item:
            continue
        is_rank = item.endswith("-rank")
        base = item[:-5] if is_rank else item
        if base in {"qwen3-max", "qwen-max", "qwen", "qwen3"}:
            base_name = "qwen3-max"
        elif base in {"deepseek-reasoner", "deepseek"}:
            base_name = "deepseek-reasoner"
        elif base in {"gemini-2.5-pro", "gemini"}:
            base_name = "gemini-2.5-pro"
        elif base in {"gemini-2.5-flash", "gemini-flash"}:
            base_name = "gemini-2.5-flash"
        else:
            base_name = base
        name = f"{base_name}-rank" if is_rank else base_name
        models.append(name)
    # Preserve order, remove duplicates
    seen: Set[str] = set()
    ordered: List[str] = []
    for name in models:
        if name not in seen:
            seen.add(name)
            ordered.append(name)
    return ordered


def _is_rank_judge(judge_name: str) -> bool:
    return judge_name.endswith("-rank")


def _base_judge_name(judge_name: str) -> str:
    return judge_name[:-5] if _is_rank_judge(judge_name) else judge_name


def _resolve_gemini_model_for_judge(judge_name: str, default_model: str) -> str:
    name = judge_name.lower()
    if "gemini-2.5-flash" in name:
        return "gemini-2.5-flash"
    if "gemini-2.5-pro" in name:
        return "gemini-2.5-pro"
    return default_model


def _resolve_gemini_credentials_path(credentials_path: Optional[str]) -> Optional[str]:
    if credentials_path and os.path.exists(credentials_path):
        return credentials_path
    env_path = os.environ.get("GEMINI_CREDENTIALS") or os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    if env_path and os.path.exists(env_path):
        return env_path
    default_path = "/Users/uestczr/Desktop/reseach_code/fixed_code/gemini-481814-7955f422ed16.json"
    if os.path.exists(default_path):
        return default_path
    return None


def _build_response_generation_system_prompt() -> str:
    """System prompt for response generation task."""
    return (
        "You are a helpful assistant. "
        "Provide a clear, accurate, and helpful response to the user's question or request. "
        "Focus on being helpful, correct, safe, and complete."
    )


# Task routing labels for HelpSteer3 memory retrieval
TASK_ROUTING_LABELS = {
    "coding": ["code", "programming", "debug", "function", "script", "python", "javascript", "java", "c++", "sql", "api", "error", "bug", "compile", "runtime"],
    "technical_explanation": ["explain", "how does", "what is", "concept", "theory", "principle", "mechanism", "architecture", "system"],
    "math_stem": ["calculate", "solve", "equation", "formula", "proof", "theorem", "math", "physics", "chemistry", "biology", "statistics"],
    "writing": ["write", "essay", "article", "blog", "story", "email", "letter", "report", "summary", "draft"],
    "analysis": ["analyze", "compare", "evaluate", "review", "assess", "critique", "pros and cons", "advantages", "disadvantages"],
    "troubleshooting": ["fix", "problem", "issue", "error", "not working", "help", "stuck", "troubleshoot", "resolve"],
    "creative": ["creative", "brainstorm", "ideas", "suggest", "recommend", "design", "create", "invent"],
    "factual": ["who", "when", "where", "what year", "how many", "list", "name", "define"],
    "instruction": ["how to", "steps", "guide", "tutorial", "instructions", "process", "procedure"],
}


def _extract_task_type(context: str) -> str:
    """
    Light-weight task routing: classify context into task type.
    Returns the most likely task type based on keyword matching.
    """
    context_lower = context.lower()
    
    # Count matches for each task type
    scores = {}
    for task_type, keywords in TASK_ROUTING_LABELS.items():
        score = sum(1 for kw in keywords if kw in context_lower)
        if score > 0:
            scores[task_type] = score
    
    if not scores:
        return "general"
    
    # Return the task type with the highest score
    return max(scores, key=scores.get)


def _extract_keywords_for_retrieval(context: str, max_keywords: int = 5) -> List[str]:
    """
    Extract key terms from context for memory retrieval.
    Focuses on nouns, technical terms, and action words.
    """
    import re
    
    # Remove common stop words
    stop_words = {
        "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "could",
        "should", "may", "might", "must", "shall", "can", "need", "dare",
        "to", "of", "in", "for", "on", "with", "at", "by", "from", "as",
        "into", "through", "during", "before", "after", "above", "below",
        "between", "under", "again", "further", "then", "once", "here",
        "there", "when", "where", "why", "how", "all", "each", "few",
        "more", "most", "other", "some", "such", "no", "nor", "not",
        "only", "own", "same", "so", "than", "too", "very", "just",
        "and", "but", "if", "or", "because", "until", "while", "this",
        "that", "these", "those", "i", "me", "my", "myself", "we", "our",
        "you", "your", "he", "him", "his", "she", "her", "it", "its",
        "they", "them", "their", "what", "which", "who", "whom",
        "please", "thanks", "thank", "help", "want", "like", "know",
    }
    
    # Extract words (alphanumeric, 3+ chars)
    words = re.findall(r'\b[a-zA-Z]{3,}\b', context.lower())
    
    # Filter stop words and get unique terms
    keywords = []
    seen = set()
    for word in words:
        if word not in stop_words and word not in seen:
            keywords.append(word)
            seen.add(word)
            if len(keywords) >= max_keywords:
                break
    
    return keywords


def _build_retrieval_query(context: str) -> str:
    """
    Build optimized query for memory retrieval.
    Uses task type + dimension hints + keywords instead of full context.
    """
    task_type = _extract_task_type(context)
    keywords = _extract_keywords_for_retrieval(context, max_keywords=5)
    
    # Map task types to likely dimensions
    task_to_dimensions = {
        "coding": "correctness clarity",
        "technical_explanation": "clarity completeness",
        "math_stem": "correctness completeness",
        "writing": "clarity coherence tone",
        "analysis": "completeness depth",
        "troubleshooting": "correctness helpfulness",
        "creative": "creativity relevance",
        "factual": "accuracy conciseness",
        "instruction": "clarity completeness step-by-step",
        "general": "",
    }
    
    dimension_hint = task_to_dimensions.get(task_type, "")
    
    # Combine: task_type + dimension_hint + keywords
    query_parts = [task_type]
    if dimension_hint:
        query_parts.append(dimension_hint)
    query_parts.extend(keywords)
    
    return " ".join(query_parts)


def _build_response_generation_system_prompt_with_memory(
    memory_bank: Any,
    retrieved_memories: List[Any],
) -> str:
    """
    System prompt with memory guidance for RESPONSE GENERATION task.
    
    Uses a priority-based structure and treats memory rules as optional guidance.
    """
    base_prompt = (
        "You are a helpful assistant. "
        "Priority: (1) safety/correctness, (2) user constraints (format/length/scope/tone), "
        "(3) usefulness, (4) clarity, (5) brevity."
    )
    
    if not retrieved_memories:
        return base_prompt + "\nWrite the answer."
    
    # Build memory rules from preference rules
    rules: List[str] = []
    for mem in retrieved_memories[:3]:  # Limit to top 3
        if hasattr(mem, 'trigger') and mem.trigger and hasattr(mem, 'comparison') and mem.comparison:
            trigger = (mem.trigger or "").strip()
            dimension = (getattr(mem, 'dimension', '') or "").strip()
            comparison = (mem.comparison or "").strip()
            
            # Format as concise rule
            if dimension:
                rule = f"• [{dimension}] {trigger}: {comparison}"
            else:
                rule = f"• {trigger}: {comparison}"
            rules.append(rule)
            
        elif hasattr(mem, 'title') and mem.title:
            title = (mem.title or "").strip()
            content = ""
            if hasattr(mem, 'content') and mem.content:
                content = (mem.content or "").strip()[:150]
            if content:
                rules.append(f"• {title}: {content}")
    
    if not rules:
        return base_prompt + "\nWrite the answer."
    
    rules_block = "\n".join(rules)
    
    return f"""{base_prompt}
Memory rules are optional: use only if relevant; ignore if conflicting; do not quote—convert to actions.
Memory rules:
{rules_block}
Write the answer."""

    scan = head[:120] if head else cleaned[:120]

    # Tie / equal
    if re.search(r"\b(tie|equal|same|both)\b", scan):
        return 0

    # Explicit response labels
    if re.search(r"\bresponse\s*1\b", scan) or re.search(r"\bresponse\s*one\b", scan):
        return -1
    if re.search(r"\bresponse\s*2\b", scan) or re.search(r"\bresponse\s*two\b", scan):
        return 1
    if re.search(r"\boption\s*1\b", scan) or re.search(r"\boption\s*one\b", scan):
        return -1
    if re.search(r"\boption\s*2\b", scan) or re.search(r"\boption\s*two\b", scan):
        return 1
    if re.fullmatch(r"(1|one|first|1st)", scan):
        return -1
    if re.fullmatch(r"(2|two|second|2nd)", scan):
        return 1

    # Fallback: look for first occurrence in full text
    if re.search(r"\bresponse\s*1\b", cleaned) or re.search(r"\bresponse\s*one\b", cleaned):
        return -1
    if re.search(r"\bresponse\s*2\b", cleaned) or re.search(r"\bresponse\s*two\b", cleaned):
        return 1
    if re.search(r"\boption\s*1\b", cleaned) or re.search(r"\boption\s*one\b", cleaned):
        return -1
    if re.search(r"\boption\s*2\b", cleaned) or re.search(r"\boption\s*two\b", cleaned):
        return 1
    if re.fullmatch(r"(1|one|first|1st)", cleaned):
        return -1
    if re.fullmatch(r"(2|two|second|2nd)", cleaned):
        return 1
    return None


def _build_extraction_system_prompt(
    task_type: str,
    prompt_profile: Optional[str] = None,
) -> str:
    if prompt_profile == "hotpotqa":
        return (
            "You are a question-answering assistant. "
            "Provide a brief rationale using the given context. "
            "Do NOT state the final answer."
        )
    prompt = (
        "You are a math tutor. Provide a concise, high-level solution outline. "
        "Do NOT give the final answer."
    )
    if task_type == "aime":
        prompt += " The final answer is an integer from 0 to 999; do not state it."
    return prompt


def _build_extraction_user_prompt(
    question: str,
    prompt_profile: Optional[str] = None,
) -> str:
    if prompt_profile == "hotpotqa":
        return (
            f"Question:\n{question}\n\n"
            "Give a short rationale based on the context without the final answer."
        )
    return f"Question:\n{question}\n\nGive a short solution outline without the final answer."


def _attach_extraction_response(
    traj: Dict[str, Any],
    model: Model,
    question: str,
    task_type: str,
    prompt_profile: Optional[str],
    temperature: float,
) -> None:
    if not traj or "extraction_response" in traj or traj.get("parsed_reasoning"):
        return
    try:
        response = model.generate(
            prompt=_build_extraction_user_prompt(question, prompt_profile),
            system_prompt=_build_extraction_system_prompt(task_type, prompt_profile),
            temperature=0.0,
            do_sample=False,
            max_tokens=512,
        )
        traj["extraction_response"] = response.text
    except ModelError as exc:
        traj["extraction_error"] = str(exc)


def _extract_preference_fields(sample: Sample) -> Tuple[str, str, str, Optional[int], Optional[Any]]:
    context = sample.meta.get("context") if isinstance(sample.meta, dict) else None
    if not context:
        context = sample.question
    response1 = sample.meta.get("response1") if isinstance(sample.meta, dict) else None
    response2 = sample.meta.get("response2") if isinstance(sample.meta, dict) else None
    pref = sample.answer
    if pref is None and isinstance(sample.meta, dict):
        pref = sample.meta.get("overall_preference")
    pref_val: Optional[int]
    try:
        pref_val = int(pref) if pref is not None else None
    except Exception:
        pref_val = None
    dimension_hint = None
    if isinstance(sample.meta, dict):
        for key in ("individual_preference", "preference_dimension", "dimension"):
            if key in sample.meta:
                dimension_hint = sample.meta.get(key)
                break
    return str(context or ""), str(response1 or ""), str(response2 or ""), pref_val, dimension_hint


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


def _normalize_rubrics(value: Any) -> List[str]:
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
            return _normalize_rubrics(parsed)
        if "\n" in text:
            return [line.strip() for line in text.splitlines() if line.strip()]
        return [text]
    text = str(value).strip()
    return [text] if text else []


def _extract_advancedif_rubrics(sample: Sample) -> List[str]:
    if not isinstance(sample.meta, dict):
        return []
    rubrics = sample.meta.get("rubrics")
    if rubrics:
        return _normalize_rubrics(rubrics)
    # IFBench-style constraints
    constraints: List[str] = []
    for key in ("llm_constraints_used", "code_constraints_used"):
        raw = sample.meta.get(key)
        parsed = _safe_json_loads(raw)
        if isinstance(parsed, dict):
            parsed = parsed.get("constraints") or parsed.get("items") or parsed.get("rules")
        if isinstance(parsed, list):
            for item in parsed:
                if isinstance(item, dict):
                    constraint = item.get("constraint") or item.get("description") or item.get("rule")
                    if constraint:
                        constraints.append(str(constraint))
                    checkers = item.get("checkers")
                    if not constraint and isinstance(checkers, list):
                        for checker in checkers:
                            if checker:
                                constraints.append(str(checker))
                elif isinstance(item, str):
                    if item.strip():
                        constraints.append(item.strip())
        elif isinstance(parsed, str) and parsed.strip():
            constraints.append(parsed.strip())
    if constraints:
        return _normalize_rubrics(constraints)
    prompt_metadata = sample.meta.get("prompt_metadata")
    parsed = _safe_json_loads(prompt_metadata)
    if isinstance(parsed, dict):
        return _normalize_rubrics(parsed.get("rubrics") or parsed.get("rubric"))
    if parsed is not None:
        return _normalize_rubrics(parsed)
    return []


def _parse_rubrichub_weight(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    raw = str(value).strip()
    if not raw:
        return None
    try:
        return float(raw)
    except Exception:
        pass
    match = re.search(r"\d+(?:\.\d+)?", raw)
    if match:
        try:
            return float(match.group(0))
        except Exception:
            return None
    return None


def _normalize_rubrichub_key(key: str) -> str:
    key = key.strip().lower()
    key = re.sub(r"[\\s\\-]+", "_", key)
    key = re.sub(r"[^a-z0-9_]", "", key)
    return key


def _extract_rubrichub_items(meta: Dict[str, Any]) -> List[Any]:
    raw = None
    if not isinstance(meta, dict):
        return []
    normalized_keys: Dict[str, str] = {}
    for key in meta.keys():
        norm = _normalize_rubrichub_key(str(key))
        if norm not in normalized_keys:
            normalized_keys[norm] = str(key)
    for key in (
        "rubrics",
        "rubric",
        "rubric_items",
        "rubric_list",
        "criteria",
        "criterion",
        "rules",
        "constraints",
    ):
        norm = _normalize_rubrichub_key(key)
        orig = normalized_keys.get(norm)
        if orig is not None and meta.get(orig) is not None:
            raw = meta.get(orig)
            break
    if raw is None:
        prompt_metadata = meta.get("prompt_metadata")
        parsed_meta = _safe_json_loads(prompt_metadata)
        if isinstance(parsed_meta, dict):
            raw = (
                parsed_meta.get("rubrics")
                or parsed_meta.get("rubric")
                or parsed_meta.get("criteria")
                or parsed_meta.get("criterion")
                or parsed_meta.get("rules")
                or parsed_meta.get("constraints")
            )
        elif parsed_meta is not None:
            raw = parsed_meta
    parsed = _safe_json_loads(raw)
    if isinstance(parsed, dict):
        for key in ("rubrics", "rubric", "criteria", "criterion", "items", "rules", "constraints"):
            if isinstance(parsed.get(key), list):
                parsed = parsed.get(key)
                break
    if isinstance(parsed, list):
        return list(parsed)
    if parsed is None:
        return []
    return [parsed]


def _parse_rubrichub_item(item: Any) -> Tuple[str, Optional[float]]:
    text = ""
    weight_val: Optional[float] = None
    if isinstance(item, dict):
        text = str(
            item.get("criteria")
            or item.get("criterion")
            or item.get("rubric")
            or item.get("requirement")
            or item.get("description")
            or item.get("text")
            or ""
        ).strip()
        weight_val = _parse_rubrichub_weight(
            item.get("point")
            or item.get("points")
            or item.get("weight")
            or item.get("score")
            or item.get("value")
        )
    else:
        text = str(item).strip()
    if text:
        match = re.search(r"(?:point|points|weight)\s*[:=]?\s*(\d+(?:\.\d+)?)", text, re.IGNORECASE)
        if match and weight_val is None:
            weight_val = _parse_rubrichub_weight(match.group(1))
        if match:
            text = text[: match.start()].strip().rstrip(":-").strip()
    return text, weight_val


def _extract_rubrichub_rubrics(sample: Sample) -> Tuple[List[str], List[float]]:
    if not isinstance(sample.meta, dict):
        return [], []
    items = _extract_rubrichub_items(sample.meta)
    rubrics: List[str] = []
    weights_raw: List[Optional[float]] = []
    for item in items:
        text, weight_val = _parse_rubrichub_item(item)
        text = _clean_rubric_text(text, max_chars=400)
        if not text:
            continue
        rubrics.append(text)
        weights_raw.append(weight_val)

    if not rubrics:
        fallback = _extract_advancedif_rubrics(sample)
        return fallback, [1.0 for _ in fallback]

    if all(w is None for w in weights_raw):
        points_raw = (
            sample.meta.get("points")
            or sample.meta.get("weights")
            or sample.meta.get("rubric_points")
            or sample.meta.get("rubric_weights")
        )
        parsed_points = _safe_json_loads(points_raw)
        if isinstance(parsed_points, str):
            parsed_points = _try_parse_list_string(parsed_points)
        if isinstance(parsed_points, list):
            for idx, val in enumerate(parsed_points):
                if idx >= len(weights_raw):
                    break
                parsed_weight = _parse_rubrichub_weight(val)
                if parsed_weight is not None:
                    weights_raw[idx] = parsed_weight

    weights: List[float] = []
    for idx, text in enumerate(rubrics):
        weight_val = weights_raw[idx] if idx < len(weights_raw) else None
        if weight_val is None:
            weight_val = 1.0
        weights.append(float(weight_val))

    if len(weights) < len(rubrics):
        weights.extend([1.0] * (len(rubrics) - len(weights)))
    if len(weights) > len(rubrics):
        weights = weights[: len(rubrics)]
    return rubrics, weights

def _clean_rubric_text(text: str, max_chars: int = 240) -> str:
    if not text:
        return ""
    cleaned = re.sub(r"\s+", " ", str(text)).strip()
    if len(cleaned) > max_chars:
        cleaned = cleaned[:max_chars].rstrip()
    return cleaned


def _infer_rubric_dimension(rubric: str) -> str:
    text = rubric.lower()
    if any(k in text for k in ("format", "markdown", "json", "bullet", "list", "table", "heading")):
        return "format"
    if any(k in text for k in ("concise", "brief", "short", "length", "word", "sentence", "limit")):
        return "conciseness"
    if any(k in text for k in ("tone", "polite", "friendly", "formal", "professional")):
        return "tone"
    if any(k in text for k in ("safe", "unsafe", "harm", "illegal", "policy", "safety")):
        return "safety"
    if any(k in text for k in ("correct", "accurate", "truth", "factual", "fact")):
        return "correctness"
    if any(k in text for k in ("complete", "coverage", "thorough", "include all", "every")):
        return "completeness"
    if any(k in text for k in ("clear", "clarity", "explicit")):
        return "clarity"
    return "instruction following"


ADVANCEDIF_SCORE_VALUES = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
BIGGEN_SCORE_VALUES = [0.0, 0.25, 0.5, 0.75, 1.0]
HELPSTEER2_DIMENSIONS = [
    "helpfulness",
    "correctness",
    "coherence",
    "complexity",
    "verbosity",
]
HELPSTEER2_SCORE_VALUES = [1, 2, 3, 4, 5]
HELPSTEER2_DIMENSION_WEIGHTS = {
    "helpfulness": 3,
    "coherence": 3,
    "correctness": 2,
    "complexity": 1,
    "verbosity": 1,
}
HELPSTEER2_WEIGHT_TOTAL = sum(HELPSTEER2_DIMENSION_WEIGHTS.values())
HELPSTEER2_MEMORY_THRESHOLD = 5


def _score_values_for_sample(sample: Sample, fallback: Optional[List[float]] = None) -> List[float]:
    if isinstance(sample.meta, dict):
        scale = str(sample.meta.get("score_scale") or "").strip().lower()
        if scale in {"quarters", "biggen", "biggen-bench", "biggen_bench"}:
            return BIGGEN_SCORE_VALUES
        source = str(sample.meta.get("source") or "").lower()
        if "biggen" in source:
            return BIGGEN_SCORE_VALUES
    return fallback or ADVANCEDIF_SCORE_VALUES


def _format_score_values(values: List[float]) -> str:
    return ", ".join(str(int(v)) if abs(v - int(v)) < 1e-9 else str(v) for v in values)


def _parse_rubric_score(text: str, allowed: Optional[List[float]] = None) -> Optional[float]:
    if not text:
        return None
    cleaned = text.strip()
    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, (int, float)):
            val = float(parsed)
            allowed_vals = allowed or ADVANCEDIF_SCORE_VALUES
            for candidate in allowed_vals:
                if abs(val - candidate) < 1e-6:
                    return float(candidate)
            return None
        if isinstance(parsed, str):
            cleaned = parsed.strip()
        if isinstance(parsed, dict):
            for key in ("score", "rating", "grade"):
                if key in parsed:
                    try:
                        val = float(parsed[key])
                        allowed_vals = allowed or ADVANCEDIF_SCORE_VALUES
                        for candidate in allowed_vals:
                            if abs(val - candidate) < 1e-6:
                                return float(candidate)
                        return None
                    except Exception:
                        pass
    except Exception:
        pass
    match = re.search(r"(0(?:\.0)?|0\.2|0\.25|0\.4|0\.5|0\.6|0\.75|0\.8|1(?:\.0)?)", cleaned)
    if not match:
        return None
    try:
        val = float(match.group(1))
    except Exception:
        return None
    allowed_vals = allowed or ADVANCEDIF_SCORE_VALUES
    for candidate in allowed_vals:
        if abs(val - candidate) < 1e-6:
            return float(candidate)
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

    # If it looks like a JSON array, parse it as a whole first.
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
        # fallback: map style {"rubric": score, ...}
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


def _parse_advancedif_score_array(text: str, allowed_scores: Optional[List[float]] = None) -> List[float]:
    if not text:
        return []
    cleaned = _strip_json_fence(text)
    blob = _extract_first_json_blob(cleaned) or cleaned.strip()
    parsed = None
    try:
        parsed = json.loads(blob)
    except Exception:
        parsed = None
    scores_raw: List[Any] = []
    if isinstance(parsed, dict):
        maybe_scores = parsed.get("scores")
        if isinstance(maybe_scores, list):
            scores_raw = maybe_scores
    elif isinstance(parsed, list):
        scores_raw = parsed
    if not scores_raw:
        matches = re.findall(r"-?\d+(?:\.\d+)?", cleaned)
        scores_raw = matches
    scores: List[float] = []
    for item in scores_raw:
        score = _parse_rubric_score(str(item), allowed_scores)
        if score is None:
            continue
        scores.append(score)
    return scores


def _parse_helpsteer2_score(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        val = float(value)
    else:
        raw = str(value).strip()
        if not raw:
            return None
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, (int, float)):
                val = float(parsed)
            else:
                val = None
        except Exception:
            val = None
        if val is None:
            match = re.search(r"\b([1-5])\b", raw)
            if not match:
                return None
            val = float(match.group(1))
    for candidate in HELPSTEER2_SCORE_VALUES:
        if abs(val - float(candidate)) < 1e-6:
            return float(candidate)
    return None


def _parse_helpsteer2_score_array(text: str) -> List[float]:
    if not text:
        return []
    cleaned = _strip_json_fence(text)
    blob = _extract_first_json_blob(cleaned) or cleaned.strip()
    parsed = None
    try:
        parsed = json.loads(blob)
    except Exception:
        parsed = None
    scores_raw: List[Any] = []
    if isinstance(parsed, dict):
        maybe_scores = parsed.get("scores")
        if isinstance(maybe_scores, list):
            scores_raw = maybe_scores
    elif isinstance(parsed, list):
        scores_raw = parsed
    if not scores_raw:
        matches = re.findall(r"\b[1-5]\b", cleaned)
        scores_raw = matches
    scores: List[float] = []
    for item in scores_raw:
        score = _parse_helpsteer2_score(item)
        if score is None:
            continue
        scores.append(score)
    return scores


def _align_advancedif_items(
    rubrics: List[str],
    items: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    if not rubrics:
        return []
    if not items:
        return [{"rubric": rubric, "score": None, "trigger": "", "dimension": "", "comparison": ""} for rubric in rubrics]

    def _norm(text: str) -> str:
        return re.sub(r"\s+", " ", text.strip().lower())

    rubric_to_item: Dict[str, Dict[str, Any]] = {}
    for item in items:
        key = _norm(str(item.get("rubric") or ""))
        if key and key not in rubric_to_item:
            rubric_to_item[key] = item

    aligned: List[Dict[str, Any]] = []
    if len(items) == len(rubrics):
        # Use index alignment as a fallback when counts match.
        for idx, rubric in enumerate(rubrics):
            item = rubric_to_item.get(_norm(rubric)) or items[idx]
            aligned.append(item)
        return aligned

    for rubric in rubrics:
        item = rubric_to_item.get(_norm(rubric))
        if item is None:
            aligned.append(
                {
                    "rubric": rubric,
                    "score": None,
                    "trigger": "",
                    "dimension": "",
                    "comparison": "",
                }
            )
        else:
            aligned.append(item)
    return aligned


def _split_advancedif_judge_items(
    items: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    scored: List[Dict[str, Any]] = []
    extras: List[Dict[str, Any]] = []
    for item in items or []:
        if item is None:
            continue
        if "score" in item and item.get("score") is not None:
            scored.append(item)
        else:
            if any(item.get(k) for k in ("trigger", "dimension", "comparison")):
                extras.append(item)
    return scored, extras


def _judge_advancedif_rubrics(
    judge_model: Model,
    prompt: str,
    response: str,
    rubrics: List[str],
    max_tokens: int = 768,
    debug_out: Optional[List[str]] = None,
    allowed_scores: Optional[List[float]] = None,
    max_memories_per_rubric: int = 1,
) -> List[Dict[str, Any]]:
    if not rubrics:
        return []
    score_values = allowed_scores or ADVANCEDIF_SCORE_VALUES
    score_values_text = _format_score_values(score_values)
    is_single_rubric = len(rubrics) == 1
    is_biggen = score_values == BIGGEN_SCORE_VALUES and is_single_rubric
    mapping_hint = ""
    if score_values == BIGGEN_SCORE_VALUES:
        mapping_hint = (
            "- If the rubric provides Score 1-5 descriptions, map Score 1..5 to "
            "0, 0.25, 0.5, 0.75, 1 respectively.\n"
        )
    rubric_lines = []
    for idx, item in enumerate(rubrics, start=1):
        rubric_lines.append(f"{idx}. {item}")
    rubric_block = "\n".join(rubric_lines)
    extra_rules = max(0, int(max_memories_per_rubric) - 1)
    extra_hint = ""
    if extra_rules > 0:
        extra_hint = (
            f"- You may output up to {extra_rules} additional memory-only lines per rubric, "
            "after the scored lines. Each extra line must omit \"score\" and include "
            "{\"rubric\",\"trigger\",\"dimension\",\"comparison\"}.\n"
        )

    if is_biggen:
        system_prompt = (
            "You are a strict rubric grader and memory extractor for a single rubric.\n"
            "Return ONLY JSONL (one JSON object per line). Do not output any extra text."
        )
    else:
        system_prompt = (
            "You are a strict rubric grader and memory extractor.\n"
            "For each rubric, output a score and (when needed) a preference-rule memory.\n"
            "Score each rubric independently. Even if some rubrics fail, do not automatically fail others.\n"
            "Return ONLY JSONL (one JSON object per line). Do not output any extra text."
        )
    user_prompt = f"""User prompt:
{_truncate_text(prompt, 2000)}

Model response:
{_truncate_text(response, 2000)}

Rubrics:
{_truncate_text(rubric_block, 2000)}

OUTPUT JSONL FORMAT (one JSON object per line):
{{"rubric": "<rubric text exactly as above>", "score": {score_values_text}, "trigger": "...", "dimension": "...", "comparison": "..."}}

Rules:
{mapping_hint}- You MUST output exactly {len(rubrics)} scored line(s), in the same order as listed.
- Each scored line must include "score" and that score must be ONLY {score_values_text}.
- If score == 1, OMIT trigger/dimension/comparison entirely (do NOT output empty strings).
{extra_hint}- If score < 1, you MUST fill trigger/dimension/comparison with generalized, reusable wording.
- If score < 1, you MUST fill trigger/dimension/comparison with generalized, reusable wording.
- trigger = the specific situation/constraint in the prompt or rubric where the rule applies.
- dimension = the key quality axis that caused the low score (why it fails).
- comparison = how to fix it, phrased as a generalized directive that is explicitly conditional on the trigger being present.
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
    except ModelError:
        return []
    if debug_out is not None:
        debug_out.append(result.text or "")
    return _parse_advancedif_judge_items(result.text)


def _judge_advancedif_memories(
    judge_model: Model,
    prompt: str,
    response: str,
    rubrics_with_ids: List[Tuple[int, str]],
    max_tokens: int = 768,
    debug_out: Optional[List[str]] = None,
    max_memories_per_rubric: int = 1,
) -> List[Dict[str, Any]]:
    if not rubrics_with_ids:
        return []
    rubric_lines = [f"{rid}. {text}" for rid, text in rubrics_with_ids]
    rubric_block = "\n".join(rubric_lines)
    extra_hint = ""
    if max_memories_per_rubric > 1:
        extra_hint = (
            f"- You may output up to {max_memories_per_rubric} lines per rubric. "
            "Each line must include rubric_id.\n"
        )
    system_prompt = (
        "You are a strict memory extractor.\n"
        "Return ONLY JSONL (one JSON object per line). Do not output any extra text."
    )
    user_prompt = f"""User prompt:
{_truncate_text(prompt, 2000)}

Model response:
{_truncate_text(response, 2000)}

Failed rubrics (use the rubric_id shown below):
{_truncate_text(rubric_block, 2000)}

OUTPUT JSONL FORMAT (one JSON object per line):
{{"rubric_id": <number>, "trigger": "...", "dimension": "...", "comparison": "..."}}

Rules:
- You MUST output at least 1 line for each failed rubric.
{extra_hint}- Each line must include rubric_id, trigger, dimension, comparison.
- trigger = the specific situation/constraint in the prompt or rubric where the rule applies.
- dimension = the key quality axis that caused the failure.
- comparison = how to fix it, phrased as a generalized directive that is explicitly conditional on the trigger being present.
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
    except ModelError:
        return []
    if debug_out is not None:
        debug_out.append(result.text or "")
    return _parse_advancedif_judge_items(result.text)


def _judge_advancedif_scores(
    judge_model: Model,
    prompt: str,
    response: str,
    rubrics: List[str],
    max_tokens: int = 256,
    debug_out: Optional[List[str]] = None,
    allowed_scores: Optional[List[float]] = None,
) -> List[float]:
    if not rubrics:
        return []
    score_values = allowed_scores or ADVANCEDIF_SCORE_VALUES
    score_values_text = _format_score_values(score_values)
    is_single_rubric = len(rubrics) == 1
    is_biggen = score_values == BIGGEN_SCORE_VALUES and is_single_rubric
    mapping_hint = ""
    if score_values == BIGGEN_SCORE_VALUES:
        mapping_hint = (
            "- If the rubric provides Score 1-5 descriptions, map Score 1..5 to "
            "0, 0.25, 0.5, 0.75, 1 respectively.\n"
        )
    rubric_lines = []
    for idx, item in enumerate(rubrics, start=1):
        rubric_lines.append(f"{idx}. {item}")
    rubric_block = "\n".join(rubric_lines)

    if is_biggen:
        system_prompt = (
            "You are a strict rubric grader for a single rubric.\n"
            "Return ONLY valid JSON with no extra text."
        )
    else:
        system_prompt = (
            "You are a strict rubric grader.\n"
            "Score each rubric independently. Even if some rubrics fail, do not automatically fail others.\n"
            "Return ONLY valid JSON with no extra text."
        )
    user_prompt = f"""User prompt:
{_truncate_text(prompt, 2000)}

Model response:
{_truncate_text(response, 2000)}

Rubrics:
{_truncate_text(rubric_block, 2000)}

OUTPUT FORMAT:
{{"scores":[score1 for rubric 1,score2 for rubric 2,...]}}

Rules:
{mapping_hint}- Output exactly {len(rubrics)} score(s), in order.
- Each score must be ONLY {score_values_text}.
- Do not output any extra keys, text, or markdown.
"""
    try:
        result = judge_model.generate(
            prompt=user_prompt,
            system_prompt=system_prompt,
            temperature=0.0,
            do_sample=False,
            max_tokens=max_tokens,
        )
    except ModelError:
        return []
    if debug_out is not None:
        debug_out.append(result.text or "")
    return _parse_advancedif_score_array(result.text, allowed_scores=score_values)


def _judge_helpsteer2_dimensions(
    judge_model: Model,
    prompt: str,
    response: str,
    max_tokens: int = 256,
    debug_out: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    dimensions = HELPSTEER2_DIMENSIONS
    dim_lines = [f"{idx}. {dim}" for idx, dim in enumerate(dimensions, start=1)]
    dim_block = "\n".join(dim_lines)
    score_values_text = ", ".join(str(v) for v in HELPSTEER2_SCORE_VALUES)
    system_prompt = (
        "You are a strict quality rater and memory extractor for HelpSteer2.\n"
        "Score each dimension independently. Return ONLY JSONL (one JSON object per line). "
        "Do not output any extra text."
    )
    user_prompt = f"""User prompt:
{_truncate_text(prompt, 2000)}

Model response:
{_truncate_text(response, 2000)}

Dimensions (score 1-5):
{dim_block}

OUTPUT JSONL FORMAT (one JSON object per line):
{{"rubric": "<dimension name>", "score": {score_values_text}, "trigger": "...", "comparison": "..."}}

Rules:
- You MUST output exactly {len(dimensions)} scored line(s), in the same order as listed.
- Each scored line must include "rubric" and "score".
- Each score must be ONLY {score_values_text}.
- Do NOT use overall impressions; score each dimension by explicitly checking it against the response.
- Scoring guide per dimension: 5 = fully satisfies with no noticeable issues; 4 = minor issues; 3 = moderate issues; 2 = major issues; 1 = unacceptable or missing.
- If uncertain between two scores, choose the lower score.
- Do NOT give a 5 if there is any noticeable issue; cap at 4.
- If score >= {HELPSTEER2_MEMORY_THRESHOLD}, OMIT trigger/comparison entirely (do NOT output empty strings).
- If score < {HELPSTEER2_MEMORY_THRESHOLD}, you MUST fill trigger/comparison with generalized, reusable wording.
- trigger = the specific issue in the prompt or response that caused the low score.
- comparison = how to fix it, phrased as a generalized directive that is actionable, testable, and explicitly conditional on the trigger.
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
    except ModelError:
        return []
    if debug_out is not None:
        debug_out.append(result.text or "")
    return _parse_advancedif_judge_items(result.text)


def _judge_helpsteer2_scores(
    judge_model: Model,
    prompt: str,
    response: str,
    max_tokens: int = 128,
    debug_out: Optional[List[str]] = None,
) -> List[float]:
    dimensions = HELPSTEER2_DIMENSIONS
    dim_lines = [f"{idx}. {dim}" for idx, dim in enumerate(dimensions, start=1)]
    dim_block = "\n".join(dim_lines)
    score_values_text = ", ".join(str(v) for v in HELPSTEER2_SCORE_VALUES)
    system_prompt = (
        "You are a strict quality rater for HelpSteer2.\n"
        "Score each dimension independently. Return ONLY valid JSON with no extra text."
    )
    user_prompt = f"""User prompt:
{_truncate_text(prompt, 2000)}

Model response:
{_truncate_text(response, 2000)}

Dimensions (score 1-5):
{dim_block}

OUTPUT FORMAT:
{{"scores":[score1 for helpfulness, score2 for correctness, score3 for coherence, score4 for complexity, score5 for verbosity]}}

Rules:
- Output exactly {len(dimensions)} scores, in the same order as listed.
- Each score must be ONLY {score_values_text}.
- Do NOT use overall impressions; score each dimension by explicitly checking it against the response.
- Scoring guide per dimension: 5 = fully satisfies with no noticeable issues; 4 = minor issues; 3 = moderate issues; 2 = major issues; 1 = unacceptable or missing.
- If uncertain between two scores, choose the lower score.
- Do NOT give a 5 if there is any noticeable issue; cap at 4.
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
    except ModelError:
        return []
    if debug_out is not None:
        debug_out.append(result.text or "")
    return _parse_helpsteer2_score_array(result.text)


def _score_advancedif_rubric(
    judge_model: Model,
    prompt: str,
    response: str,
    rubric: str,
    max_tokens: int = 64,
    allowed_scores: Optional[List[float]] = None,
) -> Optional[float]:
    score_values = allowed_scores or ADVANCEDIF_SCORE_VALUES
    score_values_text = _format_score_values(score_values)
    mapping_hint = ""
    if score_values == BIGGEN_SCORE_VALUES:
        mapping_hint = (
            "If the rubric provides Score 1-5 descriptions, map Score 1..5 to "
            "0, 0.25, 0.5, 0.75, 1 respectively.\n"
        )
    system_prompt = (
        "You are a strict rubric grader. Score the response on the rubric below.\n"
        f"Return ONLY valid JSON: {{\"score\": {score_values_text}}}."
    )
    prompt_text = _truncate_text(prompt, 2000)
    response_text = _truncate_text(response, 2000)
    rubric_text = _truncate_text(rubric, 400)
    user_prompt = (
        f"User prompt:\n{prompt_text}\n\n"
        f"Model response:\n{response_text}\n\n"
        f"Rubric:\n{rubric_text}\n\n"
        f"{mapping_hint}"
        "OUTPUT FORMAT:\n"
        f"{{\"score\": {score_values_text}}}"
    )
    try:
        result = judge_model.generate(
            prompt=user_prompt,
            system_prompt=system_prompt,
            temperature=0.0,
            do_sample=False,
            max_tokens=max_tokens,
        )
    except ModelError:
        return None
    return _parse_rubric_score(result.text, allowed_scores=score_values)


def _build_advancedif_memory_rule(
    sample_id: str,
    rubric: str,
    score: float,
) -> Optional[MemoryItem]:
    rubric_clean = _clean_rubric_text(rubric)
    if not rubric_clean:
        return None
    trigger = f"When the prompt includes the requirement: {rubric_clean}"
    dimension = _infer_rubric_dimension(rubric_clean)
    comparison = (
        f"Prefer responses that fully satisfy the requirement '{rubric_clean}' over responses that only "
        "partially address it, hedge, or omit any required detail; make missing constraints explicit and "
        "remove content that conflicts with the requirement."
    )
    source = {
        "sample_id": sample_id,
        "rubric": rubric_clean,
        "score": score,
        "timestamp": datetime.utcnow().isoformat(),
        "dataset": "advancedif",
    }
    return MemoryItem(
        type="preference_rule",
        title=trigger,
        description="",
        content="",
        source=source,
        trigger=trigger,
        dimension=dimension,
        comparison=comparison,
    )


def _truncate_text(text: str, max_chars: int) -> str:
    if not text:
        return ""
    return text if len(text) <= max_chars else text[:max_chars]


ADVANCEDIF_MIN_RESPONSE_CHARS = 20


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


def _select_fixed_subset(
    samples: List[Sample],
    target_size: int,
    seed: int,
) -> List[Sample]:
    if target_size <= 0 or target_size >= len(samples):
        return list(samples)
    indices = list(range(len(samples)))
    rng = random.Random(seed)
    rng.shuffle(indices)
    chosen = sorted(indices[:target_size])
    return [samples[i] for i in chosen]


def _select_fixed_subset_excluding(
    samples: List[Sample],
    target_size: int,
    seed: int,
    exclude_ids: Optional[Iterable[str]] = None,
) -> List[Sample]:
    if target_size <= 0:
        return []
    exclude = {str(item) for item in exclude_ids} if exclude_ids else set()
    remaining = [sample for sample in samples if sample.id not in exclude]
    if target_size >= len(remaining):
        return remaining
    indices = list(range(len(remaining)))
    rng = random.Random(seed)
    rng.shuffle(indices)
    chosen = sorted(indices[:target_size])
    return [remaining[i] for i in chosen]


def _load_locked_split_ids(path: Optional[str]) -> Tuple[List[str], List[str]]:
    if not path:
        return [], []
    if not os.path.exists(path):
        print(f"[HotpotQA] Lock split not found: {path}")
        return [], []
    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
    except Exception as exc:
        print(f"[HotpotQA] Failed to read lock split: {exc}")
        return [], []
    train_ids = data.get("train_ids") or []
    test_ids = data.get("test_ids") or []
    if not isinstance(train_ids, list):
        train_ids = []
    if not isinstance(test_ids, list):
        test_ids = []
    return [str(item) for item in train_ids], [str(item) for item in test_ids]


def _select_train_subset_by_similarity_fill(
    train_samples: List[Sample],
    test_samples: List[Sample],
    top_k: int,
    target_size: int,
    model_name: str,
    device: Optional[str],
    batch_size: int,
) -> Tuple[List[Sample], Dict[str, Any]]:
    info: Dict[str, Any] = {
        "mode": "similarity_top_k_fill",
        "top_k": int(top_k),
        "target_size": int(target_size),
        "train_count": len(train_samples),
        "test_count": len(test_samples),
        "embedding_model": model_name,
    }
    if top_k <= 0 or target_size <= 0:
        info["enabled"] = False
        info["reason"] = "top_k<=0_or_target_size<=0"
        return train_samples, info
    if not train_samples or not test_samples:
        info["enabled"] = False
        info["reason"] = "empty_train_or_test"
        return train_samples, info

    print(
        f"[Train Subset] Selecting top-{top_k} similar train questions per test "
        f"and filling to {target_size} using {model_name}..."
    )
    try:
        tokenizer, model, device_name = _load_embedding_model(model_name, device)
    except Exception as exc:
        raise RuntimeError(f"Failed to load embedding model for train subset: {exc}") from exc

    info["embedding_device"] = device_name
    train_texts = [sample.question for sample in train_samples]
    test_texts = [sample.question for sample in test_samples]

    try:
        train_emb = _embed_texts(
            model=model,
            tokenizer=tokenizer,
            texts=train_texts,
            device=device_name,
            batch_size=batch_size,
        )
        test_emb = _embed_texts(
            model=model,
            tokenizer=tokenizer,
            texts=test_texts,
            device=device_name,
            batch_size=batch_size,
        )
    except Exception as exc:
        raise RuntimeError(f"Failed to embed questions for train subset: {exc}") from exc

    try:
        import numpy as np
    except Exception:
        np = None

    selected_indices: set[int] = set()
    train_count = len(train_samples)
    effective_k = min(int(top_k), train_count)
    info["effective_k"] = effective_k

    if np is not None:
        train_arr = np.asarray(train_emb, dtype="float32")
        test_arr = np.asarray(test_emb, dtype="float32")
        train_norm = np.linalg.norm(train_arr, axis=1, keepdims=True)
        test_norm = np.linalg.norm(test_arr, axis=1, keepdims=True)
        train_arr = train_arr / np.clip(train_norm, 1e-12, None)
        test_arr = test_arr / np.clip(test_norm, 1e-12, None)

        sims = test_arr @ train_arr.T
        for row in sims:
            if effective_k >= train_count:
                idxs = range(train_count)
            else:
                idxs = np.argpartition(-row, effective_k - 1)[:effective_k]
            for idx in idxs:
                selected_indices.add(int(idx))
        max_sim_list = sims.max(axis=0).tolist()
    else:
        import torch
        train_tensor = train_emb
        test_tensor = test_emb
        if not isinstance(train_tensor, torch.Tensor):
            train_tensor = torch.tensor(train_tensor, dtype=torch.float32)
        if not isinstance(test_tensor, torch.Tensor):
            test_tensor = torch.tensor(test_tensor, dtype=torch.float32)
        train_tensor = torch.nn.functional.normalize(train_tensor, p=2, dim=1)
        test_tensor = torch.nn.functional.normalize(test_tensor, p=2, dim=1)
        sims = test_tensor @ train_tensor.T
        if effective_k >= train_count:
            topk_indices = torch.arange(train_count).repeat(sims.size(0), 1)
        else:
            topk_indices = torch.topk(sims, k=effective_k, dim=1).indices
        for row in topk_indices:
            for idx in row.tolist():
                selected_indices.add(int(idx))
        max_sim_list = sims.max(dim=0).values.tolist()

    info["initial_selected_count"] = len(selected_indices)

    def _ranked_indices(indices: Iterable[int]) -> List[int]:
        return sorted(indices, key=lambda i: (-max_sim_list[i], i))

    if len(selected_indices) < target_size:
        ranked_all = _ranked_indices(range(train_count))
        before = len(selected_indices)
        for idx in ranked_all:
            if idx in selected_indices:
                continue
            selected_indices.add(idx)
            if len(selected_indices) >= target_size:
                break
        info["fill_count"] = len(selected_indices) - before
        if len(selected_indices) < target_size:
            info["fill_exhausted"] = True
    elif len(selected_indices) > target_size:
        ranked_sel = _ranked_indices(selected_indices)
        selected_indices = set(ranked_sel[:target_size])
        info["trimmed_count"] = info["initial_selected_count"] - len(selected_indices)

    selected = [train_samples[i] for i in sorted(selected_indices)]
    info["enabled"] = True
    info["selected_count"] = len(selected)
    return selected, info


def _load_split_datasets(
    config: ExperimentConfig,
) -> Tuple[List[Sample], List[Sample], Dict[str, Any]]:
    external_split = _load_external_split_ids(config.data_split_path)
    if external_split is not None:
        split_name = (config.data_split_name or "").strip().lower()
        # When train/test datasets are provided separately, pick the source that
        # matches the requested split to avoid loading ids from the wrong corpus.
        if split_name == "test" and config.test_dataset_path:
            base_path = config.test_dataset_path
        elif split_name == "train" and config.train_dataset_path:
            base_path = config.train_dataset_path
        else:
            base_path = config.dataset_path
            if not base_path:
                base_path = config.train_dataset_path or config.test_dataset_path or ""
        # When using an external split, load the full dataset so ids resolve correctly.
        samples = list(load_dataset(base_path, max_samples=None))
        sample_map = {sample.id: sample for sample in samples}
        train_ids = external_split.get("train_ids") or []
        test_ids = external_split.get("test_ids") or []
        train_samples = [sample_map[sid] for sid in train_ids if sid in sample_map]
        test_samples = [sample_map[sid] for sid in test_ids if sid in sample_map]
        if config.max_samples:
            train_samples = train_samples[: config.max_samples]
            test_samples = test_samples[: config.max_samples]
        missing_train = [sid for sid in train_ids if sid not in sample_map]
        missing_test = [sid for sid in test_ids if sid not in sample_map]
        split_info = {
            "mode": "external_split",
            "dataset_path": base_path,
            "data_split_path": config.data_split_path,
            "data_split_name": config.data_split_name,
            "train_ids": train_ids,
            "test_ids": test_ids,
            "missing_train_ids": missing_train,
            "missing_test_ids": missing_test,
        }
        train_samples, test_samples, split_info = _maybe_filter_helpsteer3_domains(
            train_samples,
            test_samples,
            split_info,
            config,
            base_path,
            base_path,
        )
        return train_samples, test_samples, split_info
    explicit_train = config.train_dataset_path
    explicit_test = config.test_dataset_path
    train_path = explicit_train or "hotpotqa:distractor:train"
    test_path = explicit_test or "hotpotqa:distractor:validation"
    if config.prompt_profile == "hotpotqa" and (
        _is_hotpotqa_dataset(train_path) and _is_hotpotqa_dataset(test_path)
    ):
        lock_train_ids, lock_test_ids = _load_locked_split_ids(
            config.hotpotqa_lock_split_path
        )

        train_samples = list(load_dataset(train_path, max_samples=config.max_samples))
        train_pool = train_samples
        pool_size = max(0, int(config.hotpotqa_train_pool_size))
        if pool_size > 0 and len(train_samples) > pool_size:
            # Deterministic pool selection to keep runs stable across machines.
            train_pool = _select_fixed_subset(
                train_samples, target_size=pool_size, seed=config.split_seed + 1
            )
        test_full = list(load_dataset(test_path, max_samples=config.max_samples))

        # Resolve locked test samples
        test_map = {sample.id: sample for sample in test_full}
        locked_test_samples = [
            test_map[sid] for sid in lock_test_ids if sid in test_map
        ]
        missing_test_ids = [sid for sid in lock_test_ids if sid not in test_map]
        test_target = int(config.hotpotqa_test_size)
        if test_target < len(locked_test_samples):
            print(
                f"[HotpotQA] test_size={test_target} < locked_test={len(locked_test_samples)}; "
                f"keeping locked tests."
            )
            test_target = len(locked_test_samples)
        extra_test = _select_fixed_subset_excluding(
            test_full,
            target_size=max(0, test_target - len(locked_test_samples)),
            seed=config.split_seed,
            exclude_ids=[s.id for s in locked_test_samples],
        )
        test_samples = locked_test_samples + extra_test

        # Resolve locked train samples
        train_map = {sample.id: sample for sample in train_pool}
        locked_train_samples = [
            train_map[sid] for sid in lock_train_ids if sid in train_map
        ]
        missing_train_ids = [sid for sid in lock_train_ids if sid not in train_map]
        train_target = int(config.hotpotqa_train_size)
        if train_target < len(locked_train_samples):
            print(
                f"[HotpotQA] train_size={train_target} < locked_train={len(locked_train_samples)}; "
                f"keeping locked trains."
            )
            train_target = len(locked_train_samples)
        remaining_pool = [
            sample for sample in train_pool if sample.id not in {s.id for s in locked_train_samples}
        ]
        extra_train, subset_info = _select_train_subset_by_similarity_fill(
            train_samples=remaining_pool,
            test_samples=test_samples,
            top_k=config.hotpotqa_train_top_k,
            target_size=max(0, train_target - len(locked_train_samples)),
            model_name=config.train_subset_embedding_model,
            device=config.train_subset_embedding_device,
            batch_size=config.train_subset_embedding_batch_size,
        )
        selected_train = locked_train_samples + extra_train
        split_info = {
            "mode": "hotpotqa_fixed",
            "train_dataset_path": train_path,
            "test_dataset_path": test_path,
            "split_seed": config.split_seed,
            "hotpotqa_train_pool_size": pool_size if pool_size > 0 else len(train_samples),
            "hotpotqa_test_size": config.hotpotqa_test_size,
            "hotpotqa_train_size": config.hotpotqa_train_size,
            "hotpotqa_train_top_k": config.hotpotqa_train_top_k,
            "hotpotqa_lock_split_path": config.hotpotqa_lock_split_path,
            "locked_train_ids": [s.id for s in locked_train_samples],
            "locked_test_ids": [s.id for s in locked_test_samples],
            "missing_locked_train_ids": missing_train_ids,
            "missing_locked_test_ids": missing_test_ids,
            "train_pool_ids": [sample.id for sample in train_pool],
            "train_ids": [sample.id for sample in selected_train],
            "test_ids": [sample.id for sample in test_samples],
            "train_subset": subset_info,
            "train_subset_locked": True,
        }
        selected_train, test_samples, split_info = _maybe_filter_helpsteer3_domains(
            selected_train,
            test_samples,
            split_info,
            config,
            train_path,
            test_path,
        )
        return selected_train, test_samples, split_info

    resolved_train = config.train_dataset_path or config.dataset_path
    resolved_test = config.test_dataset_path or config.dataset_path
    if _is_math500_dataset(resolved_train) and _is_math500_dataset(resolved_test):
        samples = list(load_dataset(resolved_train, max_samples=config.max_samples))
        total = len(samples)
        train_target = max(0, int(config.math500_train_size))
        test_target = max(0, int(config.math500_test_size))
        if train_target + test_target > total:
            # Ensure we never exceed total samples.
            train_target = min(train_target, total)
            test_target = min(test_target, max(0, total - train_target))
        rng = random.Random(config.split_seed)
        indices = list(range(total))
        rng.shuffle(indices)
        train_indices = sorted(indices[:train_target])
        test_indices = sorted(indices[train_target:train_target + test_target])
        train_samples = [samples[i] for i in train_indices]
        test_samples = [samples[i] for i in test_indices]
        split_info = {
            "mode": "math500_fixed",
            "dataset_path": resolved_train,
            "split_seed": config.split_seed,
            "math500_train_size": config.math500_train_size,
            "math500_test_size": config.math500_test_size,
            "train_ids": [sample.id for sample in train_samples],
            "test_ids": [sample.id for sample in test_samples],
        }
        train_samples, test_samples, split_info = _maybe_filter_helpsteer3_domains(
            train_samples,
            test_samples,
            split_info,
            config,
            resolved_train,
            resolved_test,
        )
        return train_samples, test_samples, split_info

    if config.train_dataset_path or config.test_dataset_path:
        train_path = config.train_dataset_path or config.dataset_path
        test_path = config.test_dataset_path or config.dataset_path
        train_samples = list(load_dataset(train_path, max_samples=config.max_samples))
        test_samples = list(load_dataset(test_path, max_samples=config.max_samples))
        train_ids = [sample.id for sample in train_samples]
        test_ids = [sample.id for sample in test_samples]
        split_info = {
            "mode": "explicit",
            "train_dataset_path": train_path,
            "test_dataset_path": test_path,
            "train_ids": train_ids,
            "test_ids": test_ids,
        }
        train_samples, test_samples, split_info = _maybe_filter_helpsteer3_domains(
            train_samples,
            test_samples,
            split_info,
            config,
            train_path,
            test_path,
        )
        return train_samples, test_samples, split_info

    samples = list(load_dataset(config.dataset_path, max_samples=config.max_samples))
    if config.split_ratio <= 0.0 or config.split_ratio >= 1.0:
        sample_ids = [sample.id for sample in samples]
        split_info = {
            "mode": "full",
            "dataset_path": config.dataset_path,
            "train_ids": sample_ids,
            "test_ids": sample_ids,
            "split_ratio": config.split_ratio,
            "split_seed": config.split_seed,
        }
        train_samples, test_samples, split_info = _maybe_filter_helpsteer3_domains(
            samples,
            samples,
            split_info,
            config,
            config.dataset_path,
            config.dataset_path,
        )
        return train_samples, test_samples, split_info

    rng = random.Random(config.split_seed)
    indices = list(range(len(samples)))
    rng.shuffle(indices)
    train_count = int(len(samples) * config.split_ratio)
    if len(samples) >= 2:
        train_count = max(1, min(len(samples) - 1, train_count))
    train_samples = [samples[i] for i in indices[:train_count]]
    test_samples = [samples[i] for i in indices[train_count:]]
    train_ids = [sample.id for sample in train_samples]
    test_ids = [sample.id for sample in test_samples]
    split_info = {
        "mode": "random_split",
        "dataset_path": config.dataset_path,
        "train_ids": train_ids,
        "test_ids": test_ids,
        "split_ratio": config.split_ratio,
        "split_seed": config.split_seed,
    }
    train_samples, test_samples, split_info = _maybe_filter_helpsteer3_domains(
        train_samples,
        test_samples,
        split_info,
        config,
        config.dataset_path,
        config.dataset_path,
    )
    return train_samples, test_samples, split_info


def _select_train_subset_by_similarity(
    train_samples: List[Sample],
    test_samples: List[Sample],
    top_k: int,
    model_name: str,
    device: Optional[str],
    batch_size: int,
) -> Tuple[List[Sample], Dict[str, Any]]:
    info: Dict[str, Any] = {
        "mode": "similarity_top_k",
        "top_k": int(top_k),
        "train_count": len(train_samples),
        "test_count": len(test_samples),
        "embedding_model": model_name,
    }
    if top_k <= 0:
        info["enabled"] = False
        info["reason"] = "top_k<=0"
        return train_samples, info
    if not train_samples or not test_samples:
        info["enabled"] = False
        info["reason"] = "empty_train_or_test"
        return train_samples, info

    print(
        f"[Train Subset] Selecting top-{top_k} similar train questions per test "
        f"using {model_name}..."
    )
    try:
        tokenizer, model, device_name = _load_embedding_model(model_name, device)
    except Exception as exc:
        raise RuntimeError(f"Failed to load embedding model for train subset: {exc}") from exc

    info["embedding_device"] = device_name
    train_texts = [sample.question for sample in train_samples]
    test_texts = [sample.question for sample in test_samples]

    try:
        train_emb = _embed_texts(
            model=model,
            tokenizer=tokenizer,
            texts=train_texts,
            device=device_name,
            batch_size=batch_size,
        )
        test_emb = _embed_texts(
            model=model,
            tokenizer=tokenizer,
            texts=test_texts,
            device=device_name,
            batch_size=batch_size,
        )
    except Exception as exc:
        raise RuntimeError(f"Failed to embed questions for train subset: {exc}") from exc

    try:
        import numpy as np
    except Exception:
        np = None

    selected_indices: set[int] = set()
    train_count = len(train_samples)
    effective_k = min(int(top_k), train_count)
    info["effective_k"] = effective_k

    if np is not None:
        train_arr = np.asarray(train_emb, dtype="float32")
        test_arr = np.asarray(test_emb, dtype="float32")
        train_norm = np.linalg.norm(train_arr, axis=1, keepdims=True)
        test_norm = np.linalg.norm(test_arr, axis=1, keepdims=True)
        train_arr = train_arr / np.clip(train_norm, 1e-12, None)
        test_arr = test_arr / np.clip(test_norm, 1e-12, None)

        sims = test_arr @ train_arr.T
        for row in sims:
            if effective_k >= train_count:
                idxs = range(train_count)
            else:
                idxs = np.argpartition(-row, effective_k - 1)[:effective_k]
            for idx in idxs:
                selected_indices.add(int(idx))
    else:
        import torch
        train_tensor = train_emb
        test_tensor = test_emb
        if not isinstance(train_tensor, torch.Tensor):
            train_tensor = torch.tensor(train_tensor, dtype=torch.float32)
        if not isinstance(test_tensor, torch.Tensor):
            test_tensor = torch.tensor(test_tensor, dtype=torch.float32)
        train_tensor = torch.nn.functional.normalize(train_tensor, p=2, dim=1)
        test_tensor = torch.nn.functional.normalize(test_tensor, p=2, dim=1)
        sims = test_tensor @ train_tensor.T
        if effective_k >= train_count:
            topk_indices = torch.arange(train_count).repeat(sims.size(0), 1)
        else:
            topk_indices = torch.topk(sims, k=effective_k, dim=1).indices
        for row in topk_indices:
            for idx in row.tolist():
                selected_indices.add(int(idx))

    selected = [train_samples[i] for i in sorted(selected_indices)]
    info["enabled"] = True
    info["selected_count"] = len(selected)
    return selected, info


def _summarize_split_info(split_info: Dict[str, Any]) -> Dict[str, Any]:
    summary = {k: v for k, v in split_info.items() if k not in {"train_ids", "test_ids"}}
    train_ids = split_info.get("train_ids")
    test_ids = split_info.get("test_ids")
    if train_ids is not None:
        summary["train_count"] = len(train_ids)
    if test_ids is not None:
        summary["test_count"] = len(test_ids)
    return summary


def _normalize_id_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value if str(item).strip()]
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        parsed = _safe_json_loads(text)
        if isinstance(parsed, list):
            return [str(item) for item in parsed if str(item).strip()]
        if "\n" in text:
            return [line.strip() for line in text.splitlines() if line.strip()]
        return [text]
    return [str(value)]


def _load_external_split_ids(path: Optional[str]) -> Optional[Dict[str, Any]]:
    if not path:
        return None
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    train_ids = payload.get("train_ids") or payload.get("train")
    test_ids = payload.get("test_ids") or payload.get("test")
    splits = payload.get("splits")
    if isinstance(splits, dict):
        train_ids = train_ids or splits.get("train")
        test_ids = test_ids or splits.get("test")
    return {
        "train_ids": _normalize_id_list(train_ids),
        "test_ids": _normalize_id_list(test_ids),
        "raw": payload,
    }


def _filter_samples_by_id(samples: List[Sample], allowed_ids: Optional[Iterable[str]]) -> List[Sample]:
    if not allowed_ids:
        return samples
    allowed = {str(item).strip() for item in allowed_ids if str(item).strip()}
    if not allowed:
        return samples
    return [sample for sample in samples if sample.id in allowed]


def _write_split_manifest(path: str, split_info: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(split_info, handle, indent=2, ensure_ascii=True)


def _init_production_stats() -> Dict[str, Any]:
    return {
        "total": 0,
        "l1_success": 0,
        "l2_success": 0,
        "l3_success": 0,
        "all_fail": 0,
        "l3_failures": 0,
        "memories_extracted": {
            # 实际的记忆类型（与 memory.py 保持一致）
            "global_correct_flow": 0,  # 全局成功流程记忆
            "local_corrective": 0,     # 局部纠错记忆
        },
    }


def _load_production_checkpoint(
    results_path: str,
) -> Tuple[Set[str], Dict[str, Any], List[str]]:
    processed_ids: Set[str] = set()
    stats = _init_production_stats()
    l3_failed_ids: List[str] = []
    l3_failed_id_set: Set[str] = set()

    if not os.path.exists(results_path):
        return processed_ids, stats, l3_failed_ids

    with open(results_path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                result = json.loads(line)
            except json.JSONDecodeError:
                continue

            sample_id = result.get("sample_id")
            if sample_id is None:
                continue
            sample_id = str(sample_id)
            if sample_id in processed_ids:
                continue
            processed_ids.add(sample_id)

            trajectories = result.get("trajectories", [])
            traj_by_level = {}
            if isinstance(trajectories, list):
                for traj in trajectories:
                    if isinstance(traj, dict) and "level" in traj:
                        traj_by_level[traj.get("level")] = traj
            l1 = traj_by_level.get(1)
            l2 = traj_by_level.get(2)
            l3 = traj_by_level.get(3)
            l1_success = l1 and l1.get("judge", {}).get("label") == "success"
            l2_success = l2 and l2.get("judge", {}).get("label") == "success"
            l3_success = l3 and l3.get("judge", {}).get("label") == "success"

            stats["total"] += 1
            if l1_success:
                stats["l1_success"] += 1
            elif l2_success:
                stats["l2_success"] += 1
            elif l3_success:
                stats["l3_success"] += 1
            else:
                stats["all_fail"] += 1

            if l3 and l3.get("judge", {}).get("label") != "success":
                stats["l3_failures"] += 1
                if sample_id not in l3_failed_id_set:
                    l3_failed_ids.append(sample_id)
                    l3_failed_id_set.add(sample_id)

            memories = result.get("memories_extracted", [])
            if isinstance(memories, list):
                for mem in memories:
                    if not isinstance(mem, dict):
                        continue
                    mem_type = mem.get("type")
                    if not mem_type and all(k in mem for k in ("trigger", "dimension", "comparison")):
                        mem_type = "preference_rule"
                    if mem_type in stats["memories_extracted"]:
                        stats["memories_extracted"][mem_type] += 1

    return processed_ids, stats, l3_failed_ids


def _load_existing_result_ids(results_path: str) -> Set[str]:
    processed_ids: Set[str] = set()
    if not os.path.exists(results_path):
        return processed_ids
    with open(results_path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                result = json.loads(line)
            except json.JSONDecodeError:
                continue
            sample_id = result.get("sample_id")
            if sample_id is None:
                continue
            processed_ids.add(str(sample_id))
    return processed_ids


def _extract_success_reasoning(
    trajectories: List[Dict[str, Any]],
    max_chars: int = 1500,
) -> str:
    if not trajectories:
        return ""
    for traj in reversed(trajectories):
        try:
            if traj.get("judge", {}).get("label") == "success":
                text = (traj.get("parsed_reasoning") or traj.get("response") or "").strip()
                if text:
                    return text[:max_chars]
        except Exception:
            continue
    fallback = trajectories[-1] if trajectories else {}
    text = (fallback.get("parsed_reasoning") or fallback.get("response") or "").strip()
    return text[:max_chars]


def _resolve_streaming_refinement_model(
    model_name: str,
    student: Model,
    teacher: Model,
    expert: Model,
    judge_model: Model,
    extractor_model: Model,
    timeout: Optional[int],
    retries: Optional[int],
) -> Model:
    key = (model_name or "extractor").strip().lower()
    alias_map = {
        "extractor": extractor_model,
        "extractor_model": extractor_model,
        "student": student,
        "student_model": student,
        "teacher": teacher,
        "teacher_model": teacher,
        "expert": expert,
        "expert_model": expert,
        "judge": judge_model,
        "judge_model": judge_model,
    }
    if key in alias_map:
        return alias_map[key]
    return load_model(
        model_name,
        timeout=timeout,
        max_retries=retries,
    )


# ============================================================
# Memory Refinement Helper
# ============================================================

def _run_memory_refinement(
    memory_path: str,
    refined_path: str,
    results_path: str,
    extractor_model: Model,
) -> int:
    """
    Run memory refinement on all extracted memories.
    
    Strategy: Group memories by sample_id and refine each group together.
    
    Args:
        memory_path: Path to raw memories file
        refined_path: Output path for refined memories
        results_path: Path to cascade results (for reasoning context)
        extractor_model: Model to use for refinement
    
    Returns:
        Number of refined memories
    """
    # Load raw memories
    raw_memories: List[MemoryItem] = []
    if os.path.exists(memory_path):
        with open(memory_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    item = _dict_to_memory_item(data)
                    if item is not None:
                        raw_memories.append(item)
                except json.JSONDecodeError:
                    continue
    
    if not raw_memories:
        print("  No memories to refine")
        return 0
    
    print(f"  Loaded {len(raw_memories)} raw memories")
    
    # Load cascade results for reasoning context
    results_by_id: Dict[str, Dict[str, Any]] = {}
    if os.path.exists(results_path):
        with open(results_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    result = json.loads(line)
                    sample_id = result.get("sample_id")
                    if sample_id:
                        results_by_id[sample_id] = result
                except json.JSONDecodeError:
                    continue
    
    # Group memories by sample_id
    memories_by_sample: Dict[str, List[MemoryItem]] = {}
    for mem in raw_memories:
        sample_id = mem.source.get("sample_id", "unknown")
        if sample_id not in memories_by_sample:
            memories_by_sample[sample_id] = []
        memories_by_sample[sample_id].append(mem)
    
    print(f"  Grouped into {len(memories_by_sample)} sample groups")
    
    # Refine each group
    all_refined: List[MemoryItem] = []
    for sample_id, sample_memories in memories_by_sample.items():
        result = results_by_id.get(sample_id, {})
        question = result.get("question", "")
        trajectories = result.get("trajectories", [])
        reasoning = _extract_success_reasoning(trajectories, max_chars=1500)
        
        # Determine success based on final label
        success = result.get("final_label") == "success"
        
        # Call refine_memories
        try:
            refined = refine_memories(
                extractor_model=extractor_model,
                question=question,
                reasoning=reasoning,
                new_memories=sample_memories,
                retrieved_memories=[],  # No historical memories during first production
                success=success,
                sample_id=sample_id,
            )
            all_refined.extend(refined)
        except Exception as e:
            print(f"  Warning: Failed to refine memories for {sample_id}: {e}")
            # Keep original memories if refinement fails
            all_refined.extend(sample_memories)
    
    # Save refined memories
    os.makedirs(os.path.dirname(refined_path) or ".", exist_ok=True)
    with open(refined_path, "w", encoding="utf-8") as f:
        for mem in all_refined:
            f.write(json.dumps(mem.__dict__, ensure_ascii=False) + "\n")
    
    print(f"  Refined to {len(all_refined)} memories")
    return len(all_refined)


# ============================================================
# Phase 1: Memory Production
# ============================================================

def run_memory_production(config: ExperimentConfig) -> Dict[str, Any]:
    """
    Run cascade on all samples to produce memories.
    
    Returns:
        Summary statistics
    """
    print(f"=== Memory Production Phase ===")
    print(f"Strategy: {config.memory_strategy}")
    print(f"Dataset: {config.dataset_path}")
    print(f"Models: student={config.student_model}, teacher={config.teacher_model}, expert={config.expert_model}")
    
    # Setup output directory
    output_dir = os.path.join(config.output_dir, config.experiment_name, "production")
    os.makedirs(output_dir, exist_ok=True)

    is_rubrichub = any(
        _is_rubrichub_dataset(path)
        for path in (config.dataset_path, config.train_dataset_path, config.test_dataset_path)
        if path
    )
    is_advancedif = (
        config.prompt_profile == "advancedif"
        or _is_advancedif_dataset(config.dataset_path)
        or is_rubrichub
    )
    is_helpsteer2 = config.prompt_profile == "helpsteer2" or _is_helpsteer2_dataset(config.dataset_path)
    is_preference = _is_preference_task(config.task_type, config.prompt_profile)
    streaming_refinement_enabled = (
        bool(config.streaming_memory_refinement)
        and not is_advancedif
        and not is_helpsteer2
        and not is_preference
    )
    if config.streaming_memory_refinement and not streaming_refinement_enabled:
        print("Streaming memory refinement is only enabled for math/aime strategy memories; skipping for this dataset/profile.")
    
    # Initialize models
    print("\nLoading models...")
    student = load_model(
        config.student_model,
        timeout=config.api_timeout,
        max_retries=config.api_retries,
    )
    judge_model = load_model(
        config.judge_model,
        timeout=config.api_timeout,
        max_retries=config.api_retries,
    )
    if is_advancedif or is_helpsteer2 or is_preference:
        teacher = judge_model
        expert = judge_model
    else:
        teacher = load_model(
            config.teacher_model,
            timeout=config.api_timeout,
            max_retries=config.api_retries,
        )
        expert = load_model(
            config.expert_model,
            timeout=config.api_timeout,
            max_retries=config.api_retries,
        )
    if config.extractor_model.lower() in {"student", "student_model", "l1"}:
        extractor_model = student
    else:
        extractor_model = load_model(
            config.extractor_model,
            timeout=config.api_timeout,
            max_retries=config.api_retries,
        )
    deepseek_reasoner = None
    if config.deepseek_reasoner_model:
        model_name = config.deepseek_reasoner_model.strip()
        if model_name:
            if "deepseek" in model_name.lower() and not os.environ.get("DEEPSEEK_API_KEY"):
                print("  Deepseek reasoner disabled: missing DEEPSEEK_API_KEY")
            else:
                deepseek_reasoner = load_model(
                    model_name,
                    timeout=config.api_timeout,
                    max_retries=config.api_retries,
                )
    streaming_refiner_model = extractor_model
    if streaming_refinement_enabled:
        try:
            streaming_refiner_model = _resolve_streaming_refinement_model(
                model_name=config.streaming_memory_refinement_model,
                student=student,
                teacher=teacher,
                expert=expert,
                judge_model=judge_model,
                extractor_model=extractor_model,
                timeout=config.api_timeout,
                retries=config.api_retries,
            )
        except Exception as exc:
            print(f"Warning: failed to load streaming refinement model '{config.streaming_memory_refinement_model}': {exc}")
            print("Fallback to extractor model for streaming refinement.")
            streaming_refiner_model = extractor_model

    print(f"  Student: {student.name}")
    print(f"  Teacher: {teacher.name}")
    print(f"  Expert: {expert.name}")
    print(f"  Judge: {judge_model.name}")
    print(f"  Extractor: {extractor_model.name}")
    if streaming_refinement_enabled:
        print(
            f"  Streaming Refiner: {streaming_refiner_model.name} "
            f"(freq={max(1, int(config.streaming_memory_refinement_frequency))}, "
            f"history_top_k={max(1, int(config.streaming_memory_history_top_k))})"
        )
    if deepseek_reasoner:
        print(f"  Deepseek Reasoner: {deepseek_reasoner.name}")
    judge = None
    if not is_preference and not is_advancedif and not is_helpsteer2:
        judge = create_judge(
            judge_model,
            task_type=config.task_type,
            prompt_profile=config.prompt_profile,
            strict_hotpotqa=config.strict_hotpotqa_eval,
        )
    
    # Initialize memory banks for different strategies
    memory_paths = {
        "student": os.path.join(output_dir, "student_memory.jsonl"),
        "teacher": os.path.join(output_dir, "teacher_memory.jsonl"),
        "contrastive": os.path.join(output_dir, "contrastive_memory.jsonl"),
        "all": os.path.join(output_dir, "all_memory.jsonl"),
    }
    
    # Results path
    results_path = os.path.join(output_dir, "cascade_results.jsonl")
    l3_failed_ids_path = os.path.join(output_dir, "l3_failed_ids.json")
    # Distributed sharding: each rank writes to its own file to avoid contention.
    rank, world, is_main = _dist_info()
    if world > 1:
        results_path = results_path.replace(".jsonl", f".rank{rank}.jsonl")
        # Also shard memory outputs so each rank can run independently.
        for k in list(memory_paths.keys()):
            memory_paths[k] = memory_paths[k].replace(".jsonl", f".rank{rank}.jsonl")
        l3_failed_ids_path = l3_failed_ids_path.replace(".json", f".rank{rank}.json")
    
    # Load dataset (train split)
    train_dataset, test_dataset, split_info = _load_split_datasets(config)
    dataset = train_dataset
    if config.train_subset_top_k > 0 and not split_info.get("train_subset_locked"):
        dataset, subset_info = _select_train_subset_by_similarity(
            train_samples=train_dataset,
            test_samples=test_dataset,
            top_k=config.train_subset_top_k,
            model_name=config.train_subset_embedding_model,
            device=config.train_subset_embedding_device,
            batch_size=config.train_subset_embedding_batch_size,
        )
        split_info["train_subset"] = subset_info
        if subset_info.get("enabled"):
            print(
                f"[Train Subset] Selected {len(dataset)}/{len(train_dataset)} "
                f"train samples (top_k={subset_info.get('effective_k')}, "
                f"test={len(test_dataset)})"
            )
    split_path = os.path.join(config.output_dir, config.experiment_name, "data_split.json")
    _write_split_manifest(split_path, split_info)
    print(f"\nLoaded {len(dataset)} training samples (split: {split_info['mode']})")
    if config.sample_id_filter:
        before = len(dataset)
        dataset = _filter_samples_by_id(dataset, config.sample_id_filter)
        print(f"Filtered by sample_id: {len(dataset)}/{before}")

    # AdvancedIF: rubric-based instruction-following memory extraction
    if is_advancedif:
        adv_stats: Dict[str, Any] = {
            "total": 0,
            "rubrics_total": 0,
            "rubrics_scored": 0,
            "score_1": 0,
            "score_05": 0,
            "score_0": 0,
            "judge_errors": 0,
            "missing_rubrics": 0,
            "memories_extracted": {"preference_rule": 0},
        }
        processed_ids = set()
        if config.resume_production:
            processed_ids = _load_existing_result_ids(results_path)
            if processed_ids:
                print(f"Resuming production: {len(processed_ids)} samples already processed")

        skipped = 0
        mode = (config.advancedif_memory_mode or "rubric").lower()
        use_rubric = mode in {
            "rubric",
            "rubric+teacher",
            "rubric+temp",
            "all",
        }
        use_teacher = mode in {
            "teacher",
            "rubric+teacher",
            "teacher+temp",
            "all",
        }
        use_temp = mode in {
            "temp",
            "rubric+temp",
            "teacher+temp",
            "all",
        }
        debug_remaining = (
            int(config.advancedif_debug_judge_max_samples)
            if config.advancedif_debug_judge
            else 0
        )
        dataset_label = "rubrichub" if is_rubrichub else "advancedif"
        for idx, sample in enumerate(dataset):
            if world > 1 and (idx % world) != rank:
                continue
            if sample.id in processed_ids:
                skipped += 1
                continue
            print(f"\n[{idx+1}/{len(dataset)}] Processing: {sample.id}")
            start_time = time.time()

            if is_rubrichub:
                rubrics, rubric_weights = _extract_rubrichub_rubrics(sample)
            else:
                rubrics = _extract_advancedif_rubrics(sample)
                rubric_weights = [1.0 for _ in rubrics]
            if not rubrics:
                adv_stats["missing_rubrics"] += 1
                continue

            # De-duplicate rubrics while preserving order
            seen_rubrics: Set[str] = set()
            filtered_rubrics: List[str] = []
            filtered_weights: List[float] = []
            for idx_r, rubric in enumerate(rubrics):
                weight = rubric_weights[idx_r] if idx_r < len(rubric_weights) else 1.0
                cleaned = _clean_rubric_text(rubric, max_chars=400)
                if not cleaned:
                    continue
                key = cleaned.lower()
                if key in seen_rubrics:
                    continue
                seen_rubrics.add(key)
                filtered_rubrics.append(cleaned)
                filtered_weights.append(weight)

            if not filtered_rubrics:
                adv_stats["missing_rubrics"] += 1
                continue

            user_prompt = _build_user_prompt(
                sample.question,
                config.task_type,
                config.prompt_profile,
                hotpotqa_short_prompt_eval=config.hotpotqa_short_prompt_eval,
            )
            system_prompt = _build_base_system_prompt(
                config.task_type,
                config.prompt_profile,
                config.hotpotqa_short_prompt_eval,
            )

            try:
                stop_sequences = [
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
                student_stop = None if _disable_stop_for_aime(config.task_type) else stop_sequences
                response = student.generate(
                    prompt=user_prompt,
                    system_prompt=system_prompt,
                    temperature=config.temperature,
                    max_tokens=config.cascade_max_tokens,
                    stop=student_stop,
                )
                response_text = _clean_advancedif_response(response.text or "")
                if len(response_text) < ADVANCEDIF_MIN_RESPONSE_CHARS:
                    response = student.generate(
                        prompt=user_prompt,
                        system_prompt=system_prompt,
                        temperature=config.temperature,
                        max_tokens=config.cascade_max_tokens,
                        stop=None,
                    )
                    response_text = _clean_advancedif_response(response.text or "")
            except Exception as exc:
                print(f"  Error: {exc}")
                response_text = f"Error: {exc}"

            rubric_scores: List[Dict[str, Any]] = []
            memories: List[MemoryItem] = []
            memory_seen: Set[Tuple[str, str, str]] = set()
            memory_map: Dict[int, Dict[str, str]] = {}
            per_rubric_counts: Dict[int, int] = {}
            scores: List[float] = []
            had_judge_error = False
            failed_rubrics: List[Tuple[int, str]] = []

            debug_for_sample = debug_remaining > 0
            score_debug: List[str] = []
            memory_debug: List[str] = []

            if use_rubric:
                adv_stats["rubrics_total"] += len(filtered_rubrics)
                score_values = _score_values_for_sample(sample)
                max_memories_per_rubric = 3 if score_values == BIGGEN_SCORE_VALUES else 1

                scores = _judge_advancedif_scores(
                    judge_model=teacher,
                    prompt=user_prompt,
                    response=response_text,
                    rubrics=filtered_rubrics,
                    max_tokens=config.advancedif_judge_max_tokens,
                    debug_out=score_debug,
                    allowed_scores=score_values,
                )
                if (not scores or len(scores) != len(filtered_rubrics)) and config.advancedif_judge_max_tokens < 512:
                    scores = _judge_advancedif_scores(
                        judge_model=teacher,
                        prompt=user_prompt,
                        response=response_text,
                        rubrics=filtered_rubrics,
                        max_tokens=512,
                        debug_out=score_debug,
                        allowed_scores=score_values,
                    )

                if debug_for_sample:
                    debug_remaining -= 1

                had_judge_error = len(scores) != len(filtered_rubrics) or not scores
                if len(scores) < len(filtered_rubrics):
                    adv_stats["judge_errors"] += len(filtered_rubrics) - len(scores)
                    scores = scores + [0.0] * (len(filtered_rubrics) - len(scores))
                elif len(scores) > len(filtered_rubrics):
                    adv_stats["judge_errors"] += len(scores) - len(filtered_rubrics)
                    scores = scores[:len(filtered_rubrics)]

                failed_indices = [
                    idx for idx, score in enumerate(scores) if score is None or score < 0.99
                ]
                failed_rubrics = [(idx + 1, filtered_rubrics[idx]) for idx in failed_indices]

                memory_items: List[Dict[str, Any]] = []
                if failed_rubrics:
                    memory_items = _judge_advancedif_memories(
                        judge_model=teacher,
                        prompt=user_prompt,
                        response=response_text,
                        rubrics_with_ids=failed_rubrics,
                        max_tokens=config.advancedif_judge_max_tokens,
                        debug_out=memory_debug,
                        max_memories_per_rubric=max_memories_per_rubric,
                    )

                expected_ids = {rid for rid, _ in failed_rubrics}

            def _parse_rubric_id(raw: Any) -> Optional[int]:
                if raw is None:
                    return None
                if isinstance(raw, (int, float)):
                    return int(raw)
                text = str(raw).strip()
                match = re.match(r"^\\s*(\\d+)", text)
                if match:
                    return int(match.group(1))
                return None

            if use_rubric:
                for item in memory_items:
                    rid = _parse_rubric_id(item.get("rubric_id") or item.get("rubric"))
                    if rid is None or rid not in expected_ids:
                        continue
                    trigger = str(item.get("trigger") or "").strip()
                    dimension = str(item.get("dimension") or "").strip()
                    comparison = str(item.get("comparison") or "").strip()
                    if not (trigger and dimension and comparison):
                        adv_stats["judge_errors"] += 1
                        had_judge_error = True
                        continue
                    count = per_rubric_counts.get(rid, 0)
                    if count >= max_memories_per_rubric:
                        continue
                    key = (trigger, dimension, comparison)
                    if key in memory_seen:
                        continue
                    memory_seen.add(key)
                    per_rubric_counts[rid] = count + 1
                    if rid not in memory_map:
                        memory_map[rid] = {
                            "trigger": trigger,
                            "dimension": dimension,
                            "comparison": comparison,
                        }
                    rubric_text = filtered_rubrics[rid - 1] if 0 < rid <= len(filtered_rubrics) else ""
                    source = {
                        "sample_id": sample.id,
                        "rubric": rubric_text,
                        "score": scores[rid - 1] if 0 < rid <= len(scores) else 0.0,
                        "timestamp": datetime.utcnow().isoformat(),
                        "dataset": dataset_label,
                    }
                    built = _build_memory_items(
                        memories_data=[
                            {
                                "trigger": trigger,
                                "dimension": dimension,
                                "comparison": comparison,
                            }
                        ],
                        source=source,
                        memory_type="preference_rule",
                        answer_hint=None,
                        max_items=1,
                    )
                    if built:
                        memories.extend(built)

            # Additional AdvancedIF memory extraction modes
            if use_teacher or use_temp:
                stop_sequences = [
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
                if _disable_stop_for_aime(config.task_type):
                    teacher_stop = None
                    temp_stop = None
                else:
                    teacher_stop = None if "gemini" in (teacher.name or "").lower() else stop_sequences
                    temp_stop = None if "gemini" in (student.name or "").lower() else stop_sequences

            # Mode 1: teacher vs student (teacher must be better to extract)
            if use_teacher:
                teacher_text = ""
                try:
                    teacher_resp = teacher.generate(
                        prompt=user_prompt,
                        system_prompt=system_prompt,
                        temperature=0.0,
                        max_tokens=config.advancedif_compare_max_tokens,
                        stop=teacher_stop,
                    )
                    teacher_text = _clean_advancedif_response(teacher_resp.text or "")
                except Exception as exc:
                    print(f"  [Teacher Compare] Error: {exc}")
                    teacher_text = ""

                if teacher_text:
                    better = _student_pick_better_response(
                        student_model=student,
                        prompt=user_prompt,
                        response1=response_text,
                        response2=teacher_text,
                        comparison_note=(
                            "Responses may come from different models. Judge only by how well the response "
                            "follows the user's instructions, constraints, and formatting."
                        ),
                    )
                    if better == 2:
                        teacher_mems = extract_preference_memories(
                            extractor_model=student,
                            context=user_prompt,
                            response1=response_text,
                            response2=teacher_text,
                            overall_preference=1,
                            sample_id=sample.id,
                            max_items=3,
                            prompt_profile=config.prompt_profile,
                        )
                        for mem in teacher_mems:
                            key = (
                                (mem.trigger or "").strip(),
                                (mem.dimension or "").strip(),
                                (mem.comparison or "").strip(),
                            )
                            if not all(key):
                                continue
                            if key in memory_seen:
                                continue
                            memory_seen.add(key)
                            memories.append(mem)

            # Mode 2: temperature comparison (0.3 vs 1.2 by default)
            if use_temp:
                low_text = ""
                high_text = ""
                try:
                    low_resp = student.generate(
                        prompt=user_prompt,
                        system_prompt=system_prompt,
                        temperature=config.advancedif_temp_low,
                        max_tokens=config.advancedif_compare_max_tokens,
                        stop=temp_stop,
                    )
                    low_text = _clean_advancedif_response(low_resp.text or "")
                except Exception as exc:
                    print(f"  [Temp Compare] Low temp error: {exc}")
                try:
                    high_resp = student.generate(
                        prompt=user_prompt,
                        system_prompt=system_prompt,
                        temperature=config.advancedif_temp_high,
                        max_tokens=config.advancedif_compare_max_tokens,
                        stop=temp_stop,
                    )
                    high_text = _clean_advancedif_response(high_resp.text or "")
                except Exception as exc:
                    print(f"  [Temp Compare] High temp error: {exc}")

                if low_text and high_text:
                    better = _student_pick_better_response(
                        student_model=student,
                        prompt=user_prompt,
                        response1=low_text,
                        response2=high_text,
                        comparison_note=(
                            "Responses were generated with different sampling temperatures. Judge only by "
                            "instruction-following quality and factual correctness, not by verbosity."
                        ),
                    )
                    if better in (1, 2):
                        overall_preference = -1 if better == 1 else 1
                        temp_mems = extract_preference_memories(
                            extractor_model=student,
                            context=user_prompt,
                            response1=low_text,
                            response2=high_text,
                            overall_preference=overall_preference,
                            sample_id=sample.id,
                            max_items=3,
                            prompt_profile=config.prompt_profile,
                        )
                        for mem in temp_mems:
                            key = (
                                (mem.trigger or "").strip(),
                                (mem.dimension or "").strip(),
                                (mem.comparison or "").strip(),
                            )
                            if not all(key):
                                continue
                            if key in memory_seen:
                                continue
                            memory_seen.add(key)
                            memories.append(mem)

            rubric_scores = []
            if use_rubric:
                for idx, rubric_text in enumerate(filtered_rubrics):
                    score = scores[idx] if idx < len(scores) else None
                    if score is None:
                        adv_stats["judge_errors"] += 1
                        had_judge_error = True
                    else:
                        adv_stats["rubrics_scored"] += 1
                        if score >= 0.99:
                            adv_stats["score_1"] += 1
                        elif score <= 0.01:
                            adv_stats["score_0"] += 1
                        else:
                            adv_stats["score_05"] += 1
                    entry = {
                        **({"weight": filtered_weights[idx]} if is_rubrichub and idx < len(filtered_weights) else {}),
                        "rubric": rubric_text,
                        "score": score,
                    }
                    memory_fields = memory_map.get(idx + 1)
                    if memory_fields:
                        entry.update(memory_fields)
                    rubric_scores.append(entry)

            memory_path = memory_paths.get(config.memory_strategy, memory_paths["all"])
            _append_memories_to_file(memory_path, memories)
            adv_stats["total"] += 1
            adv_stats["memories_extracted"]["preference_rule"] += len(memories)

            duration = time.time() - start_time
            result = {
                "sample_id": sample.id,
                "prompt": user_prompt,
                "response": response_text,
                "rubric_scores": rubric_scores,
                "memories_extracted": [asdict_memory(m) for m in memories],
                "duration_sec": round(duration, 2),
            }
            if had_judge_error or debug_for_sample:
                debug_parts: List[str] = []
                if score_debug:
                    debug_parts.append(f"SCORE_RAW:\n{score_debug[0]}")
                if memory_debug:
                    debug_parts.append(f"MEMORY_RAW:\n{memory_debug[0]}")
                if debug_parts:
                    result["judge_raw"] = "\n\n".join(debug_parts)
            _append_jsonl(results_path, result)
            print(f"  Rubrics scored: {len(rubric_scores)} | Memories: {len(memories)}")

        summary = {
            "config": asdict(config),
            "data_split": _summarize_split_info(split_info),
            "data_split_path": split_path,
            "stats": adv_stats,
            "timestamp": datetime.utcnow().isoformat(),
        }
        if config.sample_id_filter:
            summary["sample_id_filter_count"] = len(config.sample_id_filter)
        if skipped:
            summary["skipped_samples"] = skipped
        summary_path = os.path.join(output_dir, "summary.json")
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

        print(f"\n=== AdvancedIF Memory Production Complete ===")
        print(f"Processed: {adv_stats['total']}")
        print(f"Rubrics scored: {adv_stats['rubrics_scored']}/{adv_stats['rubrics_total']}")
        print(f"Memories: {adv_stats['memories_extracted']}")

        if config.enable_memory_refinement:
            print("Memory refinement is not supported for AdvancedIF preference rules; skipping.")

        return summary

    # HelpSteer2: dimension-based scoring + memory extraction
    if is_helpsteer2:
        hs_stats: Dict[str, Any] = {
            "total": 0,
            "dimensions_total": 0,
            "dimensions_scored": 0,
            "score_5": 0,
            "score_4": 0,
            "score_3": 0,
            "score_2": 0,
            "score_1": 0,
            "judge_errors": 0,
            "memories_extracted": {"preference_rule": 0},
        }
        processed_ids = set()
        if config.resume_production:
            processed_ids = _load_existing_result_ids(results_path)
            if processed_ids:
                print(f"Resuming production: {len(processed_ids)} samples already processed")

        skipped = 0
        debug_remaining = (
            int(config.advancedif_debug_judge_max_samples)
            if config.advancedif_debug_judge
            else 0
        )
        stop_sequences = [
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
        student_stop = None if _disable_stop_for_aime(config.task_type) else stop_sequences
        for idx, sample in enumerate(dataset):
            if world > 1 and (idx % world) != rank:
                continue
            if sample.id in processed_ids:
                skipped += 1
                continue
            print(f"\n[{idx+1}/{len(dataset)}] Processing: {sample.id}")
            start_time = time.time()

            user_prompt = _build_user_prompt(
                sample.question,
                config.task_type,
                config.prompt_profile,
                hotpotqa_short_prompt_eval=config.hotpotqa_short_prompt_eval,
            )
            system_prompt = _build_base_system_prompt(
                config.task_type,
                config.prompt_profile,
                config.hotpotqa_short_prompt_eval,
            )

            try:
                response = student.generate(
                    prompt=user_prompt,
                    system_prompt=system_prompt,
                    temperature=config.temperature,
                    max_tokens=config.cascade_max_tokens,
                    stop=student_stop,
                )
                response_text = _clean_advancedif_response(response.text or "")
                if len(response_text) < ADVANCEDIF_MIN_RESPONSE_CHARS:
                    response = student.generate(
                        prompt=user_prompt,
                        system_prompt=system_prompt,
                        temperature=config.temperature,
                        max_tokens=config.cascade_max_tokens,
                        stop=None,
                    )
                    response_text = _clean_advancedif_response(response.text or "")
            except Exception as exc:
                print(f"  Error: {exc}")
                response_text = f"Error: {exc}"

            dimension_scores: List[Dict[str, Any]] = []
            memories: List[MemoryItem] = []
            hs_stats["dimensions_total"] += len(HELPSTEER2_DIMENSIONS)

            debug_holder: Optional[List[str]] = [] if debug_remaining > 0 else None
            judge_items = _judge_helpsteer2_dimensions(
                judge_model=teacher,
                prompt=user_prompt,
                response=response_text,
                max_tokens=config.advancedif_judge_max_tokens,
                debug_out=debug_holder,
            )
            if debug_remaining > 0:
                debug_remaining -= 1
            aligned_items = _align_advancedif_items(HELPSTEER2_DIMENSIONS, judge_items)
            if not aligned_items:
                hs_stats["judge_errors"] += len(HELPSTEER2_DIMENSIONS)
                dimension_scores.extend(
                    {"dimension": dim, "score": None} for dim in HELPSTEER2_DIMENSIONS
                )
            else:
                memory_seen: Set[Tuple[str, str, str]] = set()
                for dim_label, item in zip(HELPSTEER2_DIMENSIONS, aligned_items):
                    rubric_text = dim_label
                    raw_score = item.get("score")
                    score = _parse_helpsteer2_score(raw_score)
                    if score is None:
                        hs_stats["judge_errors"] += 1
                        dimension_scores.append({"dimension": rubric_text, "score": None})
                        continue
                    hs_stats["dimensions_scored"] += 1
                    if score >= 4.99:
                        hs_stats["score_5"] += 1
                    elif score >= 3.99:
                        hs_stats["score_4"] += 1
                    elif score >= 2.99:
                        hs_stats["score_3"] += 1
                    elif score >= 1.99:
                        hs_stats["score_2"] += 1
                    else:
                        hs_stats["score_1"] += 1

                    trigger = str(item.get("trigger") or "").strip()
                    comparison = str(item.get("comparison") or "").strip()
                    dimension_scores.append(
                        {
                            "dimension": rubric_text,
                            "score": score,
                            "trigger": trigger,
                            "comparison": comparison,
                        }
                    )

                    if score < HELPSTEER2_MEMORY_THRESHOLD and not (trigger and comparison):
                        hs_stats["judge_errors"] += 1
                    if score < HELPSTEER2_MEMORY_THRESHOLD and trigger and comparison:
                        key = (trigger, rubric_text, comparison)
                        if key in memory_seen:
                            continue
                        memory_seen.add(key)
                        source = {
                            "sample_id": sample.id,
                            "rubric": rubric_text,
                            "score": score,
                            "timestamp": datetime.utcnow().isoformat(),
                            "dataset": "helpsteer2",
                        }
                        built = _build_memory_items(
                            memories_data=[
                                {
                                    "trigger": trigger,
                                    "dimension": rubric_text,
                                    "comparison": comparison,
                                }
                            ],
                            source=source,
                            memory_type="preference_rule",
                            answer_hint=None,
                            max_items=1,
                        )
                        if built:
                            memories.extend(built)

            memory_path = memory_paths.get(config.memory_strategy, memory_paths["all"])
            _append_memories_to_file(memory_path, memories)
            hs_stats["total"] += 1
            hs_stats["memories_extracted"]["preference_rule"] += len(memories)

            duration = time.time() - start_time
            result = {
                "sample_id": sample.id,
                "prompt": user_prompt,
                "response": response_text,
                "dimension_scores": dimension_scores,
                "memories_extracted": [asdict_memory(m) for m in memories],
                "duration_sec": round(duration, 2),
            }
            if debug_holder:
                result["judge_raw"] = debug_holder[0]
            _append_jsonl(results_path, result)
            print(f"  Dimensions scored: {len(dimension_scores)} | Memories: {len(memories)}")

        summary = {
            "config": asdict(config),
            "data_split": _summarize_split_info(split_info),
            "data_split_path": split_path,
            "stats": hs_stats,
            "timestamp": datetime.utcnow().isoformat(),
        }
        if config.sample_id_filter:
            summary["sample_id_filter_count"] = len(config.sample_id_filter)
        if skipped:
            summary["skipped_samples"] = skipped
        summary_path = os.path.join(output_dir, "summary.json")
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

        print(f"\n=== HelpSteer2 Memory Production Complete ===")
        print(f"Processed: {hs_stats['total']}")
        print(f"Dimensions scored: {hs_stats['dimensions_scored']}/{hs_stats['dimensions_total']}")
        print(f"Memories: {hs_stats['memories_extracted']}")

        if config.enable_memory_refinement:
            print("Memory refinement is not supported for HelpSteer2; skipping.")

        return summary

    # Preference task: direct preference memory extraction (no cascade)
    if is_preference:
        pref_stats: Dict[str, Any] = {
            "total": 0,
            "skipped_ties": 0,
            "skipped_missing": 0,
            "memories_extracted": {"preference_rule": 0},
        }
        processed_ids = set()
        if config.resume_production:
            processed_ids = _load_existing_result_ids(results_path)
            if processed_ids:
                print(f"Resuming production: {len(processed_ids)} samples already processed")

        skipped = 0
        for idx, sample in enumerate(dataset):
            if world > 1 and (idx % world) != rank:
                continue
            if sample.id in processed_ids:
                skipped += 1
                continue
            print(f"\n[{idx+1}/{len(dataset)}] Processing: {sample.id}")
            start_time = time.time()

            context, r1, r2, pref_val, dimension_hint = _extract_preference_fields(sample)
            if not r1 or not r2 or pref_val is None:
                pref_stats["skipped_missing"] += 1
                continue
            if pref_val == 0:
                pref_stats["skipped_ties"] += 1
                continue

            memories = extract_preference_memories(
                extractor_model=extractor_model,
                context=context,
                response1=r1,
                response2=r2,
                overall_preference=pref_val,
                sample_id=sample.id,
                max_items=3,
                dimension_hint=dimension_hint,
                prompt_profile=config.prompt_profile,
            )

            memory_path = memory_paths.get(config.memory_strategy, memory_paths["all"])
            _append_memories_to_file(memory_path, memories)
            pref_stats["total"] += 1
            pref_stats["memories_extracted"]["preference_rule"] += len(memories)

            duration = time.time() - start_time
            preferred = "response1" if pref_val < 0 else "response2"
            result = {
                "sample_id": sample.id,
                "context": context,
                "response1": r1,
                "response2": r2,
                "overall_preference": pref_val,
                "preferred": preferred,
                "memories_extracted": [asdict_memory(m) for m in memories],
                "duration_sec": round(duration, 2),
            }
            _append_jsonl(results_path, result)
            print(f"  Preference: {pref_val} | Memories extracted: {len(memories)}")

        summary = {
            "config": asdict(config),
            "data_split": _summarize_split_info(split_info),
            "data_split_path": split_path,
            "stats": pref_stats,
            "timestamp": datetime.utcnow().isoformat(),
        }
        if config.sample_id_filter:
            summary["sample_id_filter_count"] = len(config.sample_id_filter)
        if skipped:
            summary["skipped_samples"] = skipped
        summary_path = os.path.join(output_dir, "summary.json")
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

        print(f"\n=== Preference Memory Production Complete ===")
        print(f"Processed: {pref_stats['total']}")
        print(f"Skipped ties: {pref_stats['skipped_ties']}")
        print(f"Skipped missing: {pref_stats['skipped_missing']}")
        print(f"Memories: {pref_stats['memories_extracted']}")

        if config.enable_memory_refinement:
            print("Memory refinement is not supported for preference tasks yet; skipping.")

        return summary
    
    # Cascade config
    cascade_config = CascadeConfig(
        threshold=config.cascade_threshold,
        enable_tools=config.enable_tools,
        temperature=config.temperature,
        max_tokens=config.cascade_max_tokens,
        max_tokens_l3=config.cascade_max_tokens_l3,
        upgrade_on_failure_only=config.upgrade_on_failure_only,
        task_type=config.task_type,
        prompt_profile=config.prompt_profile,
    )
    
    # Statistics / resume checkpoint
    stats = _init_production_stats()
    processed_ids: Set[str] = set()
    l3_failed_ids: List[str] = []
    l3_failed_id_set: Set[str] = set()
    if config.resume_production:
        processed_ids, stats, l3_failed_ids = _load_production_checkpoint(results_path)
        l3_failed_id_set = set(l3_failed_ids)
        if processed_ids:
            print(f"Resuming production: {len(processed_ids)} samples already processed")
        with open(l3_failed_ids_path, "w", encoding="utf-8") as handle:
            json.dump(l3_failed_ids, handle, indent=2)
    else:
        with open(l3_failed_ids_path, "w", encoding="utf-8") as handle:
            json.dump([], handle, indent=2)
    
    model_by_level = {
        1: student,
        2: teacher,
        3: expert,
    }
    memory_path = memory_paths.get(config.memory_strategy, memory_paths["all"])
    streaming_frequency = max(1, int(config.streaming_memory_refinement_frequency))
    streaming_history_top_k = max(1, int(config.streaming_memory_history_top_k))
    streaming_memory_bank: Optional[MemoryBank] = None
    streaming_pending: List[Dict[str, Any]] = []
    if streaming_refinement_enabled:
        stats["streaming_refinement"] = {
            "enabled": True,
            "frequency": streaming_frequency,
            "history_top_k": streaming_history_top_k,
            "flushes": 0,
            "updated_memories": 0,
            "fallback_raw_memories": 0,
            "pending_samples": 0,
        }
        streaming_memory_bank = MemoryBank(memory_path, top_k=streaming_history_top_k)
        if os.path.exists(memory_path):
            loaded_stream = streaming_memory_bank.load(
                memory_path,
                max_items=config.memory_max_items if config.memory_max_items > 0 else None,
            )
            if loaded_stream:
                print(f"[StreamingRefine] Loaded {loaded_stream} historical memories from {memory_path}")
        if config.memory_retrieval == "embedding":
            streaming_memory_bank.enable_embedding_retrieval(
                model_name=config.embedding_model,
                cache_path=config.embedding_cache_path,
                device=config.embedding_device,
                batch_size=config.embedding_batch_size,
            )

    def _flush_streaming_updates() -> None:
        nonlocal streaming_pending
        if not streaming_refinement_enabled or not streaming_pending or streaming_memory_bank is None:
            return
        total_written = 0
        fallback_raw = 0
        for item in streaming_pending:
            new_memories = item.get("new_memories", []) or []
            if not new_memories:
                continue
            try:
                refined = refine_memories(
                    extractor_model=streaming_refiner_model,
                    question=item.get("question", ""),
                    reasoning=item.get("reasoning", ""),
                    new_memories=new_memories,
                    retrieved_memories=item.get("retrieved_memories", []) or [],
                    success=bool(item.get("success", False)),
                    sample_id=item.get("sample_id", ""),
                )
            except Exception as exc:
                print(f"[StreamingRefine] Failed on {item.get('sample_id')}: {exc}")
                refined = []
            if not refined:
                refined = new_memories
                fallback_raw += len(refined)
            if refined:
                total_written += streaming_memory_bank.append_memories(refined)
        stats["streaming_refinement"]["flushes"] += 1
        stats["streaming_refinement"]["updated_memories"] += int(total_written)
        stats["streaming_refinement"]["fallback_raw_memories"] += int(fallback_raw)
        streaming_pending = []
        stats["streaming_refinement"]["pending_samples"] = 0

    # Process each sample
    skipped = 0
    for idx, sample in enumerate(dataset):
        # Shard work across processes if launched with accelerate/torchrun.
        if world > 1 and (idx % world) != rank:
            continue
        if sample.id in processed_ids:
            skipped += 1
            continue
        print(f"\n[{idx+1}/{len(dataset)}] Processing: {sample.id}")
        start_time = time.time()
        
        # Run cascade (no memory injection during production)
        # [FIX] 使用修复后的 system prompt
        trajectories, final_answer, final_judge = run_cascade(
            sample=sample,
            system_prompt=_build_base_system_prompt(
                config.task_type,
                config.prompt_profile,
                config.hotpotqa_short_prompt_eval,
            ),
            student=student,
            teacher=teacher,
            expert=expert,
            judge_model=judge_model,
            deepseek_reasoner=deepseek_reasoner,
            config=cascade_config,
            tools={},
        )
        
        # Analyze trajectories
        traj_by_level = {t["level"]: t for t in trajectories}
        l1 = traj_by_level.get(1)
        l2 = traj_by_level.get(2)
        l3 = traj_by_level.get(3)
        
        l1_success = l1 and l1.get("judge", {}).get("label") == "success"
        l2_success = l2 and l2.get("judge", {}).get("label") == "success"
        l3_success = l3 and l3.get("judge", {}).get("label") == "success"
        
        stats["total"] += 1
        if l1_success:
            stats["l1_success"] += 1
        elif l2_success:
            stats["l2_success"] += 1
        elif l3_success:
            stats["l3_success"] += 1
        else:
            stats["all_fail"] += 1
        if l3 and l3.get("judge", {}).get("label") != "success":
            stats["l3_failures"] += 1
            if sample.id not in l3_failed_id_set:
                l3_failed_ids.append(sample.id)
                l3_failed_id_set.add(sample.id)
            with open(l3_failed_ids_path, "w", encoding="utf-8") as handle:
                json.dump(l3_failed_ids, handle, indent=2)
        
        # Optional: add a separate reasoning outline for memory extraction
        if config.extraction_reasoning:
            need_student = config.memory_strategy in ("student", "all", "contrastive")
            need_teacher = config.memory_strategy in ("teacher", "all", "contrastive")
            if need_student and l1:
                _attach_extraction_response(
                    l1,
                    model_by_level.get(1, student),
                    sample.question,
                    config.task_type,
                    config.prompt_profile,
                    config.temperature,
                )
            if need_teacher:
                teacher_traj = l3 if l3_success else l2 if l2_success else None
                if teacher_traj:
                    _attach_extraction_response(
                        teacher_traj,
                        model_by_level.get(teacher_traj.get("level", 2), teacher),
                        sample.question,
                        config.task_type,
                        config.prompt_profile,
                        config.temperature,
                    )

        # Extract memories based on strategy
        memories = extract_memories_from_cascade(
            extractor_model=extractor_model,
            question=sample.question,
            trajectories=trajectories,
            sample_id=sample.id,
            strategy=config.memory_strategy,
            extraction_mode=config.memory_extract_mode,
            prompt_profile=config.prompt_profile,
        )
        
        # Count by type
        for mem in memories:
            if mem.type in stats["memories_extracted"]:
                stats["memories_extracted"][mem.type] += 1
        
        # Save memories to appropriate file (streaming refinement optional)
        historical_for_refine: List[MemoryItem] = []
        if streaming_refinement_enabled and streaming_memory_bank is not None and memories:
            memory_types = sorted({m.type for m in memories if m.type})
            try:
                historical_for_refine = streaming_memory_bank.retrieve(
                    sample.question,
                    top_k=streaming_history_top_k,
                    exclude_sample_id=sample.id if config.exclude_self_memory else None,
                    expected_answer=sample.answer if config.memory_answer_leak_filter else None,
                    max_similarity=config.memory_max_similarity,
                    min_similarity=config.memory_min_similarity,
                    memory_types=memory_types or None,
                    retrieval_method=config.memory_retrieval,
                )
            except Exception:
                historical_for_refine = []
            streaming_pending.append(
                {
                    "sample_id": sample.id,
                    "question": sample.question,
                    "reasoning": _extract_success_reasoning(trajectories, max_chars=3000),
                    "success": final_judge.get("label") == "success",
                    "new_memories": memories,
                    "retrieved_memories": historical_for_refine,
                }
            )
            stats["streaming_refinement"]["pending_samples"] = len(streaming_pending)
            if len(streaming_pending) >= streaming_frequency:
                _flush_streaming_updates()
        elif memories:
            _append_memories_to_file(memory_path, memories)
        
        # Save cascade result
        duration = time.time() - start_time
        result = {
            "sample_id": sample.id,
            "question": sample.question,
            "ground_truth": sample.answer,
            "trajectories": trajectories,
            "final_answer": final_answer,
            "final_label": final_judge.get("label"),
            "memories_extracted": [asdict_memory(m) for m in memories],
            "streaming_refinement_pending": bool(streaming_refinement_enabled and memories),
            "streaming_refinement_historical_count": len(historical_for_refine),
            "duration_sec": round(duration, 2),
        }
        _append_jsonl(results_path, result)
        
        # [FIX] 打印更详细的信息
        l1_pred = l1.get("parsed_answer") if l1 else None
        l2_pred = l2.get("parsed_answer") if l2 else None
        l3_pred = l3.get("parsed_answer") if l3 else None
        print(f"  GT={sample.answer} | L1: {l1_pred} {'✓' if l1_success else '✗'} | L2: {l2_pred if l2 else '-'} {'✓' if l2_success else ('✗' if l2 else '')} | L3: {l3_pred if l3 else '-'} {'✓' if l3_success else ('✗' if l3 else '')}")
        print(f"  Memories extracted: {len(memories)}")
        
        # [DEBUG] 当 L3 被调用但解析失败时，打印原始响应的前 500 字符
        if l3 and not l3_success and not l3_pred:
            l3_response = l3.get("response", "")[:500]
            print(f"  [DEBUG] L3 raw response: {l3_response}")

    if streaming_refinement_enabled:
        _flush_streaming_updates()

    # Save summary
    summary = {
        "config": asdict(config),
        "data_split": _summarize_split_info(split_info),
        "data_split_path": split_path,
        "stats": stats,
        "l3_failed_ids_path": l3_failed_ids_path,
        "timestamp": datetime.utcnow().isoformat(),
    }
    if config.sample_id_filter:
        summary["sample_id_filter_count"] = len(config.sample_id_filter)
    if skipped:
        summary["skipped_samples"] = skipped
    with open(l3_failed_ids_path, "w", encoding="utf-8") as handle:
        json.dump(l3_failed_ids, handle, indent=2)
    summary_path = os.path.join(output_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n=== Production Complete ===")
    print(f"Total: {stats['total']}")
    print(f"L1 Success: {stats['l1_success']} ({100*stats['l1_success']/stats['total']:.1f}%)")
    print(f"L2 Success: {stats['l2_success']} ({100*stats['l2_success']/stats['total']:.1f}%)")
    print(f"L3 Success: {stats['l3_success']} ({100*stats['l3_success']/stats['total']:.1f}%)")
    print(f"All Fail: {stats['all_fail']} ({100*stats['all_fail']/stats['total']:.1f}%)")
    print(f"Memories: {stats['memories_extracted']}")
    if streaming_refinement_enabled and "streaming_refinement" in stats:
        sref = stats["streaming_refinement"]
        print(
            "[StreamingRefine] "
            f"flushes={sref.get('flushes', 0)} "
            f"updated_memories={sref.get('updated_memories', 0)} "
            f"fallback_raw={sref.get('fallback_raw_memories', 0)}"
        )
    
    # Memory Refinement (optional)
    if config.enable_memory_refinement:
        print(f"\n=== Memory Refinement Phase ===")
        memory_path = memory_paths.get(config.memory_strategy, memory_paths["all"])
        refined_path = memory_path.replace(".jsonl", "_refined.jsonl")
        
        refined_count = _run_memory_refinement(
            memory_path=memory_path,
            refined_path=refined_path,
            results_path=results_path,
            extractor_model=extractor_model,
        )
        
        print(f"Refined {refined_count} memories -> {refined_path}")
        summary["refined_memory_path"] = refined_path
        summary["refined_memory_count"] = refined_count
        
        # Update summary
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
    
    return summary


# ============================================================
# Memory Retrieval Configuration (Embedding + Bandit)
# ============================================================

def _default_bandit_state_path(
    config: ExperimentConfig,
    memory_path: Optional[str],
    experiment_tag: str,
    phase_dir: str,
) -> str:
    memory_name = os.path.basename(memory_path or "memory").replace(".jsonl", "")
    memory_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", memory_name) or "memory"
    return os.path.join(
        phase_dir,
        f"bandit_state_{experiment_tag}_{memory_name}.json",
    )


def _configure_memory_bank_for_retrieval(
    memory_bank: MemoryBank,
    config: ExperimentConfig,
    memory_path: Optional[str],
    experiment_tag: str,
    phase_dir: str,
) -> None:
    if config.memory_retrieval == "embedding":
        ok = memory_bank.enable_embedding_retrieval(
            model_name=config.embedding_model,
            cache_path=config.embedding_cache_path,
            device=config.embedding_device,
            batch_size=config.embedding_batch_size,
        )
        if not ok:
            print("Warning: Embedding retrieval unavailable, falling back to TF.")
        else:
            cache_path = getattr(memory_bank, "_embedding_cache_path", None)
            if cache_path:
                print(
                    f"[Embedding] cache_path={cache_path} exists={os.path.exists(cache_path)}"
                )
    if config.enable_memory_bandit:
        state_path = config.memory_bandit_state_path or _default_bandit_state_path(
            config=config,
            memory_path=memory_path,
            experiment_tag=experiment_tag,
            phase_dir=phase_dir,
        )
        ok_bandit = memory_bank.enable_bandit(
            lambda_weight=config.memory_bandit_lambda,
            noise_var=config.memory_bandit_noise_var,
            explore_epsilon=config.memory_bandit_explore_epsilon,
            neighbor_k=config.memory_bandit_neighbor_k,
            prior_mu=config.memory_bandit_prior_mu,
            prior_sigma2=config.memory_bandit_prior_sigma2,
            state_path=state_path,
            random_seed=config.memory_bandit_seed,
        )
        if ok_bandit:
            print(
                "[Bandit] Enabled SA-CTS "
                f"(lambda={config.memory_bandit_lambda}, "
                f"noise_var={config.memory_bandit_noise_var}, "
                f"state={state_path})"
            )


def _memory_bandit_is_enabled(memory_bank: Optional[MemoryBank]) -> bool:
    if memory_bank is None:
        return False
    checker = getattr(memory_bank, "is_bandit_enabled", None)
    if callable(checker):
        try:
            return bool(checker())
        except Exception:
            return False
    return bool(getattr(memory_bank, "_bandit_enabled", False))


def _update_bandit_with_advantage(
    memory_bank: Optional[MemoryBank],
    retrieved_memories: List[MemoryItem],
    reward_memory: float,
    reward_base: float,
    debug_memory_prompt: bool = False,
) -> None:
    if memory_bank is None or not retrieved_memories:
        return
    if not _memory_bandit_is_enabled(memory_bank):
        return
    try:
        advantage = float(reward_memory) - float(reward_base)
    except Exception:
        return
    updated = memory_bank.update_bandit_posteriors(
        memories=retrieved_memories,
        advantage_reward=advantage,
    )
    if debug_memory_prompt and updated > 0:
        print(
            f"  [Bandit] Updated {updated} posteriors "
            f"with advantage={advantage:.4f} (mem={reward_memory:.4f}, base={reward_base:.4f})"
        )


# ============================================================
# Phase 2: Evaluation
# ============================================================

def run_advancedif_evaluation(
    config: ExperimentConfig,
    memory_path: Optional[str] = None,
    experiment_tag: str = "eval",
    eval_max_tokens: Optional[int] = None,
    debug_memory_prompt: bool = False,
) -> Dict[str, Any]:
    print(f"\n=== AdvancedIF Evaluation: {experiment_tag} ===")
    is_baseline = memory_path is None
    is_rubrichub = any(
        _is_rubrichub_dataset(path)
        for path in (config.dataset_path, config.train_dataset_path, config.test_dataset_path)
        if path
    )

    output_dir = os.path.join(config.output_dir, config.experiment_name, "evaluation")
    os.makedirs(output_dir, exist_ok=True)
    results_path = os.path.join(output_dir, f"results_{experiment_tag}.jsonl")
    rank, world, _is_main = _dist_info()
    if world > 1:
        results_path = results_path.replace(".jsonl", f".rank{rank}.jsonl")

    student = load_model(
        config.student_model,
        timeout=config.api_timeout,
        max_retries=config.api_retries,
    )
    judge_model = load_model(
        config.judge_model,
        timeout=config.api_timeout,
        max_retries=config.api_retries,
    )

    memory_bank = None
    if memory_path and os.path.exists(memory_path):
        memory_bank = MemoryBank(memory_path, top_k=config.memory_top_k_preference)
        loaded = memory_bank.load(
            memory_path,
            max_items=config.memory_max_items if config.memory_max_items > 0 else None,
        )
        print(f"Loaded {loaded} memories from {memory_path}")
        _configure_memory_bank_for_retrieval(
            memory_bank=memory_bank,
            config=config,
            memory_path=memory_path,
            experiment_tag=experiment_tag,
            phase_dir=output_dir,
        )

    train_dataset, test_dataset, split_info = _load_split_datasets(config)
    split_path = os.path.join(config.output_dir, config.experiment_name, "data_split.json")
    _write_split_manifest(split_path, split_info)
    split_name = (config.data_split_name or "test").strip().lower()
    dataset = train_dataset if split_name == "train" else test_dataset
    if not dataset:
        print("No samples available for AdvancedIF evaluation.")
        return {
            "experiment_tag": experiment_tag,
            "memory_path": memory_path,
            "stats": {},
            "rubric_accuracy": 0.0,
            "rubric_score_sum": 0.0,
            "rubric_count": 0,
            "data_split": _summarize_split_info(split_info),
            "data_split_path": split_path,
            "timestamp": datetime.utcnow().isoformat(),
        }

    print(f"Evaluating on {len(dataset)} samples (split: {split_info.get('mode')}, subset: {split_name})")

    stats: Dict[str, Any] = {
        "total": 0,
        "rubrics_total": 0,
        "rubrics_scored": 0,
        "score_1": 0,
        "score_05": 0,
        "score_0": 0,
        "judge_errors": 0,
        "missing_rubrics": 0,
        "memory_retrieved": 0,
    }
    total_score_sum = 0.0
    total_rubrics = 0
    total_weight_sum = 0.0
    total_weighted_sum = 0.0
    debug_remaining = (
        int(config.advancedif_debug_judge_max_samples)
        if config.advancedif_debug_judge
        else 0
    )

    stop_sequences = [
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
    use_gemini_student = "gemini" in (config.student_model or "").lower()
    student_temperature = 0.5 if use_gemini_student else config.temperature
    student_stop = (
        None if use_gemini_student or _disable_stop_for_aime(config.task_type) else stop_sequences
    )

    for idx, sample in enumerate(dataset):
        if world > 1 and (idx % world) != rank:
            continue
        print(f"\n[{idx+1}/{len(dataset)}] {sample.id}")
        start_time = time.time()

        if is_rubrichub:
            rubrics, rubric_weights = _extract_rubrichub_rubrics(sample)
        else:
            rubrics = _extract_advancedif_rubrics(sample)
            rubric_weights = [1.0 for _ in rubrics]
        if not rubrics:
            stats["missing_rubrics"] += 1
            continue

        seen_rubrics: Set[str] = set()
        filtered_rubrics: List[str] = []
        filtered_weights: List[float] = []
        for idx_r, rubric in enumerate(rubrics):
            weight = rubric_weights[idx_r] if idx_r < len(rubric_weights) else 1.0
            cleaned = _clean_rubric_text(rubric, max_chars=400)
            if not cleaned:
                continue
            key = cleaned.lower()
            if key in seen_rubrics:
                continue
            seen_rubrics.add(key)
            filtered_rubrics.append(cleaned)
            filtered_weights.append(float(weight))
        if not filtered_rubrics:
            stats["missing_rubrics"] += 1
            continue

        user_prompt = _build_user_prompt(
            sample.question,
            config.task_type,
            config.prompt_profile,
            hotpotqa_short_prompt_eval=config.hotpotqa_short_prompt_eval,
        )

        retrieved = []
        if memory_bank is not None and not is_baseline:
            retrieval_query = user_prompt or sample.question
            if config.memory_one_per_dimension:
                candidate_k = max(
                    config.memory_top_k_preference,
                    len(HELPSTEER2_DIMENSIONS) * 2,
                )
                scored = memory_bank.retrieve_with_scores(
                    retrieval_query,
                    top_k=candidate_k,
                    exclude_sample_id=sample.id if config.exclude_self_memory else None,
                    expected_answer=None,
                    max_similarity=config.memory_max_similarity,
                    min_similarity=config.memory_min_similarity,
                    memory_types="preference_rule",
                    retrieval_method=config.memory_retrieval,
                )
                seen_dimensions: Set[str] = set()
                for item in scored:
                    dimension = (getattr(item.item, "dimension", "") or "").strip().lower()
                    if not dimension:
                        dimension = "__unknown__"
                    if dimension in seen_dimensions:
                        continue
                    seen_dimensions.add(dimension)
                    retrieved.append(item.item)
                    if len(retrieved) >= config.memory_top_k_preference:
                        break
            else:
                retrieved = memory_bank.retrieve(
                    retrieval_query,
                    top_k=config.memory_top_k_preference,
                    exclude_sample_id=sample.id if config.exclude_self_memory else None,
                    expected_answer=None,
                    max_similarity=config.memory_max_similarity,
                    min_similarity=config.memory_min_similarity,
                    memory_types="preference_rule",
                    retrieval_method=config.memory_retrieval,
                )
            if retrieved:
                stats["memory_retrieved"] += 1

        base_system_prompt = _build_base_system_prompt(
            config.task_type,
            config.prompt_profile,
            config.hotpotqa_short_prompt_eval,
        )
        if is_baseline or memory_bank is None:
            system_prompt = base_system_prompt
        else:
            base_prompt = _build_base_system_prompt_with_memory(
                config.task_type,
                config.prompt_profile,
                config.hotpotqa_short_prompt_eval,
            )
            system_prompt = memory_bank.build_system_prompt(
                base_prompt, retrieved, prompt_profile=config.prompt_profile
            )
            if debug_memory_prompt and retrieved:
                titles = [m.title for m in retrieved[:3]]
                print(f"  [Debug] Memories retrieved: {len(retrieved)} | titles={titles}")

        try:
            response = student.generate(
                prompt=user_prompt,
                system_prompt=system_prompt,
                temperature=student_temperature,
                max_tokens=eval_max_tokens or config.cascade_max_tokens,
                stop=student_stop,
            )
            response_text = _clean_advancedif_response(response.text or "")
            if len(response_text) < ADVANCEDIF_MIN_RESPONSE_CHARS:
                response = student.generate(
                    prompt=user_prompt,
                    system_prompt=system_prompt,
                    temperature=student_temperature,
                    max_tokens=eval_max_tokens or config.cascade_max_tokens,
                    stop=None,
                )
                response_text = _clean_advancedif_response(response.text or "")
        except Exception as exc:
            print(f"  Error: {exc}")
            response_text = f"Error: {exc}"

        stats["total"] += 1
        stats["rubrics_total"] += len(filtered_rubrics)
        total_rubrics += len(filtered_rubrics)

        debug_for_sample = debug_remaining > 0
        debug_holder: List[str] = []
        adaptive_tokens = max(config.advancedif_judge_max_tokens, 32 + 8 * len(filtered_rubrics))
        score_values = _score_values_for_sample(sample)
        scores = _judge_advancedif_scores(
            judge_model=judge_model,
            prompt=user_prompt,
            response=response_text,
            rubrics=filtered_rubrics,
            max_tokens=adaptive_tokens,
            debug_out=debug_holder,
            allowed_scores=score_values,
        )
        if (not scores or len(scores) != len(filtered_rubrics)) and adaptive_tokens < 512:
            scores = _judge_advancedif_scores(
                judge_model=judge_model,
                prompt=user_prompt,
                response=response_text,
                rubrics=filtered_rubrics,
                max_tokens=512,
                debug_out=debug_holder,
                allowed_scores=score_values,
            )

        if debug_for_sample:
            debug_remaining -= 1

        sample_judge_error = len(scores) != len(filtered_rubrics) or not scores
        if len(scores) != len(filtered_rubrics):
            # Retry scoring only the missing rubrics (assume scores align from the start).
            if len(scores) < len(filtered_rubrics):
                missing_rubrics = filtered_rubrics[len(scores):]
                retry_tokens = max(config.advancedif_judge_max_tokens, 32 + 8 * len(missing_rubrics))
                retry_scores = _judge_advancedif_scores(
                    judge_model=judge_model,
                    prompt=user_prompt,
                    response=response_text,
                    rubrics=missing_rubrics,
                    max_tokens=retry_tokens,
                    debug_out=None,
                    allowed_scores=score_values,
                )
                if len(retry_scores) != len(missing_rubrics) and retry_tokens < 512:
                    retry_scores = _judge_advancedif_scores(
                        judge_model=judge_model,
                        prompt=user_prompt,
                        response=response_text,
                        rubrics=missing_rubrics,
                        max_tokens=512,
                        debug_out=None,
                        allowed_scores=score_values,
                    )
                if len(retry_scores) >= len(missing_rubrics):
                    retry_scores = retry_scores[:len(missing_rubrics)]
                else:
                    retry_scores = retry_scores + [0.0] * (len(missing_rubrics) - len(retry_scores))
                scores = scores + retry_scores
            if len(scores) > len(filtered_rubrics):
                scores = scores[:len(filtered_rubrics)]
            if len(scores) != len(filtered_rubrics):
                stats["judge_errors"] += abs(len(filtered_rubrics) - len(scores))
                if len(scores) < len(filtered_rubrics):
                    scores = scores + [0.0] * (len(filtered_rubrics) - len(scores))

        rubric_scores: List[Dict[str, Any]] = []
        sample_score_sum = 0.0
        sample_weight_sum = 0.0
        sample_weighted_sum = 0.0
        for idx_r, (rubric_text, score) in enumerate(zip(filtered_rubrics, scores)):
            weight = filtered_weights[idx_r] if idx_r < len(filtered_weights) else 1.0
            if score is None:
                stats["judge_errors"] += 1
                sample_judge_error = True
                score = 0.0
            if score >= 0.99:
                stats["score_1"] += 1
            elif score <= 0.01:
                stats["score_0"] += 1
            else:
                stats["score_05"] += 1
            stats["rubrics_scored"] += 1
            sample_score_sum += score
            total_score_sum += score
            sample_weight_sum += weight
            sample_weighted_sum += score * weight
            total_weight_sum += weight
            total_weighted_sum += score * weight
            rubric_scores.append(
                {
                    **({"weight": weight} if is_rubrichub else {}),
                    "rubric": rubric_text,
                    "score": score,
                }
            )

        base_response_text = None
        base_reward = None
        bandit_advantage = None
        if (
            not is_baseline
            and retrieved
            and _memory_bandit_is_enabled(memory_bank)
        ):
            try:
                base_response = student.generate(
                    prompt=user_prompt,
                    system_prompt=base_system_prompt,
                    temperature=student_temperature,
                    max_tokens=eval_max_tokens or config.cascade_max_tokens,
                    stop=student_stop,
                )
                base_response_text = _clean_advancedif_response(base_response.text or "")
                if len(base_response_text) < ADVANCEDIF_MIN_RESPONSE_CHARS:
                    base_response = student.generate(
                        prompt=user_prompt,
                        system_prompt=base_system_prompt,
                        temperature=student_temperature,
                        max_tokens=eval_max_tokens or config.cascade_max_tokens,
                        stop=None,
                    )
                    base_response_text = _clean_advancedif_response(base_response.text or "")
                base_scores = _judge_advancedif_scores(
                    judge_model=judge_model,
                    prompt=user_prompt,
                    response=base_response_text,
                    rubrics=filtered_rubrics,
                    max_tokens=adaptive_tokens,
                    debug_out=None,
                    allowed_scores=score_values,
                )
                if len(base_scores) < len(filtered_rubrics):
                    base_scores = base_scores + [0.0] * (len(filtered_rubrics) - len(base_scores))
                elif len(base_scores) > len(filtered_rubrics):
                    base_scores = base_scores[:len(filtered_rubrics)]
                base_reward = (
                    sum(float(s or 0.0) for s in base_scores) / len(filtered_rubrics)
                    if filtered_rubrics else 0.0
                )
                memory_reward = sample_score_sum / len(filtered_rubrics) if filtered_rubrics else 0.0
                bandit_advantage = memory_reward - base_reward
                _update_bandit_with_advantage(
                    memory_bank=memory_bank,
                    retrieved_memories=retrieved,
                    reward_memory=memory_reward,
                    reward_base=base_reward,
                    debug_memory_prompt=debug_memory_prompt,
                )
            except Exception as exc:
                if debug_memory_prompt:
                    print(f"  [Bandit] AdvancedIF base scoring failed: {exc}")

        duration = time.time() - start_time
        result = {
            "sample_id": sample.id,
            "prompt": user_prompt,
            "response": response_text,
            "rubric_scores": rubric_scores,
            "sample_score_sum": round(sample_score_sum, 4),
            "sample_score_avg": round(sample_score_sum / len(filtered_rubrics), 4),
            "duration_sec": round(duration, 2),
        }
        if is_rubrichub:
            weighted_avg = (sample_weighted_sum / sample_weight_sum) if sample_weight_sum else 0.0
            result.update(
                {
                    "sample_weighted_sum": round(sample_weighted_sum, 4),
                    "sample_weighted_avg": round(weighted_avg, 4),
                    "sample_weighted_pct": round(weighted_avg, 4),
                    "sample_weight_total": round(sample_weight_sum, 4),
                }
            )
        if (sample_judge_error or debug_for_sample) and debug_holder:
            result["judge_raw"] = debug_holder[0]
        if base_response_text is not None:
            result["bandit_base_response"] = base_response_text
        if base_reward is not None:
            result["bandit_base_reward"] = float(base_reward)
        if bandit_advantage is not None:
            result["bandit_advantage"] = float(bandit_advantage)
        _append_jsonl(results_path, result)
        if is_rubrichub:
            pct = (sample_weighted_sum / sample_weight_sum) if sample_weight_sum else 0.0
            print(
                f"  Rubric score (weighted): {100 * pct:.1f}% "
                f"({sample_weighted_sum:.2f}/{sample_weight_sum:.2f})"
            )
        else:
            print(
                f"  Rubric score: {100 * (sample_score_sum / len(filtered_rubrics)):.1f}% "
                f"({sample_score_sum:.2f}/{len(filtered_rubrics)})"
            )

    rubric_accuracy = total_score_sum / total_rubrics if total_rubrics > 0 else 0.0
    weighted_accuracy = total_weighted_sum / total_weight_sum if total_weight_sum > 0 else 0.0
    summary = {
        "experiment_tag": experiment_tag,
        "memory_path": memory_path,
        "stats": stats,
        "rubric_accuracy": weighted_accuracy if is_rubrichub else rubric_accuracy,
        "rubric_score_sum": total_weighted_sum if is_rubrichub else total_score_sum,
        "rubric_count": total_weight_sum if is_rubrichub else total_rubrics,
        "data_split": _summarize_split_info(split_info),
        "data_split_path": split_path,
        "timestamp": datetime.utcnow().isoformat(),
    }
    if is_rubrichub:
        summary.update(
            {
                "rubric_accuracy_unweighted": rubric_accuracy,
                "rubric_score_sum_unweighted": total_score_sum,
                "rubric_count_unweighted": total_rubrics,
                "rubric_weight_total": total_weight_sum,
            }
        )

    summary_path = os.path.join(output_dir, f"summary_{experiment_tag}.json")
    with open(summary_path, "w") as handle:
        json.dump(summary, handle, indent=2)

    print(f"\n=== AdvancedIF Results: {experiment_tag} ===")
    if is_rubrichub:
        print(
            f"Rubric completion (weighted): {100 * weighted_accuracy:.1f}% "
            f"({total_weighted_sum:.2f}/{total_weight_sum:.2f})"
        )
        print(
            f"Rubric completion (unweighted): {100 * rubric_accuracy:.1f}% "
            f"({total_score_sum:.2f}/{total_rubrics})"
        )
    else:
        print(
            f"Rubric completion: {100 * rubric_accuracy:.1f}% "
            f"({total_score_sum:.2f}/{total_rubrics})"
        )
    return summary


def run_helpsteer2_evaluation(
    config: ExperimentConfig,
    memory_path: Optional[str] = None,
    experiment_tag: str = "eval",
    eval_max_tokens: Optional[int] = None,
    debug_memory_prompt: bool = False,
) -> Dict[str, Any]:
    print(f"\n=== HelpSteer2 Evaluation: {experiment_tag} ===")
    is_baseline = memory_path is None

    output_dir = os.path.join(config.output_dir, config.experiment_name, "evaluation")
    os.makedirs(output_dir, exist_ok=True)
    results_path = os.path.join(output_dir, f"results_{experiment_tag}.jsonl")
    rank, world, _is_main = _dist_info()
    if world > 1:
        results_path = results_path.replace(".jsonl", f".rank{rank}.jsonl")

    student = load_model(
        config.student_model,
        timeout=config.api_timeout,
        max_retries=config.api_retries,
    )
    judge_model = load_model(
        config.judge_model,
        timeout=config.api_timeout,
        max_retries=config.api_retries,
    )

    memory_bank = None
    if memory_path and os.path.exists(memory_path):
        memory_bank = MemoryBank(memory_path, top_k=config.memory_top_k_preference)
        loaded = memory_bank.load(
            memory_path,
            max_items=config.memory_max_items if config.memory_max_items > 0 else None,
        )
        print(f"Loaded {loaded} memories from {memory_path}")
        _configure_memory_bank_for_retrieval(
            memory_bank=memory_bank,
            config=config,
            memory_path=memory_path,
            experiment_tag=experiment_tag,
            phase_dir=output_dir,
        )

    train_dataset, test_dataset, split_info = _load_split_datasets(config)
    split_path = os.path.join(config.output_dir, config.experiment_name, "data_split.json")
    _write_split_manifest(split_path, split_info)
    split_name = (config.data_split_name or "test").strip().lower()
    dataset = train_dataset if split_name == "train" else test_dataset
    if not dataset:
        print("No samples available for HelpSteer2 evaluation.")
        return {
            "experiment_tag": experiment_tag,
            "memory_path": memory_path,
            "stats": {},
            "dimension_score_sum": 0.0,
            "dimension_count": 0,
            "dimension_score_avg": 0.0,
            "dimension_score_pct": 0.0,
            "data_split": _summarize_split_info(split_info),
            "data_split_path": split_path,
            "timestamp": datetime.utcnow().isoformat(),
        }

    print(f"Evaluating on {len(dataset)} samples (split: {split_info.get('mode')}, subset: {split_name})")

    stats: Dict[str, Any] = {
        "total": 0,
        "dimensions_total": 0,
        "dimensions_scored": 0,
        "score_5": 0,
        "score_4": 0,
        "score_3": 0,
        "score_2": 0,
        "score_1": 0,
        "judge_errors": 0,
        "memory_retrieved": 0,
    }
    total_score_sum = 0.0
    total_weighted_sum = 0.0
    total_dimensions = 0
    debug_remaining = (
        int(config.advancedif_debug_judge_max_samples)
        if config.advancedif_debug_judge
        else 0
    )
    stop_sequences = [
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
    use_gemini_student = "gemini" in (config.student_model or "").lower()
    student_temperature = 0.5 if use_gemini_student else config.temperature
    student_stop = (
        None if use_gemini_student or _disable_stop_for_aime(config.task_type) else stop_sequences
    )

    for idx, sample in enumerate(dataset):
        if world > 1 and (idx % world) != rank:
            continue
        print(f"\n[{idx+1}/{len(dataset)}] {sample.id}")
        start_time = time.time()

        user_prompt = _build_user_prompt(
            sample.question,
            config.task_type,
            config.prompt_profile,
            hotpotqa_short_prompt_eval=config.hotpotqa_short_prompt_eval,
        )

        retrieved = []
        if memory_bank is not None and not is_baseline:
            retrieval_query = user_prompt or sample.question
            retrieved = memory_bank.retrieve(
                retrieval_query,
                top_k=config.memory_top_k_preference,
                exclude_sample_id=sample.id if config.exclude_self_memory else None,
                expected_answer=None,
                max_similarity=config.memory_max_similarity,
                min_similarity=config.memory_min_similarity,
                memory_types="preference_rule",
                retrieval_method=config.memory_retrieval,
            )
            if retrieved:
                stats["memory_retrieved"] += 1

        base_system_prompt = _build_base_system_prompt(
            config.task_type,
            config.prompt_profile,
            config.hotpotqa_short_prompt_eval,
        )
        if is_baseline or memory_bank is None:
            system_prompt = base_system_prompt
        else:
            base_prompt = _build_base_system_prompt_with_memory(
                config.task_type,
                config.prompt_profile,
                config.hotpotqa_short_prompt_eval,
            )
            system_prompt = memory_bank.build_system_prompt(
                base_prompt, retrieved, prompt_profile=config.prompt_profile
            )
            if debug_memory_prompt and retrieved:
                titles = [m.title for m in retrieved[:3]]
                print(f"  [Debug] Memories retrieved: {len(retrieved)} | titles={titles}")

        try:
            response = student.generate(
                prompt=user_prompt,
                system_prompt=system_prompt,
                temperature=student_temperature,
                max_tokens=eval_max_tokens or config.cascade_max_tokens,
                stop=student_stop,
            )
            response_text = _clean_advancedif_response(response.text or "")
            if len(response_text) < ADVANCEDIF_MIN_RESPONSE_CHARS:
                response = student.generate(
                    prompt=user_prompt,
                    system_prompt=system_prompt,
                    temperature=student_temperature,
                    max_tokens=eval_max_tokens or config.cascade_max_tokens,
                    stop=None,
                )
                response_text = _clean_advancedif_response(response.text or "")
        except Exception as exc:
            print(f"  Error: {exc}")
            response_text = f"Error: {exc}"

        stats["total"] += 1
        stats["dimensions_total"] += len(HELPSTEER2_DIMENSIONS)
        total_dimensions += len(HELPSTEER2_DIMENSIONS)

        debug_holder: Optional[List[str]] = [] if debug_remaining > 0 else None
        scores = _judge_helpsteer2_scores(
            judge_model=judge_model,
            prompt=user_prompt,
            response=response_text,
            max_tokens=config.advancedif_judge_max_tokens,
            debug_out=debug_holder,
        )
        if debug_remaining > 0:
            debug_remaining -= 1

        if len(scores) != len(HELPSTEER2_DIMENSIONS):
            stats["judge_errors"] += abs(len(HELPSTEER2_DIMENSIONS) - len(scores))
            if len(scores) < len(HELPSTEER2_DIMENSIONS):
                scores = scores + [float(HELPSTEER2_SCORE_VALUES[0])] * (len(HELPSTEER2_DIMENSIONS) - len(scores))
            else:
                scores = scores[:len(HELPSTEER2_DIMENSIONS)]

        dimension_scores: List[Dict[str, Any]] = []
        sample_score_sum = 0.0
        sample_weighted_sum = 0.0
        for dim_label, score in zip(HELPSTEER2_DIMENSIONS, scores):
            if score is None:
                stats["judge_errors"] += 1
                score = float(HELPSTEER2_SCORE_VALUES[0])
            if score >= 4.99:
                stats["score_5"] += 1
            elif score >= 3.99:
                stats["score_4"] += 1
            elif score >= 2.99:
                stats["score_3"] += 1
            elif score >= 1.99:
                stats["score_2"] += 1
            else:
                stats["score_1"] += 1
            stats["dimensions_scored"] += 1
            sample_score_sum += score
            total_score_sum += score
            sample_weighted_sum += score * HELPSTEER2_DIMENSION_WEIGHTS.get(dim_label, 1)
            dimension_scores.append({"dimension": dim_label, "score": score})
        total_weighted_sum += sample_weighted_sum

        base_response_text = None
        base_reward = None
        bandit_advantage = None
        if (
            not is_baseline
            and retrieved
            and _memory_bandit_is_enabled(memory_bank)
        ):
            try:
                base_response = student.generate(
                    prompt=user_prompt,
                    system_prompt=base_system_prompt,
                    temperature=student_temperature,
                    max_tokens=eval_max_tokens or config.cascade_max_tokens,
                    stop=student_stop,
                )
                base_response_text = _clean_advancedif_response(base_response.text or "")
                if len(base_response_text) < ADVANCEDIF_MIN_RESPONSE_CHARS:
                    base_response = student.generate(
                        prompt=user_prompt,
                        system_prompt=base_system_prompt,
                        temperature=student_temperature,
                        max_tokens=eval_max_tokens or config.cascade_max_tokens,
                        stop=None,
                    )
                    base_response_text = _clean_advancedif_response(base_response.text or "")
                base_scores = _judge_helpsteer2_scores(
                    judge_model=judge_model,
                    prompt=user_prompt,
                    response=base_response_text,
                    max_tokens=config.advancedif_judge_max_tokens,
                    debug_out=None,
                )
                if len(base_scores) < len(HELPSTEER2_DIMENSIONS):
                    base_scores = base_scores + [float(HELPSTEER2_SCORE_VALUES[0])] * (
                        len(HELPSTEER2_DIMENSIONS) - len(base_scores)
                    )
                elif len(base_scores) > len(HELPSTEER2_DIMENSIONS):
                    base_scores = base_scores[: len(HELPSTEER2_DIMENSIONS)]
                base_reward = (
                    sum(float(s or 0.0) for s in base_scores) / len(HELPSTEER2_DIMENSIONS)
                    if HELPSTEER2_DIMENSIONS else 0.0
                )
                memory_reward = sample_score_sum / len(HELPSTEER2_DIMENSIONS)
                bandit_advantage = memory_reward - base_reward
                _update_bandit_with_advantage(
                    memory_bank=memory_bank,
                    retrieved_memories=retrieved,
                    reward_memory=memory_reward,
                    reward_base=base_reward,
                    debug_memory_prompt=debug_memory_prompt,
                )
            except Exception as exc:
                if debug_memory_prompt:
                    print(f"  [Bandit] HelpSteer2 base scoring failed: {exc}")

        duration = time.time() - start_time
        result = {
            "sample_id": sample.id,
            "prompt": user_prompt,
            "response": response_text,
            "dimension_scores": dimension_scores,
            "sample_score_sum": round(sample_score_sum, 4),
            "sample_score_avg": round(sample_score_sum / len(HELPSTEER2_DIMENSIONS), 4),
            "sample_weighted_sum": round(sample_weighted_sum, 4),
            "sample_weighted_avg": round(sample_weighted_sum / HELPSTEER2_WEIGHT_TOTAL, 4),
            "sample_weighted_pct": round(
                sample_weighted_sum / (HELPSTEER2_WEIGHT_TOTAL * HELPSTEER2_SCORE_VALUES[-1]),
                4,
            ),
            "duration_sec": round(duration, 2),
        }
        if debug_holder:
            result["judge_raw"] = debug_holder[0]
        if base_response_text is not None:
            result["bandit_base_response"] = base_response_text
        if base_reward is not None:
            result["bandit_base_reward"] = float(base_reward)
        if bandit_advantage is not None:
            result["bandit_advantage"] = float(bandit_advantage)
        _append_jsonl(results_path, result)

        score_pct = sample_score_sum / (len(HELPSTEER2_DIMENSIONS) * HELPSTEER2_SCORE_VALUES[-1])
        print(
            f"  Dimension score: {score_pct*100:.1f}% "
            f"({sample_score_sum:.2f}/{len(HELPSTEER2_DIMENSIONS)*HELPSTEER2_SCORE_VALUES[-1]:.1f})"
        )

    dimension_score_avg = total_score_sum / total_dimensions if total_dimensions else 0.0
    dimension_score_pct = (
        total_score_sum / (total_dimensions * HELPSTEER2_SCORE_VALUES[-1])
        if total_dimensions
        else 0.0
    )
    total_samples = total_dimensions / len(HELPSTEER2_DIMENSIONS) if total_dimensions else 0.0
    weighted_score_avg = (
        total_weighted_sum / (total_samples * HELPSTEER2_WEIGHT_TOTAL)
        if total_samples
        else 0.0
    )
    weighted_score_pct = (
        total_weighted_sum / (total_samples * HELPSTEER2_WEIGHT_TOTAL * HELPSTEER2_SCORE_VALUES[-1])
        if total_samples
        else 0.0
    )
    summary = {
        "experiment_tag": experiment_tag,
        "memory_path": memory_path,
        "stats": stats,
        "dimension_score_sum": round(total_score_sum, 4),
        "dimension_count": total_dimensions,
        "dimension_score_avg": round(dimension_score_avg, 4),
        "dimension_score_pct": round(dimension_score_pct, 4),
        "weighted_score_sum": round(total_weighted_sum, 4),
        "weighted_score_avg": round(weighted_score_avg, 4),
        "weighted_score_pct": round(weighted_score_pct, 4),
        "data_split": _summarize_split_info(split_info),
        "data_split_path": split_path,
        "timestamp": datetime.utcnow().isoformat(),
    }

    summary_path = os.path.join(output_dir, f"summary_{experiment_tag}.json")
    with open(summary_path, "w") as handle:
        json.dump(summary, handle, indent=2)

    print(f"\n=== HelpSteer2 Results: {experiment_tag} ===")
    print(
        f"Dimension score: {100 * dimension_score_pct:.1f}% "
        f"({total_score_sum:.2f}/{total_dimensions * HELPSTEER2_SCORE_VALUES[-1]:.1f})"
    )
    if total_dimensions:
        max_weighted = total_samples * HELPSTEER2_WEIGHT_TOTAL * HELPSTEER2_SCORE_VALUES[-1]
        print(
            f"Weighted score: {100 * weighted_score_pct:.1f}% "
            f"({total_weighted_sum:.2f}/{max_weighted:.1f})"
        )
    return summary

def run_evaluation(
    config: ExperimentConfig,
    memory_path: Optional[str] = None,
    experiment_tag: str = "eval",
    eval_max_tokens: Optional[int] = None,
    debug_memory_prompt: bool = False,
) -> Dict[str, Any]:
    """
    Evaluate student model with or without memory.
    
    Args:
        config: Experiment configuration
        memory_path: Path to memory file (None for baseline)
        experiment_tag: Tag for this evaluation run
    
    Returns:
        Evaluation results
    """
    if config.prompt_profile == "advancedif":
        return run_advancedif_evaluation(
            config=config,
            memory_path=memory_path,
            experiment_tag=experiment_tag,
            eval_max_tokens=eval_max_tokens,
            debug_memory_prompt=debug_memory_prompt,
        )
    if config.prompt_profile == "helpsteer2":
        return run_helpsteer2_evaluation(
            config=config,
            memory_path=memory_path,
            experiment_tag=experiment_tag,
            eval_max_tokens=eval_max_tokens,
            debug_memory_prompt=debug_memory_prompt,
        )
    is_baseline = memory_path is None
    is_preference = _is_preference_task(config.task_type, config.prompt_profile)
    print(f"\n=== Evaluation Phase: {experiment_tag} ===")
    print(f"Memory: {'None (Baseline)' if is_baseline else memory_path}")
    
    # Setup output
    output_dir = os.path.join(config.output_dir, config.experiment_name, "evaluation")
    os.makedirs(output_dir, exist_ok=True)
    results_path = os.path.join(output_dir, f"results_{experiment_tag}.jsonl")
    rank, world, is_main = _dist_info()
    if world > 1:
        results_path = results_path.replace(".jsonl", f".rank{rank}.jsonl")
    
    # Initialize models
    student = load_model(
        config.student_model,
        timeout=config.api_timeout,
        max_retries=config.api_retries,
    )
    judge_model = None
    judge = None
    if not is_preference:
        judge_model = load_model(
            config.judge_model,
            timeout=config.api_timeout,
            max_retries=config.api_retries,
        )
        judge = create_judge(
            judge_model, task_type=config.task_type, prompt_profile=config.prompt_profile
        )
    
    # Initialize memory bank
    memory_bank = MemoryBank(memory_path or "", top_k=config.memory_top_k)
    if memory_path and os.path.exists(memory_path):
        loaded = memory_bank.load(
            memory_path,
            max_items=config.memory_max_items if config.memory_max_items > 0 else None,
        )
        print(f"Loaded {loaded} memories from {memory_path}")
        _configure_memory_bank_for_retrieval(
            memory_bank=memory_bank,
            config=config,
            memory_path=memory_path,
            experiment_tag=experiment_tag,
            phase_dir=output_dir,
        )
    
    # Load dataset (test split)
    _, test_dataset, split_info = _load_split_datasets(config)
    dataset = test_dataset
    split_path = os.path.join(config.output_dir, config.experiment_name, "data_split.json")
    _write_split_manifest(split_path, split_info)
    if not dataset:
        print("No test samples available. Adjust split_ratio or provide --test_dataset.")
        return {
            "experiment_tag": experiment_tag,
            "memory_path": memory_path,
            "stats": {"total": 0, "correct": 0, "memory_used": 0, "memory_retrieved": 0},
            "accuracy": 0.0,
            "em": None,
            "f1": None,
            "memory_utilization": 0.0,
            "data_split": _summarize_split_info(split_info),
            "data_split_path": split_path,
            "timestamp": datetime.utcnow().isoformat(),
        }
    print(f"Evaluating on {len(dataset)} samples (split: {split_info['mode']})")
    
    # Statistics
    stats: Dict[str, Any] = {
        "total": 0,
        "correct": 0,
        "memory_used": 0,
        "memory_retrieved": 0,
    }
    is_hotpotqa = config.prompt_profile == "hotpotqa"
    if is_hotpotqa:
        stats["em_sum"] = 0.0
        stats["f1_sum"] = 0.0
    if is_preference:
        stats["skipped_ties"] = 0
        stats["skipped_missing"] = 0
    
    # Process each sample
    for idx, sample in enumerate(dataset):
        if world > 1 and (idx % world) != rank:
            continue
        print(f"\n[{idx+1}/{len(dataset)}] {sample.id}")
        start_time = time.time()

        if is_preference:
            context, r1, r2, pref_val, _dimension_hint = _extract_preference_fields(sample)
            if not r1 or not r2 or pref_val is None:
                stats["skipped_missing"] += 1
                continue
            if pref_val == 0:
                stats["skipped_ties"] += 1
                continue

            expected = -1 if pref_val < 0 else 1
            # Preference retrieval: use context only to match trigger-focused memories
            retrieval_query = context
            expanded_query = None
            if not is_baseline and config.memory_query_expand:
                expanded_query = _expand_memory_query(
                    student_model=student,
                    question=retrieval_query,
                    max_tokens=config.memory_query_expand_max_tokens,
                    prompt_profile=config.prompt_profile,
                )
                if expanded_query:
                    retrieval_query = f"{retrieval_query}\nKey concepts: {expanded_query}"

            retrieved = []
            if not is_baseline:
                retrieved = memory_bank.retrieve(
                    retrieval_query,
                    top_k=config.memory_top_k_preference,
                    exclude_sample_id=sample.id if config.exclude_self_memory else None,
                    expected_answer=None,
                    max_similarity=config.memory_max_similarity,
                    min_similarity=config.memory_min_similarity,
                    memory_types="preference_rule",
                    retrieval_method=config.memory_retrieval,
                )
            if retrieved:
                stats["memory_retrieved"] += 1

            base_system_prompt = _build_base_system_prompt(
                config.task_type,
                config.prompt_profile,
                config.hotpotqa_short_prompt_eval,
            )
            if is_baseline:
                system_prompt = base_system_prompt
            else:
                base_prompt = _build_base_system_prompt_with_memory(
                    config.task_type,
                    config.prompt_profile,
                    config.hotpotqa_short_prompt_eval,
                )
                system_prompt = memory_bank.build_system_prompt(
                    base_prompt, retrieved, prompt_profile=config.prompt_profile
                )
                if debug_memory_prompt:
                    titles = [m.title for m in retrieved[:3]]
                    print(f"  [Debug] Memories retrieved: {len(retrieved)} | titles={titles}")
                    if expanded_query:
                        print(f"  [Debug] Retrieval expansion: {expanded_query[:200]}")
                    backend = getattr(memory_bank, "_last_retrieval_backend", None)
                    fallback = getattr(memory_bank, "_last_retrieval_fallback", None)
                    print(
                        f"  [Debug] Retrieval backend={backend} fallback={fallback} "
                        f"min_sim={config.memory_min_similarity} max_sim={config.memory_max_similarity}"
                    )
                    scored = memory_bank.retrieve_with_scores(
                        retrieval_query,
                        top_k=min(5, config.memory_top_k_preference),
                        exclude_sample_id=sample.id if config.exclude_self_memory else None,
                        expected_answer=None,
                        max_similarity=config.memory_max_similarity,
                        min_similarity=config.memory_min_similarity,
                        memory_types="preference_rule",
                        retrieval_method=config.memory_retrieval,
                    )
                    if scored:
                        top_scores = ", ".join(
                            f"{r.score:.4f} | {r.item.title[:60]}" for r in scored
                        )
                        print(f"  [Debug] Top scores: {top_scores}")

            user_prompt = _build_preference_user_prompt(context, r1, r2)
            base_response_text = None
            base_predicted = None
            base_is_correct = None
            try:
                response = student.generate(
                    prompt=user_prompt,
                    system_prompt=system_prompt,
                    temperature=config.temperature,
                    max_tokens=eval_max_tokens,
                )
                response_text = response.text
            except Exception as e:
                print(f"  Error: {e}")
                response_text = f"Error: {e}"

            predicted = _parse_preference_choice(response_text)
            is_correct = predicted == expected
            if (
                not is_baseline
                and retrieved
                and _memory_bandit_is_enabled(memory_bank)
            ):
                try:
                    base_response = student.generate(
                        prompt=user_prompt,
                        system_prompt=base_system_prompt,
                        temperature=config.temperature,
                        max_tokens=eval_max_tokens,
                    )
                    base_response_text = base_response.text
                    base_predicted = _parse_preference_choice(base_response_text)
                    base_is_correct = base_predicted == expected
                except Exception as exc:
                    if debug_memory_prompt:
                        print(f"  [Bandit] Failed to compute base response: {exc}")
            stats["total"] += 1
            if is_correct:
                stats["correct"] += 1

            memory_used = False
            if retrieved:
                memory_used = declared_memory_use(response_text)
                if memory_used:
                    stats["memory_used"] += 1

            bandit_advantage = None
            if base_is_correct is not None and retrieved:
                reward_memory = 1.0 if is_correct else 0.0
                reward_base = 1.0 if base_is_correct else 0.0
                bandit_advantage = reward_memory - reward_base
                _update_bandit_with_advantage(
                    memory_bank=memory_bank,
                    retrieved_memories=retrieved,
                    reward_memory=reward_memory,
                    reward_base=reward_base,
                    debug_memory_prompt=debug_memory_prompt,
                )

            duration = time.time() - start_time
            result = {
                "sample_id": sample.id,
                "context": context,
                "response1": r1,
                "response2": r2,
                "overall_preference": pref_val,
                "expected_choice": "response1" if expected < 0 else "response2",
                "predicted_choice": (
                    "response1" if predicted == -1 else "response2" if predicted == 1 else "unknown"
                ),
                "retrieved_memory": [_format_retrieved_memory(m) for m in retrieved],
                "response": response_text,
                "correct": is_correct,
                "memory_used": memory_used,
                "duration_sec": round(duration, 2),
            }
            if base_response_text is not None:
                result["bandit_base_response"] = base_response_text
            if base_predicted is not None:
                result["bandit_base_predicted_choice"] = (
                    "response1" if base_predicted == -1 else "response2" if base_predicted == 1 else "unknown"
                )
            if base_is_correct is not None:
                result["bandit_base_correct"] = bool(base_is_correct)
            if bandit_advantage is not None:
                result["bandit_advantage"] = float(bandit_advantage)
            _append_jsonl(results_path, result)

            print(
                f"  Expected: {'Response 1' if expected < 0 else 'Response 2'} | "
                f"Predicted: {'Response 1' if predicted == -1 else 'Response 2' if predicted == 1 else 'Unknown'} | "
                f"{'✓' if is_correct else '✗'}"
            )
            continue

        retrieval_query = sample.question
        expanded_query = None
        if not is_baseline and config.memory_query_expand:
            expanded_query = _expand_memory_query(
                student_model=student,
                question=sample.question,
                max_tokens=config.memory_query_expand_max_tokens,
            )
            if expanded_query:
                retrieval_query = f"{sample.question}\nKey concepts: {expanded_query}"

        # Retrieve memory
        retrieved = []
        retrieved_global = []
        retrieved_local = []
        retrieved_global_scored = None
        retrieved_local_scored = None
        global_backend = None
        global_fallback = None
        local_backend = None
        local_fallback = None
        if not is_baseline:
            retrieved_global_scored = memory_bank.retrieve_with_scores(
                retrieval_query,
                top_k=config.memory_top_k_global,
                exclude_sample_id=sample.id if config.exclude_self_memory else None,
                expected_answer=sample.answer if config.memory_answer_leak_filter else None,
                max_similarity=config.memory_max_similarity,
                min_similarity=config.memory_min_similarity,
                memory_types="global_correct_flow",
                retrieval_method=config.memory_retrieval,
            )
            global_backend = getattr(memory_bank, "_last_retrieval_backend", None)
            global_fallback = getattr(memory_bank, "_last_retrieval_fallback", None)

            retrieved_local_scored = memory_bank.retrieve_with_scores(
                retrieval_query,
                top_k=config.memory_top_k_local,
                exclude_sample_id=sample.id if config.exclude_self_memory else None,
                expected_answer=sample.answer if config.memory_answer_leak_filter else None,
                max_similarity=config.memory_max_similarity,
                min_similarity=config.memory_min_similarity,
                memory_types="local_corrective",
                retrieval_method=config.memory_retrieval,
            )
            local_backend = getattr(memory_bank, "_last_retrieval_backend", None)
            local_fallback = getattr(memory_bank, "_last_retrieval_fallback", None)

            retrieved_global = [r.item for r in retrieved_global_scored]
            retrieved_local = [r.item for r in retrieved_local_scored]

            combined_scored = []
            if retrieved_global_scored:
                combined_scored.extend(retrieved_global_scored)
            if retrieved_local_scored:
                combined_scored.extend(retrieved_local_scored)

            if combined_scored:
                combined_scored.sort(key=lambda r: r.score, reverse=True)
                seen_keys = set()
                deduped = []
                for r in combined_scored:
                    item = r.item
                    if item.type == "preference_rule":
                        key = (
                            "preference_rule",
                            (item.trigger or "").strip(),
                            (item.dimension or "").strip(),
                            (item.comparison or "").strip(),
                        )
                    else:
                        key = (item.type, item.title, item.description, item.content)
                    if key in seen_keys:
                        continue
                    seen_keys.add(key)
                    deduped.append(r)
                combined_scored = deduped

            final_k = int(config.memory_top_k) if config.memory_top_k else 0
            if final_k > 0 and combined_scored:
                combined_scored = combined_scored[:final_k]

            retrieved = [r.item for r in combined_scored]
        
        if retrieved:
            stats["memory_retrieved"] += 1
        
        # Build prompt
        # [FIX] 使用修复后的 system prompt
        base_system_prompt = _build_base_system_prompt(
            config.task_type,
            config.prompt_profile,
            config.hotpotqa_short_prompt_eval,
        )
        base_system_prompt = _apply_eval_instruction(base_system_prompt, config.eval_instruction)
        if is_baseline:
            system_prompt = base_system_prompt
        else:
            base_prompt = _build_base_system_prompt_with_memory(
                config.task_type,
                config.prompt_profile,
                config.hotpotqa_short_prompt_eval,
            )
            base_prompt = _apply_eval_instruction(base_prompt, config.eval_instruction)
            system_prompt = memory_bank.build_system_prompt(
                base_prompt, retrieved, prompt_profile=config.prompt_profile
            )
            if debug_memory_prompt:
                # 检查memory是否被注入（QA和数学任务使用不同的标题）
                injected = (
                    "Retrieved Problem-Solving Strategies" in system_prompt or
                    "Hints from Similar Questions" in system_prompt
                )
                titles = [m.title for m in retrieved[:3]]
                print(f"  [Debug] Memories retrieved: {len(retrieved)} | injected={injected} | titles={titles}")
                if expanded_query:
                    print(f"  [Debug] Retrieval expansion: {expanded_query[:200]}")
                retrieval_backend = config.memory_retrieval
                embedding_ready = getattr(memory_bank, "_embedding_enabled", False)
                bandit_enabled = _memory_bandit_is_enabled(memory_bank)
                print(
                    f"  [Debug] Retrieval backend: {retrieval_backend} | embedding_ready={embedding_ready} "
                    f"| bandit_enabled={bandit_enabled} "
                    f"| min_sim={config.memory_min_similarity} | max_sim={config.memory_max_similarity} "
                    f"| leak_filter={config.memory_answer_leak_filter} "
                    f"| fallback_to_tf(global/local)={global_fallback}/{local_fallback} "
                    f"| actual_backend(global/local)={global_backend}/{local_backend}"
                )
                if retrieved_global_scored is not None:
                    global_scores = "; ".join(
                        f"{r.score:.4f} | {r.item.title[:60]}" for r in retrieved_global_scored
                    )
                    print(f"  [Debug] Global similarities: {global_scores or 'none'}")
                    if retrieved_global_scored:
                        g_vals = [r.score for r in retrieved_global_scored]
                        g_min = min(g_vals)
                        g_max = max(g_vals)
                        g_mean = sum(g_vals) / len(g_vals)
                        print(f"  [Debug] Global similarity stats: min={g_min:.4f} mean={g_mean:.4f} max={g_max:.4f}")
                if retrieved_local_scored is not None:
                    local_scores = "; ".join(
                        f"{r.score:.4f} | {r.item.title[:60]}" for r in retrieved_local_scored
                    )
                    print(f"  [Debug] Local similarities: {local_scores or 'none'}")
                    if retrieved_local_scored:
                        l_vals = [r.score for r in retrieved_local_scored]
                        l_min = min(l_vals)
                        l_max = max(l_vals)
                        l_mean = sum(l_vals) / len(l_vals)
                        print(f"  [Debug] Local similarity stats: min={l_min:.4f} mean={l_mean:.4f} max={l_max:.4f}")
        
        # [FIX] 使用与 test_fixed.py 一致的 prompt 格式
        user_prompt = _build_user_prompt(
            sample.question,
            config.task_type,
            config.prompt_profile,
            hotpotqa_short_prompt_eval=config.hotpotqa_short_prompt_eval,
        )
        base_response_text = None
        base_parsed_answer = None
        base_judge_result = None
        base_is_correct = None
        
        # Generate response
        try:
            response = student.generate(
                prompt=user_prompt,
                system_prompt=system_prompt,
                temperature=config.temperature,
                max_tokens=eval_max_tokens,
            )
            response_text = response.text
        except Exception as e:
            print(f"  Error: {e}")
            response_text = f"Error: {e}"
        
        # Parse answer
        parsed_answer = _parse_final_answer(
            response_text,
            task_type=config.task_type,
            prompt_profile=config.prompt_profile,
            answer_only=True,
        )
        if config.llm_extract_answer:
            llm_answer = _llm_extract_answer(judge_model, response_text, config.task_type)
            if llm_answer is not None:
                parsed_answer = llm_answer
        
        # Judge
        judge_result = judge.judge(
            question=sample.question,
            response=response_text,
            parsed_answer=parsed_answer,
            ground_truth=sample.answer,
        )
        
        is_correct = judge_result["label"] == "success"
        if is_hotpotqa:
            details = judge_result.get("details") or {}
            stats["em_sum"] += float(details.get("em") or 0.0)
            stats["f1_sum"] += float(details.get("f1") or 0.0)
        stats["total"] += 1
        if is_correct:
            stats["correct"] += 1

        bandit_advantage = None
        if (
            not is_baseline
            and retrieved
            and _memory_bandit_is_enabled(memory_bank)
        ):
            try:
                base_response = student.generate(
                    prompt=user_prompt,
                    system_prompt=base_system_prompt,
                    temperature=config.temperature,
                    max_tokens=eval_max_tokens,
                )
                base_response_text = base_response.text
                base_parsed_answer = _parse_final_answer(
                    base_response_text,
                    task_type=config.task_type,
                    prompt_profile=config.prompt_profile,
                    answer_only=True,
                )
                if config.llm_extract_answer:
                    base_llm_answer = _llm_extract_answer(judge_model, base_response_text, config.task_type)
                    if base_llm_answer is not None:
                        base_parsed_answer = base_llm_answer
                base_judge_result = judge.judge(
                    question=sample.question,
                    response=base_response_text,
                    parsed_answer=base_parsed_answer,
                    ground_truth=sample.answer,
                )
                base_is_correct = base_judge_result.get("label") == "success"
                reward_memory = 1.0 if is_correct else 0.0
                reward_base = 1.0 if base_is_correct else 0.0
                bandit_advantage = reward_memory - reward_base
                _update_bandit_with_advantage(
                    memory_bank=memory_bank,
                    retrieved_memories=retrieved,
                    reward_memory=reward_memory,
                    reward_base=reward_base,
                    debug_memory_prompt=debug_memory_prompt,
                )
            except Exception as exc:
                if debug_memory_prompt:
                    print(f"  [Bandit] Failed to compute base reward: {exc}")
        
        # Check if memory was actually used
        memory_used = False
        if retrieved:
            memory_used = declared_memory_use(response_text)
            if memory_used:
                stats["memory_used"] += 1
        
        duration = time.time() - start_time
        
        # Save result
        result = {
            "sample_id": sample.id,
            "question": sample.question,
            "ground_truth": sample.answer,
                "retrieved_memory": [_format_retrieved_memory(m) for m in retrieved],
            "response": response_text,
            "parsed_answer": parsed_answer,
            "judge": judge_result,
            "correct": is_correct,
            "memory_used": memory_used,
            "duration_sec": round(duration, 2),
        }
        if base_response_text is not None:
            result["bandit_base_response"] = base_response_text
        if base_parsed_answer is not None:
            result["bandit_base_parsed_answer"] = base_parsed_answer
        if base_judge_result is not None:
            result["bandit_base_judge"] = base_judge_result
        if base_is_correct is not None:
            result["bandit_base_correct"] = bool(base_is_correct)
        if bandit_advantage is not None:
            result["bandit_advantage"] = float(bandit_advantage)
        if is_hotpotqa:
            result["em"] = float(judge_result.get("details", {}).get("em", 0.0))
            result["f1"] = float(judge_result.get("details", {}).get("f1", 0.0))
        _append_jsonl(results_path, result)
        
        print(f"  Answer: {parsed_answer} | GT: {sample.answer} | {'✓' if is_correct else '✗'}")
    
    # Calculate metrics
    accuracy = stats["correct"] / stats["total"] if stats["total"] > 0 else 0
    em = None
    f1 = None
    if is_hotpotqa and stats["total"] > 0:
        em = stats["em_sum"] / stats["total"]
        f1 = stats["f1_sum"] / stats["total"]
    memory_util = stats["memory_used"] / stats["memory_retrieved"] if stats["memory_retrieved"] > 0 else 0
    
    summary = {
        "experiment_tag": experiment_tag,
        "memory_path": memory_path,
        "stats": stats,
        "accuracy": accuracy,
        "em": em,
        "f1": f1,
        "memory_utilization": memory_util,
        "data_split": _summarize_split_info(split_info),
        "data_split_path": split_path,
        "timestamp": datetime.utcnow().isoformat(),
    }
    
    summary_path = os.path.join(output_dir, f"summary_{experiment_tag}.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n=== Results: {experiment_tag} ===")
    print(f"Accuracy: {stats['correct']}/{stats['total']} ({100*accuracy:.1f}%)")
    if is_hotpotqa and em is not None and f1 is not None:
        print(f"EM/F1: {100*em:.1f}% / {100*f1:.1f}%")
    if is_preference:
        print(f"Skipped ties: {stats.get('skipped_ties', 0)}")
        print(f"Skipped missing: {stats.get('skipped_missing', 0)}")
    if not is_baseline:
        print(f"Memory Retrieved: {stats['memory_retrieved']}/{stats['total']}")
        print(f"Memory Used: {stats['memory_used']}/{stats['memory_retrieved']} ({100*memory_util:.1f}%)")
    
    return summary


# ============================================================
# NEW: HelpSteer3 Response Comparison (base vs base+RL vs base+memory)
# ============================================================

def run_helpsteer3_response_comparison(
    config: ExperimentConfig,
    rl_model_path: Optional[str] = None,
    memory_path: Optional[str] = None,
    eval_max_tokens: Optional[int] = None,
    judge_api_key: Optional[str] = None,
    judge_models: Optional[str] = None,
    gemini_credentials: Optional[str] = None,
    gemini_location: str = "us-central1",
    print_all_judges: bool = False,
    gemini_model: str = "gemini-2.5-pro",
    single_mode: bool = False,
    single_mode_target: str = "base",
    debug_memory_prompt: bool = False,
) -> Dict[str, Any]:
    """
    Run HelpSteer3 response comparison using HelpSteer2-style 5D scoring.

    Each response is scored on 5 dimensions (helpfulness, correctness, coherence,
    complexity, verbosity), then a weighted score is computed. Pairwise outcomes
    are determined by weighted scores.
    """
    print("\n" + "="*60)
    print("HelpSteer3 Response Comparison Evaluation (5D Weighted)")
    print("="*60)
    print(f"Base model: {config.student_model}")
    print(f"RL model: {rl_model_path or 'None'}")
    print(f"Memory path: {memory_path or 'None'}")
    if single_mode:
        print(f"Mode: single-model scoring ({single_mode_target} only)")
    judge_names = _parse_judge_models(judge_models)
    gemini_credentials = _resolve_gemini_credentials_path(gemini_credentials)
    primary_judge = judge_names[0] if judge_names else (config.judge_model or "gemini-2.5-flash")
    if judge_names:
        print(f"Judge candidates: {', '.join(judge_names)}")
    print(f"Scoring judge: {primary_judge}")
    if "gemini" in primary_judge:
        print(f"Gemini creds: {gemini_credentials or 'None'} | location: {gemini_location} | model: {gemini_model}")
    
    # Setup output
    output_dir = os.path.join(config.output_dir, config.experiment_name, "comparison")
    os.makedirs(output_dir, exist_ok=True)
    results_path = os.path.join(output_dir, "comparison_results.jsonl")
    # NOTE: consistency issues not applicable for 5D scoring

    # Load base model
    base_model = load_model(
        config.student_model,
        timeout=config.api_timeout,
        max_retries=config.api_retries,
    )

    # Load scoring judge model
    score_judge = load_model(
        primary_judge,
        timeout=config.api_timeout,
        max_retries=config.api_retries,
    )
    
    # Load RL model if available
    rl_model = None
    if rl_model_path and os.path.exists(rl_model_path):
        try:
            rl_model = load_model(
                rl_model_path,
                timeout=config.api_timeout,
                max_retries=config.api_retries,
            )
            print(f"Loaded RL model from: {rl_model_path}")
        except Exception as e:
            print(f"Warning: Failed to load RL model: {e}")
            rl_model = None
    
    # Load memory bank
    memory_bank = None
    if memory_path and os.path.exists(memory_path):
        memory_bank = MemoryBank(memory_path, top_k=config.memory_top_k_preference)
        loaded = memory_bank.load(
            memory_path,
            max_items=config.memory_max_items if config.memory_max_items > 0 else None,
        )
        print(f"Loaded {loaded} memories from {memory_path}")
        _configure_memory_bank_for_retrieval(
            memory_bank=memory_bank,
            config=config,
            memory_path=memory_path,
            experiment_tag="helpsteer3_comparison",
            phase_dir=output_dir,
        )
    
    # Load test dataset
    _, test_dataset, split_info = _load_split_datasets(config)
    
    print(f"\nEvaluating on {len(test_dataset)} samples")

    # Statistics for pairwise comparisons
    stats = {
        "total": 0,
        "skipped_ties": 0,
        "skipped_missing": 0,
        # Base vs RL
        "base_vs_rl": {"base_wins": 0, "rl_wins": 0, "ties": 0, "errors": 0},
        # Base vs Memory
        "base_vs_memory": {"base_wins": 0, "memory_wins": 0, "ties": 0, "errors": 0},
        # RL vs Memory
        "rl_vs_memory": {"rl_wins": 0, "memory_wins": 0, "ties": 0, "errors": 0},
        # Weighted score tracking
        "weighted_sum": {"base": 0.0, "rl": 0.0, "memory": 0.0},
        "weighted_count": {"base": 0, "rl": 0, "memory": 0},
    }

    # Generation parameters
    gen_max_tokens = eval_max_tokens or 1024
    gen_temperature = 0.7
    
    for idx, sample in enumerate(test_dataset):
        print(f"\n[{idx+1}/{len(test_dataset)}] {sample.id}")
        
        # Extract context from sample
        context = sample.meta.get("context") if isinstance(sample.meta, dict) else None
        if not context:
            context = sample.question
        
        if not context or len(context.strip()) < 10:
            stats["skipped_missing"] += 1
            continue
        
        stats["total"] += 1
        
        # Generate responses from each model variant
        responses = {}
        
        # 1. Base model response
        try:
            base_response = base_model.generate(
                prompt=context,
                system_prompt=_build_response_generation_system_prompt(),
                temperature=gen_temperature,
                max_tokens=gen_max_tokens,
            )
            responses["base"] = base_response.text
        except Exception as e:
            print(f"  Base model error: {e}")
            responses["base"] = f"Error: {e}"
        
        # 2. RL model response (if available)
        if rl_model:
            try:
                rl_response = rl_model.generate(
                    prompt=context,
                    system_prompt=_build_response_generation_system_prompt(),
                    temperature=gen_temperature,
                    max_tokens=gen_max_tokens,
                )
                responses["rl"] = rl_response.text
            except Exception as e:
                print(f"  RL model error: {e}")
                responses["rl"] = f"Error: {e}"
        
        # 3. Base + Memory response
        retrieved: List[MemoryItem] = []
        if memory_bank:
            # Build optimized retrieval query using task routing
            retrieval_query = _build_retrieval_query(context)
            
            # Retrieve relevant memories using optimized query
            retrieved = memory_bank.retrieve(
                retrieval_query,  # Use task-based query instead of full context
                top_k=config.memory_top_k_preference,
                exclude_sample_id=sample.id if config.exclude_self_memory else None,
                memory_types="preference_rule",
                retrieval_method=config.memory_retrieval,
            )
            
            # Build prompt with memory guidance
            memory_system_prompt = _build_response_generation_system_prompt_with_memory(
                memory_bank=memory_bank,
                retrieved_memories=retrieved,
            )
            if debug_memory_prompt and retrieved:
                titles = [m.title for m in retrieved[:3]]
                print(f"  [Debug] Memories retrieved: {len(retrieved)} | titles={titles}")

            try:
                memory_response = base_model.generate(
                    prompt=context,
                    system_prompt=memory_system_prompt,
                    temperature=gen_temperature,
                    max_tokens=gen_max_tokens,
                )
                responses["memory"] = memory_response.text
            except Exception as e:
                print(f"  Memory model error: {e}")
                responses["memory"] = f"Error: {e}"

        if single_mode and single_mode_target == "memory" and memory_bank and "memory" in responses:
            # Use memory response as the base for single-mode scoring.
            responses["base"] = responses["memory"]
        if single_mode and single_mode_target == "rl" and rl_model and "rl" in responses:
            responses["base"] = responses["rl"]
        
        # Pairwise comparisons using multiple judges
        result = {
            "sample_id": sample.id,
            "context": context[:500],  # Truncate for storage
            "responses": {k: v[:1000] for k, v in responses.items()},  # Truncate
            "comparisons": {},
            "dimension_scores": {},
            "weighted_scores": {},
        }

        pairs = [] if single_mode else [
            ("base_vs_rl", "base", "rl"),
            ("base_vs_memory", "base", "memory"),
            ("rl_vs_memory", "rl", "memory"),
        ]
        scores_by_label: Dict[str, Dict[str, Any]] = {}

        def _score_response(label: str) -> Optional[Dict[str, Any]]:
            if label not in responses:
                return None
            scores = _judge_helpsteer2_scores(
                judge_model=score_judge,
                prompt=context,
                response=responses[label],
                max_tokens=config.advancedif_judge_max_tokens,
            )
            if not scores:
                return None
            if len(scores) != len(HELPSTEER2_DIMENSIONS):
                if len(scores) < len(HELPSTEER2_DIMENSIONS):
                    scores = scores + [float(HELPSTEER2_SCORE_VALUES[0])] * (len(HELPSTEER2_DIMENSIONS) - len(scores))
                else:
                    scores = scores[:len(HELPSTEER2_DIMENSIONS)]
            dimension_scores: List[Dict[str, Any]] = []
            weighted_sum = 0.0
            for dim_label, score in zip(HELPSTEER2_DIMENSIONS, scores):
                if score is None:
                    score = float(HELPSTEER2_SCORE_VALUES[0])
                dimension_scores.append({"dimension": dim_label, "score": score})
                weighted_sum += score * HELPSTEER2_DIMENSION_WEIGHTS.get(dim_label, 1)
            weighted_avg = weighted_sum / HELPSTEER2_WEIGHT_TOTAL
            stats["weighted_sum"][label] += weighted_avg
            stats["weighted_count"][label] += 1
            return {"dimension_scores": dimension_scores, "weighted_avg": weighted_avg}

        for label in responses.keys():
            scored = _score_response(label)
            if scored is not None:
                scores_by_label[label] = scored
                result["dimension_scores"][label] = scored["dimension_scores"]
                result["weighted_scores"][label] = round(scored["weighted_avg"], 6)

        if (
            memory_bank is not None
            and retrieved
            and _memory_bandit_is_enabled(memory_bank)
        ):
            base_scored = scores_by_label.get("base")
            memory_scored = scores_by_label.get("memory")
            if base_scored is not None and memory_scored is not None:
                base_reward = float(base_scored.get("weighted_avg") or 0.0)
                memory_reward = float(memory_scored.get("weighted_avg") or 0.0)
                _update_bandit_with_advantage(
                    memory_bank=memory_bank,
                    retrieved_memories=retrieved,
                    reward_memory=memory_reward,
                    reward_base=base_reward,
                    debug_memory_prompt=debug_memory_prompt,
                )
                result["bandit_base_reward"] = base_reward
                result["bandit_memory_reward"] = memory_reward
                result["bandit_advantage"] = memory_reward - base_reward

        def _compare(left_key: str, right_key: str) -> str:
            left = scores_by_label.get(left_key)
            right = scores_by_label.get(right_key)
            if left is None or right is None:
                return "missing"
            left_score = left.get("weighted_avg")
            right_score = right.get("weighted_avg")
            if left_score is None or right_score is None:
                return "missing"
            if abs(left_score - right_score) < 1e-6:
                return "tie"
            return left_key if left_score > right_score else right_key

        for pair_key, left_key, right_key in pairs:
            winner = _compare(left_key, right_key)
            result["comparisons"][pair_key] = winner

        def _record_primary(pair_key: str, winner: Optional[str]) -> None:
            if not winner or winner == "missing":
                return
            if pair_key == "base_vs_rl":
                if winner == "base":
                    stats["base_vs_rl"]["base_wins"] += 1
                elif winner == "rl":
                    stats["base_vs_rl"]["rl_wins"] += 1
                elif winner == "tie":
                    stats["base_vs_rl"]["ties"] += 1
                else:
                    stats["base_vs_rl"]["errors"] += 1
            elif pair_key == "base_vs_memory":
                if winner == "base":
                    stats["base_vs_memory"]["base_wins"] += 1
                elif winner == "memory":
                    stats["base_vs_memory"]["memory_wins"] += 1
                elif winner == "tie":
                    stats["base_vs_memory"]["ties"] += 1
                else:
                    stats["base_vs_memory"]["errors"] += 1
            elif pair_key == "rl_vs_memory":
                if winner == "rl":
                    stats["rl_vs_memory"]["rl_wins"] += 1
                elif winner == "memory":
                    stats["rl_vs_memory"]["memory_wins"] += 1
                elif winner == "tie":
                    stats["rl_vs_memory"]["ties"] += 1
                else:
                    stats["rl_vs_memory"]["errors"] += 1

        for pair_key, _, _ in pairs:
            _record_primary(pair_key, result["comparisons"].get(pair_key))

        if single_mode:
            print(f"  Base weighted score: {result['weighted_scores'].get('base', 'N/A')}")
        elif print_all_judges:
            print(
                f"  Weighted scores: "
                f"base={result['weighted_scores'].get('base', 'N/A')} | "
                f"rl={result['weighted_scores'].get('rl', 'N/A')} | "
                f"memory={result['weighted_scores'].get('memory', 'N/A')}"
            )
        else:
            if "base_vs_rl" in result["comparisons"]:
                print(f"  Base vs RL: {result['comparisons'].get('base_vs_rl', 'N/A')}")
            if "base_vs_memory" in result["comparisons"]:
                print(f"  Base vs Memory: {result['comparisons'].get('base_vs_memory', 'N/A')}")
            if "rl_vs_memory" in result["comparisons"]:
                print(f"  RL vs Memory: {result['comparisons'].get('rl_vs_memory', 'N/A')}")
        
        # Save result
        _append_jsonl(results_path, result)
    
    # Calculate win rates
    def calc_win_rate(wins: int, total: int) -> float:
        return wins / total * 100 if total > 0 else 0.0
    
    summary = {
        "config": {
            "student_model": config.student_model,
            "rl_model_path": rl_model_path,
            "memory_path": memory_path,
        },
        "stats": stats,
        "judge_model": primary_judge,
        "win_rates": {},
        "weighted_score_avg": {},
        "single_mode": single_mode,
        "timestamp": datetime.utcnow().isoformat(),
    }
    
    # Base vs RL win rates
    base_rl_total = stats["base_vs_rl"]["base_wins"] + stats["base_vs_rl"]["rl_wins"]
    if base_rl_total > 0:
        summary["win_rates"]["base_vs_rl"] = {
            "base_win_rate": calc_win_rate(stats["base_vs_rl"]["base_wins"], base_rl_total),
            "rl_win_rate": calc_win_rate(stats["base_vs_rl"]["rl_wins"], base_rl_total),
        }
    
    # Base vs Memory win rates
    base_mem_total = stats["base_vs_memory"]["base_wins"] + stats["base_vs_memory"]["memory_wins"]
    if base_mem_total > 0:
        summary["win_rates"]["base_vs_memory"] = {
            "base_win_rate": calc_win_rate(stats["base_vs_memory"]["base_wins"], base_mem_total),
            "memory_win_rate": calc_win_rate(stats["base_vs_memory"]["memory_wins"], base_mem_total),
        }
    
    # RL vs Memory win rates
    rl_mem_total = stats["rl_vs_memory"]["rl_wins"] + stats["rl_vs_memory"]["memory_wins"]
    if rl_mem_total > 0:
        summary["win_rates"]["rl_vs_memory"] = {
            "rl_win_rate": calc_win_rate(stats["rl_vs_memory"]["rl_wins"], rl_mem_total),
            "memory_win_rate": calc_win_rate(stats["rl_vs_memory"]["memory_wins"], rl_mem_total),
        }
    
    # Save summary
    summary_path = os.path.join(output_dir, "comparison_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    
    # Print results
    print("\n" + "="*60)
    print("COMPARISON RESULTS")
    print("="*60)
    print(f"Total samples: {stats['total']}")
    print(f"Skipped: {stats['skipped_missing']}")
    
    if base_rl_total > 0:
        print(f"\nBase vs RL (n={base_rl_total}):")
        print(f"  Base wins: {stats['base_vs_rl']['base_wins']} ({summary['win_rates']['base_vs_rl']['base_win_rate']:.1f}%)")
        print(f"  RL wins: {stats['base_vs_rl']['rl_wins']} ({summary['win_rates']['base_vs_rl']['rl_win_rate']:.1f}%)")
    
    if base_mem_total > 0:
        print(f"\nBase vs Memory (n={base_mem_total}):")
        print(f"  Base wins: {stats['base_vs_memory']['base_wins']} ({summary['win_rates']['base_vs_memory']['base_win_rate']:.1f}%)")
        print(f"  Memory wins: {stats['base_vs_memory']['memory_wins']} ({summary['win_rates']['base_vs_memory']['memory_win_rate']:.1f}%)")
    
    if rl_mem_total > 0:
        print(f"\nRL vs Memory (n={rl_mem_total}):")
        print(f"  RL wins: {stats['rl_vs_memory']['rl_wins']} ({summary['win_rates']['rl_vs_memory']['rl_win_rate']:.1f}%)")
        print(f"  Memory wins: {stats['rl_vs_memory']['memory_wins']} ({summary['win_rates']['rl_vs_memory']['memory_win_rate']:.1f}%)")

    for label in ("base", "rl", "memory"):
        cnt = stats["weighted_count"].get(label, 0)
        if cnt:
            avg = stats["weighted_sum"][label] / cnt
            summary["weighted_score_avg"][label] = round(avg, 6)
    if summary["weighted_score_avg"]:
        print("\nWeighted score averages:")
        for label, avg in summary["weighted_score_avg"].items():
            print(f"  {label}: {avg:.4f}")

    return summary


def _load_eval_result_records(
    output_dir: str,
    experiment_tag: str,
) -> List[Dict[str, Any]]:
    """Load evaluation JSONL records for a given experiment tag (supports rank-sharded files)."""
    records: List[Dict[str, Any]] = []
    base_name = f"results_{experiment_tag}.jsonl"
    base_path = os.path.join(output_dir, base_name)
    candidate_paths: List[str] = []
    if os.path.exists(base_path):
        candidate_paths.append(base_path)

    shard_prefix = f"results_{experiment_tag}.rank"
    if os.path.isdir(output_dir):
        for fname in sorted(os.listdir(output_dir)):
            if fname.startswith(shard_prefix) and fname.endswith(".jsonl"):
                candidate_paths.append(os.path.join(output_dir, fname))

    if not candidate_paths:
        return records

    for path in candidate_paths:
        try:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        item = json.loads(line)
                    except Exception:
                        continue
                    if isinstance(item, dict):
                        records.append(item)
        except Exception:
            continue
    return records


def _export_aime_base_wrong_memory_right_cases(
    config: ExperimentConfig,
    results: Dict[str, Any],
) -> None:
    """
    Export case-study JSON for AIME evaluation:
    questions where baseline is wrong but memory-augmented response is correct.
    """
    if config.task_type != "aime":
        return
    if "baseline" not in results:
        return

    eval_dir = os.path.join(config.output_dir, config.experiment_name, "evaluation")
    baseline_records = _load_eval_result_records(eval_dir, "baseline")
    if not baseline_records:
        return

    baseline_by_id: Dict[str, Dict[str, Any]] = {}
    for rec in baseline_records:
        sample_id = str(rec.get("sample_id") or "").strip()
        if sample_id:
            baseline_by_id[sample_id] = rec
    if not baseline_by_id:
        return

    memory_groups = [name for name in results.keys() if name != "baseline"]
    if not memory_groups:
        return

    dataset_label = "aime25"
    norm_dataset = _normalize_dataset_ref(config.test_dataset_path or config.dataset_path or "")
    if norm_dataset and "aime25" not in norm_dataset:
        dataset_label = norm_dataset.split("/")[-1]

    all_cases: List[Dict[str, Any]] = []
    for group in memory_groups:
        group_records = _load_eval_result_records(eval_dir, group)
        if not group_records:
            continue
        group_by_id: Dict[str, Dict[str, Any]] = {}
        for rec in group_records:
            sample_id = str(rec.get("sample_id") or "").strip()
            if sample_id:
                group_by_id[sample_id] = rec

        group_cases: List[Dict[str, Any]] = []
        for sample_id, mem_rec in group_by_id.items():
            base_rec = baseline_by_id.get(sample_id)
            if not base_rec:
                continue
            base_correct = bool(base_rec.get("correct"))
            mem_correct = bool(mem_rec.get("correct"))
            if base_correct or not mem_correct:
                continue

            case = {
                "sample_id": sample_id,
                "question": base_rec.get("question"),
                "ground_truth": base_rec.get("ground_truth"),
                "base": {
                    "correct": base_correct,
                    "parsed_answer": base_rec.get("parsed_answer"),
                    "response": base_rec.get("response"),
                    "judge": base_rec.get("judge"),
                },
                "memory": {
                    "group": group,
                    "correct": mem_correct,
                    "parsed_answer": mem_rec.get("parsed_answer"),
                    "response": mem_rec.get("response"),
                    "judge": mem_rec.get("judge"),
                    "memory_used": mem_rec.get("memory_used"),
                    "retrieved_memory": mem_rec.get("retrieved_memory", []),
                },
            }
            group_cases.append(case)
            all_cases.append(case)

        out_path = os.path.join(
            eval_dir,
            f"{dataset_label}_base_wrong_memory_right_{group}.json",
        )
        payload = {
            "dataset": dataset_label,
            "group": group,
            "count": len(group_cases),
            "cases": group_cases,
            "created_at": datetime.utcnow().isoformat(),
        }
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"[CaseStudy] Exported {len(group_cases)} cases -> {out_path}")

    if all_cases:
        combined_path = os.path.join(
            eval_dir,
            f"{dataset_label}_base_wrong_memory_right_cases.json",
        )
        payload = {
            "dataset": dataset_label,
            "count": len(all_cases),
            "cases": all_cases,
            "created_at": datetime.utcnow().isoformat(),
        }
        with open(combined_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"[CaseStudy] Exported combined {len(all_cases)} cases -> {combined_path}")


def run_all_evaluations(
    config: ExperimentConfig,
    eval_only: Optional[str] = None,
    eval_max_tokens: Optional[int] = None,
    debug_memory_prompt: bool = False,
    # NEW: HelpSteer3 comparison mode
    helpsteer3_comparison_mode: bool = False,
    helpsteer3_single_mode: bool = False,
    helpsteer3_single_mode_target: str = "base",
    rl_model_path: Optional[str] = None,
    judge_api_key: Optional[str] = None,
    helpsteer3_judges: Optional[str] = None,
    gemini_credentials: Optional[str] = None,
    gemini_location: str = "us-central1",
    print_all_judges: bool = False,
    gemini_model: str = "gemini-2.5-pro",
    override_memory_path: Optional[str] = None,  # NEW: Allow manual memory path override
) -> Dict[str, Any]:
    """
    Run evaluations for all experiment groups:
    - Baseline
    - Student-only
    - Teacher-only
    - Contrastive
    
    NEW: For HelpSteer3, supports comparison mode (base vs RL vs memory)
    
    Args:
        override_memory_path: If specified, use this memory path instead of auto-detecting
    """
    base_dir = os.path.join(config.output_dir, config.experiment_name, "production")
    
    # NEW: For HelpSteer3, use comparison mode if enabled
    is_helpsteer3 = config.prompt_profile == "helpsteer3" or _is_helpsteer3_dataset(config.dataset_path)
    if is_helpsteer3 and helpsteer3_comparison_mode:
        print("\n[HelpSteer3] Using comparison mode (base vs RL vs memory)")
        
        # Priority: 1) override_memory_path, 2) auto-detect from experiment directory
        memory_path_to_use = None
        if override_memory_path and os.path.exists(override_memory_path):
            memory_path_to_use = override_memory_path
            print(f"Using manually specified memory path: {memory_path_to_use}")
        else:
            # Auto-detect from experiment directory
            contrastive_memory_path = os.path.join(base_dir, "contrastive_memory.jsonl")
            if not os.path.exists(contrastive_memory_path):
                contrastive_memory_path = os.path.join(base_dir, "all_memory.jsonl")
            if os.path.exists(contrastive_memory_path):
                memory_path_to_use = contrastive_memory_path
                print(f"Using auto-detected memory path: {memory_path_to_use}")
            else:
                print("No memory file found. Memory model will be skipped (only Base vs RL will be compared).")
        
        return run_helpsteer3_response_comparison(
            config=config,
            rl_model_path=rl_model_path,
            memory_path=memory_path_to_use,
            eval_max_tokens=eval_max_tokens,
            judge_api_key=judge_api_key,
            judge_models=helpsteer3_judges,
            gemini_credentials=gemini_credentials,
            gemini_location=gemini_location,
            print_all_judges=print_all_judges,
            gemini_model=gemini_model,
            single_mode=helpsteer3_single_mode,
            single_mode_target=helpsteer3_single_mode_target,
            debug_memory_prompt=debug_memory_prompt,
        )
    
    results = {}
    eval_only = eval_only or None

    def _should_run(group: str) -> bool:
        return eval_only is None or eval_only == group
    
    # 1. Baseline (no memory)
    if _should_run("baseline"):
        print("\n" + "="*60)
        print("BASELINE (No Memory)")
        print("="*60)
        results["baseline"] = run_evaluation(
            config=config,
            memory_path=None,
            experiment_tag="baseline",
            eval_max_tokens=eval_max_tokens,
            debug_memory_prompt=debug_memory_prompt,
        )
    
    # 2. Student-only
    student_memory_path = os.path.join(base_dir, "student_memory.jsonl")
    if _should_run("student") and os.path.exists(student_memory_path):
        print("\n" + "="*60)
        print("STUDENT-ONLY MEMORY")
        print("="*60)
        results["student"] = run_evaluation(
            config=config,
            memory_path=student_memory_path,
            experiment_tag="student",
            eval_max_tokens=eval_max_tokens,
            debug_memory_prompt=debug_memory_prompt,
        )
    elif _should_run("student"):
        print(f"Student memory not found: {student_memory_path}")
    
    # 3. Teacher-only
    teacher_memory_path = os.path.join(base_dir, "teacher_memory.jsonl")
    if _should_run("teacher") and os.path.exists(teacher_memory_path):
        print("\n" + "="*60)
        print("TEACHER-ONLY MEMORY")
        print("="*60)
        results["teacher"] = run_evaluation(
            config=config,
            memory_path=teacher_memory_path,
            experiment_tag="teacher",
            eval_max_tokens=eval_max_tokens,
            debug_memory_prompt=debug_memory_prompt,
        )
    elif _should_run("teacher"):
        print(f"Teacher memory not found: {teacher_memory_path}")
    
    # 4. Contrastive
    contrastive_memory_path = os.path.join(base_dir, "contrastive_memory.jsonl")
    if _should_run("contrastive") and os.path.exists(contrastive_memory_path):
        print("\n" + "="*60)
        print("CONTRASTIVE MEMORY")
        print("="*60)
        results["contrastive"] = run_evaluation(
            config=config,
            memory_path=contrastive_memory_path,
            experiment_tag="contrastive",
            eval_max_tokens=eval_max_tokens,
            debug_memory_prompt=debug_memory_prompt,
        )
    elif _should_run("contrastive"):
        print(f"Contrastive memory not found: {contrastive_memory_path}")
    
    # 5. All combined
    all_memory_path = os.path.join(base_dir, "all_memory.jsonl")
    if _should_run("all") and os.path.exists(all_memory_path):
        print("\n" + "="*60)
        print("ALL MEMORY COMBINED")
        print("="*60)
        results["all"] = run_evaluation(
            config=config,
            memory_path=all_memory_path,
            experiment_tag="all",
            eval_max_tokens=eval_max_tokens,
            debug_memory_prompt=debug_memory_prompt,
        )
    elif _should_run("all"):
        print(f"All memory not found: {all_memory_path}")
    
    # Print comparison
    if results:
        print("\n" + "="*60)
        print("COMPARISON SUMMARY")
        print("="*60)
        is_advancedif = any("rubric_accuracy" in res for res in results.values())
        is_helpsteer2 = any("dimension_score_avg" in res for res in results.values())
        has_qa = any(res.get("em") is not None for res in results.values()) if not (is_advancedif or is_helpsteer2) else False
        if is_advancedif:
            print(f"{'Group':<15} {'Rubric%':<10} {'Score':<14}")
            print("-"*40)
        elif is_helpsteer2:
            print(f"{'Group':<15} {'Dim%':<10} {'Weighted%':<10} {'Score':<14}")
            print("-"*55)
        elif has_qa:
            print(f"{'Group':<15} {'EM':<8} {'F1':<8} {'Correct':<10}")
            print("-"*45)
        else:
            print(f"{'Group':<15} {'Accuracy':<12} {'Correct':<10}")
            print("-"*40)
        for name, res in results.items():
            if is_advancedif:
                rub_acc = res.get("rubric_accuracy") or 0.0
                rub_sum = res.get("rubric_score_sum") or 0.0
                rub_cnt = res.get("rubric_count") or 0
                print(f"{name:<15} {100*rub_acc:>6.1f}%     {rub_sum:.2f}/{rub_cnt}")
                continue
            if is_helpsteer2:
                dim_pct = res.get("dimension_score_pct") or 0.0
                dim_sum = res.get("dimension_score_sum") or 0.0
                dim_cnt = res.get("dimension_count") or 0
                weighted_pct = res.get("weighted_score_pct") or 0.0
                weighted_sum = res.get("weighted_score_sum") or 0.0
                max_total = dim_cnt * HELPSTEER2_SCORE_VALUES[-1] if dim_cnt else 0.0
                score_str = f"{dim_sum:.2f}/{max_total:.1f}" if max_total else "0.00/0.0"
                if dim_cnt:
                    total_samples = dim_cnt / len(HELPSTEER2_DIMENSIONS)
                    max_weighted = total_samples * HELPSTEER2_WEIGHT_TOTAL * HELPSTEER2_SCORE_VALUES[-1]
                    weighted_str = f"{weighted_sum:.2f}/{max_weighted:.1f}" if max_weighted else "0.00/0.0"
                else:
                    weighted_str = "0.00/0.0"
                print(
                    f"{name:<15} {100*dim_pct:>6.1f}%     {100*weighted_pct:>6.1f}%     {score_str} | {weighted_str}"
                )
                continue
            correct = res.get("stats", {}).get("correct", 0)
            total = res.get("stats", {}).get("total", 0)
            if has_qa:
                em = res.get("em")
                f1 = res.get("f1")
                em_str = f"{100*em:>5.1f}%" if em is not None else "  -  "
                f1_str = f"{100*f1:>5.1f}%" if f1 is not None else "  -  "
                print(f"{name:<15} {em_str:<8} {f1_str:<8} {correct}/{total}")
            else:
                acc = res.get("accuracy", 0.0)
                print(f"{name:<15} {100*acc:>6.1f}%      {correct}/{total}")

    # AIME case-study export: baseline wrong + memory correct.
    _export_aime_base_wrong_memory_right_cases(config, results)
    
    return results


# ============================================================
# Utility Functions
# ============================================================

def _clean_response(text: str) -> str:
    if not text:
        return ""
    think_end = text.find("</think>")
    if think_end != -1:
        text = text[think_end + len("</think>"):].strip()
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = re.sub(r"<think>.*", "", text, flags=re.DOTALL)
    return text.strip()


_QUERY_EXPANSION_PROMPT = (
    "You are extracting knowledge points for retrieval.\n"
    "Problem:\n"
    "{question}\n\n"
    "List 5-8 key concepts/techniques needed. "
    "Output only a comma-separated list of short English phrases. "
    "Do NOT solve the problem or mention any final answer."
)

# HelpSteer3 only: 扩充检索 query，与 memory 的 trigger/dimension 对齐
_QUERY_EXPANSION_PROMPT_PREFERENCE = (
    "You are extracting key phrases for preference-rule retrieval.\n"
    "Input (context and two responses):\n"
    "{question}\n\n"
    "List 5-8 short phrases that describe: (1) the situation/topic in the input (trigger), "
    "(2) quality dimensions that may matter (e.g. helpfulness, clarity, correctness). "
    "Output only a comma-separated list of short English phrases. "
    "Do NOT mention Response 1/2 or which is better."
)


def _normalize_query_expansion(text: str) -> str:
    if not text:
        return ""
    cleaned = text.strip()
    # Remove code fences if any
    cleaned = re.sub(r"```.*?```", "", cleaned, flags=re.DOTALL)
    cleaned = cleaned.replace("\n", ", ")
    cleaned = re.sub(r"[;•]", ",", cleaned)
    cleaned = re.sub(r"^(key concepts?|concepts?|topics?)\s*:\s*", "", cleaned, flags=re.I)
    cleaned = re.sub(r"\s*,\s*", ", ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned)
    cleaned = cleaned.strip(" ,")
    return cleaned[:300]


def _expand_memory_query(
    student_model: Model,
    question: str,
    max_tokens: int = 128,
    prompt_profile: Optional[str] = None,
) -> Optional[str]:
    # HelpSteer3 only: 与 memory 的 trigger/dimension 对齐的扩充
    if prompt_profile == "helpsteer3":
        prompt = _QUERY_EXPANSION_PROMPT_PREFERENCE.format(question=question.strip())
    else:
        prompt = _QUERY_EXPANSION_PROMPT.format(question=question.strip())
    try:
        response = student_model.generate(
            prompt=prompt,
            system_prompt=None,
            temperature=0.0,
            max_tokens=max_tokens,
        )
    except Exception:
        return None
    return _normalize_query_expansion(response.text)


def _llm_extract_answer(
    model: Model,
    response_text: str,
    task_type: str,
) -> Optional[str]:
    if not response_text:
        return None
    lowered = response_text.lstrip().lower()
    if lowered.startswith(("error:", "model error:")):
        return None
    rules = "Return JSON only. If no answer is present, return null."
    if task_type == "aime":
        rules = (
            "Return JSON only with an integer answer in [0, 999]. "
            "If no answer is present, return null."
        )
    prompt = (
        "You are a strict answer extractor.\n"
        f"{rules}\n\n"
        "MODEL RESPONSE:\n"
        f"{response_text[-2000:]}\n\n"
        "Output format:\n"
        "{\"answer\": <value or null>}"
    )
    try:
        resp = model.generate(
            prompt=prompt,
            system_prompt=None,
            temperature=0.0,
            do_sample=False,
            max_tokens=256,
        )
    except ModelError:
        return None
    text = resp.text or ""
    json_match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if json_match:
        try:
            data = json.loads(json_match.group(0))
            answer = data.get("answer")
        except (json.JSONDecodeError, TypeError):
            answer = None
    else:
        answer = None
    if answer is None:
        if task_type == "aime":
            num_match = re.search(r"(\d+)", text)
            if num_match:
                val = int(num_match.group(1))
                if 0 <= val <= 999:
                    return num_match.group(1)
            return None
        return text.strip() or None
    if task_type == "aime":
        try:
            val = int(float(answer))
        except (ValueError, TypeError):
            return None
        return str(val) if 0 <= val <= 999 else None
    return str(answer).strip()


def _parse_final_answer(
    text: str,
    task_type: str = "math",
    prompt_profile: Optional[str] = None,
    answer_only: bool = False,
) -> Optional[str]:
    """
    [FIX] 与 test_fixed.py 保持一致的答案解析
    优先解析 \\boxed{} 格式（官方推荐）
    """
    cleaned = _clean_response(text)
    if not cleaned:
        return None

    if prompt_profile == "hotpotqa":
        def _extract_boxed_content(text: str) -> Optional[str]:
            candidates: List[str] = []
            idx = 0
            while True:
                start = text.find("\\boxed{", idx)
                if start == -1:
                    start = text.find("boxed{", idx)
                    if start == -1:
                        break
                i = start + len("\\boxed{")
                if text[start:start + 6] != "\\boxed":
                    i = start + len("boxed{")
                depth = 1
                while i < len(text):
                    if text[i] == "{":
                        depth += 1
                    elif text[i] == "}":
                        depth -= 1
                        if depth == 0:
                            content = (
                                text[(start + len("\\boxed{")) : i]
                                if text[start:start + 6] == "\\boxed"
                                else text[(start + len("boxed{")) : i]
                            )
                            candidates.append(content)
                            break
                    i += 1
                idx = start + 6
            if candidates:
                return candidates[-1].strip()
            return None

        def _strip_latex_wrappers(value: str) -> str:
            wrapped = value.strip()
            for pat in (
                r"^\\text\{(.+)\}$",
                r"^\\mathrm\{(.+)\}$",
                r"^\\mathbf\{(.+)\}$",
            ):
                m = re.match(pat, wrapped)
                if m:
                    return m.group(1).strip()
            return wrapped

        def _trim_hotpotqa_answer(value: str) -> str:
            trimmed = value.strip()
            boxed_content = _extract_boxed_content(trimmed)
            if boxed_content:
                trimmed = boxed_content
            trimmed = re.split(
                r"(?i)\b(based on|because|since|given|according to|the context|from the context)\b",
                trimmed,
            )[0].strip()
            trimmed = re.split(r"(?i)\bstop\b", trimmed)[0].strip()
            trimmed = re.split(r"(?i)\bsystem\b", trimmed)[0].strip()
            trimmed = _strip_latex_wrappers(trimmed)
            return trimmed.strip(" .;:")

        boxed = _extract_boxed_content(cleaned)
        if boxed:
            return _trim_hotpotqa_answer(boxed)
        for line in cleaned.splitlines():
            line = line.strip()
            if line:
                return _trim_hotpotqa_answer(line)
        return _trim_hotpotqa_answer(cleaned) or None

    def _extract_boxed_content(text: str) -> Optional[str]:
        candidates: List[str] = []
        idx = 0
        while True:
            start = text.find("\\boxed{", idx)
            if start == -1:
                start = text.find("boxed{", idx)
                if start == -1:
                    break
            i = start + len("\\boxed{")
            if text[start:start + 6] != "\\boxed":
                i = start + len("boxed{")
            depth = 1
            while i < len(text):
                if text[i] == "{":
                    depth += 1
                elif text[i] == "}":
                    depth -= 1
                    if depth == 0:
                        content = (
                            text[(start + len("\\boxed{")) : i]
                            if text[start:start + 6] == "\\boxed"
                            else text[(start + len("boxed{")) : i]
                        )
                        candidates.append(content)
                        break
                i += 1
            idx = start + 6
        if candidates:
            return candidates[-1].strip()
        return None

    def _strip_latex_wrappers(value: str) -> str:
        wrapped = value.strip()
        for pat in (
            r"^\\text\{(.+)\}$",
            r"^\\mathrm\{(.+)\}$",
            r"^\\mathbf\{(.+)\}$",
        ):
            m = re.match(pat, wrapped)
            if m:
                return m.group(1).strip()
        return wrapped

    def _clean_math_answer(value: str) -> str:
        cleaned = value.strip()
        if cleaned.startswith("$") and cleaned.endswith("$") and len(cleaned) >= 2:
            cleaned = cleaned[1:-1].strip()
        cleaned = cleaned.strip(" \t\r\n.;:")
        cleaned = _strip_latex_wrappers(cleaned)
        sign = ""
        if cleaned.startswith(("+", "-")):
            sign = cleaned[0]
            cleaned = cleaned[1:].strip()
        frac_match = re.fullmatch(
            r"\\(?:d|t)?frac\{(-?\d+)\}\{(-?\d+)\}",
            cleaned,
        )
        if frac_match:
            cleaned = f"{sign}{frac_match.group(1)}/{frac_match.group(2)}"
        elif sign:
            cleaned = sign + cleaned
        return cleaned.strip()

    numeric_only = task_type == "aime"

    # [FIX] Pattern 1: \\boxed{...} (官方推荐格式，优先！)
    boxed = _extract_boxed_content(cleaned)
    if boxed:
        if numeric_only:
            num_match = re.search(r"(-?\d+)", boxed)
            return num_match.group(1) if num_match else None
        return _clean_math_answer(boxed) or None

    # Pattern 2: ANSWER: N
    answer_line = ""
    for answer_match in re.finditer(r"(?im)^\s*ANSWER\s*:\s*(.+)$", cleaned):
        answer_line = answer_match.group(1).strip()
    if answer_line:
        if numeric_only:
            num_match = re.search(r"(-?\d+)", answer_line)
            return num_match.group(1) if num_match else None
        return _clean_math_answer(answer_line) or None

    if numeric_only:
        # Pattern 3: Final Answer: N
        final = re.search(r"final\s+answer\s*[:\s]+(\d+)", cleaned, re.IGNORECASE)
        if final:
            return final.group(1)

        # Pattern 4: Answer is N
        answer = re.search(r"(?:the\s+)?answer\s+is\s*[:\s]*(\d+)", cleaned, re.IGNORECASE)
        if answer:
            return answer.group(1)

        # Pattern 5: Answer: N
        answer_colon = re.search(r"Answer\s*:\s*(\d+)", cleaned, re.IGNORECASE)
        if answer_colon:
            return answer_colon.group(1)

        # Pattern 6: = N at end
        equals = re.findall(r"=\s*(\d+)\s*(?:\n|$|\.)", cleaned)
        if equals:
            return equals[-1]

        # Pattern 7: Last 1-3 digit number (AIME specific)
        all_nums = re.findall(r"\b(\d{1,3})\b", cleaned)
        if all_nums:
            for num in reversed(all_nums):
                val = int(num)
                if 0 <= val <= 999:
                    return num
        return None

    # Math (non-AIME) fallback patterns
    final = re.search(r"final\s+answer\s*[:\s]+(.+?)(?:\n|$)", cleaned, re.IGNORECASE)
    if final:
        return _clean_math_answer(final.group(1)) or None

    answer = re.search(r"(?:the\s+)?answer\s+is\s*[:\s]*(.+?)(?:\n|$)", cleaned, re.IGNORECASE)
    if answer:
        return _clean_math_answer(answer.group(1)) or None

    answer_colon = re.search(r"Answer\s*:\s*(.+?)(?:\n|$)", cleaned, re.IGNORECASE)
    if answer_colon:
        return _clean_math_answer(answer_colon.group(1)) or None

    for line in reversed(cleaned.splitlines()):
        line = line.strip()
        if line:
            return _clean_math_answer(line) or None

    return None


def declared_memory_use(response_text: str) -> bool:
    if not response_text:
        return False
    text = response_text
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = re.sub(r"<think>.*", "", text, flags=re.DOTALL)
    text = text.lower()
    patterns = [
        r"\bmemory\b",
        r"\bmemories\b",
        r"\bretrieved\s+memory\b",
        r"\bfrom\s+(?:the\s+)?memory\b",
        r"\busing\s+(?:the\s+)?memory\b",
        r"\baccording\s+to\s+(?:the\s+)?memory\b",
        r"\bpast\s+experience\b",
        r"\bprevious\s+(?:memory|experience)\b",
        r"\bi\s+remember\b",
    ]
    return any(re.search(pattern, text) for pattern in patterns)


def _append_jsonl(path: str, record: Dict[str, Any]) -> None:
    """Append a record to a JSONL file."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        f.flush()


def _append_memories_to_file(path: str, memories: List[MemoryItem]) -> None:
    """Append memories to a JSONL file."""
    if not memories:
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        for mem in memories:
            f.write(json.dumps(_memory_item_to_dict(mem), ensure_ascii=False) + "\n")
        f.flush()


def asdict_memory(mem: MemoryItem) -> Dict[str, Any]:
    """Convert MemoryItem to dict (preference_rule 写三元组 + source，不写 type)."""
    return _memory_item_to_dict(mem)


def _format_retrieved_memory(mem: MemoryItem) -> Dict[str, Any]:
    """Format retrieved memory for results output (preference_rule 输出三元组)."""
    if mem.trigger is not None and mem.dimension is not None and mem.comparison is not None:
        return {
            "trigger": mem.trigger,
            "dimension": mem.dimension,
            "comparison": mem.comparison,
        }
    return {"title": mem.title, "type": mem.type}


# ============================================================
# Main Entry Point
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Memory-Augmented Reasoning Experiments")
    parser.add_argument("--phase", choices=["produce", "evaluate", "all"], required=True,
                        help="Phase to run: produce memories, evaluate, or both")
    parser.add_argument("--dataset", default=None,
                        help="Dataset path or HF name (used when train/test not set)")
    parser.add_argument("--train_dataset", default=None,
                        help="Training dataset path or HF name (optional)")
    parser.add_argument("--test_dataset", default=None,
                        help="Test dataset path or HF name (optional)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume memory production from existing results")
    parser.add_argument("--split_ratio", type=float, default=0.8,
                        help="Train split ratio when using a single dataset")
    parser.add_argument("--split_seed", type=int, default=42,
                        help="Random seed for train/test splitting")
    parser.add_argument("--hotpotqa_train_pool_size", type=int, default=2000,
                        help="HotpotQA candidate train pool size before similarity (default: 2000)")
    parser.add_argument("--hotpotqa_train_size", type=int, default=400,
                        help="HotpotQA train subset size (default: 400)")
    parser.add_argument("--hotpotqa_test_size", type=int, default=100,
                        help="HotpotQA test subset size (default: 100)")
    parser.add_argument("--hotpotqa_train_top_k", type=int, default=4,
                        help="HotpotQA top-k similar train samples per test (default: 4)")
    parser.add_argument("--hotpotqa_lock_split_path", default=None,
                        help="Lock HotpotQA train/test ids from an existing data_split.json")
    parser.add_argument("--data_split_path", default=None,
                        help="Use an existing data_split.json to select train/test ids")
    parser.add_argument("--data_split_name", choices=["train", "test"], default=None,
                        help="Which split to evaluate when data_split_path is provided (default: test)")
    parser.add_argument("--math500_train_size", type=int, default=400,
                        help="MATH-500 fixed train subset size (default: 400)")
    parser.add_argument("--math500_test_size", type=int, default=100,
                        help="MATH-500 fixed test subset size (default: 100)")
    parser.add_argument("--task_type", choices=["aime", "math", "preference"], default="aime",
                        help="Task type for judging and prompting")
    parser.add_argument("--strategy", choices=["student", "teacher", "contrastive", "all"],
                        default="all", help="Memory extraction strategy")
    parser.add_argument("--output_dir", default="./experiments", help="Output directory")
    parser.add_argument("--experiment_name", default="aime_ablation", help="Experiment name")
    parser.add_argument("--max_samples", type=int, default=None, help="Max samples to process")
    # [FIX] 默认模型改为 qwen
    parser.add_argument("--student_model", default="qwen:qwen-flash")
    parser.add_argument("--teacher_model", default="qwen:qwen-plus")
    parser.add_argument("--expert_model", default="qwen:qwen-max")
    parser.add_argument("--judge_model", default="qwen:qwen-flash")
    parser.add_argument(
        "--extractor_model",
        default="student",
        help="Model for memory extraction; use 'student' to reuse student_model",
    )
    parser.add_argument(
        "--use_flash_extractor",
        action="store_true",
        help="Use qwen:qwen-flash as memory extractor (overrides extractor_model)",
    )
    parser.add_argument("--deepseek_reasoner_model", default="deepseek:deepseek-reasoner")
    parser.add_argument("--no_deepseek_reasoner", action="store_true",
                        help="Disable deepseek-reasoner fallback after L3 failures")
    parser.add_argument("--cascade_max_tokens", type=int, default=8192,
                        help="Max tokens for L1/L2 generation during production")
    parser.add_argument("--cascade_max_tokens_l3", type=int, default=8192,
                        help="Max tokens for L3 generation during production")
    parser.add_argument("--api_timeout", type=int, default=None,
                        help="Timeout seconds for API models")
    parser.add_argument("--api_retries", type=int, default=None,
                        help="Max retries for API models")
    parser.add_argument("--top_k", type=int, default=2, help="Top-k memories to retrieve (legacy)")
    parser.add_argument("--top_k_global", type=int, default=3,
                        help="Top-k global memories to retrieve (non-AdvancedIF/IFBench only)")
    parser.add_argument("--top_k_local", type=int, default=5,
                        help="Top-k local memories to retrieve (non-AdvancedIF/IFBench only)")
    parser.add_argument("--top_k_preference", type=int, default=5,
                        help="Top-k preference rules to retrieve (HelpSteer3 only; no local/global split)")
    parser.add_argument("--memory_retrieval", choices=["tf", "embedding"], default="tf",
                        help="Memory retrieval backend: tf or embedding")
    parser.add_argument("--enable_memory_bandit", action="store_true",
                        help="Enable SA-CTS bandit for memory retrieval and online posterior updates")
    parser.add_argument("--memory_bandit_lambda", type=float, default=0.3,
                        help="Bandit score fusion weight λ in Score=(1-λ)*sim + λ*sampled_utility")
    parser.add_argument("--memory_bandit_noise_var", type=float, default=1.0,
                        help="Likelihood noise variance σ_noise^2 for Bayesian posterior updates")
    parser.add_argument("--memory_bandit_explore_epsilon", type=float, default=0.25,
                        help="Exploration floor ε_explore added to new memory variance")
    parser.add_argument("--memory_bandit_neighbor_k", type=int, default=5,
                        help="Neighbor count n for semantic-aware prior initialization")
    parser.add_argument("--memory_bandit_prior_mu", type=float, default=0.0,
                        help="Fallback prior mean μ for memories without usable neighbors")
    parser.add_argument("--memory_bandit_prior_sigma2", type=float, default=1.0,
                        help="Fallback prior variance σ^2 for memories without usable neighbors")
    parser.add_argument("--memory_bandit_state_path", default=None,
                        help="Optional path to persist bandit posterior state JSON")
    parser.add_argument("--memory_bandit_seed", type=int, default=42,
                        help="Random seed for Thompson sampling draws")
    parser.add_argument("--embedding_model", default="Qwen/Qwen3-Embedding-4B",
                        help="HF embedding model for memory retrieval")
    parser.add_argument("--embedding_cache", default=None,
                        help="Path to embedding cache (.npz). Defaults to <memory_path>.embeddings.npz")
    parser.add_argument("--embedding_batch_size", type=int, default=32,
                        help="Batch size for embedding computation")
    parser.add_argument("--embedding_device", default=None,
                        help="Device for embedding model (e.g., cpu, cuda)")
    parser.add_argument("--no_memory_query_expand", action="store_true",
                        help="Disable query expansion before memory retrieval")
    parser.add_argument("--memory_query_expand_max_tokens", type=int, default=128,
                        help="Max tokens for query expansion generation")
    parser.add_argument("--memory_one_per_dimension", action="store_true",
                        help="Limit retrieved memories to at most one per dimension (HelpSteer2 only)")
    parser.add_argument("--train_subset_top_k", type=int, default=0,
                        help="If >0, select top-k similar train questions per test using embeddings")
    parser.add_argument("--train_subset_embedding_model", default="Qwen/Qwen3-Embedding-4B",
                        help="Embedding model for train subset selection")
    parser.add_argument("--train_subset_embedding_batch_size", type=int, default=32,
                        help="Batch size for train subset embedding computation")
    parser.add_argument("--train_subset_embedding_device", default=None,
                        help="Device for train subset embedding model (e.g., cpu, cuda)")
    parser.add_argument("--allow_upgrade_on_success", action="store_true",
                        help="Allow upgrading to higher levels even if judged successful")
    parser.add_argument("--memory_max_similarity", type=float, default=0.9,
                        help="Filter memories with similarity above this threshold")
    parser.add_argument("--memory_min_similarity", type=float, default=0.05,
                        help="Filter memories with similarity below this threshold")
    parser.add_argument("--no_memory_similarity_filter", action="store_true",
                        help="Disable similarity-based memory filtering (min/max thresholds)")
    parser.add_argument("--no_answer_leak_filter", action="store_true",
                        help="Disable expected-answer leak filtering")
    parser.add_argument("--memory_extract_mode", choices=["two_agent", "combined"], default="two_agent",
                        help="Memory extraction mode: two_agent or combined")
    parser.add_argument("--no_exclude_self_memory", action="store_true",
                        help="Allow retrieving memories from the same sample id")
    parser.add_argument("--no_extraction_reasoning", action="store_true",
                        help="Disable extra reasoning-only call for memory extraction")
    parser.add_argument("--enable_memory_refinement", action="store_true",
                        help="Enable memory refinement/consolidation after production")
    parser.add_argument("--enable_streaming_memory_refinement", action="store_true",
                        help="Enable online/streaming memory refinement during production")
    parser.add_argument("--streaming_memory_refinement_frequency", type=int, default=1,
                        help="Flush streaming refinement every N samples with newly extracted memories")
    parser.add_argument("--streaming_memory_refinement_model", default="extractor",
                        help="Model for streaming refinement: extractor/student/teacher/expert/judge or explicit model name")
    parser.add_argument("--streaming_memory_history_top_k", type=int, default=5,
                        help="Top-k historical memories retrieved for each streaming refinement call")
    parser.add_argument("--sample_ids", default=None,
                        help="Comma/space separated sample_id list (production only)")
    parser.add_argument("--sample_id_file", default=None,
                        help="File with one sample_id per line (production only)")
    parser.add_argument("--eval_only",
                        choices=["baseline", "student", "teacher", "contrastive", "all"],
                        default=None,
                        help="Run only one evaluation group (all = all_memory)")
    parser.add_argument("--eval_dual", action="store_true",
                        help="Run baseline and custom memory evaluation sequentially (requires --memory_path)")
    parser.add_argument("--eval_instruction", default=None,
                        help="Append an instruction to the system prompt during evaluation")
    parser.add_argument("--hotpotqa_strict_eval", action="store_true",
                        help="Use shorter HotpotQA evaluation prompt (A/B test)")
    parser.add_argument("--helpsteer3_domain", default=None,
                        help="Filter HelpSteer3 preference data by domain (e.g., code)")
    parser.add_argument("--memory_path", default=None,
                        help="Override memory file path for evaluation (skips default memory lookup)")
    parser.add_argument("--memory_max_items", type=int, default=0,
                        help="Limit loaded memories to the first N items (0 = load all)")
    parser.add_argument("--llm_extract_answer", action="store_true",
                        help="Use judge model to extract final answer during evaluation")
    parser.add_argument("--eval_max_tokens", type=int, default=None,
                        help="Max tokens for student generation during evaluation")
    parser.add_argument("--allow_eval_max_tokens_override", action="store_true",
                        help="Allow eval_max_tokens to exceed the default cap")
    parser.add_argument("--debug_memory_prompt", action="store_true",
                        help="Print debug info about retrieved memories and prompt injection during evaluation")
    parser.add_argument("--advancedif_judge_max_tokens", type=int, default=768,
                        help="Max tokens for AdvancedIF judge output (JSONL rubric scoring)")
    parser.add_argument("--advancedif_debug_judge", action="store_true",
                        help="For AdvancedIF: record raw judge outputs for the first N samples")
    parser.add_argument("--advancedif_debug_judge_max_samples", type=int, default=3,
                        help="For AdvancedIF: how many samples to capture raw judge output")
    parser.add_argument("--advancedif_memory_mode", default="rubric",
                        choices=["rubric", "teacher", "temp", "rubric+teacher", "rubric+temp", "teacher+temp", "all"],
                        help="AdvancedIF memory extraction mode")
    parser.add_argument("--advancedif_compare_max_tokens", type=int, default=1024,
                        help="Max tokens for AdvancedIF teacher/temp comparison generations")
    parser.add_argument("--advancedif_temp_low", type=float, default=0.3,
                        help="Low temperature for AdvancedIF temp comparison")
    parser.add_argument("--advancedif_temp_high", type=float, default=1.2,
                        help="High temperature for AdvancedIF temp comparison")
    
    # NEW: HelpSteer3 comparison mode arguments
    parser.add_argument("--helpsteer3_comparison_mode", action="store_true",
                        help="For HelpSteer3: use comparison mode (base vs RL vs memory) instead of judge task")
    parser.add_argument("--rl_model_path", default=None,
                        help="Path to RL-trained model for HelpSteer3 comparison (online DPO trained)")
    parser.add_argument("--judge_api_key", default=None,
                        help="API key for Qwen3-Max judge (or set DASHSCOPE_API_KEY env var)")
    parser.add_argument("--helpsteer3_judges", default=None,
                        help="Comma/space separated judge list (e.g., qwen3-max,deepseek-reasoner,gemini-2.5-pro, qwen3-max-rank)")
    parser.add_argument("--helpsteer3_single_mode", action="store_true",
                        help="For HelpSteer3 comparison: score only the base response (no pairwise comparisons)")
    parser.add_argument("--helpsteer3_single_mode_target", choices=["base", "memory", "rl"], default="base",
                        help="When single mode is enabled, choose which response to score (base/memory/rl)")
    parser.add_argument("--gemini_credentials", default=None,
                        help="Path to Gemini service account JSON (or set GEMINI_CREDENTIALS)")
    parser.add_argument("--gemini_location", default="us-central1",
                        help="Vertex AI location for Gemini (default: us-central1)")
    parser.add_argument("--gemini_model", default="gemini-2.5-pro",
                        help="Gemini model name (e.g., gemini-2.5-pro or gemini-2.5-flash)")
    parser.add_argument("--print_all_judges", action="store_true",
                        help="Print per-judge pairwise results in comparison mode")
    
    args = parser.parse_args()

    if args.gemini_credentials:
        os.environ["GEMINI_CREDENTIALS"] = args.gemini_credentials
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = args.gemini_credentials
    if args.gemini_location:
        os.environ["GEMINI_LOCATION"] = args.gemini_location

    train_dataset = args.train_dataset
    test_dataset = args.test_dataset
    dataset_path = args.dataset
    if dataset_path is None and train_dataset is None and test_dataset is None:
        train_dataset = DEFAULT_TRAIN_DATASET
        test_dataset = DEFAULT_TEST_DATASET
        dataset_path = DEFAULT_TRAIN_DATASET
    elif dataset_path is None:
        dataset_path = train_dataset or test_dataset or DEFAULT_TRAIN_DATASET

    prompt_profile = _infer_prompt_profile(dataset_path, train_dataset, test_dataset)
    is_math500 = any(
        _is_math500_dataset(path)
        for path in (dataset_path, train_dataset, test_dataset)
        if path
    )
    if is_math500 and args.task_type == "aime":
        args.task_type = "math"
    if prompt_profile == "hotpotqa":
        if args.task_type == "aime":
            args.task_type = "math"
        if train_dataset is None and test_dataset is None:
            args.split_ratio = 0.8
    if prompt_profile == "helpsteer3":
        args.task_type = "preference"
    
    sample_id_filter: Optional[List[str]] = None
    if args.sample_ids or args.sample_id_file:
        ids: List[str] = []
        if args.sample_ids:
            for part in args.sample_ids.replace(",", " ").split():
                if part.strip():
                    ids.append(part.strip())
        if args.sample_id_file:
            with open(args.sample_id_file, "r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    ids.append(line)
        if ids:
            # Preserve order while de-duplicating
            seen = set()
            sample_id_filter = []
            for item in ids:
                if item not in seen:
                    seen.add(item)
                    sample_id_filter.append(item)

    config = ExperimentConfig(
        dataset_path=dataset_path,
        max_samples=args.max_samples,
        train_dataset_path=train_dataset,
        test_dataset_path=test_dataset,
        split_ratio=args.split_ratio,
        split_seed=args.split_seed,
        hotpotqa_train_pool_size=args.hotpotqa_train_pool_size,
        hotpotqa_train_size=args.hotpotqa_train_size,
        hotpotqa_test_size=args.hotpotqa_test_size,
        hotpotqa_train_top_k=args.hotpotqa_train_top_k,
        hotpotqa_lock_split_path=args.hotpotqa_lock_split_path,
        math500_train_size=args.math500_train_size,
        math500_test_size=args.math500_test_size,
        task_type=args.task_type,
        prompt_profile=prompt_profile,
        strict_hotpotqa_eval=False,
        hotpotqa_short_prompt_eval=args.hotpotqa_strict_eval,
        eval_instruction=args.eval_instruction,
        helpsteer3_domain=args.helpsteer3_domain,
        data_split_path=args.data_split_path,
        data_split_name=args.data_split_name,
        memory_strategy=args.strategy,
        memory_top_k=args.top_k,
        memory_top_k_global=args.top_k_global,
        memory_top_k_local=args.top_k_local,
        memory_top_k_preference=args.top_k_preference,
        memory_retrieval=args.memory_retrieval,
        embedding_model=args.embedding_model,
        embedding_cache_path=args.embedding_cache,
        embedding_batch_size=args.embedding_batch_size,
        embedding_device=args.embedding_device,
        memory_query_expand=not args.no_memory_query_expand,
        memory_query_expand_max_tokens=args.memory_query_expand_max_tokens,
        memory_extract_mode=args.memory_extract_mode,
        memory_one_per_dimension=args.memory_one_per_dimension,
        streaming_memory_refinement=args.enable_streaming_memory_refinement,
        streaming_memory_refinement_frequency=args.streaming_memory_refinement_frequency,
        streaming_memory_refinement_model=args.streaming_memory_refinement_model,
        streaming_memory_history_top_k=args.streaming_memory_history_top_k,
        enable_memory_bandit=args.enable_memory_bandit,
        memory_bandit_lambda=args.memory_bandit_lambda,
        memory_bandit_noise_var=args.memory_bandit_noise_var,
        memory_bandit_explore_epsilon=args.memory_bandit_explore_epsilon,
        memory_bandit_neighbor_k=args.memory_bandit_neighbor_k,
        memory_bandit_prior_mu=args.memory_bandit_prior_mu,
        memory_bandit_prior_sigma2=args.memory_bandit_prior_sigma2,
        memory_bandit_state_path=args.memory_bandit_state_path,
        memory_bandit_seed=args.memory_bandit_seed,
        advancedif_judge_max_tokens=args.advancedif_judge_max_tokens,
        advancedif_debug_judge=args.advancedif_debug_judge,
        advancedif_debug_judge_max_samples=args.advancedif_debug_judge_max_samples,
        advancedif_memory_mode=args.advancedif_memory_mode,
        advancedif_compare_max_tokens=args.advancedif_compare_max_tokens,
        advancedif_temp_low=args.advancedif_temp_low,
        advancedif_temp_high=args.advancedif_temp_high,
        train_subset_top_k=args.train_subset_top_k,
        train_subset_embedding_model=args.train_subset_embedding_model,
        train_subset_embedding_batch_size=args.train_subset_embedding_batch_size,
        train_subset_embedding_device=args.train_subset_embedding_device,
        memory_answer_leak_filter=not args.no_answer_leak_filter,
        memory_max_similarity=(
            None if args.no_memory_similarity_filter else args.memory_max_similarity
        ),
        memory_min_similarity=(
            None if args.no_memory_similarity_filter else args.memory_min_similarity
        ),
        memory_max_items=args.memory_max_items,
        exclude_self_memory=not args.no_exclude_self_memory,
        extraction_reasoning=not args.no_extraction_reasoning,
        enable_memory_refinement=args.enable_memory_refinement,
        upgrade_on_failure_only=not args.allow_upgrade_on_success,
        output_dir=args.output_dir,
        experiment_name=args.experiment_name,
        cascade_max_tokens=args.cascade_max_tokens,
        cascade_max_tokens_l3=args.cascade_max_tokens_l3,
        student_model=args.student_model,
        teacher_model=args.teacher_model,
        expert_model=args.expert_model,
        judge_model=args.judge_model,
        extractor_model=("qwen:qwen-flash" if args.use_flash_extractor else args.extractor_model),
        deepseek_reasoner_model=None if args.no_deepseek_reasoner else args.deepseek_reasoner_model,
        api_timeout=args.api_timeout,
        api_retries=args.api_retries,
        resume_production=args.resume,
        llm_extract_answer=args.llm_extract_answer,
        sample_id_filter=sample_id_filter,
    )

    if prompt_profile == "advancedif":
        # AdvancedIF/IFBench use preference_rule only (no global/local split).
        if args.top_k_preference == 5 and (args.top_k_global != 3 or args.top_k_local != 5):
            config.memory_top_k_preference = max(args.top_k_global, args.top_k_local)
            print(
                "[Info] AdvancedIF/IFBench ignores --top_k_global/--top_k_local; "
                f"using top_k_preference={config.memory_top_k_preference}."
            )
        config.memory_top_k_global = 0
        config.memory_top_k_local = 0
    
    if args.phase in ("produce", "all"):
        # For "all" strategy in production, we run once and extract all types
        run_memory_production(config)
    
    if args.phase in ("evaluate", "all"):
        eval_max_tokens = args.eval_max_tokens
        debug_memory_prompt = args.debug_memory_prompt
        prev_override = None
        if args.allow_eval_max_tokens_override:
            prev_override = os.environ.get("ALLOW_MAX_TOKENS_OVERRIDE")
            os.environ["ALLOW_MAX_TOKENS_OVERRIDE"] = "1"
        try:
            # NEW: If comparison mode is enabled, always use run_all_evaluations
            if args.helpsteer3_comparison_mode:
                run_all_evaluations(
                    config,
                    eval_only=args.eval_only,
                    eval_max_tokens=eval_max_tokens,
                    debug_memory_prompt=debug_memory_prompt,
                    helpsteer3_comparison_mode=True,
                    helpsteer3_single_mode=args.helpsteer3_single_mode,
                    helpsteer3_single_mode_target=args.helpsteer3_single_mode_target,
                    rl_model_path=args.rl_model_path,
                    judge_api_key=args.judge_api_key,
                    helpsteer3_judges=args.helpsteer3_judges,
                    gemini_credentials=args.gemini_credentials,
                    gemini_location=args.gemini_location,
                    gemini_model=args.gemini_model,
                    print_all_judges=args.print_all_judges,
                    override_memory_path=args.memory_path,  # Use manually specified memory path
                )
            elif args.eval_dual:
                if not args.memory_path:
                    raise ValueError("--eval_dual requires --memory_path")
                results = {}
                results["baseline"] = run_evaluation(
                    config=config,
                    memory_path=None,
                    experiment_tag="baseline",
                    eval_max_tokens=eval_max_tokens,
                    debug_memory_prompt=debug_memory_prompt,
                )
                results["memory"] = run_evaluation(
                    config=config,
                    memory_path=args.memory_path,
                    experiment_tag="memory",
                    eval_max_tokens=eval_max_tokens,
                    debug_memory_prompt=debug_memory_prompt,
                )
                if results:
                    print("\n" + "="*60)
                    print("COMPARISON SUMMARY")
                    print("="*60)
                    is_advancedif = any("rubric_accuracy" in res for res in results.values())
                    is_helpsteer2 = any("dimension_score_avg" in res for res in results.values())
                    has_qa = any(res.get("em") is not None for res in results.values()) if not (is_advancedif or is_helpsteer2) else False
                    if is_advancedif:
                        print(f"{'Group':<15} {'Rubric%':<10} {'Score':<14}")
                        print("-"*40)
                    elif is_helpsteer2:
                        print(f"{'Group':<15} {'Dim%':<10} {'Weighted%':<10} {'Score':<14}")
                        print("-"*55)
                    elif has_qa:
                        print(f"{'Group':<15} {'EM':<8} {'F1':<8} {'Correct':<10}")
                        print("-"*45)
                    else:
                        print(f"{'Group':<15} {'Accuracy':<12} {'Correct':<10}")
                        print("-"*40)
                    for name, res in results.items():
                        if is_advancedif:
                            rub_acc = res.get("rubric_accuracy") or 0.0
                            rub_sum = res.get("rubric_score_sum") or 0.0
                            rub_cnt = res.get("rubric_count") or 0
                            print(f"{name:<15} {100*rub_acc:>6.1f}%     {rub_sum:.2f}/{rub_cnt}")
                            continue
                        if is_helpsteer2:
                            dim_pct = res.get("dimension_score_pct") or 0.0
                            dim_sum = res.get("dimension_score_sum") or 0.0
                            dim_cnt = res.get("dimension_count") or 0
                            weighted_pct = res.get("weighted_score_pct") or 0.0
                            weighted_sum = res.get("weighted_score_sum") or 0.0
                            max_total = dim_cnt * HELPSTEER2_SCORE_VALUES[-1] if dim_cnt else 0.0
                            score_str = f"{dim_sum:.2f}/{max_total:.1f}" if max_total else "0.00/0.0"
                            if dim_cnt:
                                total_samples = dim_cnt / len(HELPSTEER2_DIMENSIONS)
                                max_weighted = total_samples * HELPSTEER2_WEIGHT_TOTAL * HELPSTEER2_SCORE_VALUES[-1]
                                weighted_str = f"{weighted_sum:.2f}/{max_weighted:.1f}" if max_weighted else "0.00/0.0"
                            else:
                                weighted_str = "0.00/0.0"
                            print(
                                f"{name:<15} {100*dim_pct:>6.1f}%     {100*weighted_pct:>6.1f}%     {score_str} | {weighted_str}"
                            )
                            continue
                        correct = res.get("stats", {}).get("correct", 0)
                        total = res.get("stats", {}).get("total", 0)
                        acc = (correct / total) if total else 0.0
                        if has_qa:
                            em = res.get("em") or 0.0
                            f1 = res.get("f1") or 0.0
                            print(f"{name:<15} {100*em:>6.1f}%  {100*f1:>6.1f}%  {correct}/{total}")
                        else:
                            print(f"{name:<15} {100*acc:>6.1f}%     {correct}/{total}")
                _export_aime_base_wrong_memory_right_cases(config, results)
            elif args.memory_path:
                # Single memory evaluation (original behavior)
                run_evaluation(
                    config=config,
                    memory_path=args.memory_path,
                    experiment_tag="memory_custom",
                    eval_max_tokens=eval_max_tokens,
                    debug_memory_prompt=debug_memory_prompt,
                )
            else:
                run_all_evaluations(
                    config,
                    eval_only=args.eval_only,
                    eval_max_tokens=eval_max_tokens,
                    debug_memory_prompt=debug_memory_prompt,
                    helpsteer3_comparison_mode=False,
                    helpsteer3_single_mode=args.helpsteer3_single_mode,
                    helpsteer3_single_mode_target=args.helpsteer3_single_mode_target,
                    rl_model_path=args.rl_model_path,
                    judge_api_key=args.judge_api_key,
                    helpsteer3_judges=args.helpsteer3_judges,
                    gemini_credentials=args.gemini_credentials,
                    gemini_location=args.gemini_location,
                    gemini_model=args.gemini_model,
                    print_all_judges=args.print_all_judges,
                    override_memory_path=None,
                )
        finally:
            if args.allow_eval_max_tokens_override:
                if prev_override is None:
                    os.environ.pop("ALLOW_MAX_TOKENS_OVERRIDE", None)
                else:
                    os.environ["ALLOW_MAX_TOKENS_OVERRIDE"] = prev_override


if __name__ == "__main__":
    main()
