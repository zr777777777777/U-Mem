from __future__ import annotations

import ast
import json
import math
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple

from dataset_utils import Sample
from models import Model, ModelError, ModelResponse


@dataclass
class CascadeConfig:
    threshold: float = 0.6
    enable_tools: bool = True
    auto_tool: bool = False
    temperature: float = 0.0
    max_tokens: int = 8192
    max_tokens_l3: int = 8192
    upgrade_on_failure_only: bool = True
    task_type: str = "math"
    prompt_profile: Optional[str] = None


ToolFunc = Callable[[Dict[str, Any]], Dict[str, Any]]


@dataclass
class ToolCall:
    name: str
    args: Dict[str, Any]


def build_default_tools() -> Dict[str, ToolFunc]:
    return {
        "python_eval": _tool_python_eval,
        "sympy_verify": _tool_sympy_verify,
        "text_check": _tool_text_check,
        "regex_extract": _tool_regex_extract,
    }


def run_cascade(
    sample: Sample,
    system_prompt: str,
    student: Model,
    teacher: Model,
    expert: Model,
    judge_model: Model,
    config: CascadeConfig,
    deepseek_reasoner: Optional[Model] = None,
    tools: Optional[Dict[str, ToolFunc]] = None,
) -> Tuple[List[Dict[str, Any]], str, Dict[str, Any]]:
    """
    Run the cascade of models on a sample.
    
    Args:
        sample: The sample to process
        system_prompt: System prompt for all models
        student: L1 student model
        teacher: L2 teacher model
        expert: L3 expert model
        judge_model: Model for judging responses
        config: Cascade configuration
        deepseek_reasoner: Optional L4 DeepSeek reasoner model
        tools: Optional tools dictionary
    
    Returns:
        Tuple of (trajectories, final_answer, final_judge)
    """
    tools = tools or {}
    trajectories: List[Dict[str, Any]] = []

    traj1 = _run_level(
        level=1,
        sample=sample,
        model=student,
        system_prompt=system_prompt,
        judge_model=judge_model,
        config=config,
        allow_tools=False,
        tools=tools,
    )
    trajectories.append(traj1)

    if _should_upgrade(traj1, config):
        traj2 = _run_level(
            level=2,
            sample=sample,
            model=teacher,
            system_prompt=system_prompt,
            judge_model=judge_model,
            config=config,
            allow_tools=config.enable_tools,
            tools=tools,
        )
        trajectories.append(traj2)
        if _should_upgrade(traj2, config):
            traj3 = _run_level(
                level=3,
                sample=sample,
                model=expert,
                system_prompt=system_prompt,
                judge_model=judge_model,
                config=config,
                allow_tools=False,
                tools=tools,
            )
            trajectories.append(traj3)
            if deepseek_reasoner and traj3.get("judge", {}).get("label") != "success":
                traj4 = _run_level(
                    level=4,
                    sample=sample,
                    model=deepseek_reasoner,
                    system_prompt=system_prompt,
                    judge_model=judge_model,
                    config=config,
                    allow_tools=False,
                    tools=tools,
                )
                trajectories.append(traj4)

    final_traj = trajectories[-1]
    for candidate in reversed(trajectories):
        if candidate["judge"]["label"] == "success":
            final_traj = candidate
            break

    final_answer = final_traj.get("parsed_answer") or final_traj.get("response")
    return trajectories, final_answer, final_traj["judge"]


def _run_level(
    level: int,
    sample: Sample,
    model: Model,
    system_prompt: str,
    judge_model: Model,
    config: CascadeConfig,
    allow_tools: bool,
    tools: Dict[str, ToolFunc],
) -> Dict[str, Any]:
    user_prompt = _build_user_prompt(sample.question, config.task_type, config.prompt_profile)
    if config.prompt_profile == "hotpotqa" and level in (2, 3):
        user_prompt = (
            f"{user_prompt}\n\n"
            "Return only a single answer. Do not output anything else."
        )
    system = _augment_system_prompt(system_prompt, allow_tools)
    timestamp = _iso_timestamp()
    prompt_for_model = _compose_prompt(system, user_prompt)

    responses: List[ModelResponse] = []
    tool_calls: List[ToolCall] = []
    tool_results: List[Dict[str, Any]] = []
    intermediate: Dict[str, Any] = {}
    level_max_tokens = config.max_tokens_l3 if level == 3 else config.max_tokens

    try:
        response = model.generate(
            prompt=user_prompt,
            system_prompt=system,
            temperature=config.temperature,
            max_tokens=level_max_tokens,
        )
        responses.append(response)
    except ModelError as exc:
        response = ModelResponse(
            text=f"Model error: {exc}",
            prompt_tokens=None,
            completion_tokens=None,
            total_tokens=None,
            raw=None,
            estimated=True,
        )

    initial_text = response.text
    if allow_tools and tools:
        tool_calls = _extract_tool_calls(initial_text)
        if tool_calls:
            tool_results = _execute_tool_calls(tool_calls, tools)
            follow_prompt = _build_tool_followup_prompt(user_prompt, initial_text, tool_results)
            prompt_for_model = _compose_prompt(system, follow_prompt)
            intermediate = {
                "initial_prompt": _compose_prompt(system, user_prompt),
                "initial_response": initial_text,
            }
            try:
                follow_response = model.generate(
                    prompt=follow_prompt,
                    system_prompt=system,
                    temperature=config.temperature,
                    max_tokens=level_max_tokens,
                )
                responses.append(follow_response)
                response = follow_response
            except ModelError as exc:
                response = ModelResponse(
                    text=f"Model error after tool call: {exc}",
                    prompt_tokens=None,
                    completion_tokens=None,
                    total_tokens=None,
                    raw=None,
                    estimated=True,
                )

    parsed_answer = _parse_final_answer(response.text, config.task_type, config.prompt_profile)
    fallback_info: Optional[Dict[str, Any]] = None
    # L3 fallback: if no parsable answer, ask for boxed answer first, then brief reasoning.
    if level == 3 and not parsed_answer:
        if config.prompt_profile == "hotpotqa":
            fallback_user = (
                f"{sample.question}\n\n"
                "Your previous response did not clearly include an answer. "
                "Please respond with a single line in the format:\n"
                "\\boxed{<your answer>}\n\n"
                "Previous response:\n"
                f"{response.text}"
            )
        else:
            fallback_user = (
                f"{sample.question}\n\n"
                "Your previous response did not clearly include a final boxed answer. "
                "First output the final answer in \\boxed{...} on its own line. "
                "Then provide a brief step-by-step reasoning (2-4 sentences). "
                "Please make your reasoning process concise yet comprehensive. "
                "Do NOT repeat the boxed answer after the first line.\n\n"
                "Previous response:\n"
                f"{response.text}"
            )
        try:
            fallback_resp = model.generate(
                prompt=fallback_user,
                system_prompt=system,
                temperature=config.temperature,
                max_tokens=min(256, level_max_tokens),
            )
            responses.append(fallback_resp)
            fallback_answer = _parse_final_answer(
                fallback_resp.text,
                config.task_type,
                config.prompt_profile,
            )
            fallback_info = {
                "response": fallback_resp.text,
                "parsed_answer": fallback_answer,
            }
            if fallback_answer:
                parsed_answer = fallback_answer
        except ModelError:
            pass
    parsed_reasoning = _extract_reasoning(response.text)
    signals = _compute_signals(response.text, parsed_answer)
    judge = _judge_response(
        sample,
        response.text,
        parsed_answer,
        judge_model,
        config.task_type,
        config.prompt_profile,
    )

    if config.auto_tool and tools:
        auto_signal = _auto_tool_check(response.text, parsed_answer, judge, tools, config.threshold)
        if auto_signal:
            signals.update(auto_signal)

    cost = _aggregate_costs(responses)

    trajectory: Dict[str, Any] = {
        "level": level,
        "model_name": model.name,
        "timestamp": timestamp,
        "prompt": prompt_for_model,
        "response": response.text,
        "parsed_answer": parsed_answer,
        "parsed_reasoning": parsed_reasoning,
        "judge": judge,
        "signals": signals,
        "cost": cost,
    }
    if tool_calls:
        trajectory["tool_calls"] = [call.__dict__ for call in tool_calls]
        trajectory["tool_results"] = tool_results
    if intermediate:
        trajectory["intermediate"] = intermediate
    if fallback_info:
        trajectory["fallback"] = fallback_info
    return trajectory


def _aggregate_costs(responses: List[ModelResponse]) -> Dict[str, Any]:
    prompt_tokens = 0
    completion_tokens = 0
    total_tokens = 0
    estimated = False
    for resp in responses:
        if resp.prompt_tokens is None or resp.completion_tokens is None or resp.total_tokens is None:
            estimated = True
            continue
        prompt_tokens += resp.prompt_tokens
        completion_tokens += resp.completion_tokens
        total_tokens += resp.total_tokens
        estimated = estimated or resp.estimated
    if not responses:
        return {"prompt_tokens": None, "completion_tokens": None, "total_tokens": None}
    if prompt_tokens == 0 and completion_tokens == 0:
        return {"prompt_tokens": None, "completion_tokens": None, "total_tokens": None}
    cost: Dict[str, Any] = {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
    }
    if estimated:
        cost["estimated"] = True
    return cost


def _should_upgrade(trajectory: Dict[str, Any], config: CascadeConfig) -> bool:
    """Decide whether to upgrade to the next level."""
    judge = trajectory.get("judge", {})
    label = judge.get("label")
    score = judge.get("score", 0.0)
    
    if config.upgrade_on_failure_only:
        return label != "success"
    
    return label != "success" or score < config.threshold


def _build_user_prompt(
    question: str,
    task_type: str,
    prompt_profile: Optional[str] = None,
) -> str:
    """
    [FIX] 与 test_fixed.py 保持一致的 user prompt 格式
    使用官方推荐的 \\boxed{} 格式
    """
    if prompt_profile == "hotpotqa":
        return (
            f"{question}\n\n"
            "Please answer using only the provided context. "
            "Put the final answer within \\boxed{}. "
            "After you output the boxed answer, STOP immediately and do not repeat it."
        )
    # 官方文档推荐: "Include 'Please reason step by step, and put your final answer within \\boxed{}.'"
    return (
        f"{question}\n\n"
        "Please reason step by step, and put your final answer within \\boxed{}. "
        "Please make your reasoning process concise yet comprehensive. "
        "After you output the boxed answer, STOP immediately and do not repeat it."
    )


def _augment_system_prompt(base_prompt: str, allow_tools: bool) -> str:
    """Add tool instructions to system prompt if tools are enabled."""
    if not allow_tools:
        return base_prompt
    
    tool_instructions = """

You have access to the following tools:
- python_eval: Evaluate a Python expression. Usage: <tool_call>{"name": "python_eval", "args": {"expression": "..."}}</tool_call>
- sympy_verify: Verify mathematical equations using SymPy. Usage: <tool_call>{"name": "sympy_verify", "args": {"equation": "..."}}</tool_call>

Use tools when helpful for verification, but always provide your reasoning."""
    
    return base_prompt + tool_instructions


def _compose_prompt(system: str, user: str) -> str:
    """Compose full prompt from system and user parts."""
    return f"System: {system}\n\nUser: {user}"


def _parse_final_answer(
    text: str,
    task_type: str,
    prompt_profile: Optional[str] = None,
) -> str:
    """
    [FIX] 与 test_fixed.py 保持一致的答案解析
    优先解析 \\boxed{} 格式
    """
    # 先移除 <think>...</think> 内容
    cleaned = text
    think_end = cleaned.find("</think>")
    if think_end != -1:
        cleaned = cleaned[think_end + len("</think>"):].strip()
    cleaned = re.sub(r"<think>.*?</think>", "", cleaned, flags=re.DOTALL)
    cleaned = re.sub(r"<think>.*", "", cleaned, flags=re.DOTALL)

    if not cleaned.strip():
        return ""

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
                    start += 0
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
                            content = text[(start + len("\\boxed{")) : i] if text[start:start + 6] == "\\boxed" else text[(start + len("boxed{")) : i]
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
        return _trim_hotpotqa_answer(cleaned) or ""

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

    # Pattern 1: \\boxed{...} (官方推荐格式，优先)
    boxed = _extract_boxed_content(cleaned)
    if boxed:
        if numeric_only:
            num_match = re.search(r"(-?\d+)", boxed)
            return num_match.group(1) if num_match else ""
        return _clean_math_answer(boxed) or ""

    # Pattern 2: ANSWER: ...
    answer_line = ""
    for answer_match in re.finditer(r"(?im)^\s*ANSWER\s*:\s*(.+)$", cleaned):
        answer_line = answer_match.group(1).strip()
    if answer_line:
        if numeric_only:
            num_match = re.search(r"(-?\d+)", answer_line)
            return num_match.group(1) if num_match else ""
        return _clean_math_answer(answer_line) or ""

    if numeric_only:
        # Pattern 3: Final Answer: N
        final_match = re.search(r"final\s+answer\s*[:\s]+(\d+)", cleaned, re.IGNORECASE)
        if final_match:
            return final_match.group(1)

        # Pattern 4: The answer is N
        is_match = re.search(r"(?:the\s+)?answer\s+is\s*[:\s]*(\d+)", cleaned, re.IGNORECASE)
        if is_match:
            return is_match.group(1)
        return ""

    # Math (non-AIME) fallback patterns
    final_match = re.search(r"final\s+answer\s*[:\s]+(.+?)(?:\n|$)", cleaned, re.IGNORECASE)
    if final_match:
        return _clean_math_answer(final_match.group(1)) or ""

    is_match = re.search(r"(?:the\s+)?answer\s+is\s*[:\s]*(.+?)(?:\n|$)", cleaned, re.IGNORECASE)
    if is_match:
        return _clean_math_answer(is_match.group(1)) or ""

    answer_colon = re.search(r"Answer\s*:\s*(.+?)(?:\n|$)", cleaned, re.IGNORECASE)
    if answer_colon:
        return _clean_math_answer(answer_colon.group(1)) or ""

    for line in reversed(cleaned.splitlines()):
        line = line.strip()
        if line:
            return _clean_math_answer(line) or ""

    return ""
def _extract_reasoning(text: str) -> str:
    """Extract reasoning/thinking process from response."""
    # Remove final answer part
    reasoning = re.sub(r"[Ff]inal [Aa]nswer.*$", "", text, flags=re.MULTILINE | re.DOTALL)
    return reasoning.strip()


def _compute_signals(text: str, parsed_answer: str) -> Dict[str, Any]:
    """Compute various signals from the response."""
    signals: Dict[str, Any] = {}
    
    # Check for uncertainty markers
    uncertainty_markers = ["i think", "maybe", "not sure", "possibly", "approximately"]
    signals["has_uncertainty"] = any(marker in text.lower() for marker in uncertainty_markers)
    
    # Check for verification
    verification_markers = ["let me verify", "checking", "double-check", "verify"]
    signals["has_verification"] = any(marker in text.lower() for marker in verification_markers)
    
    # Response length
    signals["response_length"] = len(text)
    signals["has_answer"] = bool(parsed_answer)
    
    return signals


def _normalize_qa_answer(text: Any) -> str:
    if text is None:
        return ""
    cleaned = str(text).strip().lower()
    cleaned = cleaned.replace(",", "")
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if re.fullmatch(r"\d+(?: \d+)+", cleaned):
        cleaned = cleaned.replace(" ", "")
    cleaned = cleaned.strip(" .;:!?\"'")
    return cleaned


def _judge_response(
    sample: Sample,
    response: str,
    parsed_answer: str,
    judge_model: Model,
    task_type: str = "math",
    prompt_profile: Optional[str] = None,
) -> Dict[str, Any]:
    """Judge the response using the judge model or exact match."""
    ground_truth = sample.answer

    if prompt_profile == "hotpotqa":
        pred_norm = _normalize_qa_answer(parsed_answer)
        gt_norm = _normalize_qa_answer(ground_truth)
        if pred_norm and gt_norm and pred_norm == gt_norm:
            return {
                "label": "success",
                "score": 1.0,
                "method": "exact_match",
                "details": {"predicted": parsed_answer, "ground_truth": ground_truth},
            }
        return {
            "label": "failure",
            "score": 0.0,
            "method": "exact_match",
            "details": {"predicted": parsed_answer, "ground_truth": ground_truth},
        }
    
    def _normalize_math_answer(text: Any) -> str:
        if text is None:
            return ""
        cleaned = str(text).strip()
        if cleaned.startswith("$") and cleaned.endswith("$") and len(cleaned) >= 2:
            cleaned = cleaned[1:-1].strip()
        cleaned = cleaned.replace("\\left", "").replace("\\right", "")
        boxed = re.search(r"\\boxed\{(.+)\}", cleaned)
        if boxed:
            cleaned = boxed.group(1).strip()
        for pat in (
            r"^\\text\{(.+)\}$",
            r"^\\mathrm\{(.+)\}$",
            r"^\\mathbf\{(.+)\}$",
        ):
            m = re.match(pat, cleaned)
            if m:
                cleaned = m.group(1).strip()
                break
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
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        return cleaned.lower()

    if task_type.lower() == "aime":
        # For AIME problems, use exact integer match
        if parsed_answer and ground_truth:
            try:
                pred_int = int(parsed_answer)
                gt_int = int(ground_truth)
                if pred_int == gt_int:
                    return {
                        "label": "success",
                        "score": 1.0,
                        "method": "exact_match",
                        "details": {"predicted": pred_int, "ground_truth": gt_int}
                    }
                else:
                    return {
                        "label": "failure",
                        "score": 0.0,
                        "method": "exact_match",
                        "details": {"predicted": pred_int, "ground_truth": gt_int}
                    }
            except (ValueError, TypeError):
                pass
    else:
        pred_norm = _normalize_math_answer(parsed_answer)
        gt_norm = _normalize_math_answer(ground_truth)
        if pred_norm and gt_norm and pred_norm == gt_norm:
            return {
                "label": "success",
                "score": 1.0,
                "method": "exact_match",
                "details": {"predicted": pred_norm, "ground_truth": gt_norm},
            }
        pred_num = _parse_number(pred_norm or ("" if parsed_answer is None else str(parsed_answer)))
        gt_num = _parse_number(gt_norm or ("" if ground_truth is None else str(ground_truth)))
        if pred_num is not None and gt_num is not None:
            if math.isclose(pred_num, gt_num, rel_tol=1e-6, abs_tol=1e-8):
                return {
                    "label": "success",
                    "score": 1.0,
                    "method": "numeric_match",
                    "details": {"predicted": pred_num, "ground_truth": gt_num},
                }
            return {
                "label": "failure",
                "score": 0.0,
                "method": "numeric_mismatch",
                "details": {"predicted": pred_num, "ground_truth": gt_num},
            }
    
    # Fallback to LLM judge if exact match fails
    try:
        judge_prompt = f"""Judge if the following answer is correct.

Question: {sample.question}
Expected Answer: {ground_truth}
Given Answer: {parsed_answer}
Full Response: {response[:1000]}

Is the answer correct? Respond with JSON: {{"correct": true/false, "confidence": 0.0-1.0, "reason": "..."}}"""
        
        system_prompt = (
            "You are a question-answering judge. Be precise."
            if prompt_profile == "hotpotqa"
            else "You are a math answer judge. Be precise."
        )
        judge_response = judge_model.generate(
            prompt=judge_prompt,
            system_prompt=system_prompt,
            temperature=0.0,
            do_sample=False,
            max_tokens=200,
        )
        
        # Parse judge response
        json_match = re.search(r"\{[^}]+\}", judge_response.text)
        if json_match:
            data = json.loads(json_match.group())
            is_correct = data.get("correct", False)
            confidence = float(data.get("confidence", 0.5))
            return {
                "label": "success" if is_correct else "failure",
                "score": confidence if is_correct else 1.0 - confidence,
                "method": "llm_judge",
                "details": {
                    "predicted": parsed_answer,
                    "ground_truth": ground_truth,
                    "reason": data.get("reason", ""),
                }
            }
    except (ModelError, json.JSONDecodeError, KeyError, ValueError):
        pass
    
    # Final fallback
    return {
        "label": "failure",
        "score": 0.0,
        "method": "fallback",
        "details": {"predicted": parsed_answer, "ground_truth": ground_truth}
    }


def _build_tool_followup_prompt(
    original_prompt: str,
    initial_response: str,
    tool_results: List[Dict[str, Any]],
) -> str:
    """Build follow-up prompt after tool execution."""
    results_text = "\n".join(
        f"- {r['name']}: {r.get('result', r.get('error', 'unknown'))}"
        for r in tool_results
    )
    
    return f"""{original_prompt}

Your initial response:
{initial_response}

Tool results:
{results_text}

Based on these tool results, please provide your final answer."""


def _extract_tool_calls(text: str) -> List[ToolCall]:
    calls: List[ToolCall] = []
    idx = 0
    start_tag = "<tool_call>"
    end_tag = "</tool_call>"
    while True:
        start = text.find(start_tag, idx)
        if start == -1:
            break
        end = text.find(end_tag, start)
        if end == -1:
            break
        block = text[start + len(start_tag) : end]
        obj = _parse_json_object(block)
        if obj and "name" in obj:
            calls.append(ToolCall(name=obj["name"], args=obj.get("args", {})))
        idx = end + len(end_tag)
    return calls


def _execute_tool_calls(calls: List[ToolCall], tools: Dict[str, ToolFunc]) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    for call in calls:
        tool = tools.get(call.name)
        if not tool:
            results.append({"name": call.name, "error": "tool not found"})
            continue
        try:
            result = tool(call.args)
            results.append({"name": call.name, "result": result})
        except Exception as exc:
            results.append({"name": call.name, "error": str(exc)})
    return results


def _auto_tool_check(
    response_text: str,
    parsed_answer: str,
    judge: Dict[str, Any],
    tools: Dict[str, ToolFunc],
    threshold: float,
) -> Dict[str, Any]:
    if parsed_answer and judge.get("score", 1.0) >= threshold:
        return {}
    expression = _extract_expression(response_text)
    if not expression:
        return {}
    if "=" in expression and "sympy_verify" in tools:
        result = tools["sympy_verify"]({"equation": expression})
        signal: Dict[str, Any] = {"auto_tool": {"name": "sympy_verify", "result": result}}
        if isinstance(result, dict) and "equal" in result:
            signal["verified"] = bool(result.get("equal"))
        return signal
    if "python_eval" in tools:
        result = tools["python_eval"]({"expression": expression})
        signal = {"auto_tool": {"name": "python_eval", "result": result}}
        if isinstance(result, dict) and "value" in result and parsed_answer:
            parsed_num = _parse_number(parsed_answer)
            if parsed_num is not None:
                try:
                    signal["verified"] = math.isclose(
                        float(result["value"]), parsed_num, rel_tol=1e-4, abs_tol=1e-6
                    )
                except (TypeError, ValueError):
                    pass
        return signal
    return {}


def _extract_expression(text: str) -> str:
    candidates = re.findall(r"[0-9\s\+\-\*/\^\(\)\.=]+", text)
    for cand in candidates:
        cleaned = cand.strip()
        if len(cleaned) >= 3 and any(ch.isdigit() for ch in cleaned):
            return cleaned
    return ""


def _parse_number(text: str) -> Optional[float]:
    """Parse a number from text."""
    try:
        return float(text)
    except (ValueError, TypeError):
        return None


def _parse_json_object(text: str) -> Optional[Dict[str, Any]]:
    decoder = json.JSONDecoder()
    for idx, ch in enumerate(text):
        if ch != "{":
            continue
        try:
            obj, _ = decoder.raw_decode(text[idx:])
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            continue
    return None


def _iso_timestamp() -> str:
    return datetime.utcnow().isoformat() + "Z"


def _tool_python_eval(args: Dict[str, Any]) -> Dict[str, Any]:
    expression = args.get("expression") or args.get("expr")
    if not expression:
        return {"error": "expression is required"}
    try:
        value = _safe_eval(expression)
        return {"value": value}
    except Exception as exc:
        return {"error": str(exc)}


def _tool_sympy_verify(args: Dict[str, Any]) -> Dict[str, Any]:
    try:
        import sympy as sp  # type: ignore
    except ImportError:
        return {"error": "sympy not available"}
    expression = args.get("expression")
    lhs = args.get("lhs")
    rhs = args.get("rhs")
    equation = args.get("equation")
    variable = args.get("variable", "x")
    try:
        if equation:
            var = sp.symbols(variable)
            eq = sp.sympify(equation)
            if isinstance(eq, sp.Equality):
                expr = eq.lhs - eq.rhs
            else:
                expr = eq
            solutions = sp.solve(expr, var)
            return {"solutions": [str(sol) for sol in solutions]}
        if lhs is not None and rhs is not None:
            expr = sp.simplify(sp.sympify(lhs) - sp.sympify(rhs))
            return {"equal": expr == 0, "simplified": str(expr)}
        if expression:
            expr = sp.sympify(expression)
            simplified = sp.simplify(expr)
            return {"simplified": str(simplified)}
    except Exception as exc:
        return {"error": str(exc)}
    return {"error": "no valid input"}


def _tool_text_check(args: Dict[str, Any]) -> Dict[str, Any]:
    text = args.get("text", "")
    require_final = bool(args.get("require_final", True))
    warnings: List[str] = []
    if require_final and "final answer" not in text.lower():
        warnings.append("missing_final_answer")

    assignments = re.findall(r"([a-zA-Z]\w*)\s*=\s*([-+]?\d*\.?\d+)", text)
    values: Dict[str, set] = {}
    for var, val in assignments:
        values.setdefault(var, set()).add(val)
    for var, vals in values.items():
        if len(vals) > 1:
            warnings.append(f"contradiction_{var}")

    return {"ok": len(warnings) == 0, "warnings": warnings}


def _tool_regex_extract(args: Dict[str, Any]) -> Dict[str, Any]:
    pattern = args.get("pattern")
    text = args.get("text", "")
    if not pattern:
        return {"error": "pattern is required"}
    flags = 0
    if args.get("ignore_case"):
        flags |= re.IGNORECASE
    try:
        matches = re.findall(pattern, text, flags)
        return {"matches": matches}
    except re.error as exc:
        return {"error": str(exc)}


def _safe_eval(expression: str) -> Any:
    node = ast.parse(expression, mode="eval")
    allowed_funcs = {
        "abs": abs,
        "round": round,
        "min": min,
        "max": max,
        "sum": sum,
        "sqrt": math.sqrt,
        "sin": math.sin,
        "cos": math.cos,
        "tan": math.tan,
        "log": math.log,
        "log10": math.log10,
        "exp": math.exp,
        "pow": pow,
        "pi": math.pi,
        "e": math.e,
    }

    def _eval(node: ast.AST) -> Any:
        if isinstance(node, ast.Expression):
            return _eval(node.body)
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.BinOp):
            left = _eval(node.left)
            right = _eval(node.right)
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            if isinstance(node.op, ast.Div):
                return left / right
            if isinstance(node.op, ast.FloorDiv):
                return left // right
            if isinstance(node.op, ast.Mod):
                return left % right
            if isinstance(node.op, ast.Pow):
                return left**right
            raise ValueError("unsupported operator")
        if isinstance(node, ast.UnaryOp):
            operand = _eval(node.operand)
            if isinstance(node.op, ast.UAdd):
                return +operand
            if isinstance(node.op, ast.USub):
                return -operand
            raise ValueError("unsupported unary operator")
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in allowed_funcs:
                func = allowed_funcs[node.func.id]
                args = [_eval(arg) for arg in node.args]
                return func(*args)
            raise ValueError("function not allowed")
        if isinstance(node, ast.Name):
            if node.id in allowed_funcs:
                return allowed_funcs[node.id]
            raise ValueError("name not allowed")
        if isinstance(node, ast.Tuple):
            return tuple(_eval(elt) for elt in node.elts)
        if isinstance(node, ast.List):
            return [_eval(elt) for elt in node.elts]
        raise ValueError("expression not allowed")

    return _eval(node)
