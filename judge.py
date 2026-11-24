"""
LLM-as-a-Judge for AIME Math Problems

Provides evaluation of mathematical answers using LLM-based judgment.
Supports both exact matching and semantic equivalence checking.
"""

from __future__ import annotations

import re
import json
from collections import Counter
from typing import Any, Dict, Optional, Tuple

from models import Model, ModelError


class MathJudge:
    """
    Judge for evaluating mathematical answers.
    Uses a combination of exact matching and LLM-based semantic comparison.
    """
    
    def __init__(self, judge_model: Model, use_llm_fallback: bool = True) -> None:
        """
        Args:
            judge_model: LLM model to use for semantic comparison
            use_llm_fallback: Whether to use LLM when exact match fails
        """
        self.judge_model = judge_model
        self.use_llm_fallback = use_llm_fallback
    
    def judge(
        self,
        question: str,
        response: str,
        parsed_answer: Optional[str],
        ground_truth: Any,
    ) -> Dict[str, Any]:
        """
        Judge whether the response is correct.
        
        Returns:
            Dict with keys:
                - label: "success" or "failure"
                - score: confidence score (0.0 to 1.0)
                - method: how the judgment was made
                - details: additional information
        """
        # Normalize ground truth
        gt_str = self._normalize_answer(str(ground_truth))
        
        # Get predicted answer
        if parsed_answer:
            pred_str = self._normalize_answer(parsed_answer)
        else:
            pred_str = self._extract_answer_from_response(response)
        
        # Method 1: Exact match after normalization
        if pred_str and gt_str:
            if self._exact_match(pred_str, gt_str):
                return {
                    "label": "success",
                    "score": 1.0,
                    "method": "exact_match",
                    "details": {"predicted": pred_str, "ground_truth": gt_str}
                }
        
        # Method 2: Numeric comparison
        pred_num = self._parse_number(pred_str)
        gt_num = self._parse_number(gt_str)
        if pred_num is not None and gt_num is not None:
            if abs(pred_num - gt_num) < 1e-6:
                return {
                    "label": "success",
                    "score": 1.0,
                    "method": "numeric_match",
                    "details": {"predicted": pred_num, "ground_truth": gt_num}
                }
            else:
                return {
                    "label": "failure",
                    "score": 0.0,
                    "method": "numeric_mismatch",
                    "details": {"predicted": pred_num, "ground_truth": gt_num}
                }
        
        # Method 3: LLM-based semantic comparison
        if self.use_llm_fallback:
            llm_result = self._llm_judge(question, response, pred_str, gt_str)
            return llm_result
        
        # Default: failure if no match found
        return {
            "label": "failure",
            "score": 0.0,
            "method": "no_match",
            "details": {"predicted": pred_str, "ground_truth": gt_str}
        }
    
    def _normalize_answer(self, answer: str) -> str:
        """Normalize answer string for comparison."""
        if not answer:
            return ""
        
        # Convert to string and strip
        answer = str(answer).strip()
        
        # Remove common prefixes
        answer = re.sub(r"^(the\s+)?answer\s+(is\s+)?", "", answer, flags=re.IGNORECASE)
        
        # Remove LaTeX formatting (only safe wrappers)
        answer = re.sub(r"\\boxed\{([^}]+)\}", r"\1", answer)
        answer = re.sub(r"\$([^$]+)\$", r"\1", answer)
        answer = answer.replace("\\left", "").replace("\\right", "")
        answer = re.sub(r"\\text\{([^}]+)\}", r"\1", answer)
        answer = re.sub(r"\\mathrm\{([^}]+)\}", r"\1", answer)
        answer = re.sub(r"\\mathbf\{([^}]+)\}", r"\1", answer)

        # Normalize simple LaTeX fractions
        answer = re.sub(
            r"\\(?:d|t)?frac\{(-?\d+)\}\{(-?\d+)\}",
            r"\1/\2",
            answer,
        )
        
        # Normalize fractions
        answer = re.sub(r"(\d+)\s*/\s*(\d+)", r"\1/\2", answer)
        
        # Remove extra whitespace
        answer = re.sub(r"\s+", " ", answer).strip()
        
        # Lowercase
        answer = answer.lower()
        
        return answer
    
    def _exact_match(self, pred: str, gt: str) -> bool:
        """Check if two normalized answers match exactly."""
        return pred == gt
    
    def _parse_number(self, text: str) -> Optional[float]:
        """Try to parse a number from text."""
        if not text:
            return None
        
        # Remove commas
        text = text.replace(",", "")
        
        # Try direct parsing
        try:
            return float(text)
        except ValueError:
            pass
        
        # Try fraction parsing
        frac_match = re.match(r"^(-?\d+)\s*/\s*(\d+)$", text)
        if frac_match:
            num, den = int(frac_match.group(1)), int(frac_match.group(2))
            if den != 0:
                return num / den
        
        # Try to extract number from text
        num_match = re.search(r"(-?\d+\.?\d*)", text)
        if num_match:
            try:
                return float(num_match.group(1))
            except ValueError:
                pass
        
        return None
    
    def _extract_answer_from_response(self, response: str) -> str:
        """Extract answer from response text."""
        if not response:
            return ""

        def _extract_boxed_content(text: str) -> Optional[str]:
            candidates: list[str] = []
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

        answer_line = ""
        for answer_match in re.finditer(r"(?im)^\s*ANSWER\s*:\s*(.+)$", response):
            answer_line = answer_match.group(1).strip()
        if answer_line:
            return self._normalize_answer(answer_line)
        
        # Pattern 1: \boxed{...}
        boxed_match = _extract_boxed_content(response)
        if boxed_match:
            return self._normalize_answer(boxed_match)
        
        # Pattern 2: Final Answer: ...
        final_match = re.search(
            r"final\s+answer\s*[:\s]+(.+?)(?:\n|$)",
            response,
            flags=re.IGNORECASE
        )
        if final_match:
            return self._normalize_answer(final_match.group(1))
        
        # Pattern 3: Answer: ...
        answer_match = re.search(
            r"(?:the\s+)?answer\s+is\s*[:\s]+(.+?)(?:\n|$)",
            response,
            flags=re.IGNORECASE
        )
        if answer_match:
            return self._normalize_answer(answer_match.group(1))
        
        # Pattern 4: Therefore, ... = ...
        therefore_match = re.search(
            r"therefore[,\s]+.*?=\s*(.+?)(?:\n|$)",
            response,
            flags=re.IGNORECASE
        )
        if therefore_match:
            return self._normalize_answer(therefore_match.group(1))
        
        return ""
    
    def _llm_judge(
        self,
        question: str,
        response: str,
        pred_answer: str,
        gt_answer: str,
    ) -> Dict[str, Any]:
        """Use LLM to judge if answers are semantically equivalent."""
        
        prompt = f"""You are a math judge. Determine if the predicted answer is mathematically equivalent to the ground truth.

QUESTION:
{question}

FULL RESPONSE:
{response[-1500:]}

EXTRACTED PREDICTED ANSWER: {pred_answer}

GROUND TRUTH ANSWER: {gt_answer}

Consider:
1. Mathematical equivalence (e.g., 1/2 = 0.5 = 50%)
2. Different representations (e.g., √2 ≈ 1.414)
3. Simplified vs unsimplified forms

Output a JSON object:
{{
  "equivalent": true or false,
  "confidence": 0.0 to 1.0,
  "reasoning": "brief explanation"
}}

Output ONLY the JSON."""

        try:
            result = self.judge_model.generate(
                prompt=prompt,
                system_prompt=None,
                temperature=0.0,
                do_sample=False,
                max_tokens=1024,
            )
            
            # Parse JSON response
            json_match = re.search(r"\{[^}]+\}", result.text)
            if json_match:
                data = json.loads(json_match.group())
                is_equivalent = data.get("equivalent", False)
                confidence = float(data.get("confidence", 0.5))
                
                return {
                    "label": "success" if is_equivalent else "failure",
                    "score": confidence if is_equivalent else 1.0 - confidence,
                    "method": "llm_judge",
                    "details": {
                        "predicted": pred_answer,
                        "ground_truth": gt_answer,
                        "reasoning": data.get("reasoning", ""),
                    }
                }
        
        except (ModelError, json.JSONDecodeError, KeyError, ValueError):
            pass
        
        # Fallback
        return {
            "label": "failure",
            "score": 0.0,
            "method": "llm_judge_failed",
            "details": {"predicted": pred_answer, "ground_truth": gt_answer}
        }


class AIMEJudge(MathJudge):
    """
    Specialized judge for AIME problems.
    AIME answers are always integers from 000 to 999.
    """
    
    def judge(
        self,
        question: str,
        response: str,
        parsed_answer: Optional[str],
        ground_truth: Any,
    ) -> Dict[str, Any]:
        """Judge AIME answer (integer 0-999)."""
        
        # Extract predicted integer
        pred_int = self._extract_aime_answer(response, parsed_answer)
        
        # Parse ground truth
        try:
            gt_int = int(ground_truth)
        except (ValueError, TypeError):
            gt_match = re.search(r"\d+", str(ground_truth))
            gt_int = int(gt_match.group()) if gt_match else None
        
        # Compare
        if pred_int is not None and gt_int is not None:
            if pred_int == gt_int:
                return {
                    "label": "success",
                    "score": 1.0,
                    "method": "aime_exact",
                    "details": {"predicted": pred_int, "ground_truth": gt_int}
                }
            else:
                return {
                    "label": "failure",
                    "score": 0.0,
                    "method": "aime_mismatch",
                    "details": {"predicted": pred_int, "ground_truth": gt_int}
                }

        if gt_int is not None and pred_int is None:
            return {
                "label": "failure",
                "score": 0.0,
                "method": "extraction_failed",
                "details": {"predicted": None, "ground_truth": gt_int},
            }
        
        # Fallback to parent LLM judge if extraction failed
        if self.use_llm_fallback:
            return super()._llm_judge(
                question, response,
                str(pred_int) if pred_int else "",
                str(gt_int) if gt_int else str(ground_truth)
            )
        
        return {
            "label": "failure",
            "score": 0.0,
            "method": "extraction_failed",
            "details": {"predicted": pred_int, "ground_truth": gt_int}
        }
    
    def _extract_aime_answer(
        self,
        response: str,
        parsed_answer: Optional[str]
    ) -> Optional[int]:
        """Extract AIME answer (integer 0-999) from response."""
        
        # Try parsed_answer first
        if parsed_answer:
            try:
                val = int(float(parsed_answer))
                if 0 <= val <= 999:
                    return val
            except (ValueError, TypeError):
                pass
        
        if not response:
            return None

        answer_line = ""
        for answer_match in re.finditer(r"(?im)^\s*ANSWER\s*:\s*(.+)$", response):
            answer_line = answer_match.group(1).strip()
        if answer_line:
            num_match = re.search(r"(\d+)", answer_line)
            if num_match:
                try:
                    val = int(num_match.group(1))
                    if 0 <= val <= 999:
                        return val
                except ValueError:
                    pass
        
        # Pattern 1: \boxed{...}
        boxed_matches = re.findall(r"\\boxed\{(\d+)\}", response)
        if boxed_matches:
            try:
                val = int(boxed_matches[-1])  # Take last boxed
                if 0 <= val <= 999:
                    return val
            except ValueError:
                pass
        
        # Pattern 2: Final Answer: N
        final_match = re.search(
            r"final\s+answer\s*[:\s]+(\d+)",
            response,
            flags=re.IGNORECASE
        )
        if final_match:
            try:
                val = int(final_match.group(1))
                if 0 <= val <= 999:
                    return val
            except ValueError:
                pass
        
        # Pattern 3: Answer is N / The answer is N
        answer_match = re.search(
            r"(?:the\s+)?answer\s+is\s*[:\s]*(\d+)",
            response,
            flags=re.IGNORECASE
        )
        if answer_match:
            try:
                val = int(answer_match.group(1))
                if 0 <= val <= 999:
                    return val
            except ValueError:
                pass
        
        # Pattern 4: = N at the end of a line
        equals_matches = re.findall(r"=\s*(\d+)\s*(?:\n|$)", response)
        if equals_matches:
            try:
                val = int(equals_matches[-1])
                if 0 <= val <= 999:
                    return val
            except ValueError:
                pass
        return None


class HotpotQAJudge(MathJudge):
    """
    Judge for HotpotQA-style QA.
    Uses EM/F1 after standard normalization (no LLM fallback).
    """

    def __init__(self, judge_model: Model, strict: bool = False) -> None:
        super().__init__(judge_model, use_llm_fallback=False)
        self.strict = strict

    def judge(
        self,
        question: str,
        response: str,
        parsed_answer: Optional[str],
        ground_truth: Any,
    ) -> Dict[str, Any]:
        pred_raw = parsed_answer or self._extract_pred_answer(response)
        gt_raw = "" if ground_truth is None else str(ground_truth)

        pred_norm = self._normalize_qa_answer(pred_raw)
        gt_norm = self._normalize_qa_answer(gt_raw)

        em = 1.0 if pred_norm == gt_norm else 0.0
        f1 = self._f1_score(pred_norm, gt_norm)

        return {
            "label": "success" if em == 1.0 else "failure",
            "score": f1,
            "method": "em_f1",
            "details": {
                "em": em,
                "f1": f1,
                "predicted": pred_norm,
                "ground_truth": gt_norm,
            },
        }

    def _extract_pred_answer(self, response: str) -> str:
        if not response:
            return ""
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
        return candidate.strip(" .;:")

    def _normalize_qa_answer(self, text: str) -> str:
        if not text:
            return ""
        text = text.lower()
        text = re.sub(r"\b(a|an|the)\b", " ", text)
        text = re.sub(r"[^0-9a-z\s]", " ", text)
        text = " ".join(text.split())
        if re.fullmatch(r"\d+(?: \d+)+", text):
            text = text.replace(" ", "")
        return text

    def _f1_score(self, pred: str, gt: str) -> float:
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
def create_judge(
    judge_model: Model,
    task_type: str = "aime",
    prompt_profile: Optional[str] = None,
    strict_hotpotqa: bool = False,
) -> MathJudge:
    """
    Factory function to create appropriate judge.
    
    Args:
        judge_model: LLM model for semantic comparison
        task_type: "aime" for AIME problems, "math" for general math
    
    Returns:
        MathJudge instance
    """
    if prompt_profile == "hotpotqa":
        return HotpotQAJudge(judge_model, strict=strict_hotpotqa)
    if task_type.lower() == "aime":
        return AIMEJudge(judge_model, use_llm_fallback=True)
    else:
        return MathJudge(judge_model, use_llm_fallback=True)
