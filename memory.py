from __future__ import annotations

import json
import os
import re
import hashlib
import random
from dataclasses import dataclass
from datetime import datetime
from math import sqrt
from typing import Any, Dict, Iterable, List, Optional, Literal, Union

from models import Model, ModelError


# ============================================================
# Memory Data Structures
# ============================================================

@dataclass
class MemoryItem:
    """
    Memory item types:
    - "global_correct_flow": Global correct solution process memory
    - "local_corrective": Local corrective memory (where student went wrong)
    - "preference_rule": Preference-based rule; 用 trigger/dimension/comparison 三元组，不塞进 content
    """
    type: str  # "global_correct_flow", "local_corrective", or "preference_rule"
    title: str
    description: str
    content: str  # JSON string with structured memory（非 preference_rule 时用）
    source: Dict[str, Any]
    # preference_rule 专用：内存中保持三元组，不依赖 content 解析
    trigger: Optional[str] = None
    dimension: Optional[str] = None
    comparison: Optional[str] = None


@dataclass
class RetrievedMemory:
    item: MemoryItem
    score: float


def _memory_item_to_dict(item: MemoryItem) -> Dict[str, Any]:
    """序列化 MemoryItem 为可写入 JSONL 的 dict。preference_rule 只写三元组 + source，不写 type。"""
    if item.type == "preference_rule" and item.trigger is not None and item.dimension is not None and item.comparison is not None:
        return {
            "trigger": (item.trigger or "").strip(),
            "dimension": (item.dimension or "").strip(),
            "comparison": (item.comparison or "").strip(),
            "source": item.source,
        }
    out = item.__dict__.copy()
    out.pop("trigger", None)
    out.pop("dimension", None)
    out.pop("comparison", None)
    return out


def _dict_to_memory_item(raw: Dict[str, Any]) -> Optional[MemoryItem]:
    """从 JSONL 一行解析出的 dict 转为 MemoryItem。三元组格式：仅 trigger/dimension/comparison/source，无 type。"""
    if "trigger" in raw and "dimension" in raw and "comparison" in raw:
        trigger = str(raw.get("trigger", "")).strip()
        dimension = str(raw.get("dimension", "")).strip()
        comparison = str(raw.get("comparison", "")).strip()
        if not trigger or not dimension or not comparison:
            return None
        if _is_placeholder(trigger):
            return None
        return MemoryItem(
            type="preference_rule",
            title=trigger,
            description="",
            content="",
            source=raw.get("source", {}),
            trigger=trigger,
            dimension=dimension,
            comparison=comparison,
        )
    # 非三元组格式（如 global_correct_flow / local_corrective）仍用 type, title, description, content, source
    return MemoryItem(
        type=raw.get("type", ""),
        title=raw.get("title", ""),
        description=raw.get("description", ""),
        content=raw.get("content", ""),
        source=raw.get("source", {}),
    )


# ============================================================
# 占位符检测
# ============================================================

# 已知的占位符模式
PLACEHOLDER_PATTERNS = [
    r"^<[^>]+>$",                           # <anything>
    r"^<[^>]+\s+[^>]+>$",                   # <multiple words>
    r"^\[.+\]$",                            # [anything]
    r"^placeholder",                         # placeholder...
    r"^your\s+(answer|title|description)",   # your answer/title/description
    r"^insert\s+",                           # insert ...
    r"^fill\s+in",                           # fill in ...
]

# 已知的具体占位符文本
KNOWN_PLACEHOLDERS = {
    # 来自 global_correct_flow prompt
    "<concise strategy name>",
    "<one sentence summary>",
    "<detailed transferable strategy>",
    "concise strategy name",
    "one sentence summary",
    "detailed transferable strategy",
    # 来自 local_corrective prompt  
    "<what to avoid or check>",
    "<detailed lesson or preventive strategy>",
    "what to avoid or check",
    "detailed lesson or preventive strategy",
    # 通用占位符
    "<your title here>",
    "<your description here>",
    "<your content here>",
    "<title>",
    "<description>",
    "<content>",
}


def _is_placeholder(text: str) -> bool:
    """
    检测文本是否为占位符。
    
    Args:
        text: 要检测的文本
        
    Returns:
        True 如果是占位符，False 否则
    """
    if not text:
        return True
    
    text_clean = text.strip().lower()
    
    # 空白或太短
    if len(text_clean) < 3:
        return True
    
    # 检查已知的占位符文本（忽略大小写）
    if text_clean in {p.lower() for p in KNOWN_PLACEHOLDERS}:
        return True
    
    # 检查 <xxx> 格式的占位符
    if text_clean.startswith("<") and text_clean.endswith(">"):
        return True
    
    # 检查占位符模式
    for pattern in PLACEHOLDER_PATTERNS:
        if re.match(pattern, text_clean, re.IGNORECASE):
            return True
    
    return False


def _has_any_placeholder(memories: List[Dict[str, str]]) -> bool:
    """检查记忆列表中是否包含任何占位符。HelpSteer3 为 trigger/dimension/comparison，无 title 不算占位符。"""
    for mem in memories:
        trigger = (mem.get("trigger") or "").strip()
        dimension = (mem.get("dimension") or "").strip()
        comparison = (mem.get("comparison") or "").strip()
        # HelpSteer3: 有完整三元组则不要求 title/description
        if trigger and dimension and comparison:
            if _is_placeholder(trigger) or _is_placeholder(dimension) or _is_placeholder(comparison):
                return True
            continue
        if _is_placeholder(mem.get("title", "")):
            return True
        desc = mem.get("description", "")
        if desc and _is_placeholder(desc):
            return True
    return False


def _debug_memory_extractor_output(stage: str, attempt: int, text: str) -> None:
    """Print a truncated view of extractor output for debugging."""
    if text is None:
        text = ""
    text = str(text)
    head = text[:1200]
    tail = text[-400:] if len(text) > 1600 else ""
    print(f"  [MemoryDebug:{stage}] attempt {attempt}: {head}")
    if tail:
        print(f"  [MemoryDebug:{stage}] tail: ...{tail}")


# ============================================================
# Memory Bank
# ============================================================

class MemoryBank:
    def __init__(self, memory_path: str, top_k: int = 1) -> None:
        self.memory_path = memory_path
        self.top_k = top_k
        self._items: List[MemoryItem] = []
        self._index: List[Dict[str, int]] = []
        self._embedding_enabled = False
        self._embedding_model_name: Optional[str] = None
        self._embedding_cache_path: Optional[str] = None
        self._embedding_vectors = None
        self._embedding_keys: List[str] = []
        self._embedding_model = None
        self._embedding_tokenizer = None
        self._embedding_device: Optional[str] = None
        self._embedding_batch_size: int = 32
        self._last_retrieval_backend: Optional[str] = None
        self._last_retrieval_fallback: Optional[bool] = None
        self._bandit_enabled: bool = False
        self._bandit_lambda: float = 0.3
        self._bandit_noise_var: float = 1.0
        self._bandit_explore_epsilon: float = 0.25
        self._bandit_neighbor_k: int = 5
        self._bandit_prior_mu: float = 0.0
        self._bandit_prior_sigma2: float = 1.0
        self._bandit_state_path: Optional[str] = None
        self._bandit_stats: Dict[str, Dict[str, float]] = {}
        self._bandit_rng = random.Random()

    def load(self, path: str, max_items: Optional[int] = None) -> int:
        if not path or not os.path.exists(path):
            return 0
        count = 0
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    raw = json.loads(line)
                    item = _dict_to_memory_item(raw)
                    if item is None:
                        continue
                except json.JSONDecodeError:
                    continue
                # preference_rule 已有 title=trigger；其他类型要求 title 和 content 非空
                if item.type != "preference_rule" and (not item.title or not item.content):
                    continue
                if item.type != "preference_rule" and _is_placeholder(item.title):
                    continue
                self._items.append(item)
                self._index.append(_tf_vector(_item_text_for_index(item)))
                count += 1
                if max_items and count >= max_items:
                    break
        if self._bandit_enabled:
            self._ensure_bandit_entries_for_all_items()
            self._save_bandit_state()
        return count

    def initialize_storage(self) -> None:
        if not self.memory_path:
            return
        if os.path.exists(self.memory_path):
            return
        os.makedirs(os.path.dirname(self.memory_path) or ".", exist_ok=True)
        with open(self.memory_path, "w", encoding="utf-8") as handle:
            for item in self._items:
                handle.write(json.dumps(_memory_item_to_dict(item), ensure_ascii=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    def size(self) -> int:
        return len(self._items)

    def is_bandit_enabled(self) -> bool:
        return bool(self._bandit_enabled)

    def enable_bandit(
        self,
        lambda_weight: float = 0.3,
        noise_var: float = 1.0,
        explore_epsilon: float = 0.25,
        neighbor_k: int = 5,
        prior_mu: float = 0.0,
        prior_sigma2: float = 1.0,
        state_path: Optional[str] = None,
        random_seed: Optional[int] = None,
    ) -> bool:
        if not self._items:
            print("Warning: No memories loaded; bandit retrieval not enabled.")
            return False

        self._bandit_enabled = True
        self._bandit_lambda = min(1.0, max(0.0, float(lambda_weight)))
        self._bandit_noise_var = max(float(noise_var), 1e-8)
        self._bandit_explore_epsilon = max(float(explore_epsilon), 1e-8)
        self._bandit_neighbor_k = max(1, int(neighbor_k))
        self._bandit_prior_mu = float(prior_mu)
        self._bandit_prior_sigma2 = max(float(prior_sigma2), 1e-8)
        self._bandit_state_path = state_path
        self._bandit_rng = random.Random(random_seed)
        self._bandit_stats = {}

        self._load_bandit_state()
        self._ensure_bandit_entries_for_all_items()
        self._save_bandit_state()
        return True

    def _load_bandit_state(self) -> None:
        path = self._bandit_state_path
        if not path or not os.path.exists(path):
            return
        try:
            with open(path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except Exception:
            return
        stats = payload.get("stats") if isinstance(payload, dict) else None
        if not isinstance(stats, dict):
            return
        loaded: Dict[str, Dict[str, float]] = {}
        for key, value in stats.items():
            if not isinstance(key, str) or not isinstance(value, dict):
                continue
            try:
                mu = float(value.get("mu", self._bandit_prior_mu))
                sigma2 = float(value.get("sigma2", self._bandit_prior_sigma2))
            except Exception:
                continue
            updates = float(value.get("updates", 0.0) or 0.0)
            loaded[key] = {
                "mu": mu,
                "sigma2": self._clamp_variance(sigma2),
                "updates": max(0.0, updates),
            }
        self._bandit_stats = loaded

    def _save_bandit_state(self) -> None:
        if not self._bandit_enabled:
            return
        path = self._bandit_state_path
        if not path:
            return
        try:
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            payload = {
                "version": 1,
                "updated_at": _iso_timestamp(),
                "stats": self._bandit_stats,
            }
            with open(path, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=False, indent=2)
        except Exception:
            return

    def _clamp_variance(self, sigma2: float) -> float:
        if sigma2 is None:
            return self._bandit_explore_epsilon
        return max(float(sigma2), 1e-8)

    def _bandit_similarity_by_index(self, idx_a: int, idx_b: int) -> float:
        if idx_a == idx_b:
            return 1.0
        if (
            self._embedding_enabled
            and self._embedding_vectors is not None
            and idx_a < len(self._embedding_vectors)
            and idx_b < len(self._embedding_vectors)
        ):
            try:
                return float(_dot_similarity(self._embedding_vectors[idx_a], self._embedding_vectors[idx_b]))
            except Exception:
                pass
        if idx_a < len(self._index) and idx_b < len(self._index):
            return float(_cosine_sim(self._index[idx_a], self._index[idx_b]))
        return 0.0

    def _bandit_semantic_init(self, idx: int) -> Dict[str, float]:
        if idx < 0 or idx >= len(self._items):
            return {
                "mu": self._bandit_prior_mu,
                "sigma2": self._clamp_variance(self._bandit_prior_sigma2 + self._bandit_explore_epsilon),
                "updates": 0.0,
            }
        neighbors: List[tuple[float, str]] = []
        for j, other in enumerate(self._items):
            if j == idx:
                continue
            key = _memory_key(other)
            stat = self._bandit_stats.get(key)
            if not stat:
                continue
            sim = self._bandit_similarity_by_index(idx, j)
            neighbors.append((sim, key))
        neighbors.sort(key=lambda x: x[0], reverse=True)
        top_neighbors = neighbors[: self._bandit_neighbor_k]
        if not top_neighbors:
            return {
                "mu": self._bandit_prior_mu,
                "sigma2": self._clamp_variance(self._bandit_prior_sigma2 + self._bandit_explore_epsilon),
                "updates": 0.0,
            }
        mus: List[float] = []
        vars_: List[float] = []
        for _, key in top_neighbors:
            stat = self._bandit_stats.get(key, {})
            mus.append(float(stat.get("mu", self._bandit_prior_mu)))
            vars_.append(self._clamp_variance(float(stat.get("sigma2", self._bandit_prior_sigma2))))
        if not mus:
            return {
                "mu": self._bandit_prior_mu,
                "sigma2": self._clamp_variance(self._bandit_prior_sigma2 + self._bandit_explore_epsilon),
                "updates": 0.0,
            }
        mu_new = sum(mus) / len(mus)
        sigma_new = 0.0
        for mu_j, sigma_j in zip(mus, vars_):
            sigma_new += sigma_j + (mu_j - mu_new) ** 2
        sigma_new = sigma_new / len(mus)
        sigma_new += self._bandit_explore_epsilon
        return {
            "mu": mu_new,
            "sigma2": self._clamp_variance(sigma_new),
            "updates": 0.0,
        }

    def _ensure_bandit_entry_for_index(self, idx: int) -> None:
        if idx < 0 or idx >= len(self._items):
            return
        key = _memory_key(self._items[idx])
        if key in self._bandit_stats:
            stat = self._bandit_stats[key]
            stat["sigma2"] = self._clamp_variance(float(stat.get("sigma2", self._bandit_prior_sigma2)))
            if "mu" not in stat:
                stat["mu"] = self._bandit_prior_mu
            if "updates" not in stat:
                stat["updates"] = 0.0
            return
        self._bandit_stats[key] = self._bandit_semantic_init(idx)

    def _ensure_bandit_entries_for_all_items(self) -> None:
        if not self._bandit_enabled:
            return
        for idx in range(len(self._items)):
            self._ensure_bandit_entry_for_index(idx)

    def _ensure_bandit_entry_for_item(self, item: MemoryItem) -> None:
        if not self._bandit_enabled:
            return
        key = _memory_key(item)
        if key in self._bandit_stats:
            return
        try:
            idx = self._items.index(item)
        except ValueError:
            idx = -1
        if idx >= 0:
            self._ensure_bandit_entry_for_index(idx)
        else:
            self._bandit_stats[key] = {
                "mu": self._bandit_prior_mu,
                "sigma2": self._clamp_variance(self._bandit_prior_sigma2 + self._bandit_explore_epsilon),
                "updates": 0.0,
            }

    def _bandit_score(self, item: MemoryItem, semantic_score: float, idx: Optional[int] = None) -> float:
        if not self._bandit_enabled:
            return float(semantic_score)
        if idx is not None:
            self._ensure_bandit_entry_for_index(idx)
        else:
            self._ensure_bandit_entry_for_item(item)
        key = _memory_key(item)
        stat = self._bandit_stats.get(key)
        if not stat:
            return float(semantic_score)
        mu = float(stat.get("mu", self._bandit_prior_mu))
        sigma2 = self._clamp_variance(float(stat.get("sigma2", self._bandit_prior_sigma2)))
        sampled_utility = self._bandit_rng.gauss(mu, sqrt(sigma2))
        final_score = (1.0 - self._bandit_lambda) * float(semantic_score)
        final_score += self._bandit_lambda * float(sampled_utility)
        return float(final_score)

    def update_bandit_posteriors(
        self,
        memories: Iterable[MemoryItem],
        advantage_reward: float,
    ) -> int:
        if not self._bandit_enabled:
            return 0
        updated = 0
        seen: set[str] = set()
        reward = float(advantage_reward)
        noise = max(self._bandit_noise_var, 1e-8)
        for item in memories:
            key = _memory_key(item)
            if key in seen:
                continue
            seen.add(key)
            self._ensure_bandit_entry_for_item(item)
            stat = self._bandit_stats.get(key)
            if not stat:
                continue
            mu_prior = float(stat.get("mu", self._bandit_prior_mu))
            sigma_prior = self._clamp_variance(float(stat.get("sigma2", self._bandit_prior_sigma2)))
            mu_post = (sigma_prior * reward + noise * mu_prior) / (sigma_prior + noise)
            sigma_post = (sigma_prior * noise) / (sigma_prior + noise)
            stat["mu"] = float(mu_post)
            stat["sigma2"] = self._clamp_variance(float(sigma_post))
            stat["updates"] = float(stat.get("updates", 0.0) + 1.0)
            updated += 1
        if updated > 0:
            self._save_bandit_state()
        return updated

    def retrieve(
        self,
        question: str,
        top_k: Optional[int] = None,
        exclude_sample_id: Optional[str] = None,
        expected_answer: Optional[Any] = None,
        max_similarity: Optional[float] = None,
        min_similarity: Optional[float] = None,
        memory_types: Optional[Union[str, Iterable[str]]] = None,
        retrieval_method: Literal["tf", "embedding"] = "tf",
    ) -> List[MemoryItem]:
        """
        Retrieve relevant memories for a given question.
        """
        scored = self.retrieve_with_scores(
            question=question,
            top_k=top_k,
            exclude_sample_id=exclude_sample_id,
            expected_answer=expected_answer,
            max_similarity=max_similarity,
            min_similarity=min_similarity,
            memory_types=memory_types,
            retrieval_method=retrieval_method,
        )
        return [r.item for r in scored]

    def retrieve_with_scores(
        self,
        question: str,
        top_k: Optional[int] = None,
        exclude_sample_id: Optional[str] = None,
        expected_answer: Optional[Any] = None,
        max_similarity: Optional[float] = None,
        min_similarity: Optional[float] = None,
        memory_types: Optional[Union[str, Iterable[str]]] = None,
        retrieval_method: Literal["tf", "embedding"] = "tf",
    ) -> List[RetrievedMemory]:
        """
        Retrieve relevant memories with similarity scores.
        """
        if not self._items:
            self._last_retrieval_backend = None
            self._last_retrieval_fallback = None
            return []
        if retrieval_method == "embedding":
            return self._retrieve_with_embeddings_scored(
                question=question,
                top_k=top_k,
                exclude_sample_id=exclude_sample_id,
                expected_answer=expected_answer,
                max_similarity=max_similarity,
                min_similarity=min_similarity,
                memory_types=memory_types,
            )
        self._last_retrieval_backend = "tf"
        self._last_retrieval_fallback = False
        type_set = None
        if memory_types is not None:
            if isinstance(memory_types, str):
                type_set = {memory_types}
            else:
                type_set = {str(t) for t in memory_types}
        query_vec = _tf_vector(question)
        scored: List[RetrievedMemory] = []
        for idx, (item, vec) in enumerate(zip(self._items, self._index)):
            if type_set is not None and item.type not in type_set:
                continue
            if exclude_sample_id and item.source.get("sample_id") == exclude_sample_id:
                continue
            semantic_score = _cosine_sim(query_vec, vec)
            if min_similarity is not None and semantic_score < min_similarity:
                continue
            if max_similarity is not None and semantic_score >= max_similarity:
                continue
            if expected_answer is not None and _has_answer_leak(item, expected_answer):
                continue
            final_score = self._bandit_score(item=item, semantic_score=semantic_score, idx=idx)
            scored.append(RetrievedMemory(item=item, score=final_score))
        scored.sort(key=lambda r: r.score, reverse=True)
        k = top_k or self.top_k
        return scored[:k]

    def enable_embedding_retrieval(
        self,
        model_name: str,
        cache_path: Optional[str] = None,
        device: Optional[str] = None,
        batch_size: int = 32,
    ) -> bool:
        if not self._items:
            print("Warning: No memories loaded; embedding cache not built.")
            return False
        self._embedding_model_name = model_name
        self._embedding_cache_path = cache_path or _default_embedding_cache_path(self.memory_path)
        self._embedding_device = device
        self._embedding_batch_size = max(1, int(batch_size))

        keys = [_memory_key(item) for item in self._items]
        cached_embeddings = None
        cached_keys: List[str] = []
        if self._embedding_cache_path and os.path.exists(self._embedding_cache_path):
            cached_embeddings, cached_keys = _load_embedding_cache(self._embedding_cache_path)

        try:
            tokenizer, model, device_name = _load_embedding_model(model_name, device)
        except Exception as e:
            print(f"Warning: Failed to load embedding model '{model_name}': {e}")
            return False

        self._embedding_model = model
        self._embedding_tokenizer = tokenizer
        self._embedding_device = device_name

        if cached_embeddings is not None and cached_keys == keys:
            self._embedding_vectors = cached_embeddings
            self._embedding_keys = cached_keys
            self._embedding_enabled = True
            return True

        texts = [_memory_text_for_embedding(item) for item in self._items]
        try:
            embeddings = _embed_texts(
                model=model,
                tokenizer=tokenizer,
                texts=texts,
                device=device_name,
                batch_size=self._embedding_batch_size,
            )
        except Exception as e:
            print(f"Warning: Failed to compute memory embeddings: {e}")
            return False

        self._embedding_vectors = embeddings
        self._embedding_keys = keys
        self._embedding_enabled = True

        if self._embedding_cache_path:
            _save_embedding_cache(self._embedding_cache_path, keys, embeddings)
            if os.path.exists(self._embedding_cache_path):
                print(f"[Embedding] Cache saved to {self._embedding_cache_path}")

        return True

    def _retrieve_with_embeddings(
        self,
        question: str,
        top_k: Optional[int] = None,
        exclude_sample_id: Optional[str] = None,
        expected_answer: Optional[Any] = None,
        max_similarity: Optional[float] = None,
        min_similarity: Optional[float] = None,
        memory_types: Optional[Union[str, Iterable[str]]] = None,
    ) -> List[MemoryItem]:
        scored = self._retrieve_with_embeddings_scored(
            question=question,
            top_k=top_k,
            exclude_sample_id=exclude_sample_id,
            expected_answer=expected_answer,
            max_similarity=max_similarity,
            min_similarity=min_similarity,
            memory_types=memory_types,
        )
        return [r.item for r in scored]

    def _retrieve_with_embeddings_scored(
        self,
        question: str,
        top_k: Optional[int] = None,
        exclude_sample_id: Optional[str] = None,
        expected_answer: Optional[Any] = None,
        max_similarity: Optional[float] = None,
        min_similarity: Optional[float] = None,
        memory_types: Optional[Union[str, Iterable[str]]] = None,
    ) -> List[RetrievedMemory]:
        if not self._embedding_enabled or self._embedding_vectors is None:
            self._last_retrieval_backend = "tf"
            self._last_retrieval_fallback = True
            return self.retrieve_with_scores(
                question=question,
                top_k=top_k,
                exclude_sample_id=exclude_sample_id,
                expected_answer=expected_answer,
                max_similarity=max_similarity,
                min_similarity=min_similarity,
                memory_types=memory_types,
                retrieval_method="tf",
            )
        type_set = None
        if memory_types is not None:
            if isinstance(memory_types, str):
                type_set = {memory_types}
            else:
                type_set = {str(t) for t in memory_types}

        try:
            query_vec = _embed_texts(
                model=self._embedding_model,
                tokenizer=self._embedding_tokenizer,
                texts=[question],
                device=self._embedding_device,
                batch_size=1,
            )[0]
        except Exception as e:
            print(f"Warning: Embedding query failed, falling back to TF: {e}")
            self._last_retrieval_backend = "tf"
            self._last_retrieval_fallback = True
            return self.retrieve_with_scores(
                question=question,
                top_k=top_k,
                exclude_sample_id=exclude_sample_id,
                expected_answer=expected_answer,
                max_similarity=max_similarity,
                min_similarity=min_similarity,
                memory_types=memory_types,
                retrieval_method="tf",
            )

        self._last_retrieval_backend = "embedding"
        self._last_retrieval_fallback = False
        scores: List[RetrievedMemory] = []
        for idx, item in enumerate(self._items):
            if type_set is not None and item.type not in type_set:
                continue
            if exclude_sample_id and item.source.get("sample_id") == exclude_sample_id:
                continue
            if expected_answer is not None and _has_answer_leak(item, expected_answer):
                continue
            semantic_score = float(_dot_similarity(query_vec, self._embedding_vectors[idx]))
            if min_similarity is not None and semantic_score < min_similarity:
                continue
            if max_similarity is not None and semantic_score >= max_similarity:
                continue
            final_score = self._bandit_score(item=item, semantic_score=semantic_score, idx=idx)
            scores.append(RetrievedMemory(item=item, score=final_score))

        scores.sort(key=lambda r: r.score, reverse=True)
        k = top_k or self.top_k
        return scores[:k]

    def retrieve_with_model(
        self,
        question: str,
        student_model: Model,
        top_k: Optional[int] = None,
        exclude_sample_id: Optional[str] = None,
        expected_answer: Optional[Any] = None,
        max_similarity: Optional[float] = None,
        min_similarity: Optional[float] = None,
        memory_types: Optional[Union[str, Iterable[str]]] = None,
    ) -> List[MemoryItem]:
        """
        Retrieve memories using student model for better relevance scoring.
        """
        candidates = self.retrieve(
            question=question,
            top_k=min(10, len(self._items)),
            exclude_sample_id=exclude_sample_id,
            expected_answer=expected_answer,
            max_similarity=max_similarity,
            min_similarity=min_similarity,
            memory_types=memory_types,
        )
        
        if not candidates or len(candidates) <= (top_k or self.top_k):
            return candidates[:top_k or self.top_k]
        
        prompt = _build_memory_rerank_prompt(question, candidates)
        
        try:
            response = student_model.generate(
                prompt=prompt,
                system_prompt=None,
                temperature=0.0,
                max_tokens=8192,
            )
            rankings = _parse_memory_rankings(response.text, len(candidates))
            if rankings:
                reranked = []
                for idx in rankings:
                    if 0 <= idx < len(candidates):
                        reranked.append(candidates[idx])
                return reranked[:top_k or self.top_k]
        except (ModelError, Exception):
            pass
        
        return candidates[:top_k or self.top_k]

    def build_system_prompt(
        self,
        base_prompt: str,
        memories: List[MemoryItem],
        prompt_profile: Optional[str] = None,
    ) -> str:
        if not memories:
            return base_prompt
        
        # 对于 HotpotQA 使用更紧凑的格式
        if prompt_profile == "hotpotqa":
            return self._build_qa_system_prompt(base_prompt, memories)

        # AdvancedIF: instruction-following rules (not preference choice wording)
        if prompt_profile in {"advancedif", "helpsteer2"}:
            return self._build_instruction_following_system_prompt(base_prompt, memories)

        # Preference tasks (HelpSteer3 only): trigger/dimension/comparison 格式
        if prompt_profile == "helpsteer3" or any(m.type == "preference_rule" for m in memories):
            return self._build_preference_system_prompt(base_prompt, memories)
        
        # 原有的数学任务格式
        lines = [base_prompt, "", "=== Retrieved Problem-Solving Strategies ==="]
        for idx, item in enumerate(memories, start=1):
            lines.append(f"[Strategy {idx}] {item.title}")
            lines.append(f"Type: {'Global Solution Flow' if item.type == 'global_correct_flow' else 'Local Correction'}")
            if item.description:
                lines.append(f"Description: {item.description}")
            try:
                content_data = json.loads(item.content)
                for key, value in content_data.items():
                    if key == "description":
                        continue
                    if value:
                        readable_key = key.replace("_", " ").title()
                        lines.append(f"{readable_key}: {value}")
            except json.JSONDecodeError:
                lines.append(item.content)
            lines.append("")
        lines.append("=== End Strategies ===")
        lines.append("")
        lines.append("Use these strategies if they are applicable to the current problem.")
        return "\n".join(lines)
    
    def _build_qa_system_prompt(self, base_prompt: str, memories: List[MemoryItem]) -> str:
        """
        为 QA 任务（如 HotpotQA）构建更紧凑的 system prompt。
        关键：简短，不要与"only use context"冲突
        """
        # 分离 global 和 local memories
        global_memories = [m for m in memories if m.type == "global_correct_flow"]
        local_memories = [m for m in memories if m.type == "local_corrective"]
        
        hints = []
        
        for m in global_memories:
            content_text = self._extract_content_text(m.content)
            hints.append(f"• {m.title}: {content_text}")
        
        for m in local_memories:
            content_text = self._extract_content_text(m.content)
            hints.append(f"• Avoid: {m.title} - {content_text}")
        
        if not hints:
            return base_prompt
        
        memory_block = "\n".join(hints)
        
        return f"""{base_prompt}

[Hints]
{memory_block}"""

    def _build_preference_system_prompt(self, base_prompt: str, memories: List[MemoryItem]) -> str:
        """
        Build system prompt for preference tasks (HelpSteer3 only).
        For trigger/dimension/comparison triplet: [Rule N] trigger, then Dimension | Comparison (no repeated trigger).
        """
        rules: List[str] = []
        for idx, item in enumerate(memories, start=1):
            title = (item.title or "").strip()
            body = self._format_preference_rule_body(item)
            if body:
                # HelpSteer3 triplet: trigger 作 title，正文只写 Dimension | Comparison
                rules.append(f"[Rule {idx}] {title}")
                rules.append(body)
            else:
                content_text = self._extract_content_text(item.content)
                if title:
                    rules.append(f"[Rule {idx}] {title}")
                    if content_text:
                        rules.append(content_text)
                elif content_text:
                    rules.append(f"[Rule {idx}] {content_text}")
        if not rules:
            return base_prompt
        rule_block = "\n".join(rules)
        return f"""{base_prompt}

=== Preference Rules ===
{rule_block}
=== End Preference Rules ===

Use these rules to choose the better response."""

    def _build_instruction_following_system_prompt(self, base_prompt: str, memories: List[MemoryItem]) -> str:
        """
        Build system prompt for AdvancedIF instruction-following tasks.
        Use trigger/dimension/comparison triplets as constraints, not preference wording.
        """
        rules: List[str] = []
        def _extract_triplet(item: MemoryItem) -> Tuple[str, str, str]:
            trigger = (item.trigger or item.title or "").strip()
            dimension = (item.dimension or "").strip()
            comparison = (item.comparison or "").strip()
            if trigger and dimension and comparison:
                return trigger, dimension, comparison
            if item.content:
                try:
                    data = json.loads(item.content)
                    if isinstance(data, dict):
                        trigger = (data.get("trigger") or trigger or "").strip()
                        dimension = (data.get("dimension") or dimension or "").strip()
                        comparison = (data.get("comparison") or comparison or "").strip()
                        if trigger and dimension and comparison:
                            return trigger, dimension, comparison
                except json.JSONDecodeError:
                    pass
            return "", "", ""
        for idx, item in enumerate(memories, start=1):
            trigger, dimension, comparison = _extract_triplet(item)
            if trigger and dimension and comparison:
                rules.append(f"Preference {idx}: {trigger}. For \"{dimension}\", {comparison}")
            else:
                content_text = self._extract_content_text(item.content)
                title = (item.title or "").strip()
                if title:
                    if content_text:
                        rules.append(f"Preference {idx}: {title}. {content_text}")
                    else:
                        rules.append(f"Preference {idx}: {title}")
                elif content_text:
                    rules.append(f"Preference {idx}: {content_text}")
        if not rules:
            return base_prompt
        rule_block = "\n".join(rules)
        return f"""{base_prompt}

If relevant, use these instruction-following rules as guidance:
{rule_block}

Do not mention these rules explicitly."""

    def _format_preference_rule_body(self, item: MemoryItem) -> str:
        """HelpSteer3 only: 用 item 上的三元组，返回 'Dimension: ... | Comparison: ...'（trigger 已作 title，不重复）."""
        if item.trigger is not None and item.dimension is not None and item.comparison is not None:
            d = (item.dimension or "").strip()
            c = (item.comparison or "").strip()
            if d and c:
                return f"Dimension: {d} | Comparison: {c}"
        if item.content:
            try:
                data = json.loads(item.content)
                if isinstance(data, dict) and "dimension" in data and "comparison" in data:
                    d = str(data.get("dimension", "")).strip()
                    c = str(data.get("comparison", "")).strip()
                    if d and c:
                        return f"Dimension: {d} | Comparison: {c}"
            except json.JSONDecodeError:
                pass
        return ""

    def _extract_content_text(self, content: str) -> str:
        """从 memory content 中提取可读文本。HelpSteer3 为 trigger/dimension/comparison 三元组。"""
        if not content:
            return ""
        try:
            data = json.loads(content)
            if isinstance(data, dict):
                # HelpSteer3: trigger / dimension / comparison 三元组
                if "trigger" in data and "dimension" in data and "comparison" in data:
                    t = str(data.get("trigger", "")).strip()
                    d = str(data.get("dimension", "")).strip()
                    c = str(data.get("comparison", "")).strip()
                    if t and d and c:
                        return f"Trigger: {t} | Dimension: {d} | Comparison: {c}"
                # 其他 key
                for key in ["strategy", "content", "description"]:
                    if key in data and data[key]:
                        return str(data[key])[:150]
                return json.dumps(data, ensure_ascii=False)[:150]
        except json.JSONDecodeError:
            pass
        return content[:150]

    def append_memories(self, items: Iterable[MemoryItem]) -> int:
        if not self.memory_path:
            return 0
        item_list = list(items)
        os.makedirs(os.path.dirname(self.memory_path) or ".", exist_ok=True)
        count = 0
        valid_items: List[MemoryItem] = []
        new_indices: List[int] = []
        with open(self.memory_path, "a", encoding="utf-8") as handle:
            for item in item_list:
                # 跳过占位符
                if _is_placeholder(item.title):
                    continue
                handle.write(json.dumps(_memory_item_to_dict(item), ensure_ascii=True) + "\n")
                count += 1
                valid_items.append(item)
            handle.flush()
            os.fsync(handle.fileno())
        for item in valid_items:
            self._items.append(item)
            self._index.append(_tf_vector(_item_text_for_index(item)))
            new_indices.append(len(self._items) - 1)
        if self._embedding_enabled and self._embedding_model is not None:
            if valid_items:
                keys = [_memory_key(item) for item in valid_items]
                texts = [_memory_text_for_embedding(item) for item in valid_items]
                try:
                    new_embeddings = _embed_texts(
                        model=self._embedding_model,
                        tokenizer=self._embedding_tokenizer,
                        texts=texts,
                        device=self._embedding_device,
                        batch_size=self._embedding_batch_size,
                    )
                    self._embedding_vectors = _append_embeddings(self._embedding_vectors, new_embeddings)
                    self._embedding_keys.extend(keys)
                    if self._embedding_cache_path:
                        _save_embedding_cache(self._embedding_cache_path, self._embedding_keys, self._embedding_vectors)
                except Exception as e:
                    print(f"Warning: Failed to update embedding cache: {e}")
        if self._bandit_enabled and new_indices:
            for idx in new_indices:
                self._ensure_bandit_entry_for_index(idx)
            self._save_bandit_state()
        return count

    def dump_snapshot(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            for item in self._items:
                handle.write(json.dumps(_memory_item_to_dict(item), ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    def clear(self) -> None:
        """Clear all memories from the bank."""
        self._items.clear()
        self._index.clear()
        self._embedding_vectors = None
        self._embedding_keys = []
        self._embedding_model = None
        self._embedding_tokenizer = None
        self._embedding_enabled = False
        self._bandit_stats = {}
        if self._bandit_enabled:
            self._save_bandit_state()
        if self.memory_path and os.path.exists(self.memory_path):
            with open(self.memory_path, "w", encoding="utf-8") as handle:
                handle.write("")


# ============================================================
# Memory Retrieval with Student Model
# ============================================================

def _build_memory_rerank_prompt(question: str, candidates: List[MemoryItem]) -> str:
    """Build prompt for memory re-ranking using student model."""
    candidates_text = []
    for idx, mem in enumerate(candidates):
        candidates_text.append(f"[{idx}] {mem.title}")
        desc = mem.description or ""
        if not desc:
            try:
                content_data = json.loads(mem.content)
                desc = content_data.get("description", "")
            except json.JSONDecodeError:
                desc = ""
        if desc:
            candidates_text.append(f"    {desc[:100]}")
    
    return f"""Given a math problem and candidate problem-solving strategies, rank which strategies would be most helpful.

PROBLEM:
{question[:500]}

CANDIDATE STRATEGIES:
{chr(10).join(candidates_text)}

Rank the strategies by their usefulness for this specific problem.
Output only the indices in order of usefulness, separated by commas.
Example: 2, 0, 4, 1, 3

Rankings:"""


def _parse_memory_rankings(response: str, num_candidates: int) -> List[int]:
    """Parse memory rankings from model response."""
    rankings = []
    numbers = re.findall(r"\d+", response)
    seen = set()
    for num in numbers:
        try:
            idx = int(num)
            if 0 <= idx < num_candidates and idx not in seen:
                rankings.append(idx)
                seen.add(idx)
        except ValueError:
            continue
    return rankings


# ============================================================
# Answer Leakage Prevention
# ============================================================

def _has_answer_leak(item: MemoryItem, expected_answer: Any) -> bool:
    if expected_answer is None:
        return False
    answer_str = str(expected_answer).strip().lower()
    if not answer_str:
        return False
    searchable = _item_text_for_index(item).lower()
    if answer_str in searchable:
        return True
    numbers = re.findall(r"-?\d+(?:\.\d+)?", answer_str)
    for num in numbers:
        pattern = re.compile(rf"(?<![0-9]){re.escape(num)}(?![0-9])")
        if pattern.search(searchable):
            return True
    return False


# ============================================================
# Answer Sanitization Helpers
# ============================================================

def _strip_answer_lines(text: str) -> str:
    patterns = [
        re.compile(r"(?i)\bfinal\s*answer\s*[:\-=]?\s*\S+"),
        re.compile(r"(?i)\bthe\s+answer\s+is\s*[:\-=]?\s*\S+"),
        re.compile(r"\\boxed\{[^}]*\}"),
    ]
    cleaned = text
    for pat in patterns:
        cleaned = pat.sub("[redacted]", cleaned)
    return cleaned


def _redact_answer_tokens(text: str, answer_hint: Any) -> str:
    if answer_hint is None:
        return text
    hints: List[str] = []
    if isinstance(answer_hint, (list, tuple, set)):
        hints = [str(item) for item in answer_hint if item]
    else:
        hints = [str(answer_hint)]
    numbers: List[str] = []
    for hint in hints:
        numbers.extend(re.findall(r"-?\d+(?:\.\d+)?", hint))
    if not numbers:
        return text
    redacted = text
    for num in set(numbers):
        pattern = re.compile(rf"(?<!\d){re.escape(num)}(?!\d)")
        redacted = pattern.sub("[redacted]", redacted)
    return redacted


def _sanitize_memory_field(text: Any, answer_hint: Any) -> str:
    """Handle various input types, ensure string output."""
    if text is None:
        return ""
    if isinstance(text, list):
        text = " ".join(str(item) for item in text)
    elif not isinstance(text, str):
        text = str(text)
    
    cleaned = _strip_answer_lines(text)
    cleaned = _redact_answer_tokens(cleaned, answer_hint)
    return cleaned


def _strip_response_labels(text: str) -> str:
    """Remove explicit response labels like 'Response 1/2' or 'Option 1/2'."""
    if not text:
        return text
    cleaned = re.sub(r"\bresponse\s*1\b", "", text, flags=re.IGNORECASE)
    cleaned = re.sub(r"\bresponse\s*2\b", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\boption\s*1\b", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\boption\s*2\b", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def _normalize_preference_comparison(text: str, min_words: int = 25) -> str:
    """Normalize preference comparison to a single directive sentence with minimum length."""
    if not text:
        return ""
    cleaned = re.sub(r"\s+", " ", str(text)).strip()
    if not cleaned:
        return ""
    cleaned = _strip_response_labels(cleaned)
    cleaned = cleaned.rstrip(".")
    cleaned = f"{cleaned}."
    if not cleaned.endswith("."):
        cleaned += "."
    if len(cleaned.split()) < min_words:
        cleaned = cleaned.rstrip(".") + " to improve clarity, ensure actionable specificity, and align the response with the user's request without adding irrelevant content."
    return cleaned


def _to_str(value: Any) -> str:
    """Convert any value to string, handling list, dict etc."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return " ".join(str(item) for item in value)
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


# ============================================================
# MEMORY Format Parser
# ============================================================

def _parse_memory_format(response: str) -> List[Dict[str, str]]:
    """
    Parse MEMORY format response:
    MEMORY 1:
    TITLE: ...
    DESCRIPTION: ...
    CONTENT: ...
    """
    memories = []
    parts = re.split(r'MEMORY\s*\d+\s*:', response, flags=re.IGNORECASE)
    
    for part in parts[1:]:
        try:
            title = _extract_field(part, 'TITLE:')
            description = _extract_field(part, 'DESCRIPTION:')
            content = _extract_field(part, 'CONTENT:')
            
            if title and content:
                memories.append({
                    "title": title.strip(),
                    "description": description.strip() if description else "",
                    "content": content.strip(),
                })
        except Exception:
            continue
    
    return memories


def _extract_field(text: str, field_name: str) -> str:
    """Extract field value from formatted text"""
    if field_name.upper() not in text.upper():
        return ""
    
    pattern = re.compile(re.escape(field_name), re.IGNORECASE)
    match = pattern.search(text)
    if not match:
        return ""
    
    start = match.end()
    next_field_markers = ['TITLE:', 'DESCRIPTION:', 'CONTENT:', 'MEMORY ']
    
    end = len(text)
    for marker in next_field_markers:
        marker_pattern = re.compile(re.escape(marker), re.IGNORECASE)
        marker_match = marker_pattern.search(text[start:])
        if marker_match:
            candidate_end = start + marker_match.start()
            if candidate_end < end:
                end = candidate_end
    
    value = text[start:end].strip()
    return value


# ============================================================
# New Memory Extraction System
# ============================================================

def extract_memories_from_cascade(
    extractor_model: Model,
    question: str,
    trajectories: List[Dict[str, Any]],
    sample_id: str,
    strategy: Literal["student", "teacher", "contrastive", "all"] = "all",
    extraction_mode: Literal["two_agent", "combined"] = "two_agent",
    prompt_profile: Optional[str] = None,
) -> List[MemoryItem]:
    """
    Extract memories from cascade trajectories.
    
    New Memory System Rules:
    1. Only two types of memories:
       - global_correct_flow: Complete correct solution process
       - local_corrective: Specific error correction (max 3)
    
    2. Memory extraction rules:
       - L1 success: Extract one global_correct_flow from L1
       - L1 fail, L2/L3 success:
         - two_agent: extract global from teacher + local corrective from comparison
         - combined: single call outputs both global and local memories
       - All fail: No memories (failure memories not stored)
    """
    if not trajectories:
        return []
    
    memories: List[MemoryItem] = []
    traj_by_level = {t["level"]: t for t in trajectories}
    
    l1 = traj_by_level.get(1)
    l2 = traj_by_level.get(2)
    l3 = traj_by_level.get(3)
    
    l1_success = l1 and l1.get("judge", {}).get("label") == "success"
    l2_success = l2 and l2.get("judge", {}).get("label") == "success"
    l3_success = l3 and l3.get("judge", {}).get("label") == "success"
    
    # Case 1: L1 success -> Only extract global from L1
    if l1_success:
        global_mems = extract_global_correct_flow(
            extractor_model, question, l1, sample_id, source_level=1,
            prompt_profile=prompt_profile
        )
        memories.extend(global_mems)
        return memories
    
    # Case 2: L1 fail, teacher success -> Extract global + local corrective
    teacher_traj = None
    teacher_level = None
    if l3_success:
        teacher_traj = l3
        teacher_level = 3
    elif l2_success:
        teacher_traj = l2
        teacher_level = 2
    
    if teacher_traj:
        if extraction_mode == "combined":
            combined = extract_combined_memories(
                extractor_model=extractor_model,
                question=question,
                failed_traj=l1,
                success_traj=teacher_traj,
                sample_id=sample_id,
                success_level=teacher_level,
                max_local_items=3,
                prompt_profile=prompt_profile,
            )
            memories.extend(combined)
        else:
            # Extract one global strategy from teacher
            global_mems = extract_global_correct_flow(
                extractor_model, question, teacher_traj, sample_id, source_level=teacher_level,
                prompt_profile=prompt_profile
            )
            memories.extend(global_mems)
            
            # Extract local corrective memories (max 3)
            local_mems = extract_local_corrective_memories(
                extractor_model, question, l1, teacher_traj, sample_id, max_items=3,
                prompt_profile=prompt_profile
            )
            memories.extend(local_mems)
    
    # Case 3: All fail -> Return empty (no failure memories stored)
    return memories


def extract_preference_memories(
    extractor_model: Model,
    context: str,
    response1: str,
    response2: str,
    overall_preference: int,
    sample_id: str,
    max_items: int = 3,
    dimension_hint: Optional[Any] = None,
    prompt_profile: Optional[str] = None,
) -> List[MemoryItem]:
    """
    Extract preference-style memory for HelpSteer3 preference pairs.

    Memory format: trigger / dimension / comparison 三元组。
    - trigger: 触发该规则的情境（输入/提示中的关键情境）
    - dimension: 相关质量维度（如 helpfulness, correctness, clarity）
    - comparison: 比较（偏好 B 行为而非 A 行为）
    """
    if overall_preference == 0:
        return []

    preferred_label = "Response 1" if overall_preference < 0 else "Response 2"
    rejected_label = "Response 2" if overall_preference < 0 else "Response 1"

    timestamp = _iso_timestamp()
    source = {
        "sample_id": sample_id,
        "timestamp": timestamp,
        "overall_preference": overall_preference,
        "preferred": preferred_label,
    }
    context_text = (context or "")[:1500]
    r1 = (response1 or "")[:1200]
    r2 = (response2 or "")[:1200]

    is_advancedif = (prompt_profile or "").lower() in {"advancedif", "wildif", "ifbench"}
    if is_advancedif:
        prompt = f'''You are extracting instruction-following improvement rules from a pairwise comparison.

Input/Prompt (may be conversation, code, or problem statement):
{context_text}

Response 1:
{r1}

Response 2:
{r2}

Label: {preferred_label} is better overall (overall_preference={overall_preference}).

OUTPUT FORMAT (JSON ONLY):
Return a JSON array with 1 to 3 objects:
[
  {{
    "trigger": "The specific instruction/constraint where this rule applies (e.g., 'When asked to order items by cost', 'When the user forbids numbered lists').",
    "dimension": "The key failure axis (e.g., format adherence, constraint satisfaction, ordering, completeness, relevance, accuracy).",
    "comparison": "An actionable, testable directive that improves compliance. It must be conditional on the trigger (e.g., 'When ordering is requested, sort items by the specified criterion before writing the list')."
  }}
]
Rules:
- Output ONLY valid JSON. No extra text, no Markdown, no code fences.
- Output at least 1 item if any meaningful preference can be derived; only output [] if responses are essentially identical or no preference is possible.
- Use ONLY these keys: trigger, dimension, comparison. Do NOT output title/type.
- Do NOT mention Response 1/2 or Option 1/2 in any field.
- Do NOT output generic guidelines or checklists; output only the JSON array.
- Do NOT output placeholders like "<...>".
'''
    else:
        prompt = f'''You are extracting reusable preference rules from a pairwise preference example.

Input/Prompt (may be conversation, code, or problem statement):
{context_text}

Response 1:
{r1}

Response 2:
{r2}

Label: {preferred_label} is better overall (overall_preference={overall_preference}).

OUTPUT FORMAT (JSON ONLY):
Return a JSON array with 1 to 3 objects:
[
  {{
    "trigger": "The specific context or constraint where this preference applies (e.g., 'When asked to write a concise marketing email...').",
    "dimension": "The evaluation axis driving the decision, e.g. helpfulness, correctness, safety, clarity, specificity, coherence, conciseness, completeness, relevance, accuracy",
    "comparison": "An explicit directive that determines the better response (e.g., 'The code is more concise and it solves the core problem', 'Provide direct bullet points instead of dense narrative paragraphs'). Don't mention response 1 or response 2."
  }}
]
Rules:
- Output ONLY valid JSON. No extra text, no Markdown, no code fences.
- Output at least 1 item if any meaningful preference can be derived; only output [] if responses are essentially identical or no preference is possible.
- Use ONLY these keys: trigger, dimension, comparison. Do NOT output title/type.
- Do NOT mention Response 1/2 or Option 1/2 in any field.
- Do NOT output generic guidelines or checklists; output only the JSON array.
- Do NOT output placeholders like "<...>".
'''

    return _generate_memories_with_retry(
        extractor_model,
        prompt,
        source,
        "preference_rule",
        answer_hint=None,
        max_items=max_items,
        prompt_profile=prompt_profile,
    )


def extract_global_correct_flow(
    extractor_model: Model,
    question: str,
    success_traj: Dict[str, Any],
    sample_id: str,
    source_level: int,
    max_items: int = 1,
    prompt_profile: Optional[str] = None,
) -> List[MemoryItem]:
    """
    Extract one global strategy from a successful trajectory.
    使用改进的 prompt，不使用 <> 占位符格式。
    对于 HotpotQA 使用专门的 QA 任务 prompt。
    """
    timestamp = _iso_timestamp()
    source = {
        "sample_id": sample_id,
        "level": source_level,
        "success": True,
        "timestamp": timestamp,
    }
    
    reasoning = _get_memory_response(success_traj)[:2500]
    answer_hint = success_traj.get("parsed_answer")
    
    # 根据 prompt_profile 选择不同的 prompt
    if prompt_profile == "hotpotqa":
        # QA任务专用prompt - 与combined模式风格一致
        prompt = f'''Extract ONE reasoning pattern from this correct QA answer.

Question: {question[:600]}

Correct answer: {reasoning[:1200]}

GOOD: "Comparison questions: list both entities' attributes first"
BAD: "Read carefully", "Pay attention"

JSON output (1 item):
[{{"title": "5-8 words", "description": "when to use", "content": "how to apply"}}]

Output [] if nothing useful.'''
    else:
        # 原有的数学任务prompt
        prompt = f'''You successfully solved this math problem. Extract 1 generalizable strategy that led to success.

PROBLEM:
{question}

YOUR SOLUTION (truncated):
{reasoning}

OUTPUT FORMAT (JSON ONLY):
Return a JSON array with exactly 1 object:
[
  {{
    "title": "3-8 word strategy name",
    "description": "one sentence summary",
    "content": "2-4 sentences: why it works and how to apply"
  }}
]

Rules:
- Output ONLY valid JSON. No extra text, no Markdown.
- Do NOT output placeholders like "<...>".
- If you truly cannot extract anything, output [].
'''

    return _generate_memories_with_retry(
        extractor_model,
        prompt,
        source,
        "global_correct_flow",
        answer_hint,
        max_items=max_items,
    )


def extract_local_corrective_memories(
    extractor_model: Model,
    question: str,
    failed_traj: Dict[str, Any],
    success_traj: Dict[str, Any],
    sample_id: str,
    max_items: int = 3,
    prompt_profile: Optional[str] = None,
) -> List[MemoryItem]:
    """
    Extract local corrective memories by comparing failed and successful trajectories.
    使用改进的 prompt，不使用 <> 占位符格式。
    对于 HotpotQA 使用专门的 QA 任务 prompt。
    """
    timestamp = _iso_timestamp()
    source = {
        "sample_id": sample_id,
        "failed_level": failed_traj.get("level") if failed_traj else None,
        "success_level": success_traj.get("level"),
        "timestamp": timestamp,
    }
    
    failed_response = _get_memory_response(failed_traj)[:1500] if failed_traj else ""
    success_response = _get_memory_response(success_traj)[:1500]
    
    # 根据 prompt_profile 选择不同的 prompt
    if prompt_profile == "hotpotqa":
        # QA任务专用prompt - 关注格式/陷阱，与combined模式风格一致
        prompt = f'''Extract ONE format tip or pitfall warning from this failed QA attempt.

Question: {question[:600]}

Failed: {failed_response[:800]}

Correct: {success_response[:800]}

GOOD:
- format: "Answer with name only, no explanation"
- pitfall: "Two people share first name - verify full identity"

BAD: "Read carefully", "Pay attention"

JSON output (1 item):
[{{"title": "5-8 words", "description": "when this happens", "content": "how to avoid"}}]

Output [] if nothing useful.'''
    else:
        # 原有的数学任务prompt
        prompt = f'''You attempted this math problem but made mistakes. Extract 1-3 specific lessons about what went wrong.

PROBLEM:
{question}

FAILED ATTEMPT (truncated):
{failed_response}

CORRECT SOLUTION (truncated):
{success_response}

OUTPUT FORMAT (JSON ONLY):
Return a JSON array with 1 to 3 objects:
[
  {{
    "title": "3-10 word mistake/lesson name",
    "description": "one sentence describing the error",
    "content": "2-4 sentences: what went wrong + how to avoid"
  }}
]

Rules:
- Output ONLY valid JSON. No extra text, no Markdown.
- Do NOT output placeholders like "<...>".
- If no distinct lesson exists, output [].
'''

    answer_hints = [
        failed_traj.get("parsed_answer") if failed_traj else None,
        success_traj.get("parsed_answer"),
    ]
    
    return _generate_memories_with_retry(
        extractor_model,
        prompt,
        source,
        "local_corrective",
        answer_hints,
        max_items=max_items,
    )


def _build_combined_memory_prompt(
    question: str,
    failed_response: str,
    success_response: str,
    prompt_profile: Optional[str] = None,
) -> str:
    """
    Single-call extraction prompt that outputs both global_correct_flow and local_corrective.
    对于 HotpotQA 使用专门的 QA 任务 prompt。
    """
    if prompt_profile == "hotpotqa":
        # QA任务专用prompt - 简化为1个推理模式 + 1个格式/陷阱提示
        return f'''Analyze this QA example. Extract:
1. ONE reasoning pattern from the correct answer
2. ONE format tip OR pitfall warning from the failed attempt

Question: {question[:600]}

Failed: {failed_response[:800]}

Correct: {success_response[:800]}

GOOD:
- reasoning: "Comparison questions: list both entities' attributes first"
- format: "Answer with name only, no explanation needed"
- pitfall: "Two people share first name - check full context for identity"

BAD (too vague): "Read carefully", "Pay attention"

JSON output:
[
  {{"type": "global_correct_flow", "title": "5-8 words", "description": "when to use", "content": "how to apply"}},
  {{"type": "local_corrective", "title": "5-8 words", "description": "when this happens", "content": "how to avoid"}}
]

Output [] if nothing useful.'''
    else:
        # 原有的数学任务prompt
        return f'''You are extracting reusable memories from a failed attempt and a correct solution.
global_correct_flow is a general solution strategy from the correct attempt; local_corrective is a specific error and how to avoid it.

PROBLEM:
{question}

FAILED ATTEMPT (truncated):
{failed_response}

CORRECT SOLUTION (truncated):
{success_response}

OUTPUT FORMAT (JSON ONLY):
Return a JSON array of objects. Each object must have:
{{
  "type": "global_correct_flow" | "local_corrective",
  "title": "3-10 word mistake/lesson name",
  "description": "one sentence describing the error",
  "content": "2-6 sentences: what went wrong + how to avoid"
}}

Rules:
- Output at most 1 global_correct_flow and 1-3 local_corrective.
- Make wording generalizable and concise.
- Output ONLY valid JSON. No extra text, no Markdown.
- Do NOT output placeholders like "<...>".
- If nothing useful, output [].
'''


def _normalize_memory_type(value: Any) -> str:
    if value is None:
        return ""
    lowered = str(value).strip().lower()
    if lowered in {"global_correct_flow", "global", "global_correct", "global_flow"}:
        return "global_correct_flow"
    if lowered in {"local_corrective", "local", "local_correct", "local_correction"}:
        return "local_corrective"
    if lowered in {"preference_rule", "preference", "pref_rule"}:
        return "preference_rule"
    return ""


def _parse_typed_memory_response(response_text: str) -> List[Dict[str, str]]:
    """
    Parse extractor output into a list of {type, title, description, content}.
    Only accepts JSON array responses.
    """
    if not response_text:
        return []
    arr = _parse_json_array(response_text)
    if not isinstance(arr, list):
        return []
    out: List[Dict[str, str]] = []
    for item in arr:
        if not isinstance(item, dict):
            continue
        mem_type = _normalize_memory_type(item.get("type", ""))
        if not mem_type:
            continue
        title = str(item.get("title", "")).strip()
        description = str(item.get("description", "")).strip()
        content = str(item.get("content", "")).strip()
        out.append(
            {
                "type": mem_type,
                "title": title,
                "description": description,
                "content": content,
            }
        )
    return out


def extract_combined_memories(
    extractor_model: Model,
    question: str,
    failed_traj: Optional[Dict[str, Any]],
    success_traj: Dict[str, Any],
    sample_id: str,
    success_level: int,
    max_local_items: int = 3,
    max_retries: int = 2,
    prompt_profile: Optional[str] = None,
) -> List[MemoryItem]:
    """
    Single-call extraction: output both global_correct_flow and local_corrective.
    """
    if not success_traj:
        return []

    timestamp = _iso_timestamp()
    source_global = {
        "sample_id": sample_id,
        "level": success_level,
        "success": True,
        "timestamp": timestamp,
    }
    source_local = {
        "sample_id": sample_id,
        "failed_level": failed_traj.get("level") if failed_traj else None,
        "success_level": success_traj.get("level"),
        "timestamp": timestamp,
    }

    failed_response = _get_memory_response(failed_traj)[:1500] if failed_traj else ""
    success_response = _get_memory_response(success_traj)[:1500]

    prompt = _build_combined_memory_prompt(
        question=question,
        failed_response=failed_response,
        success_response=success_response,
        prompt_profile=prompt_profile,
    )
    current_prompt = prompt

    base_max_tokens = int(os.getenv("MEMORY_MAX_TOKENS", "4096"))
    retry_max_tokens = int(os.getenv("MEMORY_RETRY_MAX_TOKENS", "8192"))

    typed_memories: List[Dict[str, str]] = []
    for attempt in range(max_retries):
        try:
            response = extractor_model.generate(
                prompt=current_prompt,
                system_prompt=None,
                temperature=0.0,
                do_sample=False,
                max_tokens=(base_max_tokens if attempt == 0 else retry_max_tokens),
            )
        except ModelError:
            return []

        typed_memories = _parse_typed_memory_response(response.text)
        if typed_memories and not _has_any_placeholder(typed_memories):
            break

        if attempt < max_retries - 1:
            _debug_memory_extractor_output("combined", attempt + 1, response.text)
            current_prompt = prompt + '''

CRITICAL REMINDER:
- Output ONLY valid JSON (a JSON array). No extra text, no Markdown.
- Do NOT output placeholders like "<...>" or "your title here".
- Each object must have non-empty "type", "title", and "content".
- If you have no useful memory, output [] exactly.'''
            print(f"  [Memory] Detected placeholder/empty, retrying... (attempt {attempt + 2}/{max_retries})")

    if not typed_memories:
        return []

    global_items = [m for m in typed_memories if m.get("type") == "global_correct_flow"][:1]
    local_items = [m for m in typed_memories if m.get("type") == "local_corrective"][:max_local_items]

    built: List[MemoryItem] = []
    if global_items:
        built.extend(
            _build_memory_items(
                global_items,
                source_global,
                "global_correct_flow",
                success_traj.get("parsed_answer"),
                max_items=1,
            )
        )
    if local_items:
        answer_hints = [
            failed_traj.get("parsed_answer") if failed_traj else None,
            success_traj.get("parsed_answer"),
        ]
        built.extend(
            _build_memory_items(
                local_items,
                source_local,
                "local_corrective",
                answer_hints,
                max_items=max_local_items,
            )
        )

    return built


def _generate_memories_with_retry(
    extractor_model: Model,
    prompt: str,
    source: Dict[str, Any],
    memory_type: str,
    answer_hint: Any = None,
    max_items: int = 3,
    max_retries: int = 2,
    prompt_profile: Optional[str] = None,
) -> List[MemoryItem]:
    """
    Generate memories with retry logic for placeholder detection.
    如果检测到占位符，会重试生成。
    """
    current_prompt = prompt

    # Memory extraction should be short and deterministic. Allow env override.
    base_max_tokens = int(os.getenv("MEMORY_MAX_TOKENS", "4096"))
    retry_max_tokens = int(os.getenv("MEMORY_RETRY_MAX_TOKENS", "8192"))

    
    system_prompt = None
    if memory_type == "preference_rule":
        system_prompt = "You are a JSON formatter for extracting LLM memory. Output ONLY a JSON array. No other text."
    is_advancedif = (prompt_profile or "").lower() in {"advancedif", "wildif", "ifbench"}

    for attempt in range(max_retries):
        try:
            response = extractor_model.generate(
                prompt=current_prompt,
                system_prompt=system_prompt,
                temperature=0.0,
                do_sample=False,
                max_tokens=(base_max_tokens if attempt == 0 else retry_max_tokens),
            )
        except ModelError:
            return []
        
        memories = _parse_memory_response(response.text)
        # preference_rule: 若空数组且尚有重试机会，强制重试一次
        if memories is not None and len(memories) == 0:
            if memory_type == "preference_rule" and attempt < max_retries - 1:
                _debug_memory_extractor_output("single", attempt + 1, response.text)
                if is_advancedif:
                    current_prompt = prompt + '''

CRITICAL REMINDER:
- Output ONLY the JSON array. No explanation, no markdown, no code fence (no ```), no text before or after.
- For preference rules: output at least 1 item unless the two responses are essentially identical.
- Each preference object must include ONLY "trigger", "dimension", "comparison" (no "title", no "type").
- Comparison must be a single actionable directive sentence that contrasts preferred vs dispreferred behavior.
- Do NOT mention Response 1/2 or Option 1/2 in any field.
- Do NOT output generic guidelines or checklists; output only the JSON array.
'''
                else:
                    current_prompt = prompt + '''

CRITICAL REMINDER:
- Output ONLY the JSON array. No explanation, no markdown, no code fence (no ```), no text before or after.
- For preference rules: output at least 1 item unless the two responses are essentially identical.
- Each preference object must include ONLY "trigger", "dimension", "comparison" (no "title", no "type").
- Comparison must be a single directive sentence (>=25 words) that contrasts preferred vs dispreferred behavior.
- Do NOT mention Response 1/2 or Option 1/2 in any field.
- Do NOT output generic guidelines or checklists; output only the JSON array.
'''
                print(f"  [Memory] Empty preference output, retrying... (attempt {attempt + 2}/{max_retries})")
                continue
            return []
        # 检查是否有占位符
        if memories and not _has_any_placeholder(memories):
            built = _build_memory_items(
                memories,
                source,
                memory_type,
                answer_hint,
                max_items=max_items,
                prompt_profile=prompt_profile,
            )
            if built:
                return built

        # 触发重试：占位符、解析到无效项、或构建后为空
        if attempt < max_retries - 1:
            _debug_memory_extractor_output("single", attempt + 1, response.text)
            if is_advancedif:
                current_prompt = prompt + '''

CRITICAL REMINDER:
- Output ONLY the JSON array. No explanation, no markdown, no code fence (no ```), no text before or after.
- Do NOT output placeholders like "<...>".
- For preference rules: each object must have ONLY "trigger", "dimension", "comparison" (no "title", no "type").
- For preference rules: comparison must be a single actionable directive sentence that contrasts preferred vs dispreferred behavior.
- Do NOT mention Response 1/2 or Option 1/2 in any field.
- Do NOT output generic guidelines or checklists; output only the JSON array.
- If you have no useful memory, output [] exactly.'''
            else:
                current_prompt = prompt + '''

CRITICAL REMINDER:
- Output ONLY the JSON array. No explanation, no markdown, no code fence (no ```), no text before or after.
- Do NOT output placeholders like "<...>".
- For preference rules: each object must have ONLY "trigger", "dimension", "comparison" (no "title", no "type").
- For preference rules: comparison must be a single directive sentence (>=25 words) that contrasts preferred vs dispreferred behavior.
- Do NOT mention Response 1/2 or Option 1/2 in any field.
- Do NOT output generic guidelines or checklists; output only the JSON array.
- If you have no useful memory, output [] exactly.'''
            print(f"  [Memory] Detected placeholder/empty, retrying... (attempt {attempt + 2}/{max_retries})")
    
    # 最后一次尝试，即使有占位符也尝试构建（会被 _build_memory_items 过滤）
    if memories:
        return _build_memory_items(
            memories,
            source,
            memory_type,
            answer_hint,
            max_items=max_items,
            prompt_profile=prompt_profile,
        )
    
    return []


def _build_memory_items(
    memories_data: List[Dict[str, str]],
    source: Dict[str, Any],
    memory_type: str,
    answer_hint: Any,
    max_items: int = 3,
    prompt_profile: Optional[str] = None,
) -> List[MemoryItem]:
    """
    构建记忆项，过滤掉无效的占位符。
    """
    items: List[MemoryItem] = []
    skipped_count = 0
    
    for mem_data in memories_data:
        title = _to_str(mem_data.get("title", ""))[:80]
        description = _to_str(mem_data.get("description", ""))[:200]
        content = _to_str(mem_data.get("content", ""))[:500]
        trigger = _to_str(mem_data.get("trigger", "")).strip()[:200]
        dimension = _to_str(mem_data.get("dimension", "")).strip()[:200]
        comparison = _to_str(mem_data.get("comparison", "")).strip()[:500]
        
        # preference_rule (HelpSteer3): 内存中保持三元组 trigger/dimension/comparison，不塞进 content
        if memory_type == "preference_rule" and trigger and dimension and comparison:
            if _is_placeholder(trigger):
                skipped_count += 1
                print(f"  [Memory] Skipped placeholder trigger: '{trigger[:50]}'")
                continue
            title = trigger
            description = ""
            t = _strip_response_labels(_sanitize_memory_field(trigger, answer_hint))
            d = _strip_response_labels(_sanitize_memory_field(dimension, answer_hint))
            c = _strip_response_labels(_sanitize_memory_field(comparison, answer_hint))
            if not t or not d or not c:
                continue
            is_advancedif = (prompt_profile or "").lower() in {"advancedif", "wildif", "ifbench"}
            if is_advancedif:
                # Keep comparison as-is for AdvancedIF-style memories (no forced suffix/min length).
                c = c.rstrip(".") + "."
            else:
                c = _normalize_preference_comparison(c, min_words=25)
                if len(c.split()) < 25:
                    skipped_count += 1
                    continue
            item = MemoryItem(
                type="preference_rule",
                title=title,
                description="",
                content="",
                source=source.copy(),
                trigger=t,
                dimension=d,
                comparison=c,
            )
            items.append(item)
            if len(items) >= max_items:
                break
            continue
        else:
            # 检查标题是否为占位符
            if _is_placeholder(title):
                skipped_count += 1
                print(f"  [Memory] Skipped placeholder title: '{title[:50]}'")
                continue
            # 描述为占位符时清空（不跳过整条记录）
            if _is_placeholder(description):
                description = ""
            # 检查内容是否为占位符
            if _is_placeholder(content):
                skipped_count += 1
                print(f"  [Memory] Skipped placeholder content for: '{title[:30]}'")
                continue
            # 检查内容长度
            if len(content.strip()) < 20:
                skipped_count += 1
                continue
            content = _sanitize_memory_field(content, answer_hint)
            if not title or not content:
                continue
            content_data = {"content": content}

        title = _redact_answer_tokens(title, answer_hint)
        description = _redact_answer_tokens(description, answer_hint)
        if not title:
            continue

        item = MemoryItem(
            type=memory_type,
            title=title,
            description=description,
            content=json.dumps(content_data, ensure_ascii=False),
            source=source.copy(),
        )
        items.append(item)
        
        if len(items) >= max_items:
            break
    
    if skipped_count > 0:
        print(f"  [Memory] Total skipped: {skipped_count} placeholder/invalid entries")
    
    return items


def _get_memory_response(traj: Dict[str, Any]) -> str:
    return traj.get("extraction_response") or traj.get("response", "") or ""


# ============================================================
# Legacy Functions (for backward compatibility)
# ============================================================

def extract_student_memory(
    extractor_model: Model,
    question: str,
    student_traj: Dict[str, Any],
    sample_id: str,
) -> List[MemoryItem]:
    """Legacy: Extract memory from student trajectory."""
    is_success = student_traj.get("judge", {}).get("label") == "success"
    if is_success:
        return extract_global_correct_flow(
            extractor_model, question, student_traj, sample_id, source_level=1
        )
    return []


def extract_teacher_memory(
    extractor_model: Model,
    question: str,
    teacher_traj: Dict[str, Any],
    sample_id: str,
) -> List[MemoryItem]:
    """Legacy: Extract memory from teacher trajectory."""
    is_success = teacher_traj.get("judge", {}).get("label") == "success"
    if is_success:
        level = teacher_traj.get("level", 2)
        return extract_global_correct_flow(
            extractor_model, question, teacher_traj, sample_id, source_level=level
        )
    return []


def extract_contrastive_memory(
    extractor_model: Model,
    question: str,
    failed_traj: Dict[str, Any],
    success_traj: Dict[str, Any],
    sample_id: str,
) -> Optional[MemoryItem]:
    """Legacy: Extract contrastive memory."""
    mems = extract_local_corrective_memories(
        extractor_model, question, failed_traj, success_traj, sample_id, max_items=1
    )
    return mems[0] if mems else None


# ============================================================
# Utility Functions
# ============================================================

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



def _parse_json_array(text: str) -> Optional[List[Any]]:
    """Extract a JSON array from a blob of text. Tolerates leading/trailing text and markdown code fences."""
    decoder = json.JSONDecoder()
    if not text:
        return None

    def _strip_code_fences(blob: str) -> List[str]:
        """Return list of inner contents of all ```...``` blocks, plus blob if no fence."""
        out: List[str] = []
        for match in re.finditer(r"```(?:json)?\s*(.*?)```", blob, flags=re.DOTALL | re.IGNORECASE):
            inner = match.group(1).strip()
            if inner:
                out.append(inner)
        if not out:
            out.append(blob)
        return out

    def _escape_invalid_backslashes(s: str) -> str:
        out = []
        i = 0
        while i < len(s):
            ch = s[i]
            if ch == "\\":
                nxt = s[i + 1] if i + 1 < len(s) else ""
                if nxt in {'"', "\\", "/", "b", "f", "n", "r", "t", "u"}:
                    out.append(ch)
                else:
                    out.append("\\\\")
                i += 1
                continue
            out.append(ch)
            i += 1
        return "".join(out)

    def _try_decode_array(s: str) -> Optional[List[Any]]:
        for idx, ch in enumerate(s):
            if ch != "[":
                continue
            try:
                arr, _ = decoder.raw_decode(s[idx:])
                if isinstance(arr, list):
                    return arr
            except json.JSONDecodeError:
                try:
                    arr, _ = decoder.raw_decode(_escape_invalid_backslashes(s[idx:]))
                    if isinstance(arr, list):
                        return arr
                except json.JSONDecodeError:
                    continue
        return None

    # Collect all candidate strings: full text + each code-fence inner content
    candidates: List[str] = [text.strip()]
    candidates.extend(_strip_code_fences(text))
    seen: set = set()
    best: Optional[List[Any]] = None
    for candidate in candidates:
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)
        # Scan every "[" in this candidate
        for idx, ch in enumerate(candidate):
            if ch != "[":
                continue
            arr = _try_decode_array(candidate[idx:])
            if not isinstance(arr, list):
                continue
            # Prefer non-empty; among non-empty prefer longer
            if not best or (len(arr) > 0 and (len(best) == 0 or len(arr) >= len(best))):
                best = arr
    return best


def _parse_memory_response(response_text: str) -> List[Dict[str, str]]:
    """
    Parse extractor output into a list of {title, description, content}.
    Prefer strict JSON (array of objects). Fallback to legacy MEMORY format.
    """
    if not response_text:
        return []

    # 1) Preferred: JSON array
    arr = _parse_json_array(response_text)
    if isinstance(arr, list):
        out: List[Dict[str, str]] = []
        for item in arr:
            if not isinstance(item, dict):
                continue
            title = str(item.get("title", "")).strip()
            description = str(item.get("description", "")).strip()
            content = str(item.get("content", "")).strip()
            entry: Dict[str, str] = {"title": title, "description": description, "content": content}
            # HelpSteer3 triplet: trigger, dimension, comparison
            for key in ("trigger", "dimension", "comparison"):
                if key in item and item[key] is not None:
                    entry[key] = str(item[key]).strip()
            out.append(entry)
        return out

    # 2) Accept single JSON object and wrap into list
    obj = _parse_json_object(response_text)
    if isinstance(obj, dict):
        return [
            {
                "title": str(obj.get("title", "")).strip(),
                "description": str(obj.get("description", "")).strip(),
                "content": str(obj.get("content", "")).strip(),
                **{
                    k: str(obj.get(k, "")).strip()
                    for k in ("trigger", "dimension", "comparison")
                    if obj.get(k) is not None
                },
            }
        ]

    # 3) Fallback: legacy MEMORY blocks
    return _parse_memory_format(response_text)

def _tf_vector(text: str) -> Dict[str, int]:
    tokens = re.findall(r"[A-Za-z0-9]+", text.lower())
    vec: Dict[str, int] = {}
    for tok in tokens:
        vec[tok] = vec.get(tok, 0) + 1
    return vec


def _cosine_sim(a: Dict[str, int], b: Dict[str, int]) -> float:
    if not a or not b:
        return 0.0
    dot = 0.0
    for key, val in a.items():
        dot += val * b.get(key, 0)
    norm_a = sqrt(sum(val * val for val in a.values()))
    norm_b = sqrt(sum(val * val for val in b.values()))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def _default_embedding_cache_path(memory_path: str) -> str:
    if not memory_path:
        return "memory_embeddings.npz"
    return f"{memory_path}.embeddings.npz"


def _item_text_for_index(item: MemoryItem) -> str:
    """用于检索索引的文本。preference_rule 用 title + trigger + dimension + comparison，其它用 title + content。"""
    if item.trigger is not None and item.dimension is not None and item.comparison is not None:
        return f"{item.title or ''} {(item.trigger or '')} {(item.dimension or '')} {(item.comparison or '')}".strip()
    return f"{item.title or ''} {item.content or ''}".strip()


def _memory_key(item: MemoryItem) -> str:
    if item.trigger is not None and item.dimension is not None and item.comparison is not None:
        base = f"{item.type}||{item.title}||{item.trigger}||{item.dimension}||{item.comparison}||{item.source}"
    else:
        base = f"{item.type}||{item.title}||{item.description}||{item.content}"
    return hashlib.sha1(base.encode("utf-8")).hexdigest()


def _memory_text_for_embedding(item: MemoryItem) -> str:
    title = (item.title or "").strip()
    description = (item.description or "").strip()
    # preference_rule: 直接用 item 上的三元组，不解析 content
    if item.trigger is not None and item.dimension is not None and item.comparison is not None:
        content_text = " ".join(
            [(item.trigger or "").strip(), (item.dimension or "").strip(), (item.comparison or "").strip()]
        ).strip()
    else:
        raw_content = (item.content or "").strip()
        content_text = raw_content
        if raw_content:
            try:
                data = json.loads(raw_content)
            except json.JSONDecodeError:
                data = None
            if isinstance(data, dict):
                content_text = ""
                if "trigger" in data and "dimension" in data and "comparison" in data:
                    content_text = " ".join(
                        str(data.get(k, "")).strip()
                        for k in ("trigger", "dimension", "comparison")
                        if data.get(k)
                    )
                elif isinstance(data.get("content"), str):
                    content_text = data["content"].strip()
                else:
                    content_text = json.dumps(data, ensure_ascii=False)
                if not description and isinstance(data.get("description"), str):
                    description = data["description"].strip()

    def _normalize_for_dupe(text: str) -> str:
        text = text.lower()
        text = re.sub(r"\s+", " ", text).strip()
        return text

    parts = [title]
    if description:
        if not content_text:
            parts.append(description)
        else:
            desc_norm = _normalize_for_dupe(description)
            content_norm = _normalize_for_dupe(content_text)
            if desc_norm not in content_norm:
                parts.append(description)
    if content_text:
        parts.append(content_text)
    return " ".join(p for p in parts if p).strip()


def _load_embedding_cache(path: str):
    try:
        import numpy as np
    except Exception:
        return None, []
    try:
        data = np.load(path, allow_pickle=False)
        keys = [str(k) for k in data["keys"].tolist()]
        embeddings = data["embeddings"].astype("float32")
        return embeddings, keys
    except Exception:
        return None, []


def _save_embedding_cache(path: str, keys: List[str], embeddings) -> None:
    try:
        import numpy as np
    except Exception:
        print("Warning: numpy not available; embedding cache not saved.")
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    try:
        np.savez(path, keys=np.array(keys), embeddings=np.asarray(embeddings, dtype="float32"))
    except Exception as e:
        print(f"Warning: Failed to save embedding cache to {path}: {e}")


def _append_embeddings(existing, new_embeddings):
    try:
        import numpy as np
    except Exception:
        return new_embeddings
    if hasattr(new_embeddings, "detach"):
        new_embeddings = new_embeddings.detach().cpu().numpy()
    new_arr = np.asarray(new_embeddings, dtype="float32")
    if existing is None:
        return new_arr
    if hasattr(existing, "detach"):
        existing = existing.detach().cpu().numpy()
    return np.concatenate([existing, new_arr], axis=0)


def _load_embedding_model(model_name: str, device: Optional[str]):
    import torch
    from transformers import AutoModel, AutoTokenizer

    device_name = device or ("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModel.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if device_name.startswith("cuda") else torch.float32,
    )
    model.to(device_name)
    model.eval()
    return tokenizer, model, device_name


def _embed_texts(
    model,
    tokenizer,
    texts: List[str],
    device: str,
    batch_size: int = 32,
    max_length: int = 512,
):
    import torch
    import torch.nn.functional as F
    try:
        import numpy as np
    except Exception:
        np = None

    embeddings = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]
        with torch.no_grad():
            if hasattr(model, "encode"):
                try:
                    out = model.encode(
                        batch,
                        convert_to_tensor=True,
                        normalize_embeddings=True,
                        device=device,
                    )
                    if isinstance(out, torch.Tensor):
                        emb = out
                    else:
                        emb = torch.tensor(out, device=device)
                except Exception:
                    emb = None
            else:
                emb = None

            if emb is None:
                enc = tokenizer(
                    batch,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                )
                enc = {k: v.to(device) for k, v in enc.items()}
                outputs = model(**enc)
                last_hidden = outputs.last_hidden_state
                mask = enc["attention_mask"].unsqueeze(-1)
                pooled = (last_hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
                emb = F.normalize(pooled, p=2, dim=1)

            embeddings.append(emb.detach().float().cpu())

    all_emb = torch.cat(embeddings, dim=0)
    if np is None:
        return all_emb
    return all_emb.numpy().astype("float32")


def _dot_similarity(a, b) -> float:
    if hasattr(a, "detach"):
        a = a.detach().cpu().numpy()
    if hasattr(b, "detach"):
        b = b.detach().cpu().numpy()
    try:
        import numpy as np
        return float(np.dot(a, b))
    except Exception:
        return float(sum(x * y for x, y in zip(a, b)))


def _iso_timestamp() -> str:
    return datetime.utcnow().isoformat() + "Z"


# ============================================================
# Memory Refinement
# ============================================================

_MEMORY_REFINEMENT_PROMPT = """You are a Memory Refinement Expert. Context: An agent successfully solved a task and generated several "New STRATEGIES" as memories. During the task, it had also retrieved several "Historical Memories" to assist its reasoning. Task: You are provided with this combined subset (New STRATEGIES + Retrieved Memories). Your goal is to refine this specific pool of memories into an optimized, consolidated set of memories.

Objectives for Refinement and Consolidation:

1.  **Elimination (Redundancy Check):** Identify any newly generated memories or historical memories that are completely or highly similar to others, and suggest eliminating these redundant items.
2.  **Consolidation (Merging):** Identify memories that can be merged (e.g., a new memory is a more specific or general form of a historical one), and combine them into a single, more comprehensive and robust knowledge unit.
3.  **Refinement (Optimization):** If the phrasing or content of a memory is not general enough or is imprecise, modify its content to increase its transferability and value for future, similar problems.
4.  **Retention (Keep):** Directly retain any memory that is valuable, non-redundant, and does not require merging or refinement.

**Your final output must be the optimal, consolidated set of strategies.**
**Note on Historical Memories:** A retrieved historical memory might have been irrelevant to the current problem's solution, but this does not necessarily make it useless. Evaluate all retrieved memories based on their **generalizability and potential to help solve other problems** in the future.

---
**PROBLEM THAT SUCCEED:**
{question}

**AGENT'S REASONING TRAJECTORY:**
{reasoning}

**NEWLY EXTRACTED ORIGINAL STRATEGIES (MEMORIES):**
{new_memories}

**RETRIEVED HISTORICAL MEMORIES:**
{retrieved_memories}

---
**Strictly output the final refined and consolidated strategy list in this format:**

MEMORY 1:
TITLE: <concise strategy name>
DESCRIPTION: <one sentence summary>
CONTENT: <detailed transferable strategy>

MEMORY 2:
...
"""


def _memory_content_for_refinement(item: MemoryItem) -> str:
    if item.trigger is not None and item.dimension is not None and item.comparison is not None:
        return (
            f"Trigger: {(item.trigger or '').strip()} | "
            f"Dimension: {(item.dimension or '').strip()} | "
            f"Comparison: {(item.comparison or '').strip()}"
        )
    raw = (item.content or "").strip()
    if not raw:
        return ""
    try:
        data = json.loads(raw)
        if isinstance(data, dict):
            if isinstance(data.get("content"), str):
                return data.get("content", "").strip()
            if isinstance(data.get("strategy"), str):
                return data.get("strategy", "").strip()
            if isinstance(data.get("description"), str):
                return data.get("description", "").strip()
            return json.dumps(data, ensure_ascii=False)
    except json.JSONDecodeError:
        pass
    return raw


def _serialize_memories_for_refinement(memories: List[MemoryItem]) -> str:
    if not memories:
        return "None"
    lines: List[str] = []
    for idx, item in enumerate(memories, start=1):
        lines.append(f"{idx}. TYPE: {(item.type or '').strip()}")
        lines.append(f"TITLE: {(item.title or '').strip()}")
        if item.description:
            lines.append(f"DESCRIPTION: {(item.description or '').strip()}")
        content = _memory_content_for_refinement(item)
        if content:
            lines.append(f"CONTENT: {content}")
        lines.append("")
    return "\n".join(lines).strip()


def _infer_refined_memory_type(
    mem_data: Dict[str, str],
    available_types: List[str],
) -> str:
    deduped_types = [t for t in dict.fromkeys(available_types) if t in {"global_correct_flow", "local_corrective"}]
    if len(deduped_types) == 1:
        return deduped_types[0]
    text = " ".join(
        [
            str(mem_data.get("title", "")),
            str(mem_data.get("description", "")),
            str(mem_data.get("content", "")),
        ]
    ).lower()
    local_markers = (
        "avoid",
        "error",
        "mistake",
        "pitfall",
        "incorrect",
        "prevent",
        "check",
    )
    global_markers = (
        "strategy",
        "approach",
        "workflow",
        "process",
        "steps",
    )
    local_hits = sum(1 for marker in local_markers if marker in text)
    global_hits = sum(1 for marker in global_markers if marker in text)
    if local_hits > global_hits:
        return "local_corrective"
    return "global_correct_flow"


def refine_memories(
    extractor_model: Model,
    question: str,
    reasoning: str,
    new_memories: List[MemoryItem],
    retrieved_memories: List[MemoryItem],
    success: bool,
    sample_id: str,
) -> List[MemoryItem]:
    """
    Refine and consolidate memories using an LLM consolidator.
    """
    if not success:
        return []

    valid_new = [
        m
        for m in new_memories
        if m.type in {"global_correct_flow", "local_corrective"} and not _is_placeholder(m.title)
    ]
    valid_retrieved = [
        m
        for m in retrieved_memories
        if m.type in {"global_correct_flow", "local_corrective"} and not _is_placeholder(m.title)
    ]
    if not valid_new and not valid_retrieved:
        return []

    prompt = _MEMORY_REFINEMENT_PROMPT.format(
        question=(question or "").strip()[:4000],
        reasoning=(reasoning or "").strip()[:8000],
        new_memories=_serialize_memories_for_refinement(valid_new),
        retrieved_memories=_serialize_memories_for_refinement(valid_retrieved),
    )

    try:
        response = extractor_model.generate(
            prompt=prompt,
            system_prompt=None,
            temperature=0.0,
            max_tokens=3072,
        )
        response_text = (response.text or "").strip()
    except Exception as exc:
        print(f"  [MemoryRefine] model call failed for sample {sample_id}: {exc}")
        return valid_new

    parsed = _parse_memory_response(response_text)
    if not parsed:
        print(f"  [MemoryRefine] parse failed for sample {sample_id}; fallback to original")
        return valid_new

    source = {}
    if valid_new:
        source.update(valid_new[0].source or {})
    source["sample_id"] = sample_id
    source["refined"] = True
    source["refined_at"] = _iso_timestamp()
    source["refinement_model"] = extractor_model.name

    available_types = [m.type for m in (valid_new + valid_retrieved)]
    grouped: Dict[str, List[Dict[str, str]]] = {
        "global_correct_flow": [],
        "local_corrective": [],
    }
    for mem_data in parsed:
        mem_type = _infer_refined_memory_type(mem_data, available_types)
        grouped[mem_type].append(mem_data)

    refined: List[MemoryItem] = []
    for mem_type, mem_list in grouped.items():
        if not mem_list:
            continue
        refined.extend(
            _build_memory_items(
                memories_data=mem_list,
                source=source,
                memory_type=mem_type,
                answer_hint=None,
                max_items=len(mem_list),
            )
        )
    if not refined:
        return valid_new
    return refined
