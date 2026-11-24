"""
Improved GRPO Training using Pre-collected Rollout Data

Key improvements over original:
1. Uses pre-collected rollouts (no regeneration during training)
2. Groups samples by question for proper advantage computation
3. More efficient training loop
4. Better KL divergence handling

Usage:
    # After unified production
    python grpo_improved.py train \
        --data ./unified/grpo_samples.jsonl \
        --model Qwen/Qwen2.5-7B-Instruct \
        --output ./grpo_model
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime
import torch
import torch.nn.functional as F

@dataclass
class TrainingHistory:
    """Track training metrics"""
    steps: List[int] = field(default_factory=list)
    losses: List[float] = field(default_factory=list)
    policy_losses: List[float] = field(default_factory=list)
    kl_losses: List[float] = field(default_factory=list)
    rewards: List[float] = field(default_factory=list)
    accuracies: List[float] = field(default_factory=list)
    epoch_summaries: List[Dict[str, float]] = field(default_factory=list)
    
    def add_step(self, step: int, loss: float, policy_loss: float = 0, 
                 kl_loss: float = 0, reward: float = 0, accuracy: float = 0):
        self.steps.append(step)
        self.losses.append(loss)
        self.policy_losses.append(policy_loss)
        self.kl_losses.append(kl_loss)
        self.rewards.append(reward)
        self.accuracies.append(accuracy)
    
    def add_epoch(self, epoch_stats: Dict[str, float]):
        self.epoch_summaries.append(epoch_stats)
    
    def save(self, path: str):
        data = {
            "steps": self.steps,
            "losses": self.losses,
            "policy_losses": self.policy_losses,
            "kl_losses": self.kl_losses,
            "rewards": self.rewards,
            "accuracies": self.accuracies,
            "epoch_summaries": self.epoch_summaries,
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)


def load_grpo_data_grouped(data_path: str) -> List[Dict[str, Any]]:
    """
    Load GRPO data and group by question/sample_id.
    
    Returns list of question groups, each containing:
    - sample_id: str
    - question: str
    - ground_truth: str
    - samples: List[{response, reward, parsed_answer, is_correct}]
    """
    # Load all samples
    samples_by_id: Dict[str, Dict[str, Any]] = {}
    
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                sample = json.loads(line)
            except json.JSONDecodeError:
                continue
            
            sample_id = sample.get("sample_id", "")
            if not sample_id:
                continue
            
            if sample_id not in samples_by_id:
                samples_by_id[sample_id] = {
                    "sample_id": sample_id,
                    "question": sample.get("question", ""),
                    "ground_truth": sample.get("ground_truth", ""),
                    "prompt_profile": sample.get("prompt_profile"),
                    "samples": [],
                }
            elif not samples_by_id[sample_id].get("prompt_profile") and sample.get("prompt_profile"):
                samples_by_id[sample_id]["prompt_profile"] = sample.get("prompt_profile")
            
            samples_by_id[sample_id]["samples"].append({
                "response": sample.get("response", ""),
                "reward": sample.get("reward", 0.0),
                "parsed_answer": sample.get("parsed_answer"),
                "is_correct": sample.get("is_correct", False),
            })
    
    # Filter out questions with insufficient samples
    result = [
        group for group in samples_by_id.values()
        if len(group["samples"]) >= 2  # Need at least 2 for advantage computation
    ]
    
    return result


def compute_group_advantages(rewards: List[float], normalize: bool = True) -> List[float]:
    """
    Compute group-relative advantages.
    
    advantage_i = reward_i - mean(rewards)
    
    Optionally normalize by std for stability.
    """
    import numpy as np
    
    if not rewards:
        return []
    
    rewards_arr = np.array(rewards)
    mean = np.mean(rewards_arr)
    
    if normalize:
        std = np.std(rewards_arr) + 1e-8
        advantages = (rewards_arr - mean) / std
        # Clip for stability
        advantages = np.clip(advantages, -5.0, 5.0)
    else:
        advantages = rewards_arr - mean
    
    return advantages.tolist()


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



def sequence_logp_on_response(
    model,
    tokenizer,
    prompt: str,
    response: str,
    max_length: int,
    device,
) -> Tuple["torch.Tensor", "torch.Tensor"]:
    """Compute log-probability of the response tokens (sum) under the model.

    Returns:
        seq_logp: scalar tensor, sum logp over response tokens
        tok_count: scalar tensor, number of response tokens considered (for debugging)
    """
    # Tokenize prompt separately to get prompt token length.
    # Use add_special_tokens=False for stable alignment; old/new must be computed identically.
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]

    full_text = prompt + response
    enc = tokenizer(
        full_text,
        return_tensors="pt",
        add_special_tokens=False,
        truncation=True,
        max_length=max_length,
        padding=False,
    )
    input_ids = enc["input_ids"].to(device)

    outputs = model(input_ids=input_ids)
    logits = outputs.logits[:, :-1, :]
    labels = input_ids[:, 1:]

    log_probs = F.log_softmax(logits, dim=-1)
    token_logp = log_probs.gather(-1, labels.unsqueeze(-1)).squeeze(-1)  # [1, T-1]

    prompt_len = len(prompt_ids)
    mask = torch.zeros_like(token_logp)
    if prompt_len < token_logp.shape[1]:
        # token_logp aligns with labels (shifted by 1), so start at prompt_len-1
        mask[:, prompt_len - 1 :] = 1.0

    seq_logp = (token_logp * mask).sum()
    tok_count = mask.sum().clamp_min(1.0)
    return seq_logp, tok_count


def run_grpo_training(
    data_path: str,
    model_name: str,
    output_dir: str,
    # Training hyperparameters
    learning_rate: float = 1e-5,
    num_epochs: int = 3,
    batch_size: int = 4,  # Questions per batch (not samples)
    gradient_accumulation_steps: int = 4,
    max_length: int = 2048,
    # GRPO specific
    kl_coef: float = 0.1,
    advantage_clip: float = 0.2,  # PPO-style advantage clipping
    # LoRA config
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    # Other
    use_4bit: bool = True,
    seed: int = 42,
    log_interval: int = 10,
    save_interval: int = 100,
    prompt_profile: Optional[str] = None,
    save_merged: bool = False,
    hotpotqa_short_prompt: bool = False,
):
    """
    Run GRPO training using pre-collected rollout data.
    
    Key differences from naive implementation:
    1. Data is already collected - no regeneration during training
    2. Samples are grouped by question for correct advantage computation
    3. More efficient training loop
    4. Better gradient handling
    """
    try:
        import torch
        import torch.nn.functional as F
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
        )
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    except ImportError as e:
        print(f"Missing dependency: {e}")
        print("Install: pip install transformers peft accelerate bitsandbytes")
        return
    
    print("=" * 60)
    print("GRPO Training (Improved)")
    print("=" * 60)
    print(f"Model: {model_name}")
    print(f"Data: {data_path}")
    print(f"Output: {output_dir}")
    print(f"KL coefficient: {kl_coef}")
    print(f"Learning rate: {learning_rate}")
    print(f"Batch size: {batch_size}")
    print(f"Grad accum: {gradient_accumulation_steps}")
    print(f"Max length: {max_length}")
    
    # Set random seed
    random.seed(seed)
    torch.manual_seed(seed)
    
    # Load data
    print("\nLoading data...")
    question_groups = load_grpo_data_grouped(data_path)
    print(f"Loaded {len(question_groups)} questions")
    
    total_samples = sum(len(g["samples"]) for g in question_groups)
    print(f"Total samples: {total_samples}")
    
    if not question_groups:
        print("No training data!")
        return
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Quantization config
    bnb_config = None
    if use_4bit and torch.cuda.is_available():
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    
    # Load tokenizer
    print("\nLoading model...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    )
    
    if use_4bit:
        model = prepare_model_for_kbit_training(model)
        # Avoid torch.utils.checkpoint warning in PyTorch >=2.4
        if hasattr(model, "gradient_checkpointing_enable") and getattr(model, "is_gradient_checkpointing", False):
            try:
                model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            except TypeError:
                # Older transformers versions don't support gradient_checkpointing_kwargs
                model.gradient_checkpointing_enable()
    
    # LoRA config
    peft_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # Precompute initial-policy log probabilities for each sample (old_logp).
    # This is required because rollouts were generated once from the same initial model.
    print("\nPrecomputing old_logp for all samples (initial policy)...")
    model.eval()
    with torch.no_grad():
        for g in question_groups:
            q = g["question"]
            group_profile = prompt_profile or g.get("prompt_profile")
            system_prompt = _build_system_prompt(group_profile, hotpotqa_short_prompt)
            user_prompt = _build_user_prompt(q, group_profile, hotpotqa_short_prompt)
            # Match rollout serialization: SYSTEM+USER only (no ASSISTANT suffix)
            prompt = f"SYSTEM:\n{system_prompt}\n\nUSER:\n{user_prompt}"
            for s in g["samples"]:
                old_logp, old_tok = sequence_logp_on_response(
                    model=model,
                    tokenizer=tokenizer,
                    prompt=prompt,
                    response=s["response"],
                    max_length=max_length,
                    device=device,
                )
                s["old_logp"] = float(old_logp.detach().cpu())
                s["old_tok"] = float(old_tok.detach().cpu())
    model.train()


    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    
    # Learning rate scheduler
    total_steps = (len(question_groups) * num_epochs) // (batch_size * gradient_accumulation_steps)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, total_steps))
    
    # Training history
    history = TrainingHistory()
    os.makedirs(output_dir, exist_ok=True)
    
    # Training loop
    print("\n" + "-" * 60)
    print("Starting training...")
    global_step = 0
    accumulation_counter = 0
    accumulated_loss = 0.0
    accumulated_policy_loss = 0.0
    accumulated_kl_loss = 0.0
    
    for epoch in range(num_epochs):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch + 1}/{num_epochs}")
        print(f"{'='*60}")
        
        # Shuffle questions each epoch
        epoch_questions = question_groups.copy()
        random.shuffle(epoch_questions)
        
        epoch_losses = []
        epoch_rewards = []
        epoch_correct = 0
        epoch_total = 0
        
        for batch_idx in range(0, len(epoch_questions), batch_size):
            batch_questions = epoch_questions[batch_idx:batch_idx + batch_size]
            batch_loss = 0.0
            batch_policy_loss = 0.0
            batch_kl_loss = 0.0
            batch_rewards = []
            batch_valid = 0
            
            for question_group in batch_questions:
                samples = question_group["samples"]
                rewards = [s["reward"] for s in samples]
                group_profile = prompt_profile or question_group.get("prompt_profile")
                
                # Compute group-relative advantages
                advantages = compute_group_advantages(rewards)
                
                # Track accuracy
                batch_rewards.extend(rewards)
                epoch_correct += sum(1 for r in rewards if r > 0.5)
                epoch_total += len(rewards)
                
                # Compute loss for this question's samples
                question_loss, question_policy, question_kl = compute_grpo_loss_for_question(
                    model=model,
                    tokenizer=tokenizer,
                    question=question_group["question"],
                    samples=samples,
                    advantages=advantages,
                    kl_coef=kl_coef,
                    max_length=max_length,
                    device=device,
                    clip_eps=advantage_clip,
                    prompt_profile=group_profile,
                    hotpotqa_short_prompt=hotpotqa_short_prompt,
                )
                
                if question_loss is not None:
                    batch_loss += question_loss
                    batch_policy_loss += question_policy
                    batch_kl_loss += question_kl
                    batch_valid += 1
            
            if batch_valid == 0:
                continue
            
            # Average over batch
            avg_loss = batch_loss / batch_valid
            avg_policy = batch_policy_loss / batch_valid
            avg_kl = batch_kl_loss / batch_valid
            
            # Gradient accumulation
            scaled_loss = avg_loss / gradient_accumulation_steps
            scaled_loss.backward()
            
            accumulated_loss += avg_loss.item()
            accumulated_policy_loss += avg_policy
            accumulated_kl_loss += avg_kl
            accumulation_counter += 1
            
            if accumulation_counter >= gradient_accumulation_steps:
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                # Optimizer step
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                
                global_step += 1
                
                # Record metrics
                step_loss = accumulated_loss / gradient_accumulation_steps
                step_policy = accumulated_policy_loss / gradient_accumulation_steps
                step_kl = accumulated_kl_loss / gradient_accumulation_steps
                step_reward = sum(batch_rewards) / max(len(batch_rewards), 1)
                step_acc = sum(1 for r in batch_rewards if r > 0.5) / max(len(batch_rewards), 1)
                
                epoch_losses.append(step_loss)
                epoch_rewards.extend(batch_rewards)
                
                history.add_step(
                    step=global_step,
                    loss=step_loss,
                    policy_loss=step_policy,
                    kl_loss=step_kl,
                    reward=step_reward,
                    accuracy=step_acc,
                )
                
                # Logging
                if global_step % log_interval == 0:
                    lr = scheduler.get_last_lr()[0]
                    print(f"  Step {global_step}: Loss={step_loss:.4f} "
                          f"(Policy={step_policy:.4f}, KL={step_kl:.4f}) "
                          f"Reward={step_reward:.3f} LR={lr:.2e}")
                
                # Save checkpoint
                if global_step % save_interval == 0:
                    ckpt_path = os.path.join(output_dir, f"checkpoint-{global_step}")
                    model.save_pretrained(ckpt_path)
                    print(f"  Checkpoint saved: {ckpt_path}")
                
                # Reset accumulation
                accumulated_loss = 0.0
                accumulated_policy_loss = 0.0
                accumulated_kl_loss = 0.0
                accumulation_counter = 0
        
        # Epoch summary
        epoch_avg_loss = sum(epoch_losses) / max(len(epoch_losses), 1)
        epoch_avg_reward = sum(epoch_rewards) / max(len(epoch_rewards), 1)
        epoch_accuracy = epoch_correct / max(epoch_total, 1)
        
        epoch_stats = {
            "epoch": epoch + 1,
            "avg_loss": epoch_avg_loss,
            "avg_reward": epoch_avg_reward,
            "accuracy": epoch_accuracy,
            "total_samples": epoch_total,
        }
        history.add_epoch(epoch_stats)
        
        print(f"\n--- Epoch {epoch + 1} Summary ---")
        print(f"  Avg Loss: {epoch_avg_loss:.4f}")
        print(f"  Avg Reward: {epoch_avg_reward:.3f}")
        print(f"  Accuracy: {epoch_correct}/{epoch_total} ({100*epoch_accuracy:.1f}%)")
        
        # Save epoch checkpoint
        epoch_path = os.path.join(output_dir, f"epoch-{epoch + 1}")
        model.save_pretrained(epoch_path)
        tokenizer.save_pretrained(epoch_path)
    
    # Save final model (LoRA adapters)
    print("\nSaving final model...")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Optionally merge LoRA into base model and save a full model
    if save_merged:
        merged_dir = f"{output_dir}_merged"
        print(f"\nMerging LoRA into base model and saving to: {merged_dir}")
        try:
            if not use_4bit and hasattr(model, "merge_and_unload"):
                merged = model.merge_and_unload()
            else:
                from peft import PeftModel

                base_model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    device_map="auto" if torch.cuda.is_available() else None,
                    trust_remote_code=True,
                    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
                )
                merged = PeftModel.from_pretrained(base_model, output_dir).merge_and_unload()
            merged.save_pretrained(merged_dir)
            tokenizer.save_pretrained(merged_dir)
            print(f"  Merged model saved: {merged_dir}")
        except Exception as e:
            print(f"Warning: Failed to merge and save full model: {e}")
    
    # Save history
    history.save(os.path.join(output_dir, "training_history.json"))
    
    # Generate plots
    try:
        generate_training_plots(history, output_dir)
    except Exception as e:
        print(f"Warning: Could not generate plots: {e}")
    
    print(f"\nTraining complete! Model saved to: {output_dir}")
    return history



def compute_grpo_loss_for_question(
    model,
    tokenizer,
    question: str,
    samples: List[Dict[str, Any]],
    advantages: List[float],
    kl_coef: float,
    max_length: int,
    device,
    clip_eps: float = 0.2,
    prompt_profile: Optional[str] = None,
    hotpotqa_short_prompt: bool = False,
) -> Tuple[Optional["torch.Tensor"], float, float]:
    """
    Compute a PPO/GRPO-style loss for all samples of a single question using fixed rollouts.

    This implementation assumes rollouts were generated once from the *initial* policy.
    Therefore it uses a clipped importance ratio between the current policy and the
    stored initial-policy log-probability ("old_logp") for each sample.

    Loss = -E[min(r*A, clip(r,1-ε,1+ε)*A)] + kl_coef * E[logp_old - logp_new]
    where r = exp((logp_new - logp_old) / tok_count)
    """
    system_prompt = _build_system_prompt(prompt_profile, hotpotqa_short_prompt)
    user_prompt = _build_user_prompt(question, prompt_profile, hotpotqa_short_prompt)
    # Match rollout serialization: SYSTEM+USER only (no ASSISTANT suffix)
    prompt = f"SYSTEM:\n{system_prompt}\n\nUSER:\n{user_prompt}"

    total_policy = torch.zeros((), device=device)
    total_kl = torch.zeros((), device=device)
    valid_count = 0

    for sample, advantage in zip(samples, advantages):
        if "old_logp" not in sample:
            # If old_logp is missing, we cannot do off-policy correction safely.
            continue

        try:
            new_logp, new_tok = sequence_logp_on_response(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                response=sample["response"],
                max_length=max_length,
                device=device,
            )
        except Exception as e:
            print(f"    Warning: Loss computation failed (forward): {e}")
            continue

        # Stored old_logp is a Python float; convert to tensor on the right device/dtype.
        old_logp = torch.tensor(sample["old_logp"], device=device, dtype=new_logp.dtype)
        old_tok = sample.get("old_tok", float(new_tok.detach().cpu()))
        tok_count = torch.tensor(old_tok, device=device, dtype=new_logp.dtype).clamp_min(1.0)

        # Normalize by token count to avoid exploding ratios for long responses.
        log_ratio = (new_logp - old_logp) / tok_count
        ratio = torch.exp(log_ratio.clamp(-10, 10))

        A = torch.tensor(advantage, device=device, dtype=new_logp.dtype)

        unclipped = ratio * A
        clipped = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * A
        policy_obj = torch.min(unclipped, clipped)
        policy_loss = -policy_obj  # we minimize loss

        # KL proxy against fixed initial policy (sampled from old):
        # E[logp_old - logp_new], positive when drifting away from old.
        kl_proxy = -log_ratio

        total_policy = total_policy + policy_loss
        total_kl = total_kl + kl_proxy
        valid_count += 1

    if valid_count == 0:
        return None, 0.0, 0.0

    total_loss = (total_policy + kl_coef * total_kl) / valid_count
    avg_policy = (total_policy / valid_count).detach().cpu().item()
    avg_kl = (total_kl / valid_count).detach().cpu().item()
    return total_loss, avg_policy, avg_kl


def generate_training_plots(history: TrainingHistory, output_dir: str):
    """Generate training visualization plots."""
    try:
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use('Agg')
    except ImportError:
        print("matplotlib not installed, skipping plots")
        return
    
    # Loss curve
    if history.steps and history.losses:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(history.steps, history.losses, 'b-', alpha=0.7, label='Total Loss')
        if history.policy_losses:
            ax.plot(history.steps, history.policy_losses, 'g--', alpha=0.7, label='Policy Loss')
        ax.set_xlabel('Step')
        ax.set_ylabel('Loss')
        ax.set_title('GRPO Training Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'loss_curve.png'), dpi=150)
        plt.close()
        print(f"  Saved: loss_curve.png")
    
    # Reward curve
    if history.steps and history.rewards:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(history.steps, history.rewards, 'b-', alpha=0.7)
        ax.set_xlabel('Step')
        ax.set_ylabel('Reward')
        ax.set_title('Training Reward')
        ax.set_ylim(-0.1, 1.1)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'reward_curve.png'), dpi=150)
        plt.close()
        print(f"  Saved: reward_curve.png")


def main():
    parser = argparse.ArgumentParser(description="Improved GRPO Training")
    subparsers = parser.add_subparsers(dest="command")
    
    # Train command
    train = subparsers.add_parser("train", help="Train GRPO model")
    train.add_argument("--data", required=True, help="GRPO samples path")
    train.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    train.add_argument("--output", required=True, help="Output directory")
    train.add_argument("--lr", type=float, default=1e-5)
    train.add_argument("--epochs", type=int, default=3)
    train.add_argument("--batch_size", type=int, default=4)
    train.add_argument("--grad_accum", type=int, default=4)
    train.add_argument("--max_length", type=int, default=2048)
    train.add_argument("--kl_coef", type=float, default=0.1)
    train.add_argument("--lora_r", type=int, default=16)
    train.add_argument("--no_4bit", action="store_true")
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--prompt_profile", choices=["aime", "hotpotqa", "advancedif"], default=None,
                       help="Override prompt profile (default: infer from data)")
    train.add_argument("--save_merged", action="store_true",
                       help="Merge LoRA into base model and save a full merged model")
    train.add_argument("--hotpotqa_short_prompt", action="store_true",
                       help="Use short HotpotQA prompt (A/B test)")
    
    args = parser.parse_args()
    
    if args.command == "train":
        run_grpo_training(
            data_path=args.data,
            model_name=args.model,
            output_dir=args.output,
            learning_rate=args.lr,
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum,
            max_length=args.max_length,
            kl_coef=args.kl_coef,
            lora_r=args.lora_r,
            use_4bit=not args.no_4bit,
            seed=args.seed,
            prompt_profile=args.prompt_profile,
            save_merged=args.save_merged,
            hotpotqa_short_prompt=args.hotpotqa_short_prompt,
        )
    else:
        parser.print_help()
        print("\nExample:")
        print("  python grpo_improved.py train --data ./grpo_samples.jsonl --model Qwen/Qwen2.5-7B-Instruct --output ./model")


if __name__ == "__main__":
    main()
