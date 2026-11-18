"""Causal language-model policy updates for response-level PPO."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from peft import PeftModel
from torch.distributions import Categorical
from transformers import AutoModelForCausalLM, AutoTokenizer

from .generation_config import DO_SAMPLE, TEMPERATURE, TOP_P


@dataclass
class GeneratedResponse:
    text: str
    token_ids: list[int]
    prompt_token_ids: list[int]
    token_log_probs: torch.Tensor
    token_entropies: torch.Tensor


class CausalLMPPOPolicy:
    """A LoRA causal LM whose sampled completion is the defender action.

    PPO applies clipped ratios to each generated token, with one response-level
    advantage. KL uses the sampled non-negative estimator
    ``exp(log π_ref - log π) + log π - log π_ref - 1``.
    """

    def __init__(self, base_model: str, adapter_path: str | Path, device: str,
                 revision: str | None = None,
                 learning_rate: float = 1e-6, trainable: bool = True,
                 dtype: torch.dtype = torch.float32):
        # The frozen KL reference can use half precision on memory-limited
        # accelerators; both adapters are reloaded in float32 for evaluation.
        self.device = torch.device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(base_model, revision=revision)
        if self.tokenizer.eos_token_id is None:
            raise ValueError("The base tokenizer must define an EOS token for response training")
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        base = AutoModelForCausalLM.from_pretrained(
            base_model, revision=revision, torch_dtype=dtype
        )
        self.model = PeftModel.from_pretrained(base, str(adapter_path), is_trainable=trainable).to(self.device)
        self.model.eval()
        self.optimizer = torch.optim.AdamW(
            (p for p in self.model.parameters() if p.requires_grad), lr=learning_rate
        ) if trainable else None

    @torch.no_grad()
    def generate(self, prompt: str, max_new_tokens: int = 160) -> GeneratedResponse:
        if not 1 <= max_new_tokens < 2048:
            raise ValueError("max_new_tokens must be between 1 and 2047")
        self.model.eval()
        max_prompt_tokens = max(1, 2048 - max_new_tokens)
        encoded = self.tokenizer(
            prompt, return_tensors="pt", truncation=False
        ).to(self.device)
        if encoded["input_ids"].shape[1] > max_prompt_tokens:
            raise ValueError(
                f"Prompt has {encoded['input_ids'].shape[1]} tokens, exceeding the "
                f"generation input budget of {max_prompt_tokens}; shorten the input "
                "or reduce --max-new-tokens"
            )
        generated = self.model.generate(
            **encoded, max_new_tokens=max_new_tokens, do_sample=DO_SAMPLE,
            temperature=TEMPERATURE, top_p=TOP_P, pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id, return_dict_in_generate=True,
            output_scores=True,
        )
        prompt_len = encoded["input_ids"].shape[1]
        token_ids = generated.sequences[0, prompt_len:].tolist()
        logps, entropies = [], []
        for step_logits, token_id in zip(generated.scores, token_ids):
            dist = Categorical(logits=step_logits[0].float())
            token = torch.tensor(token_id, device=self.device)
            logps.append(dist.log_prob(token))
            entropies.append(dist.entropy())
        if not logps:
            raise RuntimeError("The language model returned an empty completion")
        return GeneratedResponse(
            text=self.tokenizer.decode(token_ids, skip_special_tokens=True).strip(),
            token_ids=token_ids,
            prompt_token_ids=encoded["input_ids"][0].tolist(),
            token_log_probs=torch.stack(logps).detach(),
            token_entropies=torch.stack(entropies).detach(),
        )

    def _sequence_statistics(self, prompt_token_ids: list[int], token_ids: list[int],
                             reference: "CausalLMPPOPolicy | None" = None):
        full_ids = torch.tensor([prompt_token_ids + token_ids], dtype=torch.long, device=self.device)
        attention = torch.ones_like(full_ids)
        logits = self.model(input_ids=full_ids, attention_mask=attention).logits[0].float()
        start = len(prompt_token_ids) - 1
        end = start + len(token_ids)
        selected_logits = logits[start:end]
        targets = full_ids[0, len(prompt_token_ids):]
        token_logps = torch.log_softmax(selected_logits, dim=-1).gather(
            -1, targets.unsqueeze(-1)
        ).squeeze(-1)
        ref_logps = None
        if reference is not None:
            with torch.no_grad():
                ref_ids = full_ids.to(reference.device)
                ref_logits = reference.model(input_ids=ref_ids, attention_mask=torch.ones_like(ref_ids)).logits[0].float()
                ref_selected = ref_logits[start:end]
                ref_logps = torch.log_softmax(ref_selected, dim=-1).gather(
                    -1, targets.to(reference.device).unsqueeze(-1)
                ).squeeze(-1).to(self.device)
        return token_logps, Categorical(logits=selected_logits).entropy(), ref_logps

    def ppo_update(self, responses: list[GeneratedResponse],
                   advantages: torch.Tensor,
                   reference: "CausalLMPPOPolicy", clip_epsilon: float = 0.2,
                   beta_kl: float = 0.05, entropy_coeff: float = 0.01) -> dict[str, float]:
        if len(responses) != len(advantages):
            raise ValueError("PPO responses and advantages must have equal lengths")
        # Keep dropout disabled in both rollout and likelihood evaluation so
        # old and new probabilities refer to the same policy distribution.
        self.model.eval()
        self.optimizer.zero_grad(set_to_none=True)
        batch_size = len(responses)
        policy_loss_value = 0.0
        entropy_value = 0.0
        kl_value = 0.0
        for index, response in enumerate(responses):
            token_logps, token_entropies, ref_token_logps = self._sequence_statistics(
                response.prompt_token_ids, response.token_ids, reference
            )
            old_logps = response.token_log_probs.to(self.device)
            if old_logps.numel() != token_logps.numel():
                raise ValueError("Stored and recomputed completion token counts differ")
            advantage = advantages[index].to(self.device)
            ratio = torch.exp((token_logps - old_logps).clamp(-20, 20))
            clipped = ratio.clamp(1.0 - clip_epsilon, 1.0 + clip_epsilon)
            policy_loss = -torch.minimum(ratio * advantage, clipped * advantage).mean()
            entropy = token_entropies.mean()
            log_ratio = token_logps - ref_token_logps
            stable_log_ratio = log_ratio.clamp(-20.0, 20.0)
            kl = ((-stable_log_ratio).exp() + stable_log_ratio - 1.0).mean()
            loss = policy_loss + beta_kl * kl - entropy_coeff * entropy
            (loss / batch_size).backward()
            policy_loss_value += float(policy_loss.detach()) / batch_size
            entropy_value += float(entropy.detach()) / batch_size
            kl_value += float(kl.detach()) / batch_size
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        total_loss_value = policy_loss_value + beta_kl * kl_value - entropy_coeff * entropy_value
        return {"loss": total_loss_value, "policy_loss": policy_loss_value,
                "kl": kl_value, "entropy": entropy_value}

    def save_adapter(self, path: str | Path) -> None:
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
