"""Autoregressive Text Generator for Microcontroller Models.

Features:
- Stateful O(1) GRU autoregressive token generation.
- Sampling strategies:
  * Greedy argmax (temperature = 0)
  * Temperature scaling
  * Top-K filtering
  * Top-P (Nucleus) cumulative probability filtering
  * Repetition penalty (prevents degenerate loops)
- Stop token detection (e.g. <eos>)
- Streaming callback support for real-time token output.
"""

from typing import Callable, List, Optional, Union
import torch
import torch.nn.functional as F

from .model import LNResGRUModel
from .tokenizer import MicroTokenizer


def apply_repetition_penalty(
    logits: torch.Tensor,
    generated_tokens: List[int],
    penalty: float = 1.15
) -> torch.Tensor:
    """Applies standard repetition penalty to logits for previously generated tokens."""
    if penalty == 1.0 or not generated_tokens:
        return logits

    logits = logits.clone()
    unique_tokens = set(generated_tokens)
    for token_id in unique_tokens:
        if token_id < logits.size(-1):
            if logits.dim() == 1:
                val = logits[token_id].item()
                if val > 0:
                    logits[token_id] = val / penalty
                else:
                    logits[token_id] = val * penalty
            else:
                for b in range(logits.size(0)):
                    val = logits[b, token_id].item()
                    if val > 0:
                        logits[b, token_id] = val / penalty
                    else:
                        logits[b, token_id] = val * penalty
    return logits


def top_k_top_p_filtering(
    logits: torch.Tensor,
    top_k: int = 0,
    top_p: float = 1.0,
    filter_value: float = -float("Inf")
) -> torch.Tensor:
    """Filters a distribution of logits using Top-k and/or Top-p (nucleus) filtering."""
    logits = logits.clone()

    # Top-K filtering
    if top_k > 0:
        top_k = min(max(top_k, 1), logits.size(-1))
        # Keep only top_k tokens, set rest to filter_value
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    # Top-P (Nucleus) filtering
    if 0.0 < top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # Scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(
            dim=-1, index=sorted_indices, src=sorted_indices_to_remove
        )
        logits[indices_to_remove] = filter_value

    return logits


class TextGenerator:
    """Configurable Autoregressive Generator for LNResGRUModel."""

    def __init__(
        self,
        model: LNResGRUModel,
        tokenizer: MicroTokenizer,
        device: Optional[torch.device] = None
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device or next(model.parameters()).device
        self.model.to(self.device)
        self.model.eval()

    def generate(
        self,
        prompt: Union[str, List[int], torch.Tensor],
        max_new_tokens: int = 64,
        temperature: float = 0.7,
        top_k: int = 40,
        top_p: float = 0.9,
        repetition_penalty: float = 1.15,
        stop_token_ids: Optional[List[int]] = None,
        callback: Optional[Callable[[str], None]] = None,
        return_tokens: bool = False
    ) -> Union[str, List[int]]:
        """Generates a text completion or dialogue response.
        
        Args:
            prompt: Text prompt string, list of input token IDs, or token Tensor.
            max_new_tokens: Maximum number of tokens to generate.
            temperature: Sampling temperature (0.0 = greedy argmax).
            top_k: Top-k filtering threshold (0 = disabled).
            top_p: Nucleus sampling threshold (1.0 = disabled).
            repetition_penalty: Penalty factor for repeating tokens.
            stop_token_ids: List of token IDs that terminate generation early.
            callback: Optional callback invoked with each newly generated string chunk.
            return_tokens: If True, returns list of generated token IDs instead of str.
            
        Returns:
            Generated text string or list of token IDs.
        """
        if stop_token_ids is None:
            stop_token_ids = [self.tokenizer.eos_id]

        # 1. Format prompt into token sequence: <bos> <user> {prompt} <bot>
        if isinstance(prompt, str):
            clean_prompt = prompt.strip()
            if clean_prompt:
                user_tokens = self.tokenizer.encode(clean_prompt)
                prompt_tokens = [
                    self.tokenizer.bos_id,
                    self.tokenizer.user_id
                ] + user_tokens + [self.tokenizer.bot_id]
            else:
                prompt_tokens = [
                    self.tokenizer.bos_id,
                    self.tokenizer.user_id,
                    self.tokenizer.bot_id
                ]
        elif isinstance(prompt, torch.Tensor):
            prompt_tokens = prompt.view(-1).cpu().tolist()
        else:
            prompt_tokens = list(prompt)

        if not prompt_tokens:
            return [] if return_tokens else ""

        input_tensor = torch.tensor(
            prompt_tokens, dtype=torch.long, device=self.device
        ).unsqueeze(0)  # (1, seq_len)

        # 2. Prefill phase: feed prompt through model to get initial hidden states
        with torch.no_grad():
            logits, hidden_states = self.model.forward(
                input_tensor, return_hidden=True
            )
            # Last token logits predicts the first response token
            next_token_logits = logits[:, -1, :].clone()

        generated_tokens: List[int] = []

        # 3. Autoregressive token-by-token generation loop
        with torch.no_grad():
            for _ in range(max_new_tokens):
                # Apply repetition penalty
                penalized_logits = apply_repetition_penalty(
                    next_token_logits, generated_tokens, penalty=repetition_penalty
                )

                # Selection: greedy vs sampling
                if temperature <= 1e-5:
                    next_token = torch.argmax(penalized_logits, dim=-1).item()
                else:
                    scaled_logits = penalized_logits / max(temperature, 1e-5)
                    filtered_logits = top_k_top_p_filtering(
                        scaled_logits, top_k=top_k, top_p=top_p
                    )
                    probs = F.softmax(filtered_logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1).item()

                if next_token in stop_token_ids:
                    break

                generated_tokens.append(next_token)

                if callback:
                    chunk_text = self.tokenizer.decode([next_token], skip_special_tokens=True)
                    callback(chunk_text)

                # Step the model forward with the single new token
                step_input = torch.tensor([[next_token]], dtype=torch.long, device=self.device)
                next_token_logits, hidden_states = self.model.forward_step(
                    step_input, hidden_states=hidden_states
                )

        if return_tokens:
            return generated_tokens

        response_text = self.tokenizer.decode(
            generated_tokens, skip_special_tokens=True
        ).strip()
        return response_text


def generate_response(
    model: LNResGRUModel,
    tokenizer: MicroTokenizer,
    prompt: str,
    max_new_tokens: int = 64,
    temperature: float = 0.7,
    top_k: int = 40,
    top_p: float = 0.9,
    repetition_penalty: float = 1.15
) -> str:
    """Convenience functional interface for generating a response."""
    gen = TextGenerator(model, tokenizer)
    return gen.generate(
        prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        repetition_penalty=repetition_penalty
    )
