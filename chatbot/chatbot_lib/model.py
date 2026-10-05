"""Residual LayerNorm GRU Generative Language Model for ESP32 Microcontrollers.

Architecture:
- Embedding Layer: Tied input and output embedding matrix.
- Residual LayerNorm GRU Blocks: Stack of GRU layers with residual addition
  and Layer Normalization per layer.
- Final LayerNorm + Linear LM Head (tied with input embeddings).
- Compact parameter count: ~350K - 450K parameters, designed specifically
  to fit within ESP32-WROOM flash (4MB) and execute within internal SRAM (520KB).
"""

import os
from typing import Callable, Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F


class LNResGRUBlock(nn.Module):
    """Single Residual LayerNorm GRU Block.
    
    Structure:
      x_in ──────────────────────(+)──> LayerNorm ──> Dropout ──> x_out
             │                     │
             └──> GRU(hidden, hidden)
    """

    def __init__(self, hidden_size: int, dropout: float = 0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.gru = nn.GRU(
            input_size=hidden_size,
            hidden_size=hidden_size,
            batch_first=True
        )
        self.ln = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        h: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through Residual GRU + LayerNorm.
        
        Args:
            x: Tensor of shape (batch_size, seq_len, hidden_size)
            h: Optional hidden state tensor of shape (1, batch_size, hidden_size)
            
        Returns:
            Tuple of:
              - out: Tensor of shape (batch_size, seq_len, hidden_size)
              - next_h: Next hidden state of shape (1, batch_size, hidden_size)
        """
        gru_out, next_h = self.gru(x, h)
        # Residual addition followed by LayerNorm and dropout
        out = self.ln(x + gru_out)
        out = self.dropout(out)
        return out, next_h


class LNResGRUModel(nn.Module):
    """Residual LayerNorm GRU Language Model with Tied Embeddings.
    
    Default configuration:
      vocab_size = 768
      hidden_size = 160
      num_layers = 2
      tie_weights = True
      
    Total Parameters: 432,960 (~433K), exactly within the 350K-450K budget.
    ESP32 Activation Memory: ~1.28 KB for hidden state during inference.
    """

    def __init__(
        self,
        vocab_size: int = 768,
        hidden_size: int = 160,
        num_layers: int = 2,
        dropout: float = 0.1,
        tie_weights: bool = True
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout_rate = dropout
        self.tie_weights = tie_weights

        # 1. Token Embeddings
        self.tok_embeddings = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=hidden_size,
            padding_idx=0
        )
        self.drop = nn.Dropout(dropout)

        # 2. Residual LayerNorm GRU Blocks
        self.blocks = nn.ModuleList([
            LNResGRUBlock(hidden_size, dropout=dropout)
            for _ in range(num_layers)
        ])

        # 3. Final Layer Normalization (stabilizes representations before projection)
        self.final_ln = nn.LayerNorm(hidden_size)

        # 4. Language Modeling Head (Linear projection)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

        # 5. Weight Tying
        if self.tie_weights:
            self.lm_head.weight = self.tok_embeddings.weight

        self._init_weights()

    def _init_weights(self):
        """Initialize weights with scaled normal distribution suitable for tied embeddings."""
        # Embeddings: standard deviation 0.02 avoids extreme logit scales
        nn.init.normal_(self.tok_embeddings.weight, mean=0.0, std=0.02)
        if self.tok_embeddings.padding_idx is not None:
            nn.init.constant_(self.tok_embeddings.weight[self.tok_embeddings.padding_idx], 0.0)

        # LayerNorms
        for block in self.blocks:
            nn.init.ones_(block.ln.weight)
            nn.init.zeros_(block.ln.bias)
            # GRU weights initialization
            for name, param in block.gru.named_parameters():
                if "weight_ih" in name:
                    nn.init.xavier_uniform_(param)
                elif "weight_hh" in name:
                    nn.init.orthogonal_(param)
                elif "bias" in name:
                    nn.init.zeros_(param)

        nn.init.ones_(self.final_ln.weight)
        nn.init.zeros_(self.final_ln.bias)

        if not self.tie_weights:
            nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.02)

    def init_hidden(
        self,
        batch_size: int = 1,
        device: Optional[torch.device] = None
    ) -> List[torch.Tensor]:
        """Initializes zero hidden states for all layers.
        
        Returns:
            List of tensors, each of shape (1, batch_size, hidden_size)
        """
        if device is None:
            device = next(self.parameters()).device
        return [
            torch.zeros(1, batch_size, self.hidden_size, device=device)
            for _ in range(self.num_layers)
        ]

    def forward(
        self,
        input_ids: torch.Tensor,
        hidden_states: Optional[List[torch.Tensor]] = None,
        return_hidden: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]]]:
        """Forward pass over sequence.
        
        Args:
            input_ids: Tensor of token IDs, shape (batch_size, seq_len) or (batch_size,)
            hidden_states: Optional list of hidden tensors, one per layer
            return_hidden: If True, returns (logits, new_hidden_states)
            
        Returns:
            logits of shape (batch_size, seq_len, vocab_size), or
            (logits, new_hidden_states) if return_hidden=True
        """
        if input_ids.dim() == 0:
            input_ids = input_ids.view(1, 1)
        elif input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)

        batch_size, seq_len = input_ids.size()

        if hidden_states is None:
            hidden_states = [None] * self.num_layers

        if seq_len == 0:
            empty_logits = torch.empty(batch_size, 0, self.vocab_size, device=input_ids.device)
            if return_hidden:
                h_ret = hidden_states if any(h is not None for h in hidden_states) else self.init_hidden(batch_size, input_ids.device)
                return empty_logits, h_ret
            return empty_logits

        # Token embedding lookup
        h = self.tok_embeddings(input_ids)  # (batch_size, seq_len, hidden_size)
        h = self.drop(h)

        new_hidden_states: List[torch.Tensor] = []
        for i, block in enumerate(self.blocks):
            h, next_h = block(h, hidden_states[i])
            new_hidden_states.append(next_h)

        h = self.final_ln(h)
        logits = self.lm_head(h)  # (batch_size, seq_len, vocab_size)

        if return_hidden:
            return logits, new_hidden_states
        return logits

    def forward_step(
        self,
        token_ids: torch.Tensor,
        hidden_states: Optional[List[torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Single-step forward pass optimized for autoregressive generation.
        
        Args:
            token_ids: Tensor of shape (batch_size,) or (batch_size, 1) or scalar
            hidden_states: List of hidden state tensors from previous step
            
        Returns:
            Tuple of:
              - next_token_logits: Shape (batch_size, vocab_size)
              - next_hidden_states: List of updated hidden state tensors
        """
        if token_ids.dim() == 0:
            token_ids = token_ids.view(1, 1)
        elif token_ids.dim() == 1:
            token_ids = token_ids.unsqueeze(1)
        elif token_ids.dim() == 2 and token_ids.size(1) != 1:
            token_ids = token_ids[:, -1:]

        logits, new_hidden_states = self.forward(
            token_ids,
            hidden_states=hidden_states,
            return_hidden=True
        )
        # Select last token logits: shape (batch_size, vocab_size)
        return logits[:, -1, :], new_hidden_states

    def get_diagnostics(self) -> Dict:
        """Returns architecture diagnostics and ESP32 hardware feasibility metrics."""
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.parameters())
        fp32_size_kb = (trainable_params * 4) / 1024.0
        int8_size_kb = trainable_params / 1024.0
        # Hidden state RAM required for batch_size=1 inference
        sram_hidden_kb = (self.num_layers * 1 * self.hidden_size * 4) / 1024.0

        return {
            "model_type": "LNResGRUModel",
            "vocab_size": self.vocab_size,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "dropout": self.dropout_rate,
            "tie_weights": self.tie_weights,
            "trainable_params": trainable_params,
            "total_params": total_params,
            "fp32_size_kb": round(fp32_size_kb, 2),
            "int8_size_kb": round(int8_size_kb, 2),
            "sram_hidden_state_kb": round(sram_hidden_kb, 3),
            "esp32_flash_feasible": fp32_size_kb <= 3000.0,
            "esp32_sram_feasible": sram_hidden_kb <= 50.0,
        }

    def generate(
        self,
        tokenizer,
        prompt: Union[str, List[int]],
        max_new_tokens: int = 64,
        temperature: float = 0.7,
        top_k: int = 40,
        top_p: float = 0.9,
        repetition_penalty: float = 1.15,
        callback: Optional[Callable[[str], None]] = None
    ) -> str:
        """Autoregressively generates text for a given prompt using TextGenerator."""
        from .generator import TextGenerator
        gen = TextGenerator(self, tokenizer)
        return gen.generate(
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            callback=callback
        )

    def save_checkpoint(self, filepath: str, extra_meta: Optional[Dict] = None):
        """Saves model weights and architectural configuration to a checkpoint file."""
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        checkpoint = {
            "state_dict": self.state_dict(),
            "config": {
                "vocab_size": self.vocab_size,
                "hidden_size": self.hidden_size,
                "num_layers": self.num_layers,
                "dropout": self.dropout_rate,
                "tie_weights": self.tie_weights,
            },
            "diagnostics": self.get_diagnostics(),
            "meta": extra_meta or {}
        }
        torch.save(checkpoint, filepath)

    @classmethod
    def load_checkpoint(
        cls,
        filepath: str,
        device: Optional[torch.device] = None
    ) -> Tuple["LNResGRUModel", Dict]:
        """Loads model from a saved checkpoint file."""
        if device is None:
            device = torch.device("cpu")
        try:
            checkpoint = torch.load(filepath, map_location=device, weights_only=False)
        except TypeError:
            checkpoint = torch.load(filepath, map_location=device)
        config = checkpoint["config"]
        model = cls(
            vocab_size=config["vocab_size"],
            hidden_size=config["hidden_size"],
            num_layers=config["num_layers"],
            dropout=config.get("dropout", 0.0),
            tie_weights=config.get("tie_weights", True)
        )
        model.load_state_dict(checkpoint["state_dict"])
        model.to(device)
        model.eval()
        return model, checkpoint.get("meta", {})
