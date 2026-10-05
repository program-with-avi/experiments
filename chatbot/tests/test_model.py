"""Unit Tests for Residual LayerNorm GRU Architecture (LNResGRUModel)."""

import os
import tempfile
import unittest
import torch
import torch.nn as nn

from chatbot_lib.model import LNResGRUBlock, LNResGRUModel


class TestLNResGRUModel(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.vocab_size = 768
        self.hidden_size = 160
        self.num_layers = 2
        self.model = LNResGRUModel(
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=0.1,
            tie_weights=True
        )

    def test_parameter_count_budget(self):
        """Verify model parameter count satisfies the 350K - 450K ESP32 constraint."""
        diag = self.model.get_diagnostics()
        params = diag["trainable_params"]

        # 350K <= params <= 450K
        self.assertGreaterEqual(params, 350000, f"Params too small: {params}")
        self.assertLessEqual(params, 450000, f"Params too large: {params}")
        self.assertEqual(params, 432960, "Exact parameter count mismatch for standard config")

    def test_tied_embeddings(self):
        """Verify lm_head weights are strictly tied to input token embeddings."""
        self.assertIs(
            self.model.lm_head.weight,
            self.model.tok_embeddings.weight,
            "lm_head weight and tok_embeddings weight must reference the same tensor"
        )
        self.assertIsNone(self.model.lm_head.bias)

    def test_esp32_feasibility_diagnostics(self):
        """Verify diagnostics calculations for ESP32 flash and SRAM."""
        diag = self.model.get_diagnostics()
        self.assertEqual(diag["model_type"], "LNResGRUModel")
        self.assertTrue(diag["esp32_flash_feasible"])
        self.assertTrue(diag["esp32_sram_feasible"])
        self.assertLess(diag["sram_hidden_state_kb"], 5.0)  # ~1.25 KB
        self.assertLess(diag["fp32_size_kb"], 2000.0)      # ~1.69 MB

    def test_forward_tensor_shapes(self):
        """Verify standard forward pass handles 2D input batches."""
        batch_size = 4
        seq_len = 24
        x = torch.randint(0, self.vocab_size, (batch_size, seq_len))

        logits = self.model(x)
        self.assertEqual(logits.shape, (batch_size, seq_len, self.vocab_size))

        # Test return_hidden=True
        logits_h, hidden_states = self.model(x, return_hidden=True)
        self.assertEqual(logits_h.shape, (batch_size, seq_len, self.vocab_size))
        self.assertEqual(len(hidden_states), self.num_layers)
        for h in hidden_states:
            self.assertEqual(h.shape, (1, batch_size, self.hidden_size))

    def test_forward_1d_input(self):
        """Verify forward pass correctly handles single 1D sequence."""
        x = torch.randint(0, self.vocab_size, (18,))
        logits = self.model(x)
        self.assertEqual(logits.shape, (1, 18, self.vocab_size))

    def test_forward_step_equivalence(self):
        """Verify autoregressive forward_step aligns with full sequence forward."""
        self.model.eval()
        seq = torch.tensor([[10, 20, 30, 40]], dtype=torch.long)

        with torch.no_grad():
            full_logits, full_h = self.model(seq, return_hidden=True)
            expected_last_logits = full_logits[:, -1, :]

            # Step through one token at a time
            h = None
            for i in range(seq.size(1)):
                tok = seq[:, i:i+1]
                step_logits, h = self.model.forward_step(tok, hidden_states=h)

            torch.testing.assert_close(step_logits, expected_last_logits, rtol=1e-4, atol=1e-4)

    def test_checkpoint_save_and_load(self):
        """Verify checkpoint serialization and deserialization."""
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            temp_path = f.name

        try:
            extra = {"test_metric": 0.1234}
            self.model.save_checkpoint(temp_path, extra_meta=extra)

            loaded_model, meta = LNResGRUModel.load_checkpoint(temp_path)
            self.assertEqual(loaded_model.vocab_size, self.vocab_size)
            self.assertEqual(loaded_model.hidden_size, self.hidden_size)
            self.assertEqual(loaded_model.num_layers, self.num_layers)
            self.assertEqual(meta.get("test_metric"), 0.1234)

            # Check weights match
            for p1, p2 in zip(self.model.parameters(), loaded_model.parameters()):
                self.assertTrue(torch.equal(p1, p2))
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def test_forward_step_scalar_input(self):
        """Verify forward_step handles 0D scalar tensor without crash."""
        self.model.eval()
        scalar_tok = torch.tensor(42)
        logits, h = self.model.forward_step(scalar_tok)
        self.assertEqual(logits.shape, (1, self.vocab_size))
        self.assertEqual(len(h), self.num_layers)

    def test_forward_empty_sequence(self):
        """Verify forward handles sequence length of 0 without RNN error."""
        self.model.eval()
        empty = torch.empty((1, 0), dtype=torch.long)
        logits = self.model(empty)
        self.assertEqual(logits.shape, (1, 0, self.vocab_size))

        logits_h, h = self.model(empty, return_hidden=True)
        self.assertEqual(logits_h.shape, (1, 0, self.vocab_size))
        self.assertEqual(len(h), self.num_layers)


if __name__ == "__main__":
    unittest.main()
