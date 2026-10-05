import os
import unittest
import torch

from chatbot_lib.model import LNResGRUModel
from chatbot_lib.tokenizer import MicroTokenizer
from chatbot_lib.generator import (
    TextGenerator,
    apply_repetition_penalty,
    top_k_top_p_filtering,
    generate_response
)

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TOKENIZER_PATH = os.path.join(BASE_DIR, "data", "tokenizer_768.json")


class TestTextGenerator(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(42)
        cls.vocab_size = 768
        cls.tokenizer = MicroTokenizer.load(TOKENIZER_PATH)
        cls.model = LNResGRUModel(
            vocab_size=cls.vocab_size,
            hidden_size=160,
            num_layers=2
        )
        cls.generator = TextGenerator(cls.model, cls.tokenizer)

    def test_repetition_penalty_logic(self):
        """Verify repetition penalty correctly dampens positive and amplifies negative logits."""
        logits = torch.tensor([[5.0, -4.0, 2.0]])
        # Penalize token 0 and token 1
        penalized = apply_repetition_penalty(logits.clone(), [0, 1], penalty=2.0)

        # Token 0 was 5.0 -> 5.0 / 2.0 = 2.5
        self.assertAlmostEqual(penalized[0, 0].item(), 2.5, places=4)
        # Token 1 was -4.0 -> -4.0 * 2.0 = -8.0
        self.assertAlmostEqual(penalized[0, 1].item(), -8.0, places=4)
        # Token 2 was untouched -> 2.0
        self.assertAlmostEqual(penalized[0, 2].item(), 2.0, places=4)

    def test_top_k_filtering(self):
        """Verify top_k filtering retains exactly k largest values and sets rest to -inf."""
        logits = torch.tensor([[1.0, 5.0, 3.0, 2.0, 4.0]])
        filtered = top_k_top_p_filtering(logits.clone(), top_k=2)

        # Top 2 are indices 1 (5.0) and 4 (4.0)
        self.assertEqual(filtered[0, 1].item(), 5.0)
        self.assertEqual(filtered[0, 4].item(), 4.0)
        self.assertEqual(filtered[0, 0].item(), -float("inf"))
        self.assertEqual(filtered[0, 2].item(), -float("inf"))
        self.assertEqual(filtered[0, 3].item(), -float("inf"))

    def test_top_p_filtering(self):
        """Verify top_p nucleus filtering isolates dominant probabilities."""
        logits = torch.tensor([[10.0, 0.1, -1.0, -5.0]])
        filtered = top_k_top_p_filtering(logits.clone(), top_p=0.8)

        # Index 0 has >99% probability, others should be masked
        self.assertEqual(filtered[0, 0].item(), 10.0)
        self.assertEqual(filtered[0, 2].item(), -float("inf"))
        self.assertEqual(filtered[0, 3].item(), -float("inf"))

    def test_greedy_generation_deterministic(self):
        """Verify greedy decoding (temperature=0) produces deterministic token sequences."""
        prompt = "Hello"
        out1 = self.generator.generate(prompt, max_new_tokens=15, temperature=0.0)
        out2 = self.generator.generate(prompt, max_new_tokens=15, temperature=0.0)
        self.assertEqual(out1, out2)

    def test_max_new_tokens_constraint(self):
        """Verify generation halts at max_new_tokens if no stop token is encountered."""
        tokens = self.generator.generate(
            "test prompt",
            max_new_tokens=8,
            stop_token_ids=[],  # No stop token
            return_tokens=True
        )
        self.assertEqual(len(tokens), 8)

    def test_stop_tokens_early_exit(self):
        """Verify early termination upon encountering stop tokens."""
        # Force the stop token to be generated early by supplying a known stop token
        tokens = self.generator.generate(
            "test prompt",
            max_new_tokens=50,
            stop_token_ids=[self.tokenizer.eos_id],
            return_tokens=True
        )
        # Sequence should not exceed max_new_tokens and stop token should not be in tokens
        self.assertLessEqual(len(tokens), 50)
        self.assertNotIn(self.tokenizer.eos_id, tokens)

    def test_streaming_callback(self):
        """Verify token callback receives chunks as they are generated."""
        chunks = []
        def on_chunk(text):
            chunks.append(text)

        self.generator.generate(
            "hello",
            max_new_tokens=5,
            callback=on_chunk
        )
        self.assertGreaterEqual(len(chunks), 1)

    def test_repetition_penalty_1d_tensor(self):
        """Verify repetition penalty succeeds on 1D tensor without IndexError."""
        logits = torch.tensor([5.0, -4.0, 2.0])
        penalized = apply_repetition_penalty(logits, [0, 1], penalty=2.0)
        self.assertAlmostEqual(penalized[0].item(), 2.5, places=4)
        self.assertAlmostEqual(penalized[1].item(), -8.0, places=4)
        self.assertAlmostEqual(penalized[2].item(), 2.0, places=4)

    def test_repetition_penalty_immutability(self):
        """Verify repetition penalty does not mutate caller's input tensor."""
        orig = torch.tensor([[5.0, -4.0, 2.0]])
        orig_copy = orig.clone()
        _ = apply_repetition_penalty(orig, [0, 1], penalty=2.0)
        self.assertTrue(torch.equal(orig, orig_copy))

    def test_top_k_top_p_immutability(self):
        """Verify filtering does not mutate caller's input tensor."""
        orig = torch.tensor([[1.0, 5.0, 3.0, 2.0, 4.0]])
        orig_copy = orig.clone()
        _ = top_k_top_p_filtering(orig, top_k=2)
        self.assertTrue(torch.equal(orig, orig_copy))

    def test_generate_empty_and_tensor_prompt(self):
        """Verify generator handles empty string, empty list, and Tensor prompt gracefully."""
        # Empty string prompt
        out_empty_str = self.generator.generate("", max_new_tokens=5)
        self.assertIsInstance(out_empty_str, str)

        # Empty list prompt
        out_empty_list = self.generator.generate([], max_new_tokens=5)
        self.assertEqual(out_empty_list, "")

        # Tensor prompt
        prompt_tensor = torch.tensor([self.tokenizer.bos_id, 352, self.tokenizer.bot_id])
        out_tensor = self.generator.generate(prompt_tensor, max_new_tokens=5)
        self.assertIsInstance(out_tensor, str)


if __name__ == "__main__":
    unittest.main()
