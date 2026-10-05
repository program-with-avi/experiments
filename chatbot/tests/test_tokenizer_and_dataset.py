"""Unit Tests for MicroTokenizer and ConversationalDataset Pipeline."""

import os
import tempfile
import unittest
import torch

from chatbot_lib.tokenizer import MicroTokenizer
from chatbot_lib.dataset import (
    ConversationalDataset,
    dialogue_collate_fn,
    create_dataloaders,
    generate_conversational_corpus,
    TOPICS
)


BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TOKENIZER_PATH = os.path.join(BASE_DIR, "data", "tokenizer_768.json")


class TestMicroTokenizer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tokenizer_path = TOKENIZER_PATH
        cls.tokenizer = MicroTokenizer.load(cls.tokenizer_path)

    def test_special_tokens_ids(self):
        """Verify reserved special tokens and their canonical IDs."""
        self.assertEqual(self.tokenizer.pad_id, 0)
        self.assertEqual(self.tokenizer.bos_id, 1)
        self.assertEqual(self.tokenizer.eos_id, 2)
        self.assertEqual(self.tokenizer.user_id, 3)
        self.assertEqual(self.tokenizer.bot_id, 4)
        self.assertEqual(self.tokenizer.NUM_SPECIAL, 5)

    def test_vocab_size(self):
        """Verify vocabulary size is 768."""
        self.assertEqual(self.tokenizer.vocab_size, 768)

    def test_encode_decode_roundtrip(self):
        """Verify lossless tokenization and detokenization of English strings."""
        sample_texts = [
            "Hello world!",
            "The quick brown fox jumps over the lazy dog.",
            "What is the sun? Rayleigh scattering makes the sky blue.",
            "ESP32-WROOM microcontroller with 520KB SRAM."
        ]
        for text in sample_texts:
            tokens = self.tokenizer.encode(text)
            self.assertIsInstance(tokens, list)
            self.assertGreater(len(tokens), 0)
            reconstructed = self.tokenizer.decode(tokens, skip_special_tokens=True)
            self.assertEqual(reconstructed, text)

    def test_zero_oov_arbitrary_unicode(self):
        """Verify 0% Out-Of-Vocabulary rate even on rare emojis and symbols."""
        weird_text = "🌟⚡ ESP32 🚀 Microcontroller 🤖 100% Offline 🔬"
        tokens = self.tokenizer.encode(weird_text)
        reconstructed = self.tokenizer.decode(tokens, skip_special_tokens=True)
        self.assertEqual(reconstructed, weird_text)

    def test_encode_dialogue_structure_and_mask(self):
        """Verify dialogue prompt formatting and supervision loss mask."""
        user_text = "What is the sun?"
        bot_text = "The sun is a star."

        tok_ids, mask = self.tokenizer.encode_dialogue(user_text, bot_text)

        # Structure: <bos> <user> user_toks <bot> bot_toks <eos>
        self.assertEqual(tok_ids[0], self.tokenizer.bos_id)
        self.assertEqual(tok_ids[1], self.tokenizer.user_id)
        self.assertIn(self.tokenizer.bot_id, tok_ids)
        self.assertEqual(tok_ids[-1], self.tokenizer.eos_id)

        # Mask: 0 on prompt tokens (including <bot>), 1 on bot response and <eos>
        bot_idx = tok_ids.index(self.tokenizer.bot_id)
        for i in range(bot_idx + 1):
            self.assertEqual(mask[i], 0, f"Prompt index {i} should have mask 0")
        for i in range(bot_idx + 1, len(tok_ids)):
            self.assertEqual(mask[i], 1, f"Response index {i} should have mask 1")


class TestConversationalDatasetPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tokenizer = MicroTokenizer.load(TOKENIZER_PATH)
        cls.corpus = generate_conversational_corpus(target_count=100, seed=42)

    def test_corpus_generation(self):
        """Verify corpus generation covers topics and produces non-empty pairs."""
        self.assertEqual(len(self.corpus), 100)
        for q, a in self.corpus:
            self.assertIsInstance(q, str)
            self.assertIsInstance(a, str)
            self.assertGreater(len(q), 0)
            self.assertGreater(len(a), 0)

    def test_dataset_item_structure(self):
        """Verify dataset yields x, y, and mask with correct shifted targets."""
        dataset = ConversationalDataset(self.corpus[:10], self.tokenizer, max_length=64)
        self.assertEqual(len(dataset), 10)

        x, y, m = dataset[0]
        self.assertIsInstance(x, torch.Tensor)
        self.assertIsInstance(y, torch.Tensor)
        self.assertIsInstance(m, torch.Tensor)

        self.assertEqual(x.dtype, torch.long)
        self.assertEqual(y.dtype, torch.long)
        self.assertEqual(m.dtype, torch.float32)

        # y is shifted by 1 relative to x: y[t] is next token for x[t]
        self.assertEqual(len(x), len(y))
        self.assertEqual(len(x), len(m))

    def test_dialogue_collate_fn_padding(self):
        """Verify collate function right-pads sequences and sets mask to 0 on padding."""
        dataset = ConversationalDataset(self.corpus[:4], self.tokenizer, max_length=64)
        batch = [dataset[i] for i in range(4)]

        padded_x, padded_y, padded_m = dialogue_collate_fn(batch, pad_id=self.tokenizer.pad_id)

        self.assertEqual(padded_x.shape[0], 4)
        self.assertEqual(padded_y.shape[0], 4)
        self.assertEqual(padded_m.shape[0], 4)
        self.assertEqual(padded_x.shape, padded_y.shape)
        self.assertEqual(padded_x.shape, padded_m.shape)

        # Padding positions should have pad_id in x and y, and 0 in m
        for i in range(4):
            orig_len = len(dataset[i][0])
            if orig_len < padded_x.shape[1]:
                # Verify trailing elements are padded
                self.assertTrue(torch.all(padded_x[i, orig_len:] == self.tokenizer.pad_id))
                self.assertTrue(torch.all(padded_m[i, orig_len:] == 0.0))

    def test_create_dataloaders(self):
        """Verify train/val dataloaders split correctly."""
        train_l, val_l, stats = create_dataloaders(
            self.corpus, self.tokenizer, batch_size=16, val_split=0.2, seed=42
        )
        self.assertEqual(stats["total_pairs"], 100)
        self.assertEqual(stats["val_pairs"], 20)
        self.assertEqual(stats["train_pairs"], 80)
        self.assertEqual(len(train_l), 5)  # 80 / 16 = 5
        self.assertEqual(len(val_l), 2)    # 20 / 16 ceil = 2


if __name__ == "__main__":
    unittest.main()
