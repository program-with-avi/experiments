"""Custom Byte-Level Byte-Pair Encoding (BPE) Tokenizer for Microcontrollers.

Features:
- Self-contained: Pure Python standard library, zero external C/C++ dependencies.
- Byte-level base: 256 byte tokens ensure 0% Out-Of-Vocabulary (OOV) rate.
- Min-rank BPE encoding: Fast O(L log L) subword segmentation.
- Microcontroller-ready: Produces compact JSON tables and exportable C arrays
  suitable for ESP32 flash memory.
- Conversational turn markers: <pad>, <bos>, <eos>, <user>, <bot>.
"""

import json
import os
import re
from collections import Counter
from typing import Dict, List, Optional, Tuple, Union


class MicroTokenizer:
    # Reserved Special Tokens
    PAD_TOKEN = "<pad>"    # ID 0
    BOS_TOKEN = "<bos>"    # ID 1
    EOS_TOKEN = "<eos>"    # ID 2
    USER_TOKEN = "<user>"  # ID 3
    BOT_TOKEN = "<bot>"    # ID 4

    SPECIAL_TOKENS = [PAD_TOKEN, BOS_TOKEN, EOS_TOKEN, USER_TOKEN, BOT_TOKEN]
    NUM_SPECIAL = len(SPECIAL_TOKENS)  # 5

    def __init__(self, target_vocab_size: int = 1024):
        self.target_vocab_size = target_vocab_size

        # Special token IDs
        self.pad_id = 0
        self.bos_id = 1
        self.eos_id = 2
        self.user_id = 3
        self.bot_id = 4

        # Token ID <-> Bytes mapping
        self.id_to_bytes: Dict[int, bytes] = {}
        self.bytes_to_id: Dict[bytes, int] = {}

        # Merges: dict of (id1, id2) -> new_id
        # merge_ranks: dict of (id1, id2) -> priority rank
        self.merges: Dict[Tuple[int, int], int] = {}
        self.merge_ranks: Dict[Tuple[int, int], int] = {}
        self.ordered_merges: List[Tuple[int, int, int]] = []

        self._init_base_vocab()

    def _init_base_vocab(self):
        """Initialize the 5 special tokens and 256 raw byte tokens (IDs 0..260)."""
        self.id_to_bytes = {}
        self.bytes_to_id = {}
        self.merges = {}
        self.merge_ranks = {}
        self.ordered_merges = []

        # Special tokens mapped to UTF-8 byte representations
        for i, token in enumerate(self.SPECIAL_TOKENS):
            b = token.encode("utf-8")
            self.id_to_bytes[i] = b
            self.bytes_to_id[b] = i

        # 256 individual bytes mapped to IDs 5..260
        for b_val in range(256):
            token_id = self.NUM_SPECIAL + b_val
            raw_b = bytes([b_val])
            self.id_to_bytes[token_id] = raw_b
            self.bytes_to_id[raw_b] = token_id

    @property
    def vocab_size(self) -> int:
        return len(self.id_to_bytes)

    def _pre_tokenize(self, text: str) -> List[str]:
        """Split text into chunk tokens while preserving all spaces and punctuation."""
        if not text:
            return []
        # Matches whitespace followed by non-whitespace, or trailing whitespace
        chunks = re.findall(r"\s*\S+|\s+", text)
        return chunks if chunks else [text]

    def train(self, texts: List[str], min_frequency: int = 2, verbose: bool = False) -> Dict:
        """Trains BPE merges over a corpus of texts using word-frequency optimization."""
        self._init_base_vocab()

        # Step 1: Count chunk frequencies across the corpus
        chunk_counts: Counter = Counter()
        for text in texts:
            if not text:
                continue
            chunks = self._pre_tokenize(text)
            chunk_counts.update(chunks)

        # Convert each unique chunk into a tuple of base byte token IDs
        # vocab_words: dict of tuple_of_ids -> frequency
        vocab_words: Dict[Tuple[int, ...], int] = {}
        for chunk, count in chunk_counts.items():
            raw_bytes = chunk.encode("utf-8")
            byte_ids = tuple(self.NUM_SPECIAL + b for b in raw_bytes)
            vocab_words[byte_ids] = count

        current_vocab_size = self.vocab_size
        num_merges_target = self.target_vocab_size - current_vocab_size

        if verbose:
            print(f"Starting BPE training: initial vocab={current_vocab_size}, target={self.target_vocab_size}, unique chunks={len(vocab_words)}")

        # Step 2: Iteratively find the most frequent adjacent pair across unique chunks
        for step in range(num_merges_target):
            pair_counts: Counter = Counter()

            for word_tuple, freq in vocab_words.items():
                if len(word_tuple) < 2:
                    continue
                for i in range(len(word_tuple) - 1):
                    pair = (word_tuple[i], word_tuple[i + 1])
                    pair_counts[pair] += freq

            if not pair_counts:
                break

            best_pair, best_freq = pair_counts.most_common(1)[0]
            if best_freq < min_frequency:
                if verbose:
                    print(f"Stopping early at step {step}: best pair frequency {best_freq} < min_frequency {min_frequency}")
                break

            new_token_id = len(self.id_to_bytes)
            merged_bytes = self.id_to_bytes[best_pair[0]] + self.id_to_bytes[best_pair[1]]

            self.id_to_bytes[new_token_id] = merged_bytes
            self.bytes_to_id[merged_bytes] = new_token_id
            self.merges[best_pair] = new_token_id
            self.merge_ranks[best_pair] = step
            self.ordered_merges.append((best_pair[0], best_pair[1], new_token_id))

            # Step 3: Update unique chunks by merging occurrences of best_pair
            new_vocab_words: Dict[Tuple[int, ...], int] = {}
            for word_tuple, freq in vocab_words.items():
                if len(word_tuple) < 2:
                    new_vocab_words[word_tuple] = freq
                    continue

                new_word = []
                i = 0
                while i < len(word_tuple):
                    if i < len(word_tuple) - 1 and (word_tuple[i], word_tuple[i + 1]) == best_pair:
                        new_word.append(new_token_id)
                        i += 2
                    else:
                        new_word.append(word_tuple[i])
                        i += 1
                new_vocab_words[tuple(new_word)] = freq

            vocab_words = new_vocab_words

            if verbose and (step + 1) % 100 == 0:
                print(f"Step {step + 1}/{num_merges_target}: Vocab size = {len(self.id_to_bytes)} | Best pair freq = {best_freq}")

        stats = {
            "initial_vocab": self.NUM_SPECIAL + 256,
            "final_vocab": self.vocab_size,
            "num_merges": len(self.ordered_merges)
        }
        if verbose:
            print(f"BPE training complete. Final vocab size: {self.vocab_size} ({len(self.ordered_merges)} merges)")
        return stats

    def _encode_chunk(self, chunk: str) -> List[int]:
        """Encodes a single pre-tokenized chunk using min-rank BPE merging."""
        raw_bytes = chunk.encode("utf-8")
        tokens = [self.NUM_SPECIAL + b for b in raw_bytes]

        if not self.merge_ranks or len(tokens) < 2:
            return tokens

        while len(tokens) >= 2:
            min_rank = float("inf")
            best_pair = None

            for i in range(len(tokens) - 1):
                pair = (tokens[i], tokens[i + 1])
                rank = self.merge_ranks.get(pair, float("inf"))
                if rank < min_rank:
                    min_rank = rank
                    best_pair = pair

            if min_rank == float("inf"):
                break

            new_id = self.merges[best_pair]
            new_tokens = []
            i = 0
            while i < len(tokens):
                if i < len(tokens) - 1 and (tokens[i], tokens[i + 1]) == best_pair:
                    new_tokens.append(new_id)
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens

        return tokens

    def encode(
        self,
        text: str,
        add_special_tokens: bool = False,
        user_format: bool = False,
        bot_format: bool = False,
    ) -> List[int]:
        """Encodes string into a list of token IDs."""
        if not text:
            if add_special_tokens:
                return [self.bos_id, self.eos_id]
            return []

        chunks = self._pre_tokenize(text)
        tokens: List[int] = []
        for chunk in chunks:
            tokens.extend(self._encode_chunk(chunk))

        if user_format:
            tokens = [self.user_id] + tokens
        elif bot_format:
            tokens = [self.bot_id] + tokens

        if add_special_tokens:
            tokens = [self.bos_id] + tokens + [self.eos_id]

        return tokens

    def encode_dialogue(self, user_text: str, bot_text: str) -> Tuple[List[int], List[int]]:
        """Encodes a complete conversation turn with target loss mask.
        
        Returns:
            token_ids: [<bos>, <user>, ...user_tokens..., <bot>, ...bot_tokens..., <eos>]
            loss_mask: [0, 0, ...0..., 1, ...1..., 1]
                       (0 on prompt tokens, 1 on target response tokens)
        """
        user_tokens = self.encode(user_text)
        bot_tokens = self.encode(bot_text)

        # Sequence format: <bos> <user> {user_tokens} <bot> {bot_tokens} <eos>
        prompt_part = [self.bos_id, self.user_id] + user_tokens + [self.bot_id]
        target_part = bot_tokens + [self.eos_id]

        token_ids = prompt_part + target_part
        # We only compute loss on predicting the target response (and bot turn token)
        loss_mask = [0] * len(prompt_part) + [1] * len(target_part)

        return token_ids, loss_mask

    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """Decodes token IDs back into a UTF-8 string."""
        byte_chunks = []
        for tid in token_ids:
            if tid in self.id_to_bytes:
                if skip_special_tokens and tid < self.NUM_SPECIAL:
                    continue
                byte_chunks.append(self.id_to_bytes[tid])
            else:
                byte_chunks.append(b"?")

        full_bytes = b"".join(byte_chunks)
        return full_bytes.decode("utf-8", errors="replace")

    def save(self, filepath: str):
        """Serializes tokenizer configuration, vocabulary, and merges to a JSON file."""
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        data = {
            "target_vocab_size": self.target_vocab_size,
            "vocab_size": self.vocab_size,
            "num_special": self.NUM_SPECIAL,
            "special_tokens": self.SPECIAL_TOKENS,
            "merges": self.ordered_merges
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, filepath: str) -> "MicroTokenizer":
        """Loads tokenizer from a saved JSON file."""
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        tokenizer = cls(target_vocab_size=data.get("target_vocab_size", 1024))
        tokenizer._init_base_vocab()

        for rank, (p0, p1, new_id) in enumerate(data["merges"]):
            merged_bytes = tokenizer.id_to_bytes[p0] + tokenizer.id_to_bytes[p1]
            tokenizer.id_to_bytes[new_id] = merged_bytes
            tokenizer.bytes_to_id[merged_bytes] = new_id
            tokenizer.merges[(p0, p1)] = new_id
            tokenizer.merge_ranks[(p0, p1)] = rank
            tokenizer.ordered_merges.append((p0, p1, new_id))

        return tokenizer

    def export_c_header(self, filepath: str):
        """Exports the tokenizer tables as C arrays for zero-allocation ESP32 runtime."""
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write("/* Auto-generated MicroTokenizer C header for ESP32-WROOM */\n")
            f.write("#ifndef MICRO_TOKENIZER_H\n#define MICRO_TOKENIZER_H\n\n")
            f.write("#include <stdint.h>\n#include <stddef.h>\n\n")
            f.write(f"#define TOKENIZER_VOCAB_SIZE {self.vocab_size}\n")
            f.write(f"#define TOKENIZER_NUM_MERGES {len(self.ordered_merges)}\n")
            f.write(f"#define TOKEN_PAD {self.pad_id}\n")
            f.write(f"#define TOKEN_BOS {self.bos_id}\n")
            f.write(f"#define TOKEN_EOS {self.eos_id}\n")
            f.write(f"#define TOKEN_USER {self.user_id}\n")
            f.write(f"#define TOKEN_BOT {self.bot_id}\n\n")

            f.write("typedef struct {\n    uint16_t p0;\n    uint16_t p1;\n    uint16_t new_id;\n} BPEMerge;\n\n")
            f.write("static const BPEMerge BPE_MERGES[TOKENIZER_NUM_MERGES] = {\n")
            for p0, p1, new_id in self.ordered_merges:
                f.write(f"    {{{p0}, {p1}, {new_id}}},\n")
            f.write("};\n\n")
            f.write("#endif /* MICRO_TOKENIZER_H */\n")
