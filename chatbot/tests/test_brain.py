import unittest
import sys
import os

# Add parent directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from chatbot_lib.brain import Brain
from chatbot_lib.core import ChatBot


class TestBrainV05(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.brain = Brain(model_type="embedding_bag", enable_modifiers=True, verbose=False, seed=42)
        cls.bot = ChatBot(name="Ben")

    def test_unknown_words_crash_prevention(self):
        """Verify that unknown words, empty strings, None, and symbols do not cause crashes."""
        test_cases = [
            "",
            "   ",
            "???!!!...",
            None,
            12345,
            "supercalifragilisticexpialidocious foobarbazqux",
            "xyz123 abc456 def789",
            "🚀🛸🤖",
        ]
        for case in test_cases:
            frame = self.brain.predict_frame(case)
            self.assertEqual(frame["intent"], "unknown", f"Failed on input: {case!r}")
            self.assertIn("polarity", frame)
            self.assertIn("mood", frame)

    def test_predict_backward_compatibility(self):
        """Verify Brain.predict() and predict_with_confidence() maintain exact backward compatibility."""
        tag = self.brain.predict("who made you")
        self.assertEqual(tag, "identity")

        tag, conf = self.brain.predict_with_confidence("who made you")
        self.assertEqual(tag, "identity")
        self.assertGreater(conf, 0.6)

    def test_predict_frame_structure(self):
        """Verify predict_frame returns the required 4-field semantic frame."""
        frame = self.brain.predict_frame("can you calculate 5 + 5?")
        self.assertIn("intent", frame)
        self.assertIn("polarity", frame)
        self.assertIn("mood", frame)
        self.assertIn("confidence", frame)

        self.assertEqual(frame["intent"], "math")
        self.assertEqual(frame["polarity"], "affirmative")
        self.assertEqual(frame["mood"], "question")
        self.assertGreater(frame["confidence"], 0.6)

    def test_negated_math_frame(self):
        """Verify negated math query produces math intent with negated polarity and command mood."""
        frame = self.brain.predict_frame("don't calculate 5 + 5")
        self.assertEqual(frame["intent"], "math")
        self.assertEqual(frame["polarity"], "negated")
        self.assertEqual(frame["mood"], "command")

        frame2 = self.brain.predict_frame("do not calculate 5 + 5")
        self.assertEqual(frame2["intent"], "math")
        self.assertEqual(frame2["polarity"], "negated")
        self.assertEqual(frame2["mood"], "command")

        frame3 = self.brain.predict_frame("never calculate 5 + 5")
        self.assertEqual(frame3["intent"], "math")
        self.assertEqual(frame3["polarity"], "negated")
        self.assertEqual(frame3["mood"], "command")

    def test_chatbot_behavioral_suppression(self):
        """Verify ChatBot executes math normally for affirmative prompts and suppresses for negated."""
        # Affirmative: should solve
        resp_aff = self.bot.ask("calculate 5 + 5")
        self.assertIn("10", resp_aff)

        # Negated: should acknowledge refusal without calculating
        resp_neg = self.bot.ask("don't calculate 5 + 5")
        self.assertNotIn("10", resp_neg)
        self.assertIn("won't calculate", resp_neg)

        # Other negated intents
        resp_id = self.bot.ask("don't tell me your name")
        self.assertIn("identity", resp_id.lower())

        resp_friend = self.bot.ask("never talk to me again")
        self.assertIn("space", resp_friend.lower())

    def test_diagnostics_parameter_count(self):
        """Verify v0.5 diagnostics and parameter count constraint for ESP32 feasibility."""
        d = self.brain.get_diagnostics()
        self.assertEqual(d["model_type"], "embedding_bag")
        self.assertTrue(d["enable_modifiers"])
        self.assertLess(d["trainable_params"], 3000)
        self.assertLess(d["fp32_size_kb"], 12.0)


if __name__ == "__main__":
    unittest.main()
