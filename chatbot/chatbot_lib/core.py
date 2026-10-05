"""Core ChatBot Module supporting Generative AI, Hybrid Mode, and ESP32 Deployment.

Modes:
- 'hybrid' (default): Uses the generative neural network (LNResGRUModel) for general
  conversation and synthesis, while combining with rule-based safety suppression and exact math.
- 'generative': Pure autoregressive neural network generation for all prompts.
- 'v0.5': Fallback intent-classification and template-matching mode.
"""

import os
import random
import re
from typing import Dict, Optional, Union

from .brain import Brain
from .math_engine import MathEngine
from .knowledge import KnowledgeBase


class ChatBot:
    def __init__(
        self,
        name: str = "Buddy",
        mode: str = "hybrid",
        model_path: Optional[str] = None,
        tokenizer_path: Optional[str] = None,
        temperature: float = 0.7,
        top_k: int = 40,
        top_p: float = 0.9,
        repetition_penalty: float = 1.15
    ):
        self.name = name
        self.mode = mode.lower()
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.repetition_penalty = repetition_penalty

        # v0.5 Rule and Intent Engines
        self.brain = Brain()
        self.math = MathEngine()
        self.knowledge = KnowledgeBase()
        self.memory: Dict[str, str] = {}

        # Generative Model Components
        self.generative_model = None
        self.tokenizer = None
        self.generator = None
        self.has_generative = False

        # Resolve model paths
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        if model_path is None:
            model_path = os.path.join(base_dir, "data", "generative_model.pt")
        if tokenizer_path is None:
            tokenizer_path = os.path.join(base_dir, "data", "tokenizer_768.json")

        self.model_path = model_path
        self.tokenizer_path = tokenizer_path

        self._init_generative_components()

        # Fallback responses for v0.5 mode or when generative model is unavailable
        self.responses = {
            "greeting": [
                "Hello! I'm your friend, how can I help today?",
                "Hi there!",
                "Hey! Good to see you."
            ],
            "goodbye": [
                "Goodbye! Take care.",
                "See you later!",
                "Bye! Have a great day."
            ],
            "status": [
                "I'm doing great, thanks for asking!",
                "I'm functioning at 100% capacity and feeling friendly!"
            ],
            "identity": [
                f"I am {self.name}, a tiny generative AI chatbot running on an ESP32 microcontroller."
            ],
            "unknown": [
                "I'm not sure I understand, but I'm listening!",
                "Could you rephrase that? I want to make sure I get it right.",
                "That's a new one for me! Tell me more."
            ]
        }

    def _init_generative_components(self):
        """Loads the generative model and tokenizer if checkpoint files exist."""
        if os.path.exists(self.model_path) and os.path.exists(self.tokenizer_path):
            try:
                from .tokenizer import MicroTokenizer
                from .model import LNResGRUModel
                from .generator import TextGenerator

                self.tokenizer = MicroTokenizer.load(self.tokenizer_path)
                self.generative_model, _ = LNResGRUModel.load_checkpoint(self.model_path)
                self.generator = TextGenerator(self.generative_model, self.tokenizer)
                self.has_generative = True
            except Exception as e:
                self.has_generative = False
                # Silently retain fallback mode
        else:
            self.has_generative = False

    def set_mode(self, mode: str):
        """Switches the operating mode ('generative', 'hybrid', or 'v0.5')."""
        valid_modes = {"generative", "hybrid", "v0.5"}
        m = mode.lower()
        if m not in valid_modes:
            raise ValueError(f"Invalid mode '{mode}'. Choose from {valid_modes}")
        self.mode = m

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 64,
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        repetition_penalty: Optional[float] = None
    ) -> str:
        """Directly invokes the generative neural network without rule intercepts."""
        if not self.has_generative or self.generator is None:
            return "Generative model is not loaded. Train a model checkpoint first."

        temp = self.temperature if temperature is None else temperature
        tk = self.top_k if top_k is None else top_k
        tp = self.top_p if top_p is None else top_p
        rp = self.repetition_penalty if repetition_penalty is None else repetition_penalty

        return self.generator.generate(
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temp,
            top_k=tk,
            top_p=tp,
            repetition_penalty=rp
        )

    def ask(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        repetition_penalty: Optional[float] = None
    ) -> str:
        """Processes user input according to the active chatbot mode."""
        if prompt is None or not str(prompt).strip():
            return "You didn't say anything!"

        prompt = str(prompt)
        prompt_lower = prompt.lower()

        # Pure generative mode
        if self.mode == "generative" and self.has_generative:
            return self.generate(
                prompt,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty
            )

        # Check for user introducing their name (memory persistence)
        if "my name is" in prompt_lower:
            name = prompt.split("is")[-1].strip().strip(".!").capitalize()
            self.memory["user_name"] = name
            return f"Nice to meet you, {name}! I'll remember that."

        # Brain Semantic Frame (Intent & Polarity Analysis)
        frame = self.brain.predict_frame(prompt)
        intent = frame["intent"]
        polarity = frame["polarity"]

        # Negation handling: suppress action and acknowledge request
        if polarity == "negated":
            if intent == "math":
                return "Understood, I won't calculate that."
            elif intent == "identity":
                return "Understood, I will keep my identity to myself."
            elif intent == "friendship":
                return "Understood, I will give you some space."
            elif intent == "greeting":
                return "Alright, I will stay quiet."
            elif intent == "knowledge":
                return "Got it, I won't share information about that."
            else:
                return "Understood, I won't do that."

        # Personalized greeting if user name is known
        if intent == "greeting" and "user_name" in self.memory:
            return f"Hello {self.memory['user_name']}! Good to see you again."

        # Secondary check for math: solve if arithmetic calculation is present
        has_numbers = len(re.findall(r"\d+", prompt)) >= 2
        if intent == "math" and has_numbers:
            return self.math.solve(prompt)

        # In hybrid mode: use the generative model for open conversational queries
        if self.mode == "hybrid" and self.has_generative:
            resp = self.generate(
                prompt,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty
            )
            if resp and resp.strip():
                return resp

        # Fallback to v0.5 rule-based / template system
        if intent == "knowledge":
            return self.knowledge.get_info(prompt)

        if intent == "friendship":
            return self.knowledge.get_friend_reply()

        if intent in self.responses:
            return random.choice(self.responses[intent])

        return random.choice(self.responses["unknown"])

    def get_diagnostics(self) -> Dict:
        """Returns runtime diagnostics for both neural and rule components."""
        info = {
            "mode": self.mode,
            "has_generative": self.has_generative,
            "name": self.name,
            "memory_keys": list(self.memory.keys()),
        }
        if self.has_generative and self.generative_model is not None:
            info["generative_diagnostics"] = self.generative_model.get_diagnostics()
        if self.brain is not None:
            info["brain_diagnostics"] = self.brain.get_diagnostics()
        return info


# Global instance for easy use
_default_bot = ChatBot()

def ask(prompt: str) -> str:
    return _default_bot.ask(prompt)
