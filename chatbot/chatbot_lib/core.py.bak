from .brain import Brain
from .math_engine import MathEngine
from .knowledge import KnowledgeBase
import random

class ChatBot:
    def __init__(self, name="Buddy"):
        self.name = name
        self.brain = Brain()
        self.math = MathEngine()
        self.knowledge = KnowledgeBase()
        self.memory = {} # Simple in-memory storage
        
        self.responses = {
            "greeting": ["Hello! I'm your friend, how can I help today?", "Hi there!", "Hey! Good to see you."],
            "goodbye": ["Goodbye! Take care.", "See you later!", "Bye! Have a great day."],
            "status": ["I'm doing great, thanks for asking!", "I'm functioning at 100% capacity and feeling friendly!"],
            "identity": [f"I am {self.name}, a modular chatbot experiment. I know a bit about everything and I'm here to help!"],
            "unknown": ["I'm not sure I understand, but I'm listening!", "Could you rephrase that? I want to make sure I get it right.", "That's a new one for me! tell me more."]
        }

    def ask(self, prompt):
        if not prompt or prompt.strip() == "":
            return "You didn't say anything!"

        prompt_lower = prompt.lower()
        
        # Check for name introduction
        if "my name is" in prompt_lower:
            name = prompt.split("is")[-1].strip().strip(".!").capitalize()
            self.memory["user_name"] = name
            return f"Nice to meet you, {name}! I'll remember that."

        intent = self.brain.predict(prompt)
        
        # Personalized greeting if name is known
        if intent == "greeting" and "user_name" in self.memory:
            return f"Hello {self.memory['user_name']}! Good to see you again."

        # Secondary check for math: only if there are numbers
        import re
        has_numbers = len(re.findall(r"\d+", prompt)) >= 2
        
        if intent == "math" and has_numbers:
            return self.math.solve(prompt)
        
        # If it was math but no numbers, try knowledge or greeting
        if intent == "math" and not has_numbers:
            if "python" in prompt.lower() or "france" in prompt.lower():
                intent = "knowledge"
            else:
                intent = "unknown"
        
        if intent == "knowledge":
            return self.knowledge.get_info(prompt)
        
        if intent == "friendship":
            return self.knowledge.get_friend_reply()
        
        if intent in self.responses:
            return random.choice(self.responses[intent])
        
        return random.choice(self.responses["unknown"])

# Global instance for easy use
_default_bot = ChatBot()

def ask(prompt):
    return _default_bot.ask(prompt)
