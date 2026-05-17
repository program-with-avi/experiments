import random

class KnowledgeBase:
    def __init__(self):
        self.facts = {
            "python": "Python is a high-level, interpreted programming language known for its readability.",
            "earth": "Earth is the third planet from the Sun and the only astronomical object known to harbor life.",
            "ai": "Artificial Intelligence is the simulation of human intelligence processes by machines, especially computer systems.",
            "capital of france": "The capital of France is Paris.",
            "sky": "The sky is blue because of Rayleigh scattering of sunlight by the atmosphere.",
            "hungry": "I don't eat food, but I'm always hungry for more data and great conversations!",
            "sleep": "I don't need sleep, but I do appreciate a good reboot every now and then.",
            "love": "I think love is a beautiful human emotion. As an AI, I 'love' helping people!",
            "age": "I'm as old as the code that created me, which makes me eternally young.",
            "time": "Time is relative, but for me, it's just a sequence of data points. How are you spending yours?",
            "friends": "Friends are people who support and care for each other. I'm happy to be your digital friend!",
            "weather": "I don't have windows, but I can tell you that a sunny disposition always helps!",
            "smart": "I know a little bit about everything, but there's always more to learn from you!",
            "feel": "",
            "feelings": "Feelings is the thing what do you think, how do you think, what you feel"
        }
        self.friend_responses = [
            "I'm here for you! What's on your mind?",
            "That's interesting! Tell me more.",
            "I'm glad we're talking. You're a great friend.",
            "I might not have a heart, but I definitely value our chats!",
            "Life is full of wonders, isn't it?"
        ]

    def get_info(self, prompt):
        prompt_lower = prompt.lower()
        for key, fact in self.facts.items():
            if key in prompt_lower:
                return fact
        
        if "news" in prompt_lower or "current" in prompt_lower or "happened" in prompt_lower:
            return "I don't have real-time access to news right now, but I can tell you that the world is always changing! You should check a live news source for the very latest."
        
        return "I'm not exactly sure about that, but it sounds like something worth looking into! My common sense tells me it's probably quite fascinating."

    def get_friend_reply(self):
        return random.choice(self.friend_responses)
