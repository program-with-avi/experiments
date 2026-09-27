import chatbot_lib
bot = chatbot_lib.ChatBot(name="Ben")
tests = [
    "who made you",
    "who created you",
    "who built you",
    "who is behind you",
    "who programmed you",
    "who was responsible for making you",
    "tell me who made this chatbot",
    "what company created you"
]
for text in tests:
    print(bot.ask(text))