import chatbot_lib
import time

def run_demo():
    print("--- Modular Chatbot Demo ---")
    print("Initializing Brain (Training PyTorch Model)...")
    
    # Test cases
    prompts = [
        "Hello buddy!",
        "My name is Alex",
        "Hi again",
        "What is 5 plus 10?",
        "Tell me about Python",
        "What is the weather like?",
        "Do you need to sleep?",
        "Goodbye!"
    ]
    
    for p in prompts:
        print(f"\nUser: {p}")
        response = chatbot_lib.ask(p)
        time.sleep(0.5) # Simulate thinking
        print(f"ChatBot: {response}")

if __name__ == "__main__":
    run_demo()
