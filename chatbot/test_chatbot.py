"""Demonstration script for ESP32 Generative AI Chatbot.

Showcases:
1. Residual LayerNorm GRU (LNResGRUModel) parameter metrics & ESP32 memory budget.
2. Autoregressive text generation across science, technology, philosophy, and chat.
3. Hybrid mode combining neural generation, rule safety suppression, and math engine.
"""

import time
import chatbot_lib
from chatbot_lib.core import ChatBot


def run_demo():
    print("=" * 70)
    print("🤖 ESP32 GENERATIVE AI CHATBOT DEMO (Milestone 3)")
    print("=" * 70)

    # Initialize chatbot
    bot = ChatBot(name="Buddy", mode="hybrid")
    diag = bot.get_diagnostics()

    print("\n--- System Diagnostics ---")
    print(f"Active Mode: {diag['mode']}")
    print(f"Generative Neural Network Active: {diag['has_generative']}")

    if "generative_diagnostics" in diag:
        gd = diag["generative_diagnostics"]
        print(f"Model Architecture: {gd['model_type']}")
        print(f"Vocab Size: {gd['vocab_size']}")
        print(f"Hidden Dim: {gd['hidden_size']} | Layers: {gd['num_layers']}")
        print(f"Trainable Parameters: {gd['trainable_params']:,} (~{gd['trainable_params']/1000:.1f}K)")
        print(f"Model Size (FP32 Flash): {gd['fp32_size_kb']:.2f} KB | (INT8 Flash): {gd['int8_size_kb']:.2f} KB")
        print(f"Activation Memory (SRAM): {gd['sram_hidden_state_kb']:.3f} KB")
        print(f"ESP32-WROOM Feasibility: Flash={gd['esp32_flash_feasible']}, SRAM={gd['esp32_sram_feasible']}")

    print("\n" + "=" * 70)
    print("Conversational Demonstration Across Topics")
    print("=" * 70)

    test_conversations = [
        ("Greeting & Intro", "Hello buddy!"),
        ("Memory & Identity", "My name is Alex"),
        ("Personalized Greeting", "Hi again"),
        ("Arithmetic Reasoning", "What is 5 plus 10?"),
        ("Science (Astronomy)", "What is the sun?"),
        ("Science (Atmosphere)", "Why is the sky blue?"),
        ("Microcontrollers", "What is an esp32?"),
        ("Microcontrollers (Cased)", "What is an ESP32?"),
        ("Programming Language", "Tell me about Python."),
        ("Philosophy & Life", "Do you need to sleep?"),
        ("Human Connection", "What is friendship?"),
        ("Safety Negation", "Don't calculate 20 plus 30"),
        ("Farewell", "Goodbye!")
    ]

    for category, prompt in test_conversations:
        print(f"\n[{category}]")
        print(f"User:    {prompt}")
        t0 = time.time()
        response = bot.ask(prompt)
        dt = time.time() - t0
        print(f"ChatBot: {response} ({dt*1000:.1f}ms)")

    print("\n" + "=" * 70)
    print("Demo completed successfully.")
    print("=" * 70)


if __name__ == "__main__":
    run_demo()
