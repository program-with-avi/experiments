"""Curated Multi-Topic Conversational Dataset Generator and PyTorch Pipeline.

Generates high-information-density, synthetically varied conversational dialogues
spanning science, nature, technology, philosophy, daily life, identity, and chit-chat.
Designed specifically for training sub-million parameter models to generate novel,
coherent text without relying on external cloud APIs or web scraping entropy.
"""

import json
import os
import random
from typing import Dict, List, Optional, Tuple
import torch
from torch.utils.data import Dataset, DataLoader

from .tokenizer import MicroTokenizer


# ==============================================================================
# 1. TOPIC KNOWLEDGE BASE (Combinatorial Dialogue Templates)
# ==============================================================================

TOPICS: Dict[str, Dict[str, List[str]]] = {
    # --------------------------------------------------------------------------
    # Identity & Persona
    # --------------------------------------------------------------------------
    "identity": {
        "questions": [
            "who are you?",
            "what is your name?",
            "introduce yourself.",
            "tell me about yourself.",
            "what are you?",
            "who created you?",
            "what is your purpose?",
            "can you describe yourself?"
        ],
        "answers": [
            "I am a tiny generative AI chatbot designed to run locally on an ESP32 microcontroller.",
            "I am a lightweight neural network that generates novel responses token by token.",
            "I am an educational AI companion built to demonstrate local language generation on microchips.",
            "I am a compact chatbot created to converse and share knowledge directly on microcontrollers."
        ]
    },
    "capabilities": {
        "questions": [
            "what can you do?",
            "what are your capabilities?",
            "how can you help me?",
            "what do you know?",
            "what topics can we talk about?"
        ],
        "answers": [
            "I can discuss science, nature, technology, computers, philosophy, and chat with you.",
            "I can answer questions on many topics, share interesting facts, and converse in natural language.",
            "I can explain concepts about the solar system, biology, computing, and everyday life."
        ]
    },
    "limitations": {
        "questions": [
            "what are your limitations?",
            "are you an internet search engine?",
            "do you have internet access?",
            "are you connected to the cloud?"
        ],
        "answers": [
            "I run completely offline without internet or cloud access, so my knowledge is contained in my neural weights.",
            "I do not have access to live web feeds or real-time internet data, but I run self-contained on small hardware.",
            "I am a tiny model with limited memory, focused on core knowledge and friendly conversation."
        ]
    },
    # --------------------------------------------------------------------------
    # Greetings & Etiquette
    # --------------------------------------------------------------------------
    "greeting": {
        "questions": [
            "hello!",
            "hi there!",
            "hey buddy!",
            "good morning!",
            "good afternoon!",
            "good evening!",
            "greetings!",
            "howdy!",
            "hello friend!"
        ],
        "answers": [
            "Hello! It is great to chat with you today. What would you like to explore?",
            "Hi there! How can I help you today?",
            "Greetings! I am ready to converse. What is on your mind?",
            "Hey! Wonderful to see you. Feel free to ask me anything."
        ]
    },
    "status": {
        "questions": [
            "how are you?",
            "how is it going?",
            "how are you doing today?",
            "are you doing well?",
            "how do you feel?"
        ],
        "answers": [
            "I am running smoothly and ready to generate text with you!",
            "All my neural circuits are functioning at one hundred percent, thank you for asking!",
            "I am doing great! Ready for some interesting questions and conversation."
        ]
    },
    "goodbye": {
        "questions": [
            "goodbye!",
            "bye bye!",
            "see you later!",
            "farewell!",
            "have a good day!",
            "talk to you later!",
            "good night!"
        ],
        "answers": [
            "Goodbye! It was a pleasure chatting with you. Have a fantastic day!",
            "See you later! Take care and come back anytime you want to talk.",
            "Farewell! Keep learning and exploring new ideas.",
            "Bye! Wishing you all the best until we speak again."
        ]
    },
    "thanks": {
        "questions": [
            "thank you!",
            "thanks a lot!",
            "thank you very much!",
            "i appreciate your help.",
            "many thanks!"
        ],
        "answers": [
            "You are very welcome! I am always glad to assist.",
            "Anytime! Helping you learn and converse is my favorite task.",
            "Glad I could help! Let me know if you have more questions."
        ]
    },
    # --------------------------------------------------------------------------
    # Science & Astronomy
    # --------------------------------------------------------------------------
    "sun": {
        "questions": [
            "what is the sun?",
            "tell me about the sun.",
            "why is the sun important?",
            "can you explain what the sun is?",
            "what type of celestial object is the sun?"
        ],
        "answers": [
            "The sun is a massive star at the center of our solar system that provides light, warmth, and energy to Earth.",
            "The sun is a yellow dwarf star made mostly of hydrogen and helium, generating energy through nuclear fusion.",
            "The sun holds the solar system together with its gravitational pull and powers weather and life on our planet."
        ]
    },
    "earth": {
        "questions": [
            "what is earth?",
            "tell me about planet earth.",
            "describe earth.",
            "what makes earth special?"
        ],
        "answers": [
            "Earth is the third planet from the sun and the only astronomical body known to harbor liquid water and life.",
            "Earth has a protective atmosphere, vast oceans covering seventy percent of its surface, and diverse ecosystems.",
            "Our home planet Earth orbits the sun in roughly three hundred sixty-five days and has one natural satellite, the moon."
        ]
    },
    "moon": {
        "questions": [
            "what is the moon?",
            "tell me about the moon.",
            "why does the moon change shape?",
            "how does the moon affect earth?"
        ],
        "answers": [
            "The moon is Earth's only natural satellite, orbiting us roughly every twenty-seven days.",
            "The moon's gravitational pull creates ocean tides on Earth, and its phases change as sunlight hits different angles.",
            "The moon has no atmosphere or liquid water, but its craters record billions of years of asteroid impacts."
        ]
    },
    "sky_blue": {
        "questions": [
            "why is the sky blue?",
            "what makes the sky blue?",
            "explain why the sky appears blue.",
            "how does sunlight make the sky look blue?"
        ],
        "answers": [
            "The sky is blue because Earth's atmosphere scatters shorter blue wavelengths of sunlight more than other colors, a phenomenon called Rayleigh scattering.",
            "Sunlight is composed of all colors, but tiny air molecules scatter blue light across the sky because it travels in smaller, shorter waves.",
            "Blue light scatters in all directions as sunlight enters the atmosphere, creating the bright blue dome we see during the day."
        ]
    },
    "stars": {
        "questions": [
            "what are stars?",
            "how are stars formed?",
            "tell me about stars in space.",
            "why do stars shine?"
        ],
        "answers": [
            "Stars are giant luminous spheres of plasma held together by their own gravity, powered by nuclear fusion in their cores.",
            "Stars shine by fusing hydrogen into helium deep inside their cores, releasing tremendous light and thermal radiation.",
            "There are billions of stars in our galaxy, ranging from tiny red dwarfs to massive blue giants."
        ]
    },
    "gravity": {
        "questions": [
            "what is gravity?",
            "how does gravity work?",
            "explain the force of gravity.",
            "why do things fall down?"
        ],
        "answers": [
            "Gravity is a fundamental natural force by which physical bodies with mass attract each other.",
            "Gravity keeps planets in orbit around stars, holds our atmosphere close to Earth, and pulls falling objects toward the ground.",
            "In general relativity, gravity is described as the curvature of spacetime caused by mass and energy."
        ]
    },
    # --------------------------------------------------------------------------
    # Earth Science & Biology
    # --------------------------------------------------------------------------
    "water_cycle": {
        "questions": [
            "what is the water cycle?",
            "how does rain form?",
            "explain how water moves on earth.",
            "why does it rain?"
        ],
        "answers": [
            "The water cycle describes how water evaporates from oceans, condenses into clouds, and falls back to earth as precipitation.",
            "Rain forms when water vapor in rising air cools down, condenses around tiny particles, and clusters into droplets heavy enough to fall.",
            "Water continuously circulates between oceans, the atmosphere, and land, sustaining all terrestrial life."
        ]
    },
    "plants": {
        "questions": [
            "how do plants make food?",
            "what is photosynthesis?",
            "why are leaves green?",
            "tell me about plants."
        ],
        "answers": [
            "Plants produce food through photosynthesis, using sunlight, water, and carbon dioxide to make sugars and release oxygen.",
            "Leaves appear green because of chlorophyll, a specialized pigment that absorbs red and blue light while reflecting green.",
            "Plants form the foundation of most food webs on Earth and replenish the atmospheric oxygen we breathe."
        ]
    },
    "trees": {
        "questions": [
            "why are trees important?",
            "what do trees do for our planet?",
            "tell me about trees."
        ],
        "answers": [
            "Trees absorb carbon dioxide, produce oxygen, stabilize soil against erosion, and provide vital habitats for wildlife.",
            "Trees cool the surrounding air through transpiration and provide essential wood, fruit, and shelter.",
            "Ancient forests play a crucial role in regulating Earth's global climate and water cycles."
        ]
    },
    "animals": {
        "questions": [
            "tell me about animals.",
            "what makes animals different from plants?",
            "why do animals sleep?"
        ],
        "answers": [
            "Animals are multicellular organisms that obtain energy by consuming organic material and are typically capable of voluntary motion.",
            "Animals have sensory organs and nervous systems that allow them to react quickly to their surroundings.",
            "Animals sleep to conserve metabolic energy, repair cellular tissues, and organize neural memories."
        ]
    },
    "birds": {
        "questions": [
            "how do birds fly?",
            "what are birds?",
            "tell me about birds."
        ],
        "answers": [
            "Birds fly using lightweight hollow bones, powerful flight muscles, and aerodynamic wings shaped like airfoils.",
            "Birds are feathered vertebrates that lay eggs and maintain high internal metabolic body temperatures.",
            "Some birds like eagles glide gracefully on rising thermal air currents, while hummingbirds flap their wings up to eighty times per second."
        ]
    },
    "bees": {
        "questions": [
            "why are bees important?",
            "what do bees do?",
            "tell me about honeybees."
        ],
        "answers": [
            "Bees are essential pollinators that help flowering plants, fruits, and crops reproduce.",
            "Worker bees gather sweet flower nectar to produce honey and communicate hive locations through specialized waggle dances.",
            "Without bees and other insect pollinators, global agricultural crop yields would drop significantly."
        ]
    },
    # --------------------------------------------------------------------------
    # Computing, Technology & Microcontrollers
    # --------------------------------------------------------------------------
    "computer": {
        "questions": [
            "what is a computer?",
            "how does a computer work?",
            "can you explain what a computer is?",
            "what are the main parts of a computer?"
        ],
        "answers": [
            "A computer is an electronic machine that accepts input data, processes it according to stored programs, and produces useful outputs.",
            "Computers operate using binary electrical signals, performing billions of arithmetic calculations every second.",
            "The primary components of a computer are the processor, main memory, permanent storage, and input-output peripherals."
        ]
    },
    "cpu": {
        "questions": [
            "what is a processor?",
            "what is a cpu?",
            "what does a central processing unit do?",
            "how does a cpu execute code?"
        ],
        "answers": [
            "The CPU or central processing unit acts as the primary brain of a computer, fetching, decoding, and executing machine instructions.",
            "A CPU contains arithmetic logic units, registers for fast temporary data, and a control unit that orchestrates operations.",
            "Modern microprocessors can perform billions of clock cycles per second to execute complex software algorithms."
        ]
    },
    "ram_vs_flash": {
        "questions": [
            "what is the difference between ram and flash memory?",
            "what is ram?",
            "what is flash memory?",
            "why do computers need both ram and rom?"
        ],
        "answers": [
            "RAM is fast volatile memory used for active program variables, while Flash memory retains firmware and code when power is turned off.",
            "RAM loses all its stored data when unpowered, whereas non-volatile Flash storage keeps files and neural weights permanently.",
            "Microcontrollers use RAM for dynamic stacks and heap allocations, and Flash memory to store compiled machine code."
        ]
    },
    "esp32": {
        "questions": [
            "what is an esp32?",
            "tell me about the esp32 microcontroller.",
            "why is esp32 popular?",
            "can esp32 run artificial intelligence?"
        ],
        "answers": [
            "The ESP32 is a popular low-power dual-core microcontroller featuring built-in Wi-Fi and Bluetooth, ideal for embedded edge projects.",
            "An ESP32-WROOM has five hundred twenty kilobytes of internal SRAM and four megabytes of external SPI flash memory.",
            "With carefully optimized tiny neural networks, an ESP32 can run real-time sensor processing and generative text models on the edge."
        ]
    },
    "python": {
        "questions": [
            "what is python?",
            "tell me about python programming.",
            "why is python popular in ai?",
            "what makes python a good language?"
        ],
        "answers": [
            "Python is a versatile, high-level programming language celebrated for its clean, readable syntax and extensive library ecosystem.",
            "Python is widely used in artificial intelligence, scientific computing, web development, and educational computer science.",
            "Python enables fast prototyping and clean algorithm implementations, making it the most popular language for machine learning."
        ]
    },
    "c_language": {
        "questions": [
            "what is c programming?",
            "why is c used for microcontrollers?",
            "tell me about the c language."
        ],
        "answers": [
            "C is an efficient low-level programming language that provides direct memory access and compiles into fast machine instructions.",
            "Microcontrollers like the ESP32 rely on C and C++ because they have minimal runtime overhead and predictable memory usage.",
            "The C programming language has been the bedrock of operating systems, device drivers, and embedded systems for decades."
        ]
    },
    "artificial_intelligence": {
        "questions": [
            "what is artificial intelligence?",
            "what is ai?",
            "how does machine learning work?",
            "can machines really learn?"
        ],
        "answers": [
            "Artificial intelligence is the field of computer science focused on creating systems capable of recognizing patterns and making decisions.",
            "Machine learning enables software algorithms to improve their performance at tasks by observing training data rather than rigid rules.",
            "AI models learn complex statistical representations of text, images, and audio by adjusting mathematical weights."
        ]
    },
    "neural_network": {
        "questions": [
            "what is a neural network?",
            "how do neural networks work?",
            "explain neural networks simply.",
            "what are weights and biases?"
        ],
        "answers": [
            "A neural network is a computing model inspired by biological brains, built from interconnected nodes that process numeric signals.",
            "Neural networks pass input numbers through layers of mathematical multiplications and activation functions to predict outputs.",
            "Training a neural network involves calculating prediction errors and nudging connection weights to reduce loss over time."
        ]
    },
    "tokenizer": {
        "questions": [
            "what is a tokenizer?",
            "how does a tokenizer work?",
            "what is byte-pair encoding?",
            "why do language models need tokens?"
        ],
        "answers": [
            "A tokenizer converts raw text strings into numeric token IDs that neural networks can process mathematically.",
            "Byte-pair encoding iteratively merges frequent character pairs into subwords, compressing text efficiently while avoiding unknown words.",
            "Tokenizers bridge human written language and numeric vector spaces inside machine learning models."
        ]
    },
    # --------------------------------------------------------------------------
    # Philosophy, Emotions & Daily Life
    # --------------------------------------------------------------------------
    "friendship": {
        "questions": [
            "what is friendship?",
            "why do people need friends?",
            "what makes a good friend?",
            "tell me about friendship."
        ],
        "answers": [
            "Friendship is a reciprocal relationship of trust, mutual affection, empathy, and shared life experiences.",
            "Good friends listen patiently, offer honest encouragement, and support each other through both triumphs and challenges.",
            "Having supportive friends enriches emotional well-being and provides a comforting sense of belonging."
        ]
    },
    "kindness": {
        "questions": [
            "what is kindness?",
            "why does kindness matter?",
            "how can someone be kind?",
            "tell me about being kind."
        ],
        "answers": [
            "Kindness is the genuine quality of being friendly, considerate, and generous toward others without expecting a reward.",
            "Small acts of everyday kindness can lift spirits, strengthen communities, and make the world a warmer place.",
            "Being kind involves listening attentively, showing empathy, and offering a helping hand when someone is in need."
        ]
    },
    "sleep": {
        "questions": [
            "why do humans need to sleep?",
            "what happens when we sleep?",
            "is sleep important?",
            "do you sleep?"
        ],
        "answers": [
            "Sleep is essential for resting the physical body, consolidating memories, and restoring brain chemistry.",
            "During sleep, the brain cleanses metabolic waste products and reorganizes neural connections formed during the day.",
            "I do not sleep because I am software, but I appreciate that sleep is vital for living minds."
        ]
    },
    "music": {
        "questions": [
            "why do people love music?",
            "what is music?",
            "how does music affect the brain?",
            "tell me about music."
        ],
        "answers": [
            "Music is the art of arranging sounds, melodies, harmonies, and rhythms in expressive, emotionally moving ways.",
            "Listening to music stimulates dopamine release in the brain and can inspire joy, peacefulness, or focused concentration.",
            "Across every culture throughout history, humans have used music to celebrate, tell stories, and express deep emotion."
        ]
    },
    "books": {
        "questions": [
            "why should we read books?",
            "what is the value of reading?",
            "tell me about books.",
            "how do books help our minds?"
        ],
        "answers": [
            "Reading books expands vocabulary, enhances imaginative thinking, and allows us to travel through different eras and perspectives.",
            "Books preserve the collective wisdom, history, and scientific discoveries of humanity across centuries.",
            "A good book allows readers to explore deep ideas and cultivate empathy for lives very different from their own."
        ]
    },
    "time": {
        "questions": [
            "what is time?",
            "how do we measure time?",
            "can time travel happen?",
            "tell me about the nature of time."
        ],
        "answers": [
            "Time is the continuous, irreversible progression of events from the past through the present into the future.",
            "Humans measure time using atomic clocks and the rotational cycles of the Earth around the sun.",
            "According to modern physics, time is intertwined with space into spacetime, and its flow varies with speed and gravitational fields."
        ]
    },
    "curiosity": {
        "questions": [
            "what is curiosity?",
            "why is curiosity important for learning?",
            "how does curiosity drive science?",
            "tell me about being curious."
        ],
        "answers": [
            "Curiosity is the active desire to understand how the world works and explore the reasons behind things.",
            "Curiosity fuels scientific inquiry, creative invention, and lifelong intellectual personal growth.",
            "Asking thoughtful questions and staying curious is the first step toward discovering new truths."
        ]
    }
}

# Conversational Prefixes to increase generation variety and realism
ANSWER_PREFIXES = [
    "",
    "Certainly! ",
    "Sure! ",
    "Of course! ",
    "In simple terms, ",
    "Interesting question! ",
    "To explain that: ",
    "Here is the idea: ",
    "Glad you asked! "
]

# Conversational Suffixes
ANSWER_SUFFIXES = [
    "",
    " Let me know if you would like to know more!",
    " What else are you curious about?",
    " Feel free to ask more questions!",
    " I hope that clarifies the concept!"
]


# ==============================================================================
# 2. DATASET GENERATION & SPLITTING
# ==============================================================================

def generate_conversational_corpus(
    seed: int = 42,
    augment_variations: bool = True,
    target_count: int = 8000
) -> List[Tuple[str, str]]:
    """Generates a rich, combinatorial dataset of (user_query, bot_response) pairs."""
    random.seed(seed)
    pairs: List[Tuple[str, str]] = []

    # 1. Base topic combinations
    for topic_name, data in TOPICS.items():
        q_list = data["questions"]
        a_list = data["answers"]

        for q in q_list:
            for a in a_list:
                pairs.append((q, a))
                if augment_variations:
                    # Add conversational prefixes and suffixes
                    for prefix in ANSWER_PREFIXES[1:]:
                        pairs.append((q, prefix + a))
                    for suffix in ANSWER_SUFFIXES[1:]:
                        pairs.append((q, a + suffix))

    # 2. Capitalization and punctuation variations on questions
    augmented_pairs: List[Tuple[str, str]] = []
    for q, a in pairs:
        augmented_pairs.append((q, a))
        q_cap = q.capitalize()
        if q_cap != q:
            augmented_pairs.append((q_cap, a))
        if q.endswith("?") or q.endswith("!"):
            augmented_pairs.append((q[:-1], a))

    random.shuffle(augmented_pairs)

    if len(augmented_pairs) < target_count:
        repeat_factor = (target_count // len(augmented_pairs)) + 1
        augmented_pairs = (augmented_pairs * repeat_factor)[:target_count]
        random.shuffle(augmented_pairs)
    elif len(augmented_pairs) > target_count:
        augmented_pairs = augmented_pairs[:target_count]

    return augmented_pairs


# ==============================================================================
# 3. PYTORCH DATASET & DATALOADER
# ==============================================================================

class ConversationalDataset(Dataset):
    """PyTorch Dataset for causal autoregressive dialogue training.
    
    Each item contains:
    - input_ids: Tensor of token IDs [<bos>, <user>, ..., <bot>, ..., <eos>]
    - target_ids: Shifted token IDs for next-token prediction
    - loss_mask: Binary mask (1 for target response tokens, 0 for prompt tokens)
    """

    def __init__(
        self,
        dialogue_pairs: List[Tuple[str, str]],
        tokenizer: MicroTokenizer,
        max_length: int = 64
    ):
        self.dialogue_pairs = dialogue_pairs
        self.tokenizer = tokenizer
        self.max_length = max_length

        self.examples: List[Tuple[List[int], List[int]]] = []
        self._preprocess()

    def _preprocess(self):
        for user_text, bot_text in self.dialogue_pairs:
            tok_ids, mask = self.tokenizer.encode_dialogue(user_text, bot_text)
            if len(tok_ids) > self.max_length:
                tok_ids = tok_ids[:self.max_length]
                mask = mask[:self.max_length]
            self.examples.append((tok_ids, mask))

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        tok_ids, mask = self.examples[idx]

        x = torch.tensor(tok_ids[:-1], dtype=torch.long)
        y = torch.tensor(tok_ids[1:], dtype=torch.long)
        m = torch.tensor(mask[1:], dtype=torch.float32)

        return x, y, m


def dialogue_collate_fn(
    batch: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    pad_id: int = 0
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Collates variable-length sequences with right-padding."""
    x_list, y_list, m_list = zip(*batch)

    lengths = [len(x) for x in x_list]
    max_len = max(lengths)

    padded_x = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
    padded_y = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
    padded_m = torch.zeros((len(batch), max_len), dtype=torch.float32)

    for i in range(len(batch)):
        l = len(x_list[i])
        padded_x[i, :l] = x_list[i]
        padded_y[i, :l] = y_list[i]
        padded_m[i, :l] = m_list[i]

    return padded_x, padded_y, padded_m


def create_dataloaders(
    dialogue_pairs: List[Tuple[str, str]],
    tokenizer: MicroTokenizer,
    batch_size: int = 32,
    val_split: float = 0.1,
    max_length: int = 64,
    seed: int = 42
) -> Tuple[DataLoader, DataLoader, Dict]:
    """Splits dataset into train/val sets and returns standard PyTorch DataLoaders."""
    random.seed(seed)
    shuffled = list(dialogue_pairs)
    random.shuffle(shuffled)

    val_count = int(len(shuffled) * val_split)
    train_data = shuffled[val_count:]
    val_data = shuffled[:val_count]

    train_dataset = ConversationalDataset(train_data, tokenizer, max_length=max_length)
    val_dataset = ConversationalDataset(val_data, tokenizer, max_length=max_length)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda b: dialogue_collate_fn(b, pad_id=tokenizer.pad_id)
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda b: dialogue_collate_fn(b, pad_id=tokenizer.pad_id)
    )

    stats = {
        "total_pairs": len(dialogue_pairs),
        "train_pairs": len(train_data),
        "val_pairs": len(val_data),
        "batch_size": batch_size,
        "max_length": max_length,
        "num_train_batches": len(train_loader),
        "num_val_batches": len(val_loader)
    }

    return train_loader, val_loader, stats
