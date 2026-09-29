import torch
import torch.nn as nn
import numpy as np
import re


class IntentModelBag(nn.Module):
    """Brain v0.3 / v0.5 Stage 1: EmbeddingBag + Linear classifier."""
    def __init__(self, vocab_size, embed_size, hidden_size, output_size):
        super(IntentModelBag, self).__init__()

        self.embedding = nn.EmbeddingBag(
            vocab_size,
            embed_size,
            mode="mean"
        )

        self.l1 = nn.Linear(embed_size, hidden_size)
        self.l2 = nn.Linear(hidden_size, hidden_size)
        self.l3 = nn.Linear(hidden_size, output_size)

        self.relu = nn.ReLU()

    def forward(self, x, offsets):
        out = self.embedding(x, offsets)
        out = self.l1(out)
        out = self.relu(out)
        out = self.l2(out)
        out = self.relu(out)
        out = self.l3(out)
        return out


class IntentModelGRU(nn.Module):
    """Brain v0.4: Embedding -> 1-layer GRU -> final hidden state -> Linear classifier."""
    def __init__(self, vocab_size, embed_size, hidden_size, output_size, pad_idx=0):
        super(IntentModelGRU, self).__init__()
        self.pad_idx = pad_idx

        self.embedding = nn.Embedding(
            vocab_size,
            embed_size,
            padding_idx=pad_idx
        )

        self.gru = nn.GRU(
            input_size=embed_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=False
        )

        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, lengths=None):
        embed = self.embedding(x)
        if lengths is not None:
            packed = nn.utils.rnn.pack_padded_sequence(
                embed, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            _, h_n = self.gru(packed)
        else:
            _, h_n = self.gru(embed)
        out = self.fc(h_n[0])
        return out


class ModifierEngine:
    """Stage 2: Deterministic Rule-Based Linguistic Modifier Engine.
    
    Identifies polarity (affirmative / negated) and mood (statement / question / command).
    Designed for zero dynamic memory allocation and microcontroller (ESP32) efficiency.
    """
    NEGATION_WORDS = {
        "not", "never", "no", "cannot", "cant", "wont", "stop", "quit", "neither"
    }

    QUESTION_STARTERS = {
        "who", "what", "where", "when", "why", "how", "which", "whose", "whom",
        "can", "could", "will", "would", "shall", "should", "may", "might",
        "is", "are", "am", "was", "were", "do", "does", "did", "have", "has", "had"
    }

    COMMAND_VERBS = {
        "calculate", "compute", "solve", "add", "subtract", "multiply", "divide",
        "tell", "give", "show", "explain", "describe", "find", "count",
        "say", "speak", "talk", "help", "be", "make", "hang", "chat",
        "stop", "quit", "shut"
    }

    def detect(self, sentence):
        if not sentence or not isinstance(sentence, str):
            return {"polarity": "affirmative", "mood": "statement"}

        text = sentence.strip().lower()
        if not text:
            return {"polarity": "affirmative", "mood": "statement"}

        has_question_mark = "?" in text

        # Normalize typographic quotes and contractions for token inspection
        normalized = Brain._expand_contractions_static(text)

        # Tokenize with regex word boundaries
        raw_tokens = re.findall(r"\b[a-z0-9']+\b", normalized)
        if not raw_tokens:
            return {"polarity": "affirmative", "mood": "statement"}

        # 1. Polarity Detection
        is_negated = any(tok in self.NEGATION_WORDS for tok in raw_tokens)
        polarity = "negated" if is_negated else "affirmative"

        # 2. Mood Detection
        first_word = raw_tokens[0]

        is_question = has_question_mark
        if not is_question and len(raw_tokens) >= 2:
            if first_word in self.QUESTION_STARTERS:
                if first_word in {"do", "did", "does"} and len(raw_tokens) >= 2 and raw_tokens[1] in ("not", "never"):
                    is_question = False
                else:
                    is_question = True
        elif not is_question and first_word in {"who", "what", "where", "when", "why", "how"}:
            is_question = True

        is_command = False
        if not is_question:
            if "please" in raw_tokens:
                is_command = True
            elif first_word in self.COMMAND_VERBS:
                is_command = True
            elif first_word in {"do", "did", "does"} and len(raw_tokens) >= 3 and raw_tokens[1] in ("not", "never") and raw_tokens[2] in self.COMMAND_VERBS:
                is_command = True
            elif first_word in ("not", "never") and len(raw_tokens) >= 2 and raw_tokens[1] in self.COMMAND_VERBS:
                is_command = True
            elif len(raw_tokens) >= 2 and first_word == "let" and raw_tokens[1] == "us":
                is_command = True
            elif len(raw_tokens) >= 2 and first_word == "i" and raw_tokens[1] in ("want", "need") and "to" in raw_tokens:
                # "I want you to calculate"
                is_command = True

        if is_question:
            mood = "question"
        elif is_command:
            mood = "command"
        else:
            mood = "statement"

        return {
            "polarity": polarity,
            "mood": mood
        }


class Brain:
    CONTRACTIONS = [
        (r"\bwhat's\b", "what is"),
        (r"\bwho's\b", "who is"),
        (r"\bwhere's\b", "where is"),
        (r"\bhow's\b", "how is"),
        (r"\bwhen's\b", "when is"),
        (r"\bwhy's\b", "why is"),
        (r"\bit's\b", "it is"),
        (r"\bthat's\b", "that is"),
        (r"\bthere's\b", "there is"),
        (r"\blet's\b", "let us"),
        (r"\bi'm\b", "i am"),
        (r"\byou're\b", "you are"),
        (r"\bwe're\b", "we are"),
        (r"\bthey're\b", "they are"),
        (r"\bcan't\b", "cannot"),
        (r"\bwon't\b", "will not"),
        (r"n't\b", " not"),
        (r"'ve\b", " have"),
        (r"'ll\b", " will"),
        (r"'d\b", " would"),
    ]

    MATH_SYMBOLS = [
        (r"\+", " plus "),
        (r"-", " minus "),
        (r"\*", " times "),
        (r"/", " divide "),
        (r"%", " percent "),
    ]

    NORMALIZATION_MAP = {
        # Identity / creation / development
        "made": "make", "makes": "make", "making": "make", "maker": "make", "makers": "make",
        "created": "create", "creates": "create", "creating": "create", "creator": "create", "creators": "create", "creation": "create",
        "built": "build", "builds": "build", "building": "build", "builder": "build", "builders": "build",
        "programmed": "program", "programs": "program", "programming": "program", "programmer": "program", "programmers": "program",
        "developed": "develop", "develops": "develop", "developing": "develop", "developer": "develop", "developers": "develop", "development": "develop",
        "designed": "design", "designs": "design", "designing": "design", "designer": "design", "designers": "design",
        "coded": "code", "codes": "code", "coding": "code", "coder": "code", "coders": "code",
        "authored": "author", "authors": "author", "authoring": "author",
        "invented": "invent", "invents": "invent", "inventing": "invent", "inventor": "invent", "inventors": "invent",
        "written": "write", "wrote": "write", "writing": "write", "writes": "write", "writer": "write",

        # Math operations
        "calculated": "calculate", "calculates": "calculate", "calculating": "calculate", "calculation": "calculate", "calculations": "calculate", "calculator": "calculate",
        "multiplied": "multiply", "multiplies": "multiply", "multiplying": "multiply", "multiplication": "multiply",
        "divided": "divide", "divides": "divide", "dividing": "divide", "division": "divide",
        "added": "add", "adds": "add", "adding": "add", "addition": "add",
        "subtracted": "subtract", "subtracts": "subtract", "subtracting": "subtract", "subtraction": "subtract",
        "computed": "compute", "computes": "compute", "computing": "compute", "computation": "compute",
        "solved": "solve", "solves": "solve", "solving": "solve", "solution": "solve",

        # Interaction / status / friendship / knowledge
        "doing": "do", "does": "do", "did": "do",
        "going": "go", "goes": "go", "went": "go",
        "helped": "help", "helps": "help", "helping": "help",
        "talked": "talk", "talks": "talk", "talking": "talk",
        "friends": "friend", "friendly": "friend", "friendship": "friend",
        "hung": "hang", "hanging": "hang", "hangs": "hang",
        "met": "meet", "meets": "meet", "meeting": "meet",
        "loved": "love", "loves": "love", "loving": "love",
        "slept": "sleep", "sleeps": "sleep", "sleeping": "sleep",
        "thanks": "thank", "thanking": "thank", "thanked": "thank",
        "wants": "want", "wanting": "want", "wanted": "want",
        "tells": "tell", "telling": "tell", "told": "tell",
        "asking": "ask", "asked": "ask", "asks": "ask",
        "feeling": "feel", "feels": "feel", "felt": "feel", "feelings": "feel",
        "called": "call", "calling": "call", "calls": "call",
        "chatting": "chat", "chatted": "chat", "chats": "chat",
        "events": "event",
        "numbers": "number",
        "greetings": "greet", "greeting": "greet", "greeted": "greet", "greets": "greet",
    }

    @staticmethod
    def _expand_contractions_static(text):
        text = text.replace("’", "'").replace("‘", "'").replace("`", "'")
        for pattern, replacement in Brain.CONTRACTIONS:
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        return text

    def __init__(self, model_type="embedding_bag", enable_modifiers=True, verbose=False, seed=42):
        self.model_type = model_type.lower()
        self.enable_modifiers = enable_modifiers
        self.confidence_threshold = 0.6
        self.last_loss = None

        self.modifier_engine = ModifierEngine() if enable_modifiers else None

        self.intents = {
            "greeting": [
                "hi",
                "hello",
                "hey there",
                "good morning",
                "good afternoon",
                "good evening",
                "hello buddy",
                "hey friend",
                "greetings",
                "wassup",
                "yo bro",
                "howdy",
                "welcome",
                "nice to see you",
                "hello there"
            ],
            "goodbye": [
                "bye",
                "see you",
                "goodnight",
                "cya",
                "goodbye",
                "bye bye",
                "later champ",
                "talk to you later",
                "see you soon",
                "have a good day",
                "farewell",
                "take care",
                "catch you later"
            ],
            "math": [
                "calculate",
                "what is the result",
                "calculate this for me",
                "can you add these numbers",
                "please subtract 10 from 20",
                "multiply 5 by 6",
                "divide 100 by 4",
                "plus",
                "minus",
                "times",
                "divided by",
                "sum of two numbers",
                "solve this math problem",
                "compute this calculation",
                "do some math"
            ],
            "status": [
                "how are you",
                "how is it going",
                "are you okay",
                "what is going on",
                "how can i help",
                "how are you doing",
                "are you doing well",
                "how are things",
                "how do you feel today",
                "is everything all right"
            ],
            "identity": [
                "who are you",
                "what is your name",
                "tell me about yourself",
                "who created you",
                "who made you",
                "who built you",
                "who is your creator",
                "who programmed you",
                "who developed you",
                "who is behind you",
                "who was responsible for making you",
                "tell me who made this chatbot",
                "what company created you",
                "who designed you",
                "who wrote your code",
                "who is your author",
                "who invented you",
                "what should i call you",
                "what are you called",
                "introduce yourself"
            ],
            "friendship": [
                "be my friend",
                "talk to me",
                "let us hang out",
                "can we be friends",
                "best friend forever",
                "i need a friend",
                "will you be my companion",
                "i want to chat with you",
                "spend time with me",
                "you are a great friend"
            ],
            "knowledge": [
                "tell me about python",
                "what is artificial intelligence",
                "what is the capital of france",
                "why is the sky blue",
                "tell me about the earth",
                "what is the weather like",
                "what is the current time",
                "tell me the news",
                "what happened today",
                "do you sleep",
                "are you hungry",
                "what do you think about love",
                "how old are you",
                "where is france located",
                "tell me an interesting fact",
                "give me some information",
                "explain what ai is"
            ]
        }

        self.tags = sorted(self.intents.keys())

        # Stage 1 Stop words: functional particles, articles, and copula variants.
        # Negation particles are excluded from Stage 1 vocabulary so they do not distort intent detection.
        self.ignore_words = {
            '?', '!', '.', ',', 'is', 'the', 'a', 'an', 'are', 'am', 'was', 'were',
            'on', 'in', 'at', 'to', 'for', 'from', 'of', 'by', 'with',
            'do', 'does', 'did', 'not', 'no', 'never', 'cannot', 'cant', 'wont', 'please'
        }

        # Build vocabulary for Stage 1
        words = []
        for tag, patterns in self.intents.items():
            for pattern in patterns:
                w = self._tokenize(pattern)
                words.extend(w)

        self.words = sorted(list(set(words)))

        self.output_size = len(self.tags)
        self.embed_size = 16
        self.hidden_size = 16 if self.model_type == "gru" else 8

        if self.model_type == "gru":
            self.pad_idx = 0
            self.word2idx = {w: i + 1 for i, w in enumerate(self.words)}
            self.input_size = len(self.words) + 1
            self.model = IntentModelGRU(
                vocab_size=self.input_size,
                embed_size=self.embed_size,
                hidden_size=self.hidden_size,
                output_size=self.output_size,
                pad_idx=self.pad_idx
            )
        else:
            self.pad_idx = None
            self.word2idx = {w: i for i, w in enumerate(self.words)}
            self.input_size = len(self.words)
            self.model = IntentModelBag(
                vocab_size=self.input_size,
                embed_size=self.embed_size,
                hidden_size=self.hidden_size,
                output_size=self.output_size
            )

        torch.manual_seed(seed)
        np.random.seed(seed)

        self._train(seed=seed, verbose=verbose)

    def _expand_contractions(self, text):
        return Brain._expand_contractions_static(text)

    def _normalize_word(self, word):
        word = word.lower()
        return self.NORMALIZATION_MAP.get(word, word)

    def _tokenize(self, sentence):
        if not sentence or not isinstance(sentence, str):
            return []

        sentence = sentence.lower()
        sentence = self._expand_contractions(sentence)

        for sym_pattern, replacement in self.MATH_SYMBOLS:
            sentence = re.sub(sym_pattern, replacement, sentence)

        sentence = re.sub(r'[^a-z0-9\s]', ' ', sentence)
        tokens = sentence.split()

        cleaned_tokens = [
            self._normalize_word(token)
            for token in tokens
            if token not in self.ignore_words
        ]

        return cleaned_tokens

    def _sentence_to_ids(self, sentence):
        tokens = self._tokenize(sentence)
        ids = []
        for token in tokens:
            idx = self.word2idx.get(token)
            if idx is not None:
                ids.append(idx)
        return ids

    def _train(self, seed=42, epochs=500, lr=0.01, verbose=False):
        torch.manual_seed(seed)
        np.random.seed(seed)

        if self.model_type == "gru":
            self._train_gru(seed=seed, epochs=epochs, lr=lr, verbose=verbose)
        else:
            self._train_bag(seed=seed, epochs=epochs, lr=lr, verbose=verbose)

    def _train_gru(self, seed=42, epochs=500, lr=0.01, verbose=False):
        sequences = []
        labels = []

        for i, tag in enumerate(self.tags):
            for pattern in self.intents[tag]:
                ids = self._sentence_to_ids(pattern)
                if ids:
                    sequences.append(ids)
                    labels.append(i)

        if not sequences:
            return

        lengths = [len(seq) for seq in sequences]
        max_len = max(lengths)

        padded_sequences = [
            seq + [self.pad_idx] * (max_len - len(seq))
            for seq in sequences
        ]

        X_train = torch.tensor(padded_sequences, dtype=torch.long)
        lengths_tensor = torch.tensor(lengths, dtype=torch.long)
        y_train = torch.tensor(labels, dtype=torch.long)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        self.model.train()
        for epoch in range(epochs):
            outputs = self.model(X_train, lengths_tensor)
            loss = criterion(outputs, y_train)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        self.model.eval()
        self.last_loss = loss.item()

        if verbose:
            self.print_diagnostics()

    def _train_bag(self, seed=42, epochs=500, lr=0.01, verbose=False):
        X_train = []
        y_train = []

        for i, tag in enumerate(self.tags):
            for pattern in self.intents[tag]:
                ids = self._sentence_to_ids(pattern)
                if ids:
                    X_train.append(ids)
                    y_train.append(i)

        if not X_train:
            return

        flat_words = []
        offsets = []
        current_offset = 0

        for sentence in X_train:
            offsets.append(current_offset)
            flat_words.extend(sentence)
            current_offset += len(sentence)

        X_train_tensor = torch.tensor(flat_words, dtype=torch.long)
        offsets_tensor = torch.tensor(offsets, dtype=torch.long)
        y_train_tensor = torch.tensor(y_train, dtype=torch.long)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        self.model.train()
        for epoch in range(epochs):
            outputs = self.model(X_train_tensor, offsets_tensor)
            loss = criterion(outputs, y_train_tensor)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        self.model.eval()
        self.last_loss = loss.item()

        if verbose:
            self.print_diagnostics()

    def predict(self, sentence):
        """Preserved backward-compatible public API. Returns primary intent string."""
        tag, _ = self.predict_with_confidence(sentence)
        return tag

    def predict_with_confidence(self, sentence):
        """Preserved public API. Returns (intent, confidence)."""
        ids = self._sentence_to_ids(sentence)

        if not ids:
            return "unknown", 0.0

        self.model.eval()
        with torch.no_grad():
            if self.model_type == "gru":
                X = torch.tensor([ids], dtype=torch.long)
                output = self.model(X)
            else:
                X = torch.tensor(ids, dtype=torch.long)
                offsets = torch.tensor([0], dtype=torch.long)
                output = self.model(X, offsets)

            probs = torch.softmax(output, dim=1)
            confidence, predicted = torch.max(probs, dim=1)

            tag = self.tags[predicted.item()]
            conf_val = confidence.item()

            if conf_val > self.confidence_threshold:
                return tag, conf_val

            return "unknown", conf_val

    def predict_frame(self, sentence):
        """Brain v0.5 Two-Stage API: returns a structured semantic frame.
        
        {
            "intent": <primary_topic>,
            "polarity": "affirmative" | "negated",
            "mood": "statement" | "question" | "command",
            "confidence": <float>
        }
        """
        intent, confidence = self.predict_with_confidence(sentence)

        if self.modifier_engine:
            modifiers = self.modifier_engine.detect(sentence)
            polarity = modifiers["polarity"]
            mood = modifiers["mood"]
        else:
            polarity = "affirmative"
            mood = "statement"

        return {
            "intent": intent,
            "polarity": polarity,
            "mood": mood,
            "confidence": confidence
        }

    def get_diagnostics(self):
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        weight_bytes = trainable_params * 4
        if self.enable_modifiers:
            version_str = "v0.5 (Two-Stage: EmbeddingBag + Rule Modifier)"
        elif self.model_type == "gru":
            version_str = "v0.4 (Sequential GRU)"
        else:
            version_str = "v0.3 (EmbeddingBag)"

        return {
            "version": version_str,
            "model_type": self.model_type,
            "enable_modifiers": self.enable_modifiers,
            "vocab_size": self.input_size,
            "embed_dim": self.embed_size,
            "hidden_size": self.hidden_size,
            "output_size": self.output_size,
            "trainable_params": trainable_params,
            "fp32_size_kb": weight_bytes / 1024.0,
            "training_loss": self.last_loss,
        }

    def print_diagnostics(self):
        d = self.get_diagnostics()
        loss_str = f"{d['training_loss']:.4f}" if d['training_loss'] is not None else "N/A"
        print("=" * 60)
        print(f"       BRAIN MODEL DIAGNOSTICS: {d['version']}")
        print("=" * 60)
        print(f"  Architecture:            {d['model_type'].upper()}")
        print(f"  Two-Stage Modifiers:     {'ENABLED' if d['enable_modifiers'] else 'DISABLED'}")
        print(f"  Vocabulary Size:         {d['vocab_size']} words")
        print(f"  Embedding Dimension:     {d['embed_dim']}")
        print(f"  Hidden Size:             {d['hidden_size']}")
        print(f"  Output Intents:          {d['output_size']}")
        print(f"  Trainable Parameters:    {d['trainable_params']}")
        print(f"  Estimated FP32 Size:     {d['fp32_size_kb']:.2f} KB")
        print(f"  Final Training Loss:     {loss_str}")
        print("=" * 60)


if __name__ == "__main__":
    brain = Brain(model_type="embedding_bag", enable_modifiers=True, verbose=True)
    sample_tests = [
        "calculate 5 + 5",
        "don't calculate 5 + 5",
        "can you calculate 5 + 5?",
        "who made you?",
        "don't tell me your name",
        "never talk to me again",
        "tell me about Python",
        "don't tell me about Python"
    ]
    print("\nSample Semantic Frame Predictions (Brain v0.5):")
    for text in sample_tests:
        frame = brain.predict_frame(text)
        print(f"  '{text}' -> intent={frame['intent']:11} | polarity={frame['polarity']:11} | mood={frame['mood']:9} (conf: {frame['confidence']:.2%})")
