import torch
import torch.nn as nn
import numpy as np
import json
import os

class IntentModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(IntentModel, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.l2 = nn.Linear(hidden_size, hidden_size)
        self.l3 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        out = self.relu(out)
        out = self.l3(out)
        return out

class Brain:
    def __init__(self):
        self.intents = {
            "greeting": ["hi! How are you?", "hello, nice to meet you", "hey there", "good morning", "good evening", "hi there", "hello buddy", "wassup!", "Yo bro", "Bonjour! un cafe?", "How you doing?", "nice to see you"],
            "goodbye": ["bye", "see you", "goodnight", "Cya!", "GGs!", "goodbye", "bye bye", "later champ", "goodluck!","nice meeting you"],
            "math": ["plus","minus","times","divided by","calculate","sum","math","add","subtract","multiply", "divide","what is the result","calculate this","do some math"],
            "status": ["how are you", "how is it going", "are you okay","What's going on", "how can i help"],
            "identity": ["who are you", "what is your name", "tell me about yourself", "who created you"],
            "friendship": ["be my friend", "talk to me", "let's hang out", "friend", "best friend", "best friends forever"],
            "identity": ["who are you","what is your name","tell me about yourself","who created you","who made you", "who built you", "who is your creator"],
            "knowledge": ["tell me about", "who is", "where is", "what happened", "current events", "news", "capital of", "what is the capital", "what is python", "weather", "sleep", "hungry", "love", "time", "smart", "age", "do you sleep", "are you hungry"]
        }
        self.tags = sorted(self.intents.keys())
        self.words = []
        self.ignore_words = ['?', '!', '.', ',', 'is', 'the', 'a', 'an']
        
        # Build vocabulary
        for tag, patterns in self.intents.items():
            for pattern in patterns:
                w = self._tokenize(pattern)
                self.words.extend(w)
        
        self.words = sorted(set([w for w in self.words if w not in self.ignore_words]))
        
        self.input_size = len(self.words)
        self.output_size = len(self.tags)
        self.hidden_size = 8
        
        self.model = IntentModel(self.input_size, self.hidden_size, self.output_size)
        self._train()

    def _tokenize(self, sentence):
        import re
        sentence = sentence.lower()
        # Remove punctuation
        sentence = re.sub(r'[?!\.,]', '', sentence)
        tokens = sentence.split()
        tokens = [
            self._normalize_word(token)
            for token in tokens
            if token not in self.ignore_words
        ]

        return tokens

    def _bag_of_words(self, tokenized_sentence):
        bag = np.zeros(len(self.words), dtype=np.float32)
        for idx, w in enumerate(self.words):
            if w in tokenized_sentence:
                bag[idx] = 1.0
        return bag

    def _train(self):
        X_train = []
        y_train = []
        
        for i, tag in enumerate(self.tags):
            for pattern in self.intents[tag]:
                w = self._tokenize(pattern)
                X_train.append(self._bag_of_words(w))
                y_train.append(i)
        
        X_train = torch.tensor(np.array(X_train))
        y_train = torch.tensor(np.array(y_train), dtype=torch.long)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)
        
        for epoch in range(500):
            outputs = self.model(X_train)
            loss = criterion(outputs, y_train)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    def predict(self, sentence):
        tokenized = self._tokenize(sentence)
        X = self._bag_of_words(tokenized)
        X = torch.from_numpy(X).reshape(1, -1)
        
        output = self.model(X)
        _, predicted = torch.max(output, dim=1)
        tag = self.tags[predicted.item()]
        
        probs = torch.softmax(output, dim=1)
        prob = probs[0][predicted.item()]
        
        if prob.item() > 0.6:
            return tag
        return "unknown"
    def _normalize_word(self, word):
        word = word.lower()

        replacements = {
            "made": "make",
            "makes": "make",
            "making": "make",
            "created": "create",
            "creating": "create",
            "creator": "create",
            "calculated": "calculate",
            "calculating": "calculate",
            "calculation": "calculate",
            "multiplied": "multiply",
            "multiplying": "multiply",
            "divided": "divide",
            "dividing": "divide",
        }

        return replacements.get(word, word)
