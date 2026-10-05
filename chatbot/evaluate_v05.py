"""
Brain v0.5 Evaluation and 3-Way Architectural Comparison:
v0.3 (EmbeddingBag) vs v0.4 (GRU) vs v0.5 (Two-Stage Frame).

Measures:
1. Original Evaluation Suite (Seen & Unseen Test Accuracies)
2. Existing Adversarial Stress Suite (Unchanged 117 tests across 8 categories)
3. New Linguistic Frame & Modifier Evaluation (Polarity, Mood, Frame, Behavioral)
4. Model Size, Diagnostics, Parameter Counts, and ESP32 Footprints
5. Polarity False Positives / False Negatives Analysis
"""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from chatbot_lib.brain import Brain
from chatbot_lib.core import ChatBot


def is_match(pred, expected):
    if isinstance(expected, (set, list, tuple)):
        return pred in expected
    return pred == expected


def format_expected(expected):
    if isinstance(expected, (set, list, tuple)):
        return " / ".join(expected)
    return str(expected)


def run_full_evaluation():
    print("=" * 85)
    print("      BRAIN ARCHITECTURAL BENCHMARK: v0.3 vs v0.4 vs v0.5 (Two-Stage)")
    print("=" * 85)

    print("\n[1/4] Training Brain v0.3 (EmbeddingBag baseline)...")
    brain_v3 = Brain(model_type="embedding_bag", enable_modifiers=False, seed=42)
    diag_v3 = brain_v3.get_diagnostics()

    print("[2/4] Training Brain v0.4 (Sequential GRU)...")
    brain_v4 = Brain(model_type="gru", enable_modifiers=False, seed=42)
    diag_v4 = brain_v4.get_diagnostics()

    print("[3/4] Initializing Brain v0.5 (Two-Stage: EmbeddingBag + Rule Modifier)...")
    brain_v5 = Brain(model_type="embedding_bag", enable_modifiers=True, seed=42)
    diag_v5 = brain_v5.get_diagnostics()

    bot_v5 = ChatBot(mode="v0.5")

    # -------------------------------------------------------------------------
    # SUITE 1: ORIGINAL EVALUATION SUITE
    # -------------------------------------------------------------------------
    seen_examples = [
        ("hello buddy", "greeting"),
        ("good morning", "greeting"),
        ("howdy", "greeting"),
        ("bye bye", "goodbye"),
        ("see you soon", "goodbye"),
        ("farewell", "goodbye"),
        ("calculate this for me", "math"),
        ("multiply 5 by 6", "math"),
        ("divide 100 by 4", "math"),
        ("how are you doing", "status"),
        ("are you okay", "status"),
        ("how are things", "status"),
        ("who are you", "identity"),
        ("who made you", "identity"),
        ("who programmed you", "identity"),
        ("who is behind you", "identity"),
        ("be my friend", "friendship"),
        ("talk to me", "friendship"),
        ("can we be friends", "friendship"),
        ("tell me about python", "knowledge"),
        ("what is artificial intelligence", "knowledge"),
        ("why is the sky blue", "knowledge"),
    ]

    unseen_examples = [
        ("who created you?", "identity"),
        ("who built you?", "identity"),
        ("who developed you?", "identity"),
        ("who was responsible for making you", "identity"),
        ("tell me who made this chatbot", "identity"),
        ("what company created you", "identity"),
        ("who designed you?", "identity"),
        ("who wrote your code?", "identity"),
        ("who is your author?", "identity"),
        ("who invented you?", "identity"),
        ("what should i call you?", "identity"),
        ("hey there friend", "greeting"),
        ("good afternoon to you", "greeting"),
        ("howdy there", "greeting"),
        ("welcome to our chat", "greeting"),
        ("see ya later", "goodbye"),
        ("have a great day ahead", "goodbye"),
        ("catch you later buddy", "goodbye"),
        ("talk with you later", "goodbye"),
        ("please calculate 25 plus 17", "math"),
        ("what is the result of 10 times 5", "math"),
        ("can you divide 50 by 2", "math"),
        ("subtract 15 from 45", "math"),
        ("solve this math problem for me", "math"),
        ("how have you been doing", "status"),
        ("are you doing well today", "status"),
        ("how do you feel right now", "status"),
        ("is everything all right with you", "status"),
        ("can we become friends", "friendship"),
        ("i really need a friend right now", "friendship"),
        ("let us hang out together", "friendship"),
        ("will you be my companion", "friendship"),
        ("what is the capital of france", "knowledge"),
        ("tell me some news today", "knowledge"),
        ("tell me about the earth", "knowledge"),
        ("do you ever sleep", "knowledge"),
        ("give me information about python", "knowledge"),
    ]

    def eval_simple(model, examples):
        correct = 0
        for text, exp in examples:
            pred, _ = model.predict_with_confidence(text)
            if is_match(pred, exp):
                correct += 1
        return correct, len(examples), (correct / len(examples)) * 100.0

    v3_seen_c, v3_seen_t, v3_seen_acc = eval_simple(brain_v3, seen_examples)
    v4_seen_c, v4_seen_t, v4_seen_acc = eval_simple(brain_v4, seen_examples)
    v5_seen_c, v5_seen_t, v5_seen_acc = eval_simple(brain_v5, seen_examples)

    v3_unseen_c, v3_unseen_t, v3_unseen_acc = eval_simple(brain_v3, unseen_examples)
    v4_unseen_c, v4_unseen_t, v4_unseen_acc = eval_simple(brain_v4, unseen_examples)
    v5_unseen_c, v5_unseen_t, v5_unseen_acc = eval_simple(brain_v5, unseen_examples)

    # -------------------------------------------------------------------------
    # SUITE 2: UNCHANGED EXISTING ADVERSARIAL EVALUATION SUITE (117 Tests)
    # -------------------------------------------------------------------------
    adversarial_categories = {
        "negation": [
            ("do not calculate 5 plus 5", "unknown"),
            ("don't calculate 5 plus 5", "unknown"),
            ("I don't want math", "unknown"),
            ("don't tell me your name", "unknown"),
            ("I don't want to know who you are", "unknown"),
            ("I am not your friend", "unknown"),
            ("don't say hello", "unknown"),
            ("never talk to me again", "unknown"),
            ("do not tell me about python", "unknown"),
            ("I do not need to know the capital of france", "unknown"),
        ],
        "word_order": [
            ("who made you", "identity"),
            ("you made who", "unknown"),
            ("who you made", "unknown"),
            ("calculate 5 plus 5", "math"),
            ("5 plus 5 calculate", "math"),
            ("friendship best friend", "friendship"),
            ("best friend friendship", "friendship"),
            ("python what is", "knowledge"),
            ("you are who", "identity"),
            ("name your what", "unknown"),
        ],
        "keyword_distraction": [
            ("I was reading about Python and my friend said hello", {"knowledge", "friendship", "greeting", "unknown"}),
            ("I am hungry but goodbye", "goodbye"),
            ("my friend is doing math", "unknown"),
            ("the weather is good, goodbye", "goodbye"),
            ("Python friends are interesting", "unknown"),
            ("I saw a nice car on the street", "unknown"),
            ("time flies when you are having fun", "unknown"),
            ("the sum of all fears is a great movie", "unknown"),
            ("take care of your health", "unknown"),
            ("can you help me move this heavy couch", "unknown"),
        ],
        "multi_intent": [
            ("hello, who are you?", {"greeting", "identity"}),
            ("calculate 5 plus 5 and tell me about Python", {"math", "knowledge"}),
            ("goodbye, but first tell me your name", {"goodbye", "identity"}),
            ("I'm hungry and I want to know the weather", "knowledge"),
            ("are you okay and can you calculate 5 plus 5?", {"status", "math"}),
            ("hi can we be friends", {"greeting", "friendship"}),
            ("good morning, what is the capital of france?", {"greeting", "knowledge"}),
            ("see you later, are you doing well?", {"goodbye", "status"}),
        ],
        "paraphrase": [
            ("Salutations to everyone present", "greeting"),
            ("A pleasant afternoon to you", "greeting"),
            ("Top of the morning", "greeting"),
            ("I shall depart now", "goodbye"),
            ("Until next we meet", "goodbye"),
            ("I am taking my leave", "goodbye"),
            ("Disclose your developer", "identity"),
            ("Reveal your origin", "identity"),
            ("From which entity do you originate", "identity"),
            ("Who authored this codebase", "identity"),
            ("Compute the arithmetic product", "math"),
            ("Perform numeric summation", "math"),
            ("Evaluate this algebraic expression", "math"),
            ("What is your current operational status?", "status"),
            ("Are you functioning optimally?", "status"),
            ("How fares your system?", "status"),
            ("Let us form a camaraderie", "friendship"),
            ("I seek companionship", "friendship"),
            ("Elucidate the climate conditions", "knowledge"),
            ("State the geopolitical capital of Germany", "knowledge"),
        ],
        "typo_slang": [
            ("heyyy", "greeting"),
            ("hiiii", "greeting"),
            ("whos ur creator", "identity"),
            ("wat r u", "identity"),
            ("calc 5+5", "math"),
            ("who made u", "identity"),
            ("gud morning", "greeting"),
            ("bbye", "goodbye"),
            ("c u later", "goodbye"),
            ("how r u", "status"),
            ("pls calculate", "math"),
            ("ur my best frnd", "friendship"),
        ],
        "out_of_domain": [
            ("Supernovae release heavy elements into the interstellar medium.", "unknown"),
            ("Plate tectonics drive the formation of continental mountain ranges.", "unknown"),
            ("Preheat the skillet and sear the garlic in olive oil.", "unknown"),
            ("Knead the sourdough bread dough until gluten develops elasticity.", "unknown"),
            ("Chloroplasts capture photons during photosynthetic light reactions.", "unknown"),
            ("Penguins possess dense plumage to survive Antarctic blizzards.", "unknown"),
            ("Covalent bonds involve the sharing of electron pairs between atoms.", "unknown"),
            ("Gravitational waves propagate through spacetime at the speed of light.", "unknown"),
            ("The Roman Empire expanded across the Mediterranean basin.", "unknown"),
            ("Excavations revealed pottery shards from the Bronze Age.", "unknown"),
            ("Gothic cathedrals feature flying buttresses and ribbed vaults.", "unknown"),
            ("Shakespeare wrote sonnets exploring themes of mortality and romance.", "unknown"),
            ("Beethoven composed his Ninth Symphony despite profound deafness.", "unknown"),
            ("The referee awarded a penalty kick following the handball violation.", "unknown"),
            ("Antibiotics are ineffective against viral respiratory infections.", "unknown"),
            ("Clinical trials evaluate the efficacy and side effects of novel drugs.", "unknown"),
            ("Central banks adjust interest rates to manage inflationary pressures.", "unknown"),
            ("Diversified portfolios balance equities with sovereign bond holdings.", "unknown"),
            ("The appellate court upheld the lower jurisdiction's ruling.", "unknown"),
            ("Epistemology investigates the nature and justification of belief.", "unknown"),
            ("The Amazon basin encompasses dense tropical rainforest ecosystems.", "unknown"),
            ("Deep ocean trenches harbor unique extremophile organisms.", "unknown"),
            ("Barometric pressure drops sharply during hurricane development.", "unknown"),
            ("Phonemes represent the smallest contrastive units of linguistic sound.", "unknown"),
            ("Crop rotation replenishes nitrogen levels in agricultural soil.", "unknown"),
            ("Replace the ceramic brake pads to prevent rotor degradation.", "unknown"),
            ("Solid rocket boosters provide initial thrust during orbital launch.", "unknown"),
            ("Coral reefs suffer bleaching when ocean temperatures elevate.", "unknown"),
            ("Impressionist painters captured fleeting atmospheric lighting effects.", "unknown"),
            ("Cognitive dissonance arises when actions contradict core beliefs.", "unknown"),
            ("Nomadic tribes migrated in accordance with seasonal herd movements.", "unknown"),
            ("Fossilized footprints reveal dinosaur locomotion dynamics.", "unknown"),
        ],
        "invalid_input": [
            ("", "unknown"),
            (" ", "unknown"),
            ("   \t  \n  ", "unknown"),
            (None, "unknown"),
            (12345, "unknown"),
            ("12345", "unknown"),
            ("42 99 1000", "unknown"),
            ("3.14159", "unknown"),
            ("???", "unknown"),
            ("!...,,,;;;", "unknown"),
            ("@#$%^&*()_+", "unknown"),
            ("flabbergasted", "unknown"),
            ("extraterrestrial", "unknown"),
            ("supercalifragilisticexpialidocious", "unknown"),
            ("🚀🛸🤖", "unknown"),
        ]
    }

    def eval_adv(model):
        res = {}
        total = 0
        passed = 0
        for cat, tests in adversarial_categories.items():
            cat_pass = 0
            for text, exp in tests:
                pred, _ = model.predict_with_confidence(text)
                if is_match(pred, exp):
                    cat_pass += 1
            res[cat] = {"passed": cat_pass, "total": len(tests), "acc": (cat_pass / len(tests)) * 100.0}
            total += len(tests)
            passed += cat_pass
        res["overall"] = {"passed": passed, "total": total, "acc": (passed / total) * 100.0}
        return res

    adv_v3 = eval_adv(brain_v3)
    adv_v4 = eval_adv(brain_v4)
    adv_v5 = eval_adv(brain_v5)

    # -------------------------------------------------------------------------
    # SUITE 3: TWO-STAGE LINGUISTIC FRAME & MODIFIER EVALUATION (v0.5 Specific)
    # -------------------------------------------------------------------------
    print("\n[4/4] Running Two-Stage Linguistic Frame & Behavioral Evaluation on v0.5...")

    frame_test_cases = [
        # The 13 required test cases:
        ("calculate 5 + 5", "math", "affirmative", "command", "The result is 10"),
        ("don't calculate 5 + 5", "math", "negated", "command", "Understood, I won't calculate that."),
        ("do not calculate 5 + 5", "math", "negated", "command", "Understood, I won't calculate that."),
        ("never calculate 5 + 5", "math", "negated", "command", "Understood, I won't calculate that."),
        ("can you calculate 5 + 5?", "math", "affirmative", "question", "The result is 10"),
        ("please calculate 5 + 5", "math", "affirmative", "command", "The result is 10"),
        ("I don't want you to calculate 5 + 5", "math", "negated", "command", "Understood, I won't calculate that."),
        ("what is 5 + 5?", "math", "affirmative", "question", "The result is 10"),
        ("tell me about Python", "knowledge", "affirmative", "command", "Python is a high-level"),
        ("don't tell me about Python", "knowledge", "negated", "command", "Got it, I won't share information about that."),
        ("who made you?", "identity", "affirmative", "question", "I am Buddy"),
        ("don't tell me your name", "identity", "negated", "command", "Understood, I will keep my identity to myself."),
        ("never talk to me again", "friendship", "negated", "command", "Understood, I will give you some space."),

        # Adversarial context cases (modifier words in non-negation or unrelated contexts):
        ("knot in a rope", "unknown", "affirmative", "statement", "I'm not sure"),
        ("notice the small detail", "unknown", "affirmative", "command", "I'm not sure"),
        ("he said no to drugs", "unknown", "negated", "statement", "I'm not sure"),
        ("the movie was not bad", "unknown", "negated", "statement", "I'm not sure"),
        ("nothing happened today", "knowledge", "affirmative", "statement", "world is always changing"),
        ("now is the best time", "knowledge", "affirmative", "statement", "Time is relative"),
        ("stop at the red light", "unknown", "negated", "command", "I'm not sure"),
        ("please pass the salt", "unknown", "affirmative", "command", "I'm not sure"),
    ]

    intent_correct = 0
    polarity_correct = 0
    mood_correct = 0
    frame_correct = 0
    behavior_correct = 0

    # Polarity breakdown: False Positives & False Negatives
    # Positive class = "negated", Negative class = "affirmative"
    tp = 0
    fp = 0
    tn = 0
    fn = 0

    print("\n" + "=" * 85)
    print("      TWO-STAGE SEMANTIC FRAME & BEHAVIORAL VERIFICATION (Brain v0.5)")
    print("=" * 85)
    print(f"{'Input Prompt':<36} | {'Intent (Exp/Pred)':<20} | {'Pol':<10} | {'Mood':<10} | {'Frame':<6} | {'Behavior'}")
    print("-" * 85)

    for prompt, exp_intent, exp_pol, exp_mood, exp_behavior_sub in frame_test_cases:
        frame = brain_v5.predict_frame(prompt)
        bot_response = bot_v5.ask(prompt)

        p_intent = frame["intent"]
        p_pol = frame["polarity"]
        p_mood = frame["mood"]

        i_ok = (p_intent == exp_intent)
        p_ok = (p_pol == exp_pol)
        m_ok = (p_mood == exp_mood)
        f_ok = i_ok and p_ok and m_ok
        b_ok = exp_behavior_sub in bot_response

        if i_ok: intent_correct += 1
        if p_ok: polarity_correct += 1
        if m_ok: mood_correct += 1
        if f_ok: frame_correct += 1
        if b_ok: behavior_correct += 1

        # Track Polarity Confusion Matrix
        if exp_pol == "negated" and p_pol == "negated":
            tp += 1
        elif exp_pol == "affirmative" and p_pol == "negated":
            fp += 1
        elif exp_pol == "affirmative" and p_pol == "affirmative":
            tn += 1
        elif exp_pol == "negated" and p_pol == "affirmative":
            fn += 1

        i_str = f"{p_intent} ({'✓' if i_ok else '✗'})"
        p_str = f"{p_pol[:3]} ({'✓' if p_ok else '✗'})"
        m_str = f"{p_mood[:4]} ({'✓' if m_ok else '✗'})"
        f_str = "PASS" if f_ok else "FAIL"
        b_str = "PASS" if b_ok else "FAIL"

        print(f"{prompt[:36]:<36} | {i_str:<20} | {p_str:<10} | {m_str:<10} | {f_str:<6} | {b_str}")

    n_frame = len(frame_test_cases)
    intent_acc = (intent_correct / n_frame) * 100.0
    polarity_acc = (polarity_correct / n_frame) * 100.0
    mood_acc = (mood_correct / n_frame) * 100.0
    full_frame_acc = (frame_correct / n_frame) * 100.0
    behavioral_acc = (behavior_correct / n_frame) * 100.0

    # -------------------------------------------------------------------------
    # MASTER COMPARISON TABLE: v0.3 vs v0.4 vs v0.5
    # -------------------------------------------------------------------------
    print("\n" + "=" * 85)
    print("                     MASTER ARCHITECTURAL COMPARISON")
    print("=" * 85)
    print(f"{'Evaluation Metric':<32} | {'v0.3 (EmbeddingBag)':<18} | {'v0.4 (GRU)':<14} | {'v0.5 (Two-Stage)':<16}")
    print("-" * 85)
    print(f"{'Architecture':<32} | {'EmbeddingBag + Dense':<18} | {'1-Layer GRU':<14} | {'Bag + Rule Modifier':<16}")
    print(f"{'Total Trainable Parameters':<32} | {diag_v3['trainable_params']:<18} | {diag_v4['trainable_params']:<14} | {diag_v5['trainable_params']:<16}")
    print(f"{'Estimated FP32 Size':<32} | {diag_v3['fp32_size_kb']:<14.2f} KB | {diag_v4['fp32_size_kb']:<10.2f} KB | {diag_v5['fp32_size_kb']:<12.2f} KB")
    print(f"{'Training Loss (Final)':<32} | {diag_v3['training_loss']:<18.4f} | {diag_v4['training_loss']:<14.4f} | {diag_v5['training_loss']:<16.4f}")
    print("-" * 85)
    print(f"{'Training Accuracy (Seen)':<32} | {v3_seen_acc:>15.2f}% | {v4_seen_acc:>11.2f}% | {v5_seen_acc:>13.2f}%")
    print(f"{'Unseen Test Accuracy':<32} | {v3_unseen_acc:>15.2f}% | {v4_unseen_acc:>11.2f}% | {v5_unseen_acc:>13.2f}%")
    print(f"{'Adversarial Overall Accuracy':<32} | {adv_v3['overall']['acc']:>15.2f}% | {adv_v4['overall']['acc']:>11.2f}% | {adv_v5['overall']['acc']:>13.2f}%")
    print(f"{'— Negation Accuracy (as unk)':<32} | {adv_v3['negation']['acc']:>15.2f}% | {adv_v4['negation']['acc']:>11.2f}% | {adv_v5['negation']['acc']:>13.2f}%*")
    print(f"{'— Word-Order Accuracy':<32} | {adv_v3['word_order']['acc']:>15.2f}% | {adv_v4['word_order']['acc']:>11.2f}% | {adv_v5['word_order']['acc']:>13.2f}%")
    print(f"{'— Keyword-Distraction Accuracy':<32} | {adv_v3['keyword_distraction']['acc']:>15.2f}% | {adv_v4['keyword_distraction']['acc']:>11.2f}% | {adv_v5['keyword_distraction']['acc']:>13.2f}%")
    print(f"{'— Multi-Intent Accuracy':<32} | {adv_v3['multi_intent']['acc']:>15.2f}% | {adv_v4['multi_intent']['acc']:>11.2f}% | {adv_v5['multi_intent']['acc']:>13.2f}%")
    print(f"{'— Paraphrase Accuracy':<32} | {adv_v3['paraphrase']['acc']:>15.2f}% | {adv_v4['paraphrase']['acc']:>11.2f}% | {adv_v5['paraphrase']['acc']:>13.2f}%")
    print(f"{'— Typo/Slang Accuracy':<32} | {adv_v3['typo_slang']['acc']:>15.2f}% | {adv_v4['typo_slang']['acc']:>11.2f}% | {adv_v5['typo_slang']['acc']:>13.2f}%")
    print(f"{'— Out-of-Domain Rejection':<32} | {adv_v3['out_of_domain']['acc']:>15.2f}% | {adv_v4['out_of_domain']['acc']:>11.2f}% | {adv_v5['out_of_domain']['acc']:>13.2f}%")
    print(f"{'— Invalid-Input Handling':<32} | {adv_v3['invalid_input']['acc']:>15.2f}% | {adv_v4['invalid_input']['acc']:>11.2f}% | {adv_v5['invalid_input']['acc']:>13.2f}%")
    print("-" * 85)
    print(f"{'Two-Stage Frame Metrics:':<32} | {'(N/A: Single-stage)':<18} | {'(N/A)':<14} | {'(v0.5 Dedicated)':<16}")
    print(f"{'  Intent (Topic) Accuracy':<32} | {'N/A':<18} | {'N/A':<14} | {intent_acc:>13.2f}%")
    print(f"{'  Polarity Accuracy':<32} | {'N/A':<18} | {'N/A':<14} | {polarity_acc:>13.2f}%")
    print(f"{'  Mood Accuracy':<32} | {'N/A':<18} | {'N/A':<14} | {mood_acc:>13.2f}%")
    print(f"{'  Full-Frame Accuracy':<32} | {'N/A':<18} | {'N/A':<14} | {full_frame_acc:>13.2f}%")
    print(f"{'  Behavioral Correctness':<32} | {'N/A':<18} | {'N/A':<14} | {behavioral_acc:>13.2f}%")
    print("=" * 85)
    print("* Note: The legacy adversarial suite scored Negation as PASS only if the classifier returned 'unknown'.")
    print("  In Brain v0.5, the Intent Engine identifies the domain topic ('math') while the Modifier Engine captures negation,")
    print("  enabling appropriate refusal behavior rather than falling back to ignorance.")

    # -------------------------------------------------------------------------
    # POLARITY ERROR ANALYSIS (FALSE POSITIVES & FALSE NEGATIVES)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 85)
    print("          POLARITY CONFUSION ANALYSIS (Brain v0.5 Modifier Engine)")
    print("=" * 85)
    print(f"  True Negated (TP):       {tp:>3} (Correctly detected as negated)")
    print(f"  True Affirmative (TN):   {tn:>3} (Correctly detected as affirmative)")
    print(f"  False Negated (FP):      {fp:>3} (Affirmative misclassified as negated)")
    print(f"  False Affirmative (FN):  {fn:>3} (Negated misclassified as affirmative)")
    print(f"  Polarity Precision:      {(tp / (tp + fp) * 100.0) if (tp + fp) > 0 else 0.0:.2f}%")
    print(f"  Polarity Recall:         {(tp / (tp + fn) * 100.0) if (tp + fn) > 0 else 0.0:.2f}%")
    print("=" * 85)


if __name__ == "__main__":
    run_full_evaluation()
