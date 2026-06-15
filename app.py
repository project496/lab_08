import streamlit as st
import itertools
import numpy as np
import tensorflow as tf
from tensorflow import keras

st.set_page_config(page_title="AI Scrabble Word Finder", layout="centered")

SCRABBLE_SCORES = {
    'a':1,'b':3,'c':3,'d':2,'e':1,'f':4,'g':2,'h':4,'i':1,'j':8,'k':5,'l':1,
    'm':3,'n':1,'o':1,'p':3,'q':10,'r':1,'s':1,'t':1,'u':1,'v':4,'w':4,'x':8,'y':4,'z':10
}

WORD_LIST = [
    "cat","dog","bird","word","world","data","python","code","scrabble","learn",
    "plane","train","brain","rain","gain","main","paint","pair","air","fair",
    "chair","hair","near","ear","gear","run","running","ring","sing","king",
    "bring","string","thing","thinking","apple","apply","play","player","paper",
    "tiger","lion","zebra","mouse","house","home","game","name","same","fame"
]

def scrabble_score(word):
    return sum(SCRABBLE_SCORES.get(c,0) for c in word.lower())

def word_features(word):
    vec = np.zeros(28)

    for c in word.lower():
        if c.isalpha():
            vec[ord(c)-97] += 1

    vec[26] = len(word)

    vowels = sum(1 for c in word.lower() if c in "aeiou")
    vec[27] = vowels / max(len(word), 1)

    return vec


def can_make(word, letters):
    from collections import Counter
    word_count = Counter(word)
    letters_count = Counter(letters)

    for ch in word_count:
        if word_count[ch] > letters_count.get(ch, 0):
            return False
    return True


def generate_words(letters):
    letters = letters.lower()
    valid_words = []

    for word in WORD_LIST:
        if can_make(word, letters):
            valid_words.append(word)

    return valid_words


# ================= AI MODEL =================

X = np.array([word_features(w) for w in WORD_LIST])

y = np.array([
    scrabble_score(w) * 1.2 + len(w) * 0.5
    for w in WORD_LIST
])

model = keras.Sequential([
    keras.layers.Dense(64, activation="relu", input_shape=(28,)),
    keras.layers.Dense(32, activation="relu"),
    keras.layers.Dense(16, activation="relu"),
    keras.layers.Dense(1)
])

model.compile(optimizer="adam", loss="mse")
model.fit(X, y, epochs=150, verbose=0)


# ================= UI =================

st.title("🧠 AI Scrabble Word Finder (Advanced Version)")

letters = st.text_input("Enter letters (example: applerun)")

if st.button("Generate Words"):
    if letters:

        words = generate_words(letters)

        if not words:
            st.warning("No valid words found.")
        else:
            results = []

            for w in words:
                score = scrabble_score(w)

                ai_score = float(
                    model.predict(word_features(w).reshape(1,-1), verbose=0)[0][0]
                )

                results.append((w, score, ai_score))

            results.sort(key=lambda x: x[2], reverse=True)

            st.subheader("🏆 Best AI Ranked Words")

            for w, score, ai_score in results:
                st.write(f"{w} | Score: {score} | AI Rank: {round(ai_score,2)}")

    else:
        st.warning("Please enter letters")
