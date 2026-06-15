import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow import keras
from collections import Counter

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

def can_make(word, letters):
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


# Simple AI model (ranking only)
def word_to_vector(word):
    vec = np.zeros(26)
    for c in word.lower():
        if c.isalpha():
            vec[ord(c)-97] += 1
    return vec

X = np.array([word_to_vector(w) for w in WORD_LIST])
y = np.array([scrabble_score(w) for w in WORD_LIST])

model = keras.Sequential([
    keras.layers.Dense(32, activation="relu", input_shape=(26,)),
    keras.layers.Dense(16, activation="relu"),
    keras.layers.Dense(1)
])

model.compile(optimizer="adam", loss="mse")
model.fit(X, y, epochs=80, verbose=0)

st.title("AI Scrabble Word Finder (Smart Version)")

letters = st.text_input("Enter letters (e.g. applerun)")

if st.button("Generate Words"):
    if letters:
        words = generate_words(letters)

        if not words:
            st.warning("No valid words found.")
        else:
            results = []

            for w in words:
                score = scrabble_score(w)
                ai_score = float(model.predict(word_to_vector(w).reshape(1,-1), verbose=0)[0][0])
                results.append((w, score, ai_score))

            results.sort(key=lambda x: x[2], reverse=True)

            st.subheader("Best Scrabble Words")

            for w, score, ai_score in results:
                st.write(f"{w} | Score: {score} | AI Rank: {round(ai_score,2)}")
    else:
        st.warning("Enter letters first")
