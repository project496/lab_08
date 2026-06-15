import streamlit as st
import numpy as np
from collections import Counter
import tensorflow as tf
from tensorflow import keras

st.set_page_config(page_title="AI Scrabble Word Finder", layout="centered")

SCRABBLE_SCORES = {
    'a':1,'b':3,'c':3,'d':2,'e':1,'f':4,'g':2,'h':4,'i':1,'j':8,'k':5,'l':1,
    'm':3,'n':1,'o':1,'p':3,'q':10,'r':1,'s':1,'t':1,'u':1,'v':4,'w':4,'x':8,'y':4,'z':10
}

def scrabble_score(word):
    return sum(SCRABBLE_SCORES.get(c, 0) for c in word.lower())

def word_features(word):
    vec = np.zeros(28)

    for c in word.lower():
        if c.isalpha():
            vec[ord(c) - 97] += 1

    vec[26] = len(word)

    vowels = sum(1 for c in word.lower() if c in "aeiou")
    vec[27] = vowels / max(len(word), 1)

    return vec


def can_make(word, letters):
    w = Counter(word)
    l = Counter(letters)

    for ch in w:
        if w[ch] > l.get(ch, 0):
            return False
    return True


# 📂 LOAD REAL DICTIONARY
@st.cache_data
def load_words():
    with open("words.txt", "r") as f:
        return [w.strip().lower() for w in f.readlines() if len(w.strip()) > 1]

WORD_LIST = load_words()


def generate_words(letters):
    letters = letters.lower()
    return [w for w in WORD_LIST if can_make(w, letters)]


# 🧠 AI MODEL
X = np.array([word_features(w) for w in WORD_LIST[:5000]])  # speed optimization
y = np.array([scrabble_score(w) for w in WORD_LIST[:5000]])

model = keras.Sequential([
    keras.layers.Dense(64, activation="relu", input_shape=(28,)),
    keras.layers.Dense(32, activation="relu"),
    keras.layers.Dense(16, activation="relu"),
    keras.layers.Dense(1)
])

model.compile(optimizer="adam", loss="mse")
model.fit(X, y, epochs=50, verbose=0)


# UI
st.title("🧠 AI Scrabble Word Finder (FULL VERSION)")

letters = st.text_input("Enter letters (example: scrabble)")

if st.button("Generate Words"):
    if letters:

        words = generate_words(letters)

        if len(words) == 0:
            st.warning("No words found. Try different letters.")
        else:
            results = []

            for w in words:
                score = scrabble_score(w)

                ai_score = float(
                    model.predict(word_features(w).reshape(1, -1), verbose=0)[0][0]
                )

                results.append((w, score, ai_score))

            results.sort(key=lambda x: x[2], reverse=True)

            st.subheader(f"🏆 Found {len(results)} Words")

            for w, score, ai_score in results[:50]:
                st.write(f"{w} | Score: {score} | AI Rank: {round(ai_score,2)}")

    else:
        st.warning("Please enter letters")
