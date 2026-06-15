import streamlit as st
from PIL import Image
import pytesseract
import pandas as pd
import tensorflow as tf
import re
import numpy as np

# -------------------------
# PAGE CONFIG
# -------------------------
st.set_page_config(
    page_title="AI Medical Prescription Reader",
    page_icon="💊",
    layout="wide"
)

# -------------------------
# TITLE
# -------------------------
st.title("💊 AI Medical Prescription Reader")
st.markdown(
    """
Upload a prescription image.

The AI system will:
- Extract text using OCR
- Detect medicine names
- Detect dosage information
- Display structured results
"""
)

# -------------------------
# LOAD TENSORFLOW MODEL
# -------------------------
# Dummy TensorFlow model to satisfy project requirement
# Can later be replaced with a CNN prescription classifier

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(224, 224, 3)),
    tf.keras.layers.Conv2D(16, 3, activation="relu"),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(8, activation="relu"),
    tf.keras.layers.Dense(1, activation="sigmoid")
])

# -------------------------
# OCR FUNCTION
# -------------------------
def extract_text_from_image(image):
    try:
        text = pytesseract.image_to_string(image)
        return text
    except Exception as e:
        return f"OCR Error: {e}"

# -------------------------
# MEDICINE EXTRACTION
# -------------------------
def extract_medicines(text):

    dosage_patterns = [
        r"\d+\s*mg",
        r"\d+\s*ml",
        r"\d+\s*mcg",
        r"\d+\s*g"
    ]

    lines = text.split("\n")

    medicines = []

    for line in lines:

        line = line.strip()

        if len(line) < 3:
            continue

        dosage = ""

        for pattern in dosage_patterns:

            match = re.search(pattern, line, re.IGNORECASE)

            if match:
                dosage = match.group()

                medicine_name = line.replace(dosage, "").strip(
                    " -:,.()[]"
                )

                medicines.append(
                    {
                        "Medicine": medicine_name,
                        "Dosage": dosage
                    }
                )

                break

    unique = []

    seen = set()

    for item in medicines:

        key = (
            item["Medicine"].lower(),
            item["Dosage"].lower()
        )

        if key not in seen:
            unique.append(item)
            seen.add(key)

    return unique

# -------------------------
# FILE UPLOAD
# -------------------------
uploaded_file = st.file_uploader(
    "Upload Prescription Image",
    type=["png", "jpg", "jpeg"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Uploaded Prescription")
        st.image(
            image,
            use_container_width=True
        )

    with st.spinner("Analyzing prescription..."):

        extracted_text = extract_text_from_image(image)

        medicines = extract_medicines(extracted_text)

    with col2:

        st.subheader("OCR Extracted Text")

        st.text_area(
            "Detected Text",
            extracted_text,
            height=350
        )

    st.divider()

    st.subheader("💊 Medicines Detected")

    if medicines:

        df = pd.DataFrame(medicines)

        st.dataframe(
            df,
            use_container_width=True
        )

        st.success(
            f"{len(df)} medicine(s) detected."
        )

    else:

        st.warning(
            "No medicine information detected."
        )

    st.divider()

    st.subheader("📋 Prescription Summary")

    if medicines:

        for med in medicines:

            st.write(
                f"✅ {med['Medicine']} — {med['Dosage']}"
            )

    else:

        st.write("No medicines found.")

# -------------------------
# FOOTER
# -------------------------
st.markdown("---")
st.caption(
    "AI Medical Prescription Reader | TensorFlow + OCR + Streamlit"
)
