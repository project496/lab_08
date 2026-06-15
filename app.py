import streamlit as st
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import pytesseract
import re

st.set_page_config(
    page_title="AI Medical Prescription Reader",
    page_icon="💊",
    layout="wide"
)

medicine_database = {
    "paracetamol": ["Pain Relief", "500mg every 4-6 hours"],
    "amoxicillin": ["Antibiotic", "500mg three times daily"],
    "ibuprofen": ["Pain Relief", "200-400mg every 6 hours"],
    "diclofenac": ["Pain Relief", "As prescribed"],
    "azithromycin": ["Antibiotic", "As prescribed"],
    "cetirizine": ["Allergy", "As prescribed"],
    "metformin": ["Diabetes", "As prescribed"],
    "augmentin": ["Antibiotic", "As prescribed"],
    "panadol": ["Pain Relief", "As prescribed"],
    "brufen": ["Pain Relief", "As prescribed"],
    "flagyl": ["Antibiotic", "As prescribed"],
    "calpol": ["Fever", "As prescribed"]
}

def preprocess_image(image):
    img = np.array(image)

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    gray = cv2.resize(
        gray,
        None,
        fx=2,
        fy=2,
        interpolation=cv2.INTER_CUBIC
    )

    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    thresh = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        11
    )

    return thresh

def extract_text(image):
    processed = preprocess_image(image)

    text = pytesseract.image_to_string(processed)

    data = []

    for line in text.split("\n"):
        line = line.strip()

        if len(line) > 0:
            data.append({
                "Detected Text": line
            })

    return text, pd.DataFrame(data)

def detect_medicines(text):
    medicines = []

    lower_text = text.lower()

    for med in medicine_database.keys():
        if med in lower_text:
            medicines.append(med)

    words = re.findall(r"[A-Za-z]+", lower_text)

    for word in words:
        if len(word) > 4:
            if word not in medicines:
                medicines.append(word)

    return list(dict.fromkeys(medicines))

st.title("💊 AI Medical Prescription Reader")

uploaded_file = st.file_uploader(
    "Upload Prescription Image",
    type=["png", "jpg", "jpeg"]
)

if uploaded_file:

    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns(2)

    with col1:
        st.image(image, caption="Prescription", width=500)

    with st.spinner("Processing..."):
        extracted_text, ocr_df = extract_text(image)

    with col2:
        st.subheader("Extracted Text")

        st.text_area(
            "OCR Result",
            extracted_text,
            height=350
        )

    st.subheader("OCR Detection Details")

    if not ocr_df.empty:
        st.dataframe(
            ocr_df,
            use_container_width=True
        )
    else:
        st.warning("No text detected.")

    medicines = detect_medicines(extracted_text)

    st.subheader("Detected Medicines")

    if medicines:

        result = []

        for med in medicines:

            if med.lower() in medicine_database:

                result.append({
                    "Medicine": med.title(),
                    "Usage": medicine_database[med.lower()][0],
                    "Dosage": medicine_database[med.lower()][1]
                })

            else:

                result.append({
                    "Medicine": med.title(),
                    "Usage": "Unknown",
                    "Dosage": "Unknown"
                })

        st.dataframe(
            pd.DataFrame(result),
            use_container_width=True
        )

    else:
        st.warning("No medicines detected.")

    st.subheader("Medical Disclaimer")

    st.info(
        "This application is for educational purposes only. Always consult a healthcare professional."
    )
