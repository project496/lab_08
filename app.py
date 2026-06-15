import streamlit as st
import easyocr
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import re

st.set_page_config(
    page_title="AI Medical Prescription Reader",
    page_icon="💊",
    layout="wide"
)

@st.cache_resource
def load_reader():
    return easyocr.Reader(['en'], gpu=False)

reader = load_reader()

medicine_database = {
    "paracetamol": ["Pain Relief", "500mg every 4-6 hours"],
    "amoxicillin": ["Antibiotic", "500mg three times daily"],
    "ibuprofen": ["Pain and Inflammation", "200-400mg every 6 hours"],
    "cetirizine": ["Allergy", "10mg daily"],
    "azithromycin": ["Antibiotic", "500mg daily"],
    "metformin": ["Diabetes", "500mg twice daily"],
    "diclofenac": ["Pain Relief", "As prescribed"],
    "ultrafen": ["Pain Relief", "As prescribed"],
    "cartilix": ["Joint Support", "As prescribed"],
    "relentus": ["Muscle Relaxant", "As prescribed"],
    "panadol": ["Pain Relief", "As prescribed"],
    "augmentin": ["Antibiotic", "As prescribed"],
    "brufen": ["Pain Relief", "As prescribed"],
    "calpol": ["Fever", "As prescribed"],
    "flagyl": ["Infection", "As prescribed"],
    "dolo": ["Pain Relief", "As prescribed"],
    "napa": ["Pain Relief", "As prescribed"]
}

def preprocess_image(image):
    img = np.array(image)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gray = cv2.equalizeHist(gray)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    thresh = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11,
        2
    )
    return thresh

def extract_text(image):
    processed = preprocess_image(image)

    results = reader.readtext(
        processed,
        detail=1,
        paragraph=True
    )

    full_text = ""
    rows = []

    for item in results:
        text = item[1]
        conf = round(item[2] * 100, 2)

        full_text += text + "\n"

        rows.append({
            "Detected Text": text,
            "Confidence (%)": conf
        })

    return full_text, pd.DataFrame(rows)

def detect_medicines(text):
    detected = []

    text = text.lower()

    for medicine in medicine_database:
        if medicine.lower() in text:
            detected.append(medicine)

    words = re.findall(r"[A-Za-z]+", text)

    for word in words:
        if len(word) > 4:
            if word not in detected:
                detected.append(word)

    return list(dict.fromkeys(detected))

st.title("💊 AI Medical Prescription Reader")

uploaded_file = st.file_uploader(
    "Upload Prescription Image",
    type=["png", "jpg", "jpeg"]
)

if uploaded_file:

    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns(2)

    with col1:
        st.image(image, caption="Uploaded Prescription", width=500)

    with st.spinner("Reading Prescription..."):
        extracted_text, ocr_df = extract_text(image)

    with col2:
        st.subheader("Extracted Text")
        st.text_area(
            "OCR Result",
            extracted_text,
            height=400
        )

    st.subheader("OCR Detection Details")
    st.dataframe(ocr_df, use_container_width=True)

    medicines = detect_medicines(extracted_text)

    st.subheader("Detected Medicines")

    if medicines:

        result = []

        for med in medicines:

            med_lower = med.lower()

            if med_lower in medicine_database:
                result.append({
                    "Medicine": med.title(),
                    "Usage": medicine_database[med_lower][0],
                    "Dosage": medicine_database[med_lower][1]
                })
            else:
                result.append({
                    "Medicine": med.title(),
                    "Usage": "Not Available",
                    "Dosage": "Not Available"
                })

        st.dataframe(
            pd.DataFrame(result),
            use_container_width=True
        )

    else:
        st.warning("No medicines detected.")

    st.subheader("Medical Disclaimer")

    st.info(
        "This tool is for educational purposes only. Always consult a healthcare professional before taking any medication."
    )
