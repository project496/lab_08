import streamlit as st
import easyocr
import cv2
import numpy as np
from PIL import Image
import pandas as pd

# --------------------------------------------------
# Page Config
# --------------------------------------------------
st.set_page_config(
    page_title="AI Medical Prescription Reader",
    page_icon="💊",
    layout="wide"
)

# --------------------------------------------------
# Load OCR Model
# --------------------------------------------------
@st.cache_resource
def load_reader():
    return easyocr.Reader(['en'], gpu=False)

reader = load_reader()

# --------------------------------------------------
# Medicine Database
# --------------------------------------------------
medicine_database = {
    "paracetamol": {
        "usage": "Pain relief and fever reduction",
        "dosage": "500mg every 4-6 hours",
        "warning": "Do not exceed 4000mg/day"
    },
    "amoxicillin": {
        "usage": "Antibiotic for bacterial infections",
        "dosage": "500mg three times daily",
        "warning": "Complete the full prescribed course"
    },
    "ibuprofen": {
        "usage": "Pain, inflammation, fever",
        "dosage": "200-400mg every 6 hours",
        "warning": "Take after meals"
    },
    "cetirizine": {
        "usage": "Allergy treatment",
        "dosage": "10mg once daily",
        "warning": "May cause drowsiness"
    },
    "azithromycin": {
        "usage": "Antibiotic",
        "dosage": "500mg once daily",
        "warning": "Take as prescribed"
    },
    "metformin": {
        "usage": "Diabetes management",
        "dosage": "500mg twice daily",
        "warning": "Take with food"
    }
}

# --------------------------------------------------
# OCR Function
# --------------------------------------------------
def perform_ocr(image):
    try:
        img_array = np.array(image)

        if len(img_array.shape) == 3:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_array

        result = reader.readtext(gray)

        extracted_text = ""

        for item in result:
            extracted_text += item[1] + "\n"

        return extracted_text

    except Exception as e:
        st.error(f"OCR Error: {e}")
        return ""

# --------------------------------------------------
# Extract Medicines
# --------------------------------------------------
def extract_medicines(text):

    medicines_found = []

    text = text.lower()

    for medicine in medicine_database.keys():
        if medicine in text:
            medicines_found.append(medicine)

    return medicines_found

# --------------------------------------------------
# UI
# --------------------------------------------------
st.title("💊 AI Medical Prescription Reader")

st.markdown("""
Upload a doctor's prescription image and let AI extract medicine names and dosage information.
""")

uploaded_file = st.file_uploader(
    "Upload Prescription Image",
    type=["png", "jpg", "jpeg"]
)

if uploaded_file is not None:

    try:
        image = Image.open(uploaded_file).convert("RGB")

        col1, col2 = st.columns(2)

        with col1:
            st.image(
                image,
                caption="Uploaded Prescription",
                width=500
            )

        with st.spinner("Reading prescription..."):
            extracted_text = perform_ocr(image)

        with col2:
            st.subheader("Extracted Text")
            st.text_area(
                "OCR Result",
                extracted_text,
                height=350
            )

        st.divider()

        medicines = extract_medicines(extracted_text)

        st.subheader("Detected Medicines")

        if medicines:

            data = []

            for med in medicines:
                data.append({
                    "Medicine": med.title(),
                    "Usage": medicine_database[med]["usage"],
                    "Dosage": medicine_database[med]["dosage"],
                    "Warning": medicine_database[med]["warning"]
                })

            df = pd.DataFrame(data)

            st.dataframe(df)

            st.success(
                f"{len(medicines)} medicine(s) detected successfully."
            )

        else:
            st.warning(
                "No medicines found in the built-in database."
            )

        st.divider()

        st.subheader("Medical Disclaimer")

        st.info(
            "This application is for educational purposes only. "
            "Always consult a qualified healthcare professional."
        )

    except Exception as e:
        st.error(f"Image Processing Error: {e}")

else:
    st.info("Please upload a prescription image to begin.")
