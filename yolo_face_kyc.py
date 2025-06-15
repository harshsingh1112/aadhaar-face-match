import streamlit as st
import cv2
import numpy as np
from pdf2image import convert_from_bytes
from deepface import DeepFace
from PIL import Image
import tempfile
import os

st.set_page_config(page_title="Aadhaar KYC Face Match", layout="centered")
st.title("🔍 Aadhaar KYC Face Match")

# Step 1: Upload Aadhaar
uploaded_file = st.file_uploader("Upload Aadhaar (PDF or Image)", type=["pdf", "jpg", "jpeg", "png"])

if uploaded_file:
    with st.spinner("Processing Aadhaar..."):
        if uploaded_file.type == "application/pdf":
            images = convert_from_bytes(uploaded_file.read())
            aadhaar_img = np.array(images[0])  # Use the first page
        else:
            aadhaar_img = np.array(Image.open(uploaded_file))

        aadhaar_img = cv2.cvtColor(aadhaar_img, cv2.COLOR_RGB2BGR)

        # Step 2: Extract Aadhaar Face
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        gray = cv2.cvtColor(aadhaar_img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        if len(faces) == 0:
            st.error("No face found in Aadhaar image. Try another.")
        else:
            (x, y, w, h) = faces[0]
            aadhaar_face = aadhaar_img[y:y+h, x:x+w]
            st.image(cv2.cvtColor(aadhaar_face, cv2.COLOR_BGR2RGB), caption="Detected Aadhaar Face")

            # Step 3: Upload/Take Live Photo
            st.subheader("📸 Live Face Capture")
            live_file = st.camera_input("Take a live photo")

            if live_file:
                live_img = Image.open(live_file)
                live_img = cv2.cvtColor(np.array(live_img), cv2.COLOR_RGB2BGR)

                # Step 4: Face Match
                with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp1, \
                     tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp2:
                    cv2.imwrite(tmp1.name, aadhaar_face)
                    cv2.imwrite(tmp2.name, live_img)

                    result = DeepFace.verify(img1_path=tmp1.name, img2_path=tmp2.name, enforce_detection=False)

                    if result["verified"]:
                        st.success("✅ Face Matched!")
                    else:
                        st.error("❌ Face Mismatch")