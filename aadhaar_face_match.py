import streamlit as st
from deepface import DeepFace
import cv2
import numpy as np
from pdf2image import convert_from_bytes
import tempfile
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

st.set_page_config(page_title="Aadhaar Face Match", layout="centered")

st.title("Aadhaar Face Match with Live Webcam")
st.markdown("Upload your Aadhaar PDF and use live webcam to verify your identity.")

# Upload Aadhaar PDF
aadhaar_pdf = st.file_uploader("Upload your Aadhaar PDF", type=["pdf"])

# Checkbox to enable live webcam capture
enable_webcam = st.checkbox("Enable Live Webcam Capture")

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

def extract_face(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    if len(faces) == 0:
        return None
    x, y, w, h = faces[0]
    return img[y:y+h, x:x+w]

aadhaar_face_img = None
if aadhaar_pdf:
    try:
        pages = convert_from_bytes(aadhaar_pdf.read(), dpi=300)
        aadhaar_img = np.array(pages[0])
        aadhaar_img = cv2.cvtColor(aadhaar_img, cv2.COLOR_RGB2BGR)
        face = extract_face(aadhaar_img)
        if face is not None:
            aadhaar_face_img = face
            st.image(cv2.cvtColor(face, cv2.COLOR_BGR2RGB), caption="Extracted Aadhaar Face")
        else:
            st.warning("No face detected in Aadhaar PDF.")
    except Exception as e:
        st.error(f"Error processing Aadhaar PDF: {e}")

class FaceCaptureTransformer(VideoTransformerBase):
    def __init__(self):
        self.captured_face = None

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        face = extract_face(img)
        if face is not None:
            self.captured_face = face
            # Draw rectangle around face
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
            for (x,y,w,h) in faces:
                cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

ctx = None
if enable_webcam:
    ctx = webrtc_streamer(
        key="face-capture",
        video_transformer_factory=FaceCaptureTransformer,
        media_stream_constraints={"video": True, "audio": False},
    )

if aadhaar_face_img is not None and ctx is not None and ctx.video_transformer:
    live_face_img = ctx.video_transformer.captured_face
    if live_face_img is not None:
        st.image(cv2.cvtColor(live_face_img, cv2.COLOR_BGR2RGB), caption="Captured Live Face")

        temp_dir = tempfile.mkdtemp()
        aadhaar_path = f"{temp_dir}/aadhaar_face.jpg"
        live_path = f"{temp_dir}/live_face.jpg"
        cv2.imwrite(aadhaar_path, aadhaar_face_img)
        cv2.imwrite(live_path, live_face_img)

        with st.spinner("Verifying faces..."):
            result = DeepFace.verify(img1_path=aadhaar_path, img2_path=live_path, enforce_detection=False)

        if result["verified"]:
            st.success(f"✅ Face Match! (Distance: {result['distance']:.4f})")
        else:
            st.error(f"❌ Face does not match. (Distance: {result['distance']:.4f})")
    else:
        st.info("Waiting for face capture from webcam...")

elif aadhaar_pdf and not enable_webcam:
    st.info("Enable live webcam capture to perform face verification.")

elif not aadhaar_pdf:
    st.info("Upload Aadhaar PDF to start.")
