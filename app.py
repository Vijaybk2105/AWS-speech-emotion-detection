import streamlit as st
import cv2
from deepface import DeepFace
import tempfile
import os

st.set_page_config(
    page_title="Face Analysis",
    page_icon="👤"
)

st.title("Face Detection and Analysis")
st.write("Upload an image to estimate age and gender.")

uploaded_file = st.file_uploader(
    "Upload Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:

    image_bytes = uploaded_file.getvalue()

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".jpg"
    ) as temp:
        temp.write(image_bytes)
        image_path = temp.name

    try:
        image = cv2.imread(image_path)

        results = DeepFace.analyze(
            img_path=image_path,
            actions=["age", "gender"],
            enforce_detection=True,
            detector_backend="opencv"
        )

        if isinstance(results, dict):
            results = [results]

        for i, face in enumerate(results):

            region = face["region"]

            x = region["x"]
            y = region["y"]
            w = region["w"]
            h = region["h"]

            age = face["age"]
            gender_scores = face["gender"]

            gender = max(
                gender_scores,
                key=gender_scores.get
            )

            label = f"{gender}, Age: {age}"

            cv2.rectangle(
                image,
                (x, y),
                (x + w, y + h),
                (0, 255, 0),
                2
            )

            cv2.putText(
                image,
                label,
                (x, max(20, y - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

            st.subheader(f"Face {i + 1}")
            st.write(f"Estimated Age: {age}")
            st.write(f"Predicted Gender: {gender}")

        image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )

        st.subheader("Analyzed Image")
        st.image(image, use_container_width=True)

    except Exception as e:
        st.error(f"Analysis failed: {e}")

    finally:
        if os.path.exists(image_path):
            os.remove(image_path)