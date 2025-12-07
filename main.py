import cv2
import numpy as np
import streamlit as st
from tensorflow.keras.applications.mobilenet_v2 import (
    MobileNetV2,
    preprocess_input,
    decode_predictions
)
from PIL import Image


@st.cache_resource
def load_model():
    model = MobileNetV2(weights="imagenet")
    return model


def preprocess_image(image: Image.Image):
    # 🔹 Always convert to 3-channel RGB (drops alpha if present)
    image = image.convert("RGB")

    # PIL → NumPy
    img = np.array(image)

    # 🔹 Resize with OpenCV
    img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)

    # Ensure float32
    img = img.astype(np.float32)

    # Preprocess for MobileNetV2
    img = preprocess_input(img)
    img = np.expand_dims(img, axis=0)  # shape: (1, 224, 224, 3)

    return img


def classify_image(model, image: Image.Image):
    try:
        processed_image = preprocess_image(image)
        predictions = model.predict(processed_image)
        decoded_predictions = decode_predictions(predictions, top=3)[0]
        return decoded_predictions
    except Exception as e:
        st.error(f"Error classifying image: {str(e)}")
        return None


def main():
    st.set_page_config(
        page_title="AI Image Classifier",
        page_icon="🖼️",
        layout="centered"
    )

    st.title("AI Image Classifier")
    st.write("Upload an image and let AI tell you what is in it!")

    model = load_model()

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)

        if st.button("Classify Image"):
            with st.spinner("Analyzing Image..."):
                pil_img = Image.open(uploaded_file)
                predictions = classify_image(model, pil_img)

                if predictions:
                    st.subheader("Predictions")
                    for _, label, score in predictions:
                        st.write(f"**{label}**: {score:.2%}")


if __name__ == "__main__":
    main()
