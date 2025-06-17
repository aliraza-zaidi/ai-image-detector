import streamlit as st
from PIL import Image
import torch
from transformers import AutoFeatureExtractor, SwinForImageClassification
import torchvision.transforms as T

st.set_page_config(page_title="AI Image Detector", layout="centered")

st.title("🧠 AI Image Detector")
st.write("Upload an image to detect whether it is **AI-generated** or **human-made**.")


@st.cache_resource
def load_model():
    model = SwinForImageClassification.from_pretrained(".")
    model.eval()
    return model

@st.cache_resource
def load_preprocessor():
    return AutoFeatureExtractor.from_pretrained(".")

model = load_model()
feature_extractor = load_preprocessor()

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)
    
    inputs = feature_extractor(images=image, return_tensors="pt")
        
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_class_idx = logits.argmax(-1).item()
        labels = model.config.id2label
        predicted_label = labels[predicted_class_idx]

    
    st.markdown("---")
    st.subheader("🔍 Prediction:")
    st.success(f"The image is **{predicted_label.upper()}**.")

