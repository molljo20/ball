import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# ==============================
# KONFIGURATION
# ==============================
MODEL_PATH = "converted_keras.zip"
IMG_SIZE = (224, 224)

# ⚠️ Passe die Klassen an DEIN Modell an
CLASS_NAMES = [
    "Fußball",
    "Basketball",
    "Tennisball",
    "Volleyball",
    "Handball"
]

# ==============================
# MODELL LADEN
# ==============================
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model(MODEL_PATH)
    return model

model = load_model()

# ==============================
# BILD VORVERARBEITUNG
# ==============================
def preprocess_image(image: Image.Image):
    image = image.convert("RGB")
    image = image.resize(IMG_SIZE)
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# ==============================
# STREAMLIT UI
# ==============================
st.set_page_config(page_title="Ball-Erkennung", page_icon="⚽")

st.title("⚽ Ball-Erkennungs-App")
st.write("Lade ein Bild hoch und die KI erkennt, **welche Art von Ball** darauf zu sehen ist.")

uploaded_file = st.file_uploader(
    "📸 Bild hochladen (jpg, jpeg, png)",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Hochgeladenes Bild", use_column_width=True)

    with st.spinner("🔍 Modell analysiert das Bild..."):
        img_array = preprocess_image(image)
        predictions = model.predict(img_array)
        predicted_class = np.argmax(predictions)
        confidence = predictions[0][predicted_class]

    st.success(f"🏆 Erkannter Ball: **{CLASS_NAMES[predicted_class]}**")
    st.write(f"🔢 Sicherheit: **{confidence:.2%}**")

    # Optional: Wahrscheinlichkeiten anzeigen
    st.subheader("📊 Wahrscheinlichkeiten")
    for i, class_name in enumerate(CLASS_NAMES):
        st.write(f"{class_name}: {predictions[0][i]:.2%}")
