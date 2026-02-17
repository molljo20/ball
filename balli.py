import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import zipfile
import os

# -------------------------------
# App Titel
# -------------------------------
st.title("🏐⚽ Ball-Erkennung")
st.write("Lade ein Bild hoch und das Modell erkennt, ob es ein **Fußball** oder **Volleyball** ist.")

# -------------------------------
# Modell entpacken (einmalig)
# -------------------------------
MODEL_ZIP_PATH = "converted_keras.zip"
MODEL_DIR = "model"

if not os.path.exists(MODEL_DIR):
    with zipfile.ZipFile(MODEL_ZIP_PATH, "r") as zip_ref:
        zip_ref.extractall(MODEL_DIR)

# -------------------------------
# Modell laden
# -------------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_DIR)

model = load_model()

# Klassen (ANPASSEN falls nötig!)
CLASS_NAMES = ["Fußball", "Volleyball"]

# -------------------------------
# Bild-Upload
# -------------------------------
uploaded_file = st.file_uploader(
    "📷 Bild hochladen",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    # Bild anzeigen
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Hochgeladenes Bild", use_column_width=True)

    # -------------------------------
    # Bild vorbereiten
    # -------------------------------
    IMG_SIZE = (224, 224)  # an dein Modell anpassen!
    img = image.resize(IMG_SIZE)
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # -------------------------------
    # Vorhersage
    # -------------------------------
    prediction = model.predict(img_array)
    class_index = np.argmax(prediction)
    confidence = prediction[0][class_index]

    # -------------------------------
    # Ergebnis anzeigen
    # -------------------------------
    st.subheader("🔍 Ergebnis")
    st.write(f"**Erkannter Ball:** {CLASS_NAMES[class_index]}")
    st.write(f"**Sicherheit:** {confidence * 100:.2f} %")
