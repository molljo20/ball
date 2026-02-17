import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
import os
import sys
from pathlib import Path

# Seitenkonfiguration
st.set_page_config(
    page_title="Ball-Erkennung",
    page_icon="⚽",
    layout="centered"
)

# Titel und Beschreibung
st.title("⚽ Ball-Erkennung 🏐")
st.markdown("---")

# Sidebar mit Informationen und Debug-Info
with st.sidebar:
    st.header("ℹ️ Über diese App")
    st.write("""
    Diese App erkennt, ob auf einem hochgeladenen Bild ein **Fußball** oder **Volleyball** zu sehen ist.
    """)
    
    st.markdown("---")
    st.header("🔍 Debug-Informationen")
    
    # Aktuelles Verzeichnis
    current_dir = Path("/mount/src/ball")
    st.write(f"📂 App-Verzeichnis: `{current_dir}`")
    
    # Alle Dateien im Verzeichnis auflisten
    st.write("📋 Vorhandene Dateien:")
    try:
        files = list(current_dir.glob("*"))
        for f in files:
            size = f.stat().st_size if f.is_file() else 0
            if f.is_file():
                st.write(f"- 📄 {f.name} ({size:,} bytes)")
            else:
                st.write(f"- 📁 {f.name}/")
    except Exception as e:
        st.write(f"Fehler beim Auflisten: {e}")

# Modell-Ladefunktion mit verschiedenen Backend-Optionen
@st.cache_resource
def load_ball_model():
    """Lädt das Keras-Modell mit verschiedenen Kompatibilitätsoptionen"""
    
    model_path = Path("/mount/src/ball/keras_Model.h5")
    labels_path = Path("/mount/src/ball/labels.txt")
    
    st.sidebar.markdown("---")
    st.sidebar.header("🔎 Modell-Ladeversuch")
    
    # Prüfe ob Dateien existieren
    if not model_path.exists():
        st.sidebar.error(f"❌ Modell nicht gefunden: {model_path}")
        return None, None
    
    if not labels_path.exists():
        st.sidebar.error(f"❌ Labels nicht gefunden: {labels_path}")
        return None, None
    
    st.sidebar.success(f"✅ Dateien gefunden!")
    st.sidebar.write(f"Modell Größe: {model_path.stat().st_size:,} bytes")
    
    # Labels laden
    try:
        with open(labels_path, "r") as f:
            class_names = [line.strip() for line in f.readlines()]
        st.sidebar.success(f"✅ Labels geladen: {class_names}")
    except Exception as e:
        st.sidebar.error(f"❌ Fehler beim Laden der Labels: {e}")
        return None, None
    
    # Versuche Modell mit verschiedenen Optionen zu laden
    try:
        # Versuch 1: Normales Laden
        st.sidebar.write("Versuch 1: Normales Laden...")
        model = tf.keras.models.load_model(model_path, compile=False)
        st.sidebar.success("✅ Modell erfolgreich geladen (Versuch 1)!")
        return model, class_names
    except Exception as e1:
        st.sidebar.warning(f"Versuch 1 fehlgeschlagen: {str(e1)[:100]}...")
        
        try:
            # Versuch 2: Mit custom_objects für ältere Modelle
            st.sidebar.write("Versuch 2: Mit custom_objects...")
            model = tf.keras.models.load_model(
                model_path, 
                compile=False,
                custom_objects={'tf': tf}
            )
            st.sidebar.success("✅ Modell erfolgreich geladen (Versuch 2)!")
            return model, class_names
        except Exception as e2:
            st.sidebar.warning(f"Versuch 2 fehlgeschlagen: {str(e2)[:100]}...")
            
            try:
                # Versuch 3: Als H5-Datei mit spezifischen Einstellungen
                st.sidebar.write("Versuch 3: Mit safe_mode=False...")
                model = tf.keras.models.load_model(
                    model_path, 
                    compile=False,
                    safe_mode=False
                )
                st.sidebar.success("✅ Modell erfolgreich geladen (Versuch 3)!")
                return model, class_names
            except Exception as e3:
                st.sidebar.error(f"❌ Alle Ladeversuche fehlgeschlagen!")
                st.sidebar.error(f"Letzter Fehler: {str(e3)}")
                
                # Zeige TensorFlow Version für Debugging
                st.sidebar.write(f"TensorFlow Version: {tf.__version__}")
                st.sidebar.write(f"Keras Version: {tf.keras.__version__}")
                
                return None, None

# Modell laden
model, class_names = load_ball_model()

def preprocess_image(image):
    """Bereitet das Bild für das Modell vor"""
    # Bild in RGB konvertieren (falls es RGBA ist)
    image = image.convert("RGB")
    
    # Bild auf 224x224 zuschneiden (zentriert)
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
    
    # Bild in Numpy-Array umwandeln
    image_array = np.asarray(image)
    
    # Normalisieren
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
    
    # In die richtige Form bringen
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    data[0] = normalized_image_array
    
    return data, image

def predict_ball_type(image_data):
    """Führt die Vorhersage durch"""
    prediction = model.predict(image_data, verbose=0)
    index = np.argmax(prediction)
    class_name = class_names[index].strip()
    confidence_score = prediction[0][index]
    
    return class_name, confidence_score, index

# Hauptbereich - Datei-Upload
st.header("📤 Bild hochladen")

# Prüfe ob Modell geladen wurde
if model is None or class_names is None:
    st.error("⚠️ Modell konnte nicht geladen werden!")
    
    st.info("""
    ### 📋 Kompatibilitätsproblem:
    
    Dein Modell konnte nicht mit der aktuellen TensorFlow Version geladen werden.
    
    **Mögliche Lösungen:**
    
    1. **TensorFlow Version in requirements.txt anpassen:**
       ```
       tensorflow==2.13.0
       keras==2.13.0
       ```
    
    2. **Modell neu exportieren** mit der aktuellen Version
    
    3. **Anderes Modell-Format verwenden** (z.B. SavedModel)
    """)
    
    # Zeige TensorFlow Version
    st.write(f"📦 TensorFlow Version: {tf.__version__}")
    
else:
    uploaded_file = st.file_uploader(
        "Wähle ein Bild aus...", 
        type=["jpg", "jpeg", "png", "bmp", "webp"],
        help="Lade ein Bild mit einem Fußball oder Volleyball hoch"
    )
    
    # Wenn ein Bild hochgeladen wurde
    if uploaded_file is not None:
        # Bild anzeigen
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📸 Hochgeladenes Bild")
            image = Image.open(uploaded_file)
            st.image(image, use_container_width=True)
        
        # Bild vorverarbeiten und Vorhersage durchführen
        with st.spinner("🔍 Analysiere Bild..."):
            processed_image, original_image = preprocess_image(image)
            class_name, confidence, index = predict_ball_type(processed_image)
        
        # Ergebnis anzeigen
        with col2:
            st.subheader("🎯 Ergebnis")
            
            # Emoji basierend auf Vorhersage
            ball_emoji = "⚽" if any(keyword in class_name.lower() for keyword in ["fußball", "fussball", "football", "0"]) else "🏐"
            
            # Fortschrittsbalken für Konfidenz
            st.metric("Erkannte Ballart", f"{ball_emoji} {class_name}")
            st.progress(float(confidence))
            st.caption(f"Konfidenz: {confidence:.2%}")
            
            # Zusätzliche Informationen
            st.markdown("---")
            st.markdown("**📊 Detailierte Vorhersage:**")
            
            # Alle Klassenwahrscheinlichkeiten anzeigen
            prediction = model.predict(processed_image, verbose=0)[0]
            for i, class_label in enumerate(class_names):
                prob = prediction[i] if i < len(prediction) else 0
                clean_label = str(class_label).strip()
                emoji = "⚽" if any(keyword in clean_label.lower() for keyword in ["fußball", "fussball", "football", "0"]) else "🏐"
                st.markdown(f"{emoji} **{clean_label}:** {prob:.2%}")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>⚽ Erkenne den Unterschied zwischen Fußball und Volleyball 🏐</p>
    <p style='color: gray; font-size: 0.8em;'>Hochgeladene Bilder werden nur für die Vorhersage verwendet und nicht gespeichert.</p>
</div>
""", unsafe_allow_html=True)
