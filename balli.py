import streamlit as st
from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np
import os
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

# Sidebar mit Informationen
with st.sidebar:
    st.header("ℹ️ Über diese App")
    st.write("""
    Diese App erkennt, ob auf einem hochgeladenen Bild ein **Fußball** oder **Volleyball** zu sehen ist.
    
    **So funktioniert's:**
    1. Lade ein Bild hoch (JPG, PNG, etc.)
    2. Die KI analysiert das Bild
    3. Du erhältst das Ergebnis mit Konfidenzwert
    """)
    
    st.markdown("---")
    st.header("📊 Modell-Info")
    
    # Zeige an, wo die Dateien gefunden werden
    current_dir = os.path.dirname(os.path.abspath(__file__))
    st.write(f"Aktuelles Verzeichnis: `{current_dir}`")
    
    # Prüfe, welche Dateien existieren
    files = os.listdir(current_dir)
    st.write("Gefundene Dateien:")
    for file in files:
        st.write(f"- {file}")

# Überprüfen, ob Modell-Dateien existieren
@st.cache_resource
def load_ball_model():
    """Lädt das Keras-Modell und die Labels"""
    try:
        # Absoluten Pfad zur Datei ermitteln
        current_dir = Path(__file__).parent.absolute()
        model_path = current_dir / "keras_Model.h5"
        labels_path = current_dir / "labels.txt"
        
        st.sidebar.write(f"📁 Versuche Modell zu laden von: {model_path}")
        
        # Prüfe ob Dateien existieren
        if not model_path.exists():
            st.sideki.error(f"❌ Modell nicht gefunden: {model_path}")
            # Zeige alle Dateien im Verzeichnis
            files = list(current_dir.glob("*"))
            st.sidebar.write("Vorhandene Dateien:")
            for f in files:
                st.sidebar.write(f"- {f.name}")
            return None, None
            
        if not labels_path.exists():
            st.sidebar.error(f"❌ Labels nicht gefunden: {labels_path}")
            return None, None
        
        # Modell laden
        model = load_model(str(model_path), compile=False)
        with open(labels_path, "r") as f:
            class_names = f.readlines()
        
        st.sidebar.success("✅ Modell erfolgreich geladen!")
        return model, class_names
        
    except Exception as e:
        st.sidebar.error(f"❌ Fehler beim Laden: {str(e)}")
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
    ### 📋 Problembehandlung:
    1. Stelle sicher, dass folgende Dateien im Repository sind:
       - `keras_Model.h5`
       - `labels.txt`
    
    2. Die Dateien müssen im **Hauptverzeichnis** liegen (nicht in einem Unterordner)
    
    3. Überprüfe die Dateinamen (Groß-/Kleinschreibung):
       - `keras_Model.h5` (nicht `keras_model.h5`)
       - `labels.txt` (nicht `Labels.txt`)
    
    4. In Streamlit Cloud:
       - Gehe zu "Manage app" → "Logs" für detaillierte Fehlermeldungen
    """)
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
            ball_emoji = "⚽" if "fußball" in class_name.lower() or "fussball" in class_name.lower() or "football" in class_name.lower() else "🏐"
            
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
                prob = prediction[i]
                clean_label = class_label.strip()
                emoji = "⚽" if "fußball" in clean_label.lower() or "fussball" in clean_label.lower() or "football" in clean_label.lower() else "🏐"
                st.markdown(f"{emoji} **{clean_label}:** {prob:.2%}")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>⚽ Erkenne den Unterschied zwischen Fußball und Volleyball 🏐</p>
    <p style='color: gray; font-size: 0.8em;'>Hochgeladene Bilder werden nur für die Vorhersage verwendet und nicht gespeichert.</p>
</div>
""", unsafe_allow_html=True)
