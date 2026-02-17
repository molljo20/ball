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
    current_dir = Path(__file__).parent.absolute()
    st.write(f"📂 App-Verzeichnis: `{current_dir}`")
    
    # Arbeitsverzeichnis
    work_dir = Path.cwd()
    st.write(f"📂 Arbeitsverzeichnis: `{work_dir}`")
    
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
    
    # Python-Pfad
    st.write(f"🐍 Python-Pfad: {sys.path}")

# Modell-Ladefunktion mit mehreren Suchstrategien
@st.cache_resource
def load_ball_model():
    """Lädt das Keras-Modell und die Labels mit verschiedenen Suchstrategien"""
    
    # Verschiedene mögliche Pfade
    possible_paths = []
    
    # 1. Aktuelles Verzeichnis der Python-Datei
    current_dir = Path(__file__).parent.absolute()
    possible_paths.append(current_dir)
    
    # 2. Arbeitsverzeichnis
    possible_paths.append(Path.cwd())
    
    # 3. Direkt im Hauptverzeichnis (für Streamlit Cloud)
    possible_paths.append(Path("/mount/src/ball"))
    
    # 4. Im selben Verzeichnis wie das Skript
    possible_paths.append(Path(__file__).parent)
    
    st.sidebar.markdown("---")
    st.sidebar.header("🔎 Modell-Suche")
    
    model = None
    class_names = None
    found_model = False
    found_labels = False
    
    for path in possible_paths:
        model_path = path / "keras_Model.h5"
        labels_path = path / "labels.txt"
        
        st.sidebar.write(f"Suche in: {path}")
        
        if model_path.exists() and not found_model:
            st.sidebar.success(f"✅ Modell gefunden: {model_path}")
            found_model = True
            try:
                model = tf.keras.models.load_model(str(model_path), compile=False)
                st.sidebar.success("✅ Modell erfolgreich geladen!")
            except Exception as e:
                st.sidebar.error(f"❌ Fehler beim Laden: {e}")
                model = None
        
        if labels_path.exists() and not found_labels:
            st.sidebar.success(f"✅ Labels gefunden: {labels_path}")
            found_labels = True
            try:
                with open(labels_path, "r") as f:
                    class_names = [line.strip() for line in f.readlines()]
                st.sidebar.success(f"✅ Labels geladen: {class_names}")
            except Exception as e:
                st.sidebar.error(f"❌ Fehler beim Laden der Labels: {e}")
                class_names = None
        
        if found_model and found_labels:
            break
    
    if not found_model:
        st.sidebar.error("❌ keras_Model.h5 nicht gefunden!")
    
    if not found_labels:
        st.sidebar.error("❌ labels.txt nicht gefunden!")
    
    return model, class_names

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

# Hauptbereich - Datei-Upload
st.header("📤 Bild hochladen")

# Prüfe ob Modell geladen wurde
if model is None or class_names is None:
    st.error("⚠️ Modell konnte nicht geladen werden!")
    
    st.info("""
    ### 📋 Mögliche Lösungen:
    
    1. **Repository auf GitHub prüfen:**
       ```bash
       # Überprüfe, ob die Dateien wirklich da sind:
       ls -la /mount/src/ball/
       ```
    
    2. **Manuell nachsehen:** Gehe zu deinem GitHub-Repository und prüfe:
       - [ ] `keras_Model.h5` ist vorhanden
       - [ ] `labels.txt` ist vorhanden
       - [ ] Die Dateinamen sind **exakt** gleich (Groß-/Kleinschreibung!)
    
    3. **Dateien neu hochladen:**
       ```bash
       git add keras_Model.h5 labels.txt
       git commit -m "Add model files"
       git push
       ```
    
    4. **In Streamlit Cloud:** 
       - Gehe zu "Manage app" → "Reboot" (neu starten)
       - Prüfe die Logs auf spezifische Fehler
    """)
    
    # Zeige detaillierte Info
    st.markdown("---")
    st.subheader("📊 Detaillierte System-Info:")
    
    # Versuche direkt auf Dateien zuzugreifen
    try:
        base_path = Path("/mount/src/ball")
        st.write(f"Inhalt von {base_path}:")
        if base_path.exists():
            for item in base_path.iterdir():
                st.write(f"- {item.name}")
        else:
            st.write("❌ Pfad nicht gefunden!")
    except Exception as e:
        st.write(f"Fehler: {e}")

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
            ball_emoji = "⚽" if any(keyword in class_name.lower() for keyword in ["fußball", "fussball", "football"]) else "🏐"
            
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
                emoji = "⚽" if any(keyword in clean_label.lower() for keyword in ["fußball", "fussball", "football"]) else "🏐"
                st.markdown(f"{emoji} **{clean_label}:** {prob:.2%}")

def predict_ball_type(image_data):
    """Führt die Vorhersage durch"""
    prediction = model.predict(image_data, verbose=0)
    index = np.argmax(prediction)
    class_name = class_names[index].strip()
    confidence_score = prediction[0][index]
    
    return class_name, confidence_score, index

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>⚽ Erkenne den Unterschied zwischen Fußball und Volleyball 🏐</p>
    <p style='color: gray; font-size: 0.8em;'>Hochgeladene Bilder werden nur für die Vorhersage verwendet und nicht gespeichert.</p>
</div>
""", unsafe_allow_html=True)
