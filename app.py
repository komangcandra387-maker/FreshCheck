from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
from PIL import Image
from pathlib import Path
import numpy as np
import io

app = Flask(__name__)

# =========================================================
# CONFIG
# =========================================================

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "freshcheck_model.keras"

IMG_SIZE = (160, 160)

# Confidence minimum untuk kelas utama
CONFIDENCE_THRESHOLD = 70.0

# =========================================================
# LOAD MODEL
# =========================================================

print()
print("==========================================")
print("        FRESHCHECK - LOAD MODEL")
print("==========================================")
print("Model :", MODEL_PATH)

model = load_model(MODEL_PATH)

OUTPUT_COUNT = model.output_shape[-1]

print("Input  :", model.input_shape)
print("Output :", model.output_shape)

# =========================================================
# 7 CLASS MODEL
# =========================================================

class_names = [
    "freshapples",
    "freshbanana",
    "freshoranges",
    "other",
    "rottenapples",
    "rottenbanana",
    "rottenoranges"
]

print()
print("==========================================")
print("KELAS MODEL")
print("==========================================")

for i, name in enumerate(class_names):
    print(f"{i} -> {name}")

print("Jumlah kelas :", len(class_names))

# =========================================================
# DATA BUAH
# =========================================================

buah_data = {
    "Apple": {
        "nama": "Apel",
        "kalori": 52,
        "karbohidrat": 13.8,
        "protein": 0.3,
        "lemak": 0.2,
        "serat": 2.4
    },
    "Banana": {
        "nama": "Pisang",
        "kalori": 89,
        "karbohidrat": 22.8,
        "protein": 1.1,
        "lemak": 0.3,
        "serat": 2.6
    },
    "Orange": {
        "nama": "Jeruk",
        "kalori": 47,
        "karbohidrat": 11.8,
        "protein": 0.9,
        "lemak": 0.1,
        "serat": 2.4
    }
}

# =========================================================
# MAPPING
# =========================================================

class_mapping = {
    "freshapples": {
        "fruit": "Apple",
        "condition": "Fresh"
    },
    "freshbanana": {
        "fruit": "Banana",
        "condition": "Fresh"
    },
    "freshoranges": {
        "fruit": "Orange",
        "condition": "Fresh"
    },
    "rottenapples": {
        "fruit": "Apple",
        "condition": "Rotten"
    },
    "rottenbanana": {
        "fruit": "Banana",
        "condition": "Rotten"
    },
    "rottenoranges": {
        "fruit": "Orange",
        "condition": "Rotten"
    }
}

# =========================================================
# NUTRISI
# =========================================================

def get_nutrition(fruit_key):
    info = buah_data.get(fruit_key)
    if info is None:
        return {
            "kalori": 0,
            "karbohidrat": 0,
            "protein": 0,
            "lemak": 0,
            "serat": 0
        }
    return {
        "kalori": info["kalori"],
        "karbohidrat": info["karbohidrat"],
        "protein": info["protein"],
        "lemak": info["lemak"],
        "serat": info["serat"]
    }

# =========================================================
# PREDIKSI GAMBAR
# =========================================================

def predict_image(image):
    image = image.convert("RGB")
    image = image.resize(
        IMG_SIZE,
        Image.Resampling.LANCZOS
    )

    img_array = np.array(
        image,
        dtype=np.float32
    )
    img_array = np.expand_dims(
        img_array,
        axis=0
    )

    prediction = model.predict(
        img_array,
        verbose=0
    )[0]

    index = int(np.argmax(prediction))
    confidence = float(prediction[index] * 100)
    label = class_names[index]

    if label == "other":
        return {
            "error": (
                "Buah tidak didukung. "
                "FreshCheck hanya mendukung "
                "Apel, Pisang, dan Jeruk."
            ),
            "confidence": round(confidence, 2),
            "label": label
        }

    if confidence < CONFIDENCE_THRESHOLD:
        return {
            "error": (
                "Buah tidak dapat dikenali dengan cukup yakin. "
                "Silakan gunakan foto Apel, Pisang, atau Jeruk "
                "yang lebih jelas."
            ),
            "confidence": round(confidence, 2)
        }

    mapping = class_mapping.get(label)
    if mapping is None:
        return {
            "error": f"Label tidak dikenali: {label}"
        }

    fruit_key = mapping["fruit"]
    kondisi = mapping["condition"]

    info = buah_data.get(fruit_key)
    if info is None:
        return {
            "error": f"Data buah tidak ditemukan: {fruit_key}"
        }

    if kondisi == "Fresh":
        status = "Fresh"
        masa_simpan = "2-5 hari"
        rekomendasi = "Buah terdeteksi masih segar dan dapat dikonsumsi."
    else:
        status = "Rotten"
        masa_simpan = "Tidak dapat disimpan"
        rekomendasi = "Buah terdeteksi sudah busuk dan tidak disarankan untuk dikonsumsi."

    nutrition = get_nutrition(fruit_key)

    return {
        "buah": info["nama"],
        "status": status,
        "confidence": round(confidence, 2),
        "score": round(confidence),
        "kalori": nutrition["kalori"],
        "karbohidrat": nutrition["karbohidrat"],
        "protein": nutrition["protein"],
        "lemak": nutrition["lemak"],
        "serat": nutrition["serat"],
        "masa_simpan": masa_simpan,
        "rekomendasi": rekomendasi,
        "label": label,
        "model_output": OUTPUT_COUNT,
        "class_count": len(class_names)
    }

# =========================================================
# HOME (Render index.html)
# =========================================================

@app.route("/")
def home():
    return render_template("index.html")

# =========================================================
# PREDICT API
# =========================================================

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({
            "error": "Tidak ada gambar yang dikirim."
        }), 400

    file = request.files["image"]

    if file.filename == "":
        return jsonify({
            "error": "File gambar belum dipilih."
        }), 400

    try:
        image_bytes = file.read()
        image = Image.open(io.BytesIO(image_bytes))
        hasil = predict_image(image)
        return jsonify(hasil)

    except Exception as e:
        return jsonify({
            "error": f"Gagal memproses gambar: {str(e)}"
        }), 500

# =========================================================
# STATUS MODEL
# =========================================================

@app.route("/status")
def model_status():
    return jsonify({
        "server": "online",
        "model": "FreshCheck",
        "input": model.input_shape,
        "output": OUTPUT_COUNT,
        "class_count": len(class_names),
        "classes": class_names,
        "confidence_threshold": CONFIDENCE_THRESHOLD
    })

# =========================================================
# RUN SERVER
# =========================================================

if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )
