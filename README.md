# 🍎 FreshCheck — Fruit Freshness Detection

**FreshCheck** is a machine learning-based application designed to detect fruit freshness using image classification. It helps users identify whether an apple, banana, or orange is fresh or rotten by analyzing an image of the fruit.

The project combines an image classification model with a user-friendly interface to make fruit freshness detection simple and accessible.

## ✨ Features

- **Fruit Freshness Detection** — Classifies fruit images as fresh or rotten.
- **Multiple Fruit Classes** — Supports apples, bananas, and oranges.
- **Image Upload** — Upload an image to analyze fruit freshness.
- **Webcam Detection** — Supports real-time detection if webcam functionality is enabled.
- **Nutrition Calculator** — Provides nutritional information for supported fruits.
- **Interactive Interface** — Displays detection results and freshness status.

## 🧠 Machine Learning Model

FreshCheck uses a trained deep learning image classification model built with TensorFlow and Keras.

The model recognizes six classes:

| Class | Description |
|---|---|
| `freshapple` | Fresh apple |
| `freshbanana` | Fresh banana |
| `freshorange` | Fresh orange |
| `rottenapple` | Rotten apple |
| `rottenbanana` | Rotten banana |
| `rottenorange` | Rotten orange |

**Model file:** `freshcheck_model.keras`

## 🛠️ Tech Stack

- **Python** — Machine learning development
- **TensorFlow & Keras** — Model training and inference
- **HTML** — Web page structure
- **CSS** — Interface styling
- **JavaScript** — Frontend interactions

## 📂 Project Structure

The project structure may vary depending on the version of the repository.

```text
FreshCheck/
├── freshcheck_model.keras
├── train/
├── test/
├── index.html
├── style.css
├── script.js
└── README.md
```

*Note: Update the file and folder names above to match your actual repository.*

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR-USERNAME/YOUR-REPOSITORY.git
```

Replace the repository URL with your own GitHub repository URL.

### 2. Navigate to the Project

```bash
cd YOUR-REPOSITORY
```

### 3. Install Dependencies

If you are running the Python model locally, install the required dependencies:

```bash
pip install tensorflow numpy pillow
```

Additional dependencies may be required depending on the implementation.

### 4. Run the Application

Open `index.html` in your browser if the frontend is a standalone web application.

If the project uses a Python backend for model inference, start the backend using the appropriate command for your application.

## 📊 How It Works

1. The user uploads or captures an image of a fruit.
2. The image is prepared for the model.
3. The trained classification model analyzes the image.
4. The application displays the predicted fruit class and freshness status.
5. Additional nutritional information can be displayed when supported.

The accuracy of predictions depends on the training data, image quality, and model performance.

## 🎯 Project Goals

FreshCheck aims to explore how machine learning and computer vision can help users recognize fruit freshness through an accessible digital application.

## 🔮 Future Improvements

- Improve model accuracy with a more diverse dataset.
- Display prediction confidence scores.
- Expand support for additional fruit types.
- Improve mobile responsiveness.
- Enhance real-time camera detection.

## 👨‍💻 Developer

Developed as a machine learning and web development project.

**Project:** FreshCheck  
**Category:** Machine Learning · Computer Vision · Web Development

---

*FreshCheck — Making fruit freshness detection smarter and simpler.*
