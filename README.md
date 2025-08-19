# Eyesense

An API for ocular disease detection from fundus images using a deep learning model.

<p align="center">
  <!-- TODO: Add a GIF or screenshot of your application in action. -->
  <!-- <img src="link-to-your-gif-or-screenshot.gif" alt="Eyesense Demo" width="600"/> -->
</p>

---

## � Project Overview

Eyesense is a machine learning project that leverages a pre-trained **Xception** model to analyze eye fundus images and predict the likelihood of various ocular diseases. The project is exposed as a RESTful API built with **FastAPI**.

The primary goal is to provide a tool that can serve as a preliminary diagnostic aid, capable of identifying conditions such as cataracts, glaucoma, diabetes-related retinopathy, and more.

### ✨ Features

- **AI-Powered Predictions**: Utilizes a deep learning model for accurate disease classification.
- **RESTful API**: A clean, fast, and documented API for easy integration.
- **Dockerized**: Containerized for consistent deployment and scalability.
- **Cloud-Integrated**: Downloads the trained model from Google Cloud Storage on startup.

---

## �️ Tech Stack

- **Backend**: FastAPI, Uvicorn
- **ML/DL**: TensorFlow, Keras, Scikit-learn, Numpy
- **Image Processing**: OpenCV, Pillow
- **Deployment**: Docker, Google Cloud Storage
- **Testing**: Pytest, HTTPX
- **Linting**: Flake8

---

## 🚀 Getting Started

These instructions will help you set up a local copy of the project for development and testing.

### 📋 Prerequisites

- Python 3.10
- Docker
- Google Cloud SDK (for authentication to GCS)

---

### 🔧 Local Installation (API)

Follow these steps to set up the development environment:

1️⃣ **Clone the repository:**
```bash
git clone https://github.com/caazzi/eyesense.git
cd eyesense
```

2️⃣ **(Optional) Create a virtual environment:**
```bash
python -m venv venv
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate     # Windows
```

3️⃣ **Install dependencies:**
```bash
pip install -r requirements.txt
```

4️⃣ **Run the Streamlit app:**
```bash
streamlit run app/app.py
```

The application will open in your default browser.

---

## ⚙️ Running the Tests

To ensure the project is working correctly, run the test scripts using:

```bash
pytest tests/
```

---

### 🔩 End-to-End Tests

These tests verify the entire workflow, from uploading an image to receiving a prediction.

Example:
```python
def test_prediction():
    img = load_test_image("data/example_images/eye.jpg")
    result = model.predict(img)
    assert result is not None
```

---

### ⌨️ Code Style Tests

To maintain clean and consistent code formatting, use **Flake8**:

```bash
flake8 app/ --max-line-length=120
```

---

## 📦 Deployment

To deploy the application on a live server, follow these steps:

1. Ensure all dependencies are installed in the production environment.
2. Run `streamlit run app/app.py` inside a cloud server or container.
3. Configure Streamlit sharing or deploy using services like **Heroku**, **AWS**, or **Google Cloud**.

---

## 🛠️ Built With

- **TensorFlow/Keras** - Machine learning framework
- **Streamlit** - Web application framework
- **OpenCV** - Image processing library

---

## 🖇️ Contributing

Please read `COLLABORATION.md` for details on our code of conduct and submission guidelines.

---

## ✒️ Authors

**Claudio Azzi**   -     [@caazzi](https://github.com/caazzi)

**Erika Chang**    -     [@erika-chang](https://github.com/erika-chang)

**George Silva**   -     [@gbs1234](https://github.com/gbs1234)

**Joao Sales**     -     [@masalesvic](https://github.com/masalesvic)

---

## 📄 License

This project is licensed under the MIT License - see the `LICENSE.md` file for details.

---

## 🎁 Acknowledgments

- Thanks to the **open-source community** for their amazing tools 🔧
- Shoutout to **researchers** advancing AI in healthcare 📢
- Special thanks to **contributors & testers** 🫂


