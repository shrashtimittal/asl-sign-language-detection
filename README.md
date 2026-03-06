# 🤟 American Sign Language (ASL) Detection

## 📌 Project Overview
This project implements a deep learning based system capable of recognizing **American Sign Language (ASL) hand gestures** from images and predicting the corresponding alphabet.

The model is trained to classify hand gestures into **29 classes**, including the letters **A–Z** and three additional classes: **SPACE, DELETE, and NOTHING**.

This system demonstrates how computer vision and deep learning can be used to bridge communication gaps between hearing-impaired individuals and others.

---

## 🎯 Objective
The goal of this project is to develop a robust image classification model that can accurately detect and interpret American Sign Language hand gestures.

The system takes an input image of a hand sign and predicts the corresponding alphabet.

---

## 🗂 Dataset
The dataset used in this project contains **29 classes**:

- A–Z (26 alphabet signs)
- SPACE
- DELETE
- NOTHING

The dataset includes separate **training and testing sets** organized in folders for each class.

Due to size limitations, the dataset is **not included in this repository**.

Expected dataset structure:
dataset/
│
├── A/
├── B/
├── C/
...
├── Z/
├── SPACE/
├── DELETE/
└── NOTHING/

---

## 🧠 Approach

The workflow for this project includes:

1. **Data Exploration**
   - Visual inspection of hand gesture images
   - Understanding class distribution

2. **Data Preprocessing**
   - Image resizing
   - Normalization
   - Train-test split

3. **Model Training**
   - Deep learning based image classification model
   - Training using labeled ASL gesture images

4. **Model Evaluation**
   - Accuracy evaluation on test dataset
   - Performance analysis

5. **Deployment**
   - Streamlit web interface for real-time predictions

---

## 🛠 Technologies Used

- Python
- PyTorch
- OpenCV
- NumPy
- Matplotlib
- Streamlit
- Scikit-learn

---

## 🚀 Running the Project

### 1️⃣ Clone the Repository
git clone https://github.com/shrashtimittal/asl-sign-language-detection.git

cd asl-sign-language-detection

### 2️⃣ Install Dependencies
pip install -r requirements.txt

### 3️⃣ Run the Streamlit App
streamlit run app_streamlit.py

This will launch a local web interface where users can upload ASL images and obtain predictions.

---

## 📂 Project Structure
asl-sign-language-detection/
│
├── models/ # Trained model files
├── scripts/ # Training and preprocessing scripts
├── app_streamlit.py # Streamlit interface for predictions
├── eda_samples.png # Dataset visualization
└── README.md

---

## 📊 Example Data Visualization

The dataset exploration step includes visualization of sample ASL gesture images.

![ASL Sample Images](eda_samples.png)

---

## 🔮 Future Improvements

- Real-time sign detection using webcam
- Model performance optimization
- Deployment as a cloud application
- Integration into assistive communication systems

---

## 👩‍💻 Author

**Shrashti Mittal**

Machine Learning & AI Enthusiast  
Interested in AI, Robotics, and Aerospace Applications
