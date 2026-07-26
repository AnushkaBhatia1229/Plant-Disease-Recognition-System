# 🌿 Plant Disease Recognition System

![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python)
![Flask](https://img.shields.io/badge/Flask-Web%20Application-black?logo=flask)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Deep%20Learning-orange?logo=tensorflow)
![Keras](https://img.shields.io/badge/Keras-Neural%20Networks-red?logo=keras)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen)

---

# 🌱 Plant Disease Recognition System

An AI-powered web application that detects plant diseases from leaf images using **Deep Learning** and **TensorFlow**. The application helps farmers, agriculture researchers, students, and gardening enthusiasts identify diseases quickly and provides detailed information including symptoms, causes, treatment, and prevention methods.

---

# 📖 Table of Contents

- About the Project
- Features
- Tech Stack
- Project Structure
- Installation
- Usage
- How It Works
- Project Screenshots
- Future Improvements
- Contributing
- Author
- License

---

# 📌 About the Project

Plant diseases significantly affect crop production and agricultural productivity. Early detection can reduce crop losses and improve farming efficiency.

This project uses a trained TensorFlow deep learning model to classify plant leaf diseases from uploaded images. After prediction, the system displays comprehensive disease information stored in a JSON database.

The application provides an easy-to-use web interface built with Flask and HTML/CSS.

---

# ✨ Features

✅ Upload plant leaf images

✅ AI-powered disease prediction

✅ Deep Learning model using TensorFlow & Keras

✅ Disease name prediction

✅ Symptoms information

✅ Disease causes

✅ Treatment recommendations

✅ Prevention tips

✅ Responsive and clean user interface

✅ Fast prediction results

---

# 🛠 Tech Stack

## Frontend

- HTML5
- CSS3

## Backend

- Python
- Flask

## AI / Machine Learning

- TensorFlow
- Keras
- NumPy
- Pillow

## Database

- JSON

---

# 📂 Project Structure

```text
Plant-Disease-Recognition-System/
│
├── app.py
├── requirements.txt
├── plant_disease.json
├── plant_disease_model.keras
├── README.md
│
├── templates/
│   └── home.html
│
├── static/
│   ├── css/
│   └── images/
│
├── uploads/
├── uploadimages/
└── screenshots/
    ├── homepage.png
    ├── upload.png
    ├── prediction.png
    └── result2.png
```

---

# ⚙️ Installation

## Clone Repository

```bash
git clone https://github.com/AnushkaBhatia1229/Plant-Disease-Recognition-System.git
```

---

## Move to Project Folder

```bash
cd Plant-Disease-Recognition-System
```

---

## Install Required Packages

```bash
pip install -r requirements.txt
```

---

## Run the Application

```bash
python app.py
```

---

## Open Browser

```
http://127.0.0.1:5000
```

---

# 🚀 Usage

1. Open the application in your browser.

2. Upload a plant leaf image.

3. Click **Predict**.

4. The AI model analyzes the image.

5. The predicted disease is displayed.

6. Read:

- Disease Name
- Symptoms
- Causes
- Treatment
- Prevention

---

# 🧠 How It Works

```text
User Uploads Image
        │
        ▼
Image Preprocessing
        │
        ▼
TensorFlow Deep Learning Model
        │
        ▼
Disease Prediction
        │
        ▼
JSON Database
        │
        ▼
Disease Information Display
```

---

# 📸 Project Screenshots

## 🏠 Home Page

> Save the screenshot as:<img width="1920" height="1200" alt="Screenshot (1379)" src="https://github.com/user-attachments/assets/8eb2ef1c-49a9-4bd0-beed-2072720f0ed9" />


```


```

<img src="screenshots/homepage.png" width="900">

---

## 📤 Upload Image

> Save the screenshot as:

```
<img width="1920" height="1200" alt="image" src="https://github.com/user-attachments/assets/d25021ab-0ba6-4271-a28f-c6f29132bd90" />

```

<img src="screenshots/upload.png" width="900">

---

## 🌿 Prediction Result

> Save the screenshot as:

```
<img width="1920" height="1200" alt="Screenshot (1378)" src="https://github.com/user-attachments/assets/0a7ee89c-111b-481e-b5fb-c869126b9181" />

```

<img src="screenshots/prediction.png" width="900">

---

## 🤖 Disease Information

> Save the screenshot as:

```
<img width="1920" height="1200" alt="Screenshot (1381)" src="https://github.com/user-attachments/assets/e2f2034e-8d61-47f8-9254-604495f866ba" />
<img width="1920" height="1200" alt="Screenshot (1380)" src="https://github.com/user-attachments/assets/31d452b6-d25f-4751-b881-50f1198de0c6" />

```

<img src="screenshots/result2.png" width="900">

---

# 💡 Example Workflow

```
Leaf Image
      │
      ▼
Upload
      │
      ▼
Prediction
      │
      ▼
Disease Information
      │
      ▼
Treatment & Prevention
```

---

# 📦 Requirements

- Python 3.12+
- Flask
- TensorFlow
- Keras
- Pillow
- NumPy

Install dependencies using:

```bash
pip install -r requirements.txt
```

---

# 📈 Future Improvements

- 🌐 Multi-language Support

- 📱 Mobile Responsive UI

- 📷 Real-time Camera Detection

- 📊 Prediction Confidence Score

- ☁ Cloud Deployment

- 📄 PDF Report Generation

- 🌾 More Crop Disease Classes

- 📉 Prediction History

- 🔔 Smart Farming Recommendations

---

# 🤝 Contributing

Contributions are welcome!

### Fork Repository

```bash
git fork
```

### Create New Branch

```bash
git checkout -b feature-name
```

### Commit Changes

```bash
git commit -m "Added new feature"
```

### Push Changes

```bash
git push origin feature-name
```

### Create Pull Request

---

# 👩‍💻 Author

## **Anushka Bhatia**

🎓 BCA (Artificial Intelligence & Machine Learning)

🏫 Sharda University

💻 AI | Machine Learning | Data Science Enthusiast

🌱 Passionate about building AI solutions for real-world problems.

---

# 🌟 Support

If you found this project useful,

⭐ Please Star this repository.

It motivates future development and helps others discover the project.

# ❤️ Thank You

Thank you for visiting this repository.

If you like this project,

⭐ Star the repository

🍴 Fork it

💡 Share your feedback

Happy Coding! 🚀
