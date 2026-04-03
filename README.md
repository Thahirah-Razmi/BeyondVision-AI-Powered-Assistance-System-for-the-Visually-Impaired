# 🦯 BeyondVision: AI-Powered Assistance System for the Visually Impaired

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-3.0.0-green.svg)](https://flask.palletsprojects.com/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-red.svg)](https://github.com/ultralytics/ultralytics)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 📌 Overview

BeyondVision is a web-based AI-powered assistance system designed to enhance the independence and safety of visually impaired individuals. The system integrates computer vision, natural language processing, and web technologies to provide real-time environmental awareness, intelligent navigation, and accessible text recognition through an intuitive, voice-enhanced interface.

**Fine-tuned YOLOv8 model achieved 99.39% mAP@50 and 98.43% F1 score for blind track detection.**

---

## 📌 Table of Contents
- [Features](#-features)
- [Tech Stack](#-tech-stack)
- [Architecture](#-architecture)
- [Model Performance](#-model-performance)
- [Getting Started](#-getting-started-installation)
- [User Guide](#-user-guide)
- [Visual Previews](#-visual-previews)
- [Author](#-author)
- [License](#-license)

---

## ✨ Features

### For Visually Impaired Users (Blind End)

**Blind Track Detection & Direction Analysis**
- Upload video for frame-by-frame analysis using fine-tuned YOLOv8
- Slope-based direction algorithm (±0.2 radian threshold) detects left/right turns
- LLM-generated natural language voice prompts (Qwen2.5:3b via Ollama)
- Real-time speech synthesis via pyttsx3

**Intelligent Navigation System**
- GPS integration with OpenStreetMap for route planning
- Safety scoring algorithm (0-100) considering weather, time of day, and distance
- Turn-by-turn voice guidance
- OSRM API for walking route calculation

**OCR Text Recognition**
- Tesseract OCR for extracting text from images (JPEG, PNG, GIF, BMP)
- Automatic text-to-speech conversion
- Support for printed text and documents

**Environment Risk Assessment**
- Real-time weather data from Open-Meteo API
- Light level calculation based on time and cloud cover
- Four-tier risk classification: Safe, Moderate, Risky, Dangerous
- Context-specific safety recommendations

**Voice Command System**
- Web Speech API for hands-free control
- Commands: destination setting, OCR activation, obstacle detection, help, stop, location check, weather

### For Caretakers (Family End)

**Real-time Monitoring Dashboard**
- Live user location tracking on Leaflet.js map
- Active/inactive status indicators (5-minute timeout)

**Analytics & Reporting**
- Obstacle detection statistics (left/right turn counts, avg confidence)
- Environment risk distribution charts
- OCR usage trends (7-day line chart)
- Daily activity feed from all log tables

**Alert & Messaging System**
- Risk alert notifications with color-coded severity
- Two-way text-to-voice messaging (caretaker → user)
- Server-Sent Events (SSE) for real-time updates

### Security & Authentication
- Secure registration with email verification (SMTP)
- SHA-256 password hashing
- Flask-Login session management
- Role-based access control (user/caretaker)

---

## 💻 Tech Stack

| Category           | Technology                                          |
|--------------------|-----------------------------------------------------|
| Backend            | Python, Flask                                       |
| Frontend           | HTML5, CSS3, JavaScript, Jinja2 templates          |
| Database           | MySQL                                               |
| Computer Vision    | YOLOv8 (Ultralytics), OpenCV, Pillow               |
| OCR                | Tesseract (pyesseract)                             |
| NLP / LLM          | Ollama (Qwen2.5:3b)                                 |
| Text-to-Speech     | pyttsx3                                             |
| Mapping            | Leaflet.js, OpenStreetMap, OSRM API                |
| Weather API        | Open-Meteo API                                      |
| Geolocation        | Web Geolocation API, Nominatim (geocoding)         |
| Real-time Comms    | Server-Sent Events (SSE)                            |
| Concurrency        | threading, concurrent.futures                      |

---

## 🏗️ Architecture

The system follows a **Model-View-Controller (MVC)** pattern with a three-tier architecture:

<img width="1206" height="826" alt="image" src="https://github.com/user-attachments/assets/48027caa-2940-48ef-ac47-e6c89ee0e89a" />

---

### Design Patterns Implemented
- **Decorator**: `@login_required`, `@caretaker_required` route protection
- **Singleton**: YOLO model (loaded once, reused globally)
- **Repository**: All database access functions
- **Generator/Factory**: Video frame streaming (`generate_frames()`)
- **Strategy**: Voice speed/volume selection; LLM prompt personalisation
- **Observer**: Server-Sent Events for real-time speech text
- **Facade**: `/environment` route hiding multi-step weather/risk logic
- **Thread-per-Task**: Non-blocking speech synthesis and LLM calls
- **Null Object/Fallback**: Graceful API failure handling

---

## 📊 Model Performance

### YOLOv8n Fine-Tuning Results (120 epochs)

| Metric              | Best Value | Best Epoch | Final (Epoch 120) |
|---------------------|------------|------------|-------------------|
| Precision           | 99.79%     | 107        | 99.08%            |
| Recall              | 98.42%     | 82         | 97.79%            |
| mAP@50              | 99.48%     | 107        | 99.39%            |
| mAP@50-95           | 92.24%     | 104        | 91.32%            |
| F1 Score (calc.)    | —          | —          | 98.43%            |

---

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- MySQL Server
- Ollama with qwen2.5:3b model
- Tesseract OCR installed locally

1. Clone repository

```
git clone https://github.com/Thahirah-Razmi/BeyondVision-AI-Powered-Assistance-System-for-the-Visually-Impaired.git 
cd BeyondVision-AI-Powered-Assistance-System-for-the-Visually-Impaired
```

2. Create virtual environment

```
# Create virtual environment
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

3. Install dependencies

```
pip install -r requirements.txt
```

4. Install Ollama and pull model

```
# Visit https://ollama.com/ to download and install Ollama for your OS

# After installation, pull the qwen2.5:3b model
ollama pull qwen2.5:3b

# Verify the model is installed
ollama list
```
Make sure Ollama service is running on `http://localhost:11434` (default port).

5. Database Setup
The application automatically creates the MySQL database and all required tables on the first run. No manual setup is needed.

7. Run application

```
python BeyondVision.py
```

The application will run at http://127.0.0.1:5000/ 

## 📖 User Guide

### Blind Track Video Analysis
1. Click "Select Video File" → choose MP4/AVI/MOV/MKV/WEBM (max 300MB)
2. "Start Analysis" runs automatically → video processes frame-by-frame
3. Listen for voice prompts: "Please turn left" / "Please turn right" / "Continue straight ahead"

### Location & Navigation
1. Location is detected automatically. If it doesn't work, click "Locate My Position" for GPS location (or select "Manually Select Current Location" via More menu)
2. Search for a destination by typing in the "Enter destination" field or (or select "Manually Select Current Location" on map via More menu)
3. Click “Locate and Start Navigation” to display the route and receive turn-by-turn directions with voice instructions.
4. Click "More" → "Plan Route" → displays route and turn-by-turn directions
6. Click "Start Navigation" for safety score and voice-guided navigation
7. Click "Stop" to cancel the current navigation.”
8. Click “Show Directions” to display the route.
9. Click "More" → "Find Nearby Blind Tracks" to view blind tracks nearby

### Smart OCR Text Reader
1. Click "Select Image File" → upload JPEG/PNG/GIF/BMP
2.“Read Text from Image” runs automatically → text is extracted and spoken aloud.

### Environment Safety Check
1. Click "Check Environment Safety" → real-time weather, light level, risk score displayed and spoken aloud with feedback.
2. Click "Auto Check" for automatic 5-minute updates

### Voice Commands (Click microphone button)
| Command              | Action                              |
|----------------------|-------------------------------------|
| "Go to [place]"      | Search and set destination          |
| "Read"               | Open OCR file selector for text recognition              |
| "Detect"             | Open video file selector for blind track detection            |
| "Help"               | Open user guide                     |
| "Where am I"         | Announce current address            |
| "What's the weather" | Trigger environment check           |
| "Repeat"             | Repeat last instruction             |
| "Stop"               | Cancel current operation            |

### Caretaker Features
- Switch to "Family end" mode in Settings → message input appears
- Type message and Send → spoken aloud to blind user in real-time
- Access `/caretaker_dashboard` for analytics and live location monitoring

---

## 📸 Visual Previews

### Login Page
![Login Page] <img width="1181" height="627" alt="image" src="https://github.com/user-attachments/assets/bce307be-9bcc-4643-bd33-ca8b82eda6e6" />

### Registration Page

![Restration Page] <img width="1180" height="791" alt="image" src="https://github.com/user-attachments/assets/32e69a7c-8cf9-47df-b6e0-ab05785eb45e" />

###Forget Password Page

![Forget Password Page] <img width="1181" height="658" alt="image" src="https://github.com/user-attachments/assets/15474230-272a-4035-be27-a0d9227d3378" />

### Main User Interface (Blind End)
![Main Interface] <img width="1181" height="988" alt="image" src="https://github.com/user-attachments/assets/13dc3555-ba92-42ca-9480-3edf64079253" />
![Main Interface] <img width="1181" height="1036" alt="image" src="https://github.com/user-attachments/assets/4118c8f7-beb7-42b6-b728-6efbfbd2c5fd" />
![Main Interface] <img width="1181" height="695" alt="image" src="https://github.com/user-attachments/assets/beac2b2c-f38b-4252-a7e2-4a6d8905db10" />
![Main Interface] <img width="1181" height="552" alt="image" src="https://github.com/user-attachments/assets/71fd516c-8f82-4c03-85a5-213a11a9da16" />
![Main Interface] <img width="1181" height="614" alt="image" src="https://github.com/user-attachments/assets/3e030400-07d8-4315-9fa9-bb9c943ca463" />
![Main Interface] <img width="1181" height="672" alt="image" src="https://github.com/user-attachments/assets/74496e41-9760-4b22-ab02-ebc4c59db9f5" />
![Main Interface] <img width="1063" height="1299" alt="image" src="https://github.com/user-attachments/assets/6b4c8644-b37e-4d7e-a48a-1bfef91ed09b" />
![Main Interface] <img width="1063" height="985" alt="image" src="https://github.com/user-attachments/assets/5ac3c85e-268d-4f3f-9fef-88cf1d8268a3" />
![Main Interface] <img width="1063" height="991" alt="image" src="https://github.com/user-attachments/assets/7babc173-738c-4dfa-9169-977595676b95" />
![Main Interface] <img width="1063" height="1057" alt="image" src="https://github.com/user-attachments/assets/f2b089e2-270d-48b0-94f7-ec86fbadd031" />
![Main Interface] <img width="1063" height="841" alt="image" src="https://github.com/user-attachments/assets/0ec84c86-c43d-4a62-ae89-c057ca12b668" />
![Main Interface] <img width="1063" height="969" alt="image" src="https://github.com/user-attachments/assets/f4137f13-f4ff-4785-91cd-fc29c11ab01e" />
![Main Interface] <img width="1063" height="965" alt="image" src="https://github.com/user-attachments/assets/0131cbef-8bdb-4d20-8c10-d093529ad290" />
![Main Interface] <img width="1181" height="559" alt="image" src="https://github.com/user-attachments/assets/d65f81cd-2988-4422-a42b-aadb2641824f" />

### Blind Track Detection in Action
![Blind Track Detection] <img width="649" height="1244" alt="image" src="https://github.com/user-attachments/assets/0609bcd4-1a2e-4c6f-9fec-1b53190407fe" />
![Blind Track Detection] <img width="479" height="1342" alt="image" src="https://github.com/user-attachments/assets/5a0704f4-2af8-46a0-9033-90557df164f1" />

### Navigation with Route Planning
![Navigation] <img width="708" height="669" alt="image" src="https://github.com/user-attachments/assets/6637adc1-bd71-4985-a5fa-32892854a5a4" />
![Navigation] <img width="708" height="691" alt="image" src="https://github.com/user-attachments/assets/bedef282-0583-49a6-98fa-05797799bf46" />
![Navigation] <img width="708" height="634" alt="image" src="https://github.com/user-attachments/assets/afc8bb38-b16d-4f1b-b189-d5d5743e8a23" />
![Navigation] <img width="708" height="1114" alt="image" src="https://github.com/user-attachments/assets/b56f35f7-dbf3-484b-b44f-84b55f20c244" />
![Navigation] <img width="709" height="730" alt="image" src="https://github.com/user-attachments/assets/086b046f-a69d-4af8-b6ca-d26c07490c54" />

### OCR Text Recognition
![OCR] <img width="1181" height="462" alt="image" src="https://github.com/user-attachments/assets/51474aa5-2157-4a9c-a9c0-e141648db1e2" />
![OCR] <img width="1181" height="558" alt="image" src="https://github.com/user-attachments/assets/42362a8f-fdc6-4cdc-8c69-80a4ac48b5dc" />
![OCR] <img width="1180" height="930" alt="image" src="https://github.com/user-attachments/assets/38a592f6-f621-4ef9-b954-fb53cfe1e2f9" />

### Environment Risk Assessment
![Risk Assessment] <img width="708" height="600" alt="image" src="https://github.com/user-attachments/assets/3b7b5018-718e-4a76-972b-18c193968bf1" />
![Risk Assessment] <img width="708" height="566" alt="image" src="https://github.com/user-attachments/assets/ba0e5322-8f45-40de-86b6-85a75d4d77a2" />
![Risk Assessment] <img width="708" height="559" alt="image" src="https://github.com/user-attachments/assets/fc2221d8-c12d-4509-aac9-1585262a4878" />

### Voice Command Interface
![Voice Commands] <img width="1181" height="799" alt="image" src="https://github.com/user-attachments/assets/b0addf87-8f44-4fd8-afd9-dffb8b8d7cb6" />

### Main User Interface (Family End / Caretaker)
![Caretaker] <img width="1181" height="993" alt="image" src="https://github.com/user-attachments/assets/62f9a636-1120-4658-93bb-ad3fd2ddebb0" />
![Caretaker] <img width="1181" height="1289" alt="image" src="https://github.com/user-attachments/assets/2ceec4d3-e948-42c7-ab79-b88c0d15d599" />
![Caretaker] <img width="1181" height="475" alt="image" src="https://github.com/user-attachments/assets/4271d73c-7936-4c7b-aed7-0c760d22c9b6" />

### Caretaker Dashboard 
![Caretaker Dashboard] <img width="1181" height="1030" alt="image" src="https://github.com/user-attachments/assets/8cf542f6-9156-49f4-91ec-70ff1fc2b0cb" />
![Caretaker Dashboard] <img width="1181" height="542" alt="image" src="https://github.com/user-attachments/assets/e9cdd3db-0899-43c7-8f95-cfc182468b22" />
![Caretaker Dashboard] <img width="1181" height="661" alt="image" src="https://github.com/user-attachments/assets/7e63df9d-8228-42dc-bc41-78c89d1743d4" />

### Settings Modal (Voice & Profile)
![Settings]

---

## 📁 Database Schema (6 tables)

| Table                  | Purpose                                 |
|------------------------|-----------------------------------------|
| `users`                | Authentication, roles, timestamps       |
| `user_settings`        | Voice speed/volume, age, gender, name   |
| `obstacle_detection_logs` | Blind track turn events with confidence |
| `navigation_logs`      | GPS routes, safety scores, completion   |
| `ocr_usage_logs`       | Character counts, success/failure       |
| `environment_risk_logs`| Risk scores, weather, light level       |
| `family_messages`      | Caretaker → user messages (read/unread) |

---

## 👩‍💻 Author

**Fathima Thahirah Razmi**  
[GitHub](https://github.com/Thahirah-Razmi)

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](./LICENSE) file for details.

---
