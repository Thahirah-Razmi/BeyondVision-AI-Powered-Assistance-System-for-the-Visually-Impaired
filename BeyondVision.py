from flask import Flask, render_template, Response, request, jsonify, redirect, url_for, session
import cv2
from ultralytics import YOLO
import ollama
import pyttsx3
import threading
import time
import os
import pymysql
import hashlib
import random
import string
import smtplib
from email.mime.text import MIMEText
from email.header import Header
from werkzeug.utils import secure_filename
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import functools
import geopy.distance
from PIL import Image
import pytesseract
from flask import request, jsonify
import pyttsx3
import os
from datetime import datetime, timedelta
import json
import requests
import math
from geopy.distance import distance as geo_distance
import concurrent.futures

app = Flask(__name__)
app.secret_key = 'secret_key_for_beyondvision' #Used to sign it so the server can detect tampering

DB_CONFIG = {
    'host': '127.0.0.1',
    'user': 'root',
    'password': '', 
    'db': 'beyondvision',
    'charset': 'utf8mb4',
    'cursorclass': pymysql.cursors.DictCursor #Makes PyMySQL return query results as Python dictionaries instead of plain tuples
}

EMAIL_CONFIG = {
    'sender': '',  #Gmail
    'password': '',  #Gmail App Password
    'smtp_server': 'smtp.gmail.com',  
    'smtp_port': 465  
}

verification_codes = {}  # Format: {email: {'code': '123456', 'expires': timestamp}}

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'webm'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 300 * 1024 * 1024  

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

model_weights = "models/weights/best.pt" #Besr weights after training loaded during system initialization using the Ultralytics YOLO framework.
model = YOLO(model_weights)

current_video_path = None
video_active = False
last_call_time = 0
call_interval = 14
current_speech_text = ""
latest_speech_text = "Waiting for video upload and analysis..."
camera = None
voices_cache = None

login_attempts = {}

user_settings = {
    "gender": "Not specified",  # Gender: Male/Female/Not specified
    "name": "User",  # User name
    "age": "Not specified",  # Age group: Youth/Middle-aged/Elderly/Not specified
    "voice_speed": "Medium",  # Voice speed: Slow/Medium/Fast
    "voice_volume": "Medium",  # Voice volume: Low/Medium/High
    "user_mode": "Blind end",  # User mode: Blind end/Family end
    "encourage": "On"  # Give encouragement when appropriate: On/Off
}

user_locations = {}  # Format: {user_id: {'lat': latitude, 'lng': longitude, 'timestamp': timestamp}}

navigation_logs = []
obstacle_detection_logs = []
environment_risk_logs = []
ocr_usage_logs = []

def caretaker_required(f): #Access Control Decorators (login_required decorator and caretaker_required decorator)
    @functools.wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('login'))
        
        conn = get_db_connection()
        if conn:
            try:
                with conn.cursor() as cursor:
                    cursor.execute("SELECT role FROM users WHERE id = %s", (session['user_id'],))
                    user = cursor.fetchone()
                    if not user or user['role'] != 'caretaker':
                        return jsonify({"status": "error", "message": "Caretaker access required"}), 403
            finally:
                conn.close()
        
        return f(*args, **kwargs)
    return decorated_function

def log_obstacle_detection(user_id, detection_type, confidence, video_path=None):
    """Log obstacle detection events"""
    conn = get_db_connection()
    if not conn:
        return
    
    try:
        with conn.cursor() as cursor:
            cursor.execute(
                "INSERT INTO obstacle_detection_logs (user_id, detection_type, confidence, video_path) VALUES (%s, %s, %s, %s)",
                (user_id, detection_type, confidence, video_path)
            )
        conn.commit()
    except Exception as e:
        print(f"Error logging obstacle detection: {e}")
    finally:
        conn.close()

def log_environment_risk(user_id, risk_level, risk_score, weather, light_level, advice):
    """Log environment risk assessments"""
    conn = get_db_connection()
    if not conn:
        return
    
    try:
        with conn.cursor() as cursor:
            cursor.execute(
                "INSERT INTO environment_risk_logs (user_id, risk_level, risk_score, weather, light_level, advice) VALUES (%s, %s, %s, %s, %s, %s)",
                (user_id, risk_level, risk_score, weather, light_level, advice)
            )
        conn.commit()
    except Exception as e:
        print(f"Error logging environment risk: {e}")
    finally:
        conn.close()

def log_ocr_usage(user_id, characters_extracted, success=True, error_message=None):
    """Log OCR usage"""
    conn = get_db_connection()
    if not conn:
        return
    
    try:
        with conn.cursor() as cursor:
            cursor.execute(
                "INSERT INTO ocr_usage_logs (user_id, characters_extracted, success, error_message) VALUES (%s, %s, %s, %s)",
                (user_id, characters_extracted, success, error_message)
            )
        conn.commit()
    except Exception as e:
        print(f"Error logging OCR usage: {e}")
    finally:
        conn.close()

def log_navigation(user_id, start_lat, start_lng, end_lat, end_lng, distance_km, duration_minutes, safety_score, completed=True):
    """Log navigation events"""
    conn = get_db_connection()
    if not conn:
        print("Database connection failed for logging navigation")
        return False
    
    try:
        with conn.cursor() as cursor:
            start_lat = start_lat if start_lat is not None else None
            start_lng = start_lng if start_lng is not None else None
            end_lat = end_lat if end_lat is not None else None
            end_lng = end_lng if end_lng is not None else None
            
            print(f"Logging navigation: user_id={user_id}, start=({start_lat},{start_lng}), end=({end_lat},{end_lng})")
            
            cursor.execute(
                "INSERT INTO navigation_logs (user_id, start_lat, start_lng, end_lat, end_lng, distance_km, duration_minutes, safety_score, completed) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)",
                (user_id, start_lat, start_lng, end_lat, end_lng, distance_km, duration_minutes, safety_score, completed)
            )
            nav_id = cursor.lastrowid
        conn.commit()
        print(f"Navigation logged successfully with ID: {nav_id}")
        return nav_id
    except Exception as e:
        print(f"Error logging navigation: {e}")
        import traceback
        traceback.print_exc()
        conn.rollback()
        return False
    finally:
        conn.close()

def get_db_connection():
    """Create database connection"""
    try:
        connection = pymysql.connect(**DB_CONFIG)
        return connection
    except Exception as e:
        print(f"Database connection error: {e}")
        return None

def generate_verification_code(length=6): #Registration with Email Verification (random 6-digit numeric code)
    """Generate numeric verification code of specified length"""
    return ''.join(random.choices(string.digits, k=length))


def is_valid_email(email):
    """Simple email format validation"""
    import re
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(pattern, email) is not None


def send_verification_email(to_email, verification_code):
    """Send verification code email"""
    try:
        html_content = f"""
        <html>
            <head>
                <meta http-equiv="Content-Type" content="text/html; charset=utf-8" />
            </head>
            <body>
                <p style="font-size: 16px; color: #333;">Your verification code is:</p>
                <div style="
                    font-size: 24px;
                    color: #ff4444;
                    font-weight: bold;
                    margin: 10px 0;
                    padding: 12px;
                    background: #f8f9fa;
                    border-radius: 8px;
                    display: inline-block;
                ">{verification_code}</div>
                <p style="font-size: 14px; color: #666; margin-top: 10px;">
                    The verification code is valid for 10 minutes. Do not share it with others. If this was not your action, please ignore this email.
                </p>
            </body>
        </html>
        """

        message = MIMEText(html_content, 'html', 'utf-8')

        from email.utils import formataddr
        message['From'] = formataddr(("BeyondVision", EMAIL_CONFIG['sender']))
        message['To'] = Header(to_email)
        message['Subject'] = Header('【BeyondVision】Verification Code', 'utf-8')

        server = smtplib.SMTP_SSL(EMAIL_CONFIG['smtp_server'], EMAIL_CONFIG['smtp_port'])
        server.login(EMAIL_CONFIG['sender'], EMAIL_CONFIG['password'])
        server.sendmail(EMAIL_CONFIG['sender'], [to_email], message.as_string())
        server.quit()

        verification_codes[to_email] = { #The code is stored temporarily in an in-memory dictionary with a 10-minute expiry timestamp
            'code': verification_code,
            'expires': time.time() + 600  
        }

        return True, "Verification code sent"
    except Exception as e:
        print(f"Failed to send email:  {e}")
        return False, f"Failed to send verification code: {str(e)}"


def verify_code(email, code): #Email verification using SMTP (Gmail)
    """Verify email verification code"""
    if email not in verification_codes:
        return False, "Verification code does not exist or has expired"

    stored_data = verification_codes[email]
    current_time = time.time()

    if current_time > stored_data['expires']:
        del verification_codes[email] 
        return False, "Verification code has expired"

    if stored_data['code'] != code:
        return False, "Incorrect verification code"

    del verification_codes[email]
    return True, "Verification successful"


def init_database():
    """Initialize database, create necessary tables"""

    try:
        bootstrap_conn = pymysql.connect(
            host=DB_CONFIG['host'],
            user=DB_CONFIG['user'],
            password=DB_CONFIG['password'],
            charset=DB_CONFIG['charset'],
            cursorclass=DB_CONFIG['cursorclass']
        )
        with bootstrap_conn.cursor() as cursor:
            cursor.execute(
                "CREATE DATABASE IF NOT EXISTS beyondvision "
                "CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci"
            )
        bootstrap_conn.commit()
        bootstrap_conn.close()
        print("Database 'beyondvision' ready")
    except Exception as e:
        print(f"Failed to create database: {e}")
        return False

    conn = get_db_connection()
    if not conn:
        print("Unable to connect to database, please check database configuration")
        return False

    try:
        with conn.cursor() as cursor:
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS users (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    username VARCHAR(50) 
                           CHARACTER SET utf8mb4 
                           COLLATE utf8mb4_bin
                           NOT NULL UNIQUE,
                    password VARCHAR(255) NOT NULL,
                    email VARCHAR(100) NOT NULL UNIQUE,
                    phone VARCHAR(20),
                    role ENUM('user', 'caretaker') DEFAULT 'user',
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    last_login DATETIME
                )
            ''')

            cursor.execute('''
                CREATE TABLE IF NOT EXISTS user_settings (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    user_id INT NOT NULL,
                    gender ENUM('Male','Female','Not specified') DEFAULT 'Not specified',
                    name VARCHAR(50) DEFAULT 'User',
                    age ENUM('Youth','Middle-aged','Elderly','Not specified') DEFAULT 'Not specified',
                    voice_speed ENUM('Slow','Medium','Fast') DEFAULT 'Medium',
                    voice_volume ENUM('Low','Medium','High') DEFAULT 'Medium',
                    user_mode ENUM('Blind end','Family end') DEFAULT 'Blind end',
                    encourage ENUM('On','Off') DEFAULT 'On',
                    FOREIGN KEY (user_id) REFERENCES users(id)
                )
            ''')

            cursor.execute('''
                CREATE TABLE IF NOT EXISTS obstacle_detection_logs (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    user_id INT NOT NULL,
                    detection_type VARCHAR(50) NOT NULL,
                    confidence DECIMAL(5,4),
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    video_path VARCHAR(255),
                    FOREIGN KEY (user_id) REFERENCES users(id)
                )
            ''')

            cursor.execute('''
                CREATE TABLE IF NOT EXISTS environment_risk_logs (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    user_id INT NOT NULL,
                    risk_level VARCHAR(20) NOT NULL,
                    risk_score INT NOT NULL,
                    weather VARCHAR(50),
                    light_level INT,
                    advice TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users(id)
                )
            ''')

            cursor.execute('''
                CREATE TABLE IF NOT EXISTS ocr_usage_logs (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    user_id INT NOT NULL,
                    characters_extracted INT,
                    success BOOLEAN,
                    error_message TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users(id)
                )
            ''')

            cursor.execute('''
                CREATE TABLE IF NOT EXISTS navigation_logs (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    user_id INT NOT NULL,
                    start_lat DECIMAL(10,8),
                    start_lng DECIMAL(10,8),
                    end_lat DECIMAL(10,8),
                    end_lng DECIMAL(10,8),
                    distance_km DECIMAL(8,2),
                    duration_minutes INT,
                    safety_score INT,
                    completed BOOLEAN DEFAULT FALSE,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users(id)
                )
            ''')

            cursor.execute('''
                CREATE TABLE IF NOT EXISTS family_messages (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    sender_id INT NOT NULL,
                    recipient_id INT NOT NULL,
                    message TEXT NOT NULL,
                    is_read BOOLEAN DEFAULT FALSE,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (sender_id) REFERENCES users(id),
                    FOREIGN KEY (recipient_id) REFERENCES users(id)
                )
            ''')

        conn.commit()
        print("Database initialization successful")
        return True
    except Exception as e:
        print(f"Database initialization failed: {e}")
        return False
    finally:
        conn.close()


def register_user(username, password, email, verification_code, phone=None): #Implements secure login, registration, and password recovery with email verification.
    """Register new user, add verification code validation"""
    code_valid, message = verify_code(email, verification_code)
    if not code_valid:
        return False, message

    conn = get_db_connection()
    if not conn:
        return False, "Database connection failed"

    try:
        password_hash = hashlib.sha256(password.encode()).hexdigest() #Password Hashing 

        with conn.cursor() as cursor:
            cursor.execute("SELECT id FROM users WHERE username = %s", (username,))
            if cursor.fetchone():
                return False, "Username already exists"

            cursor.execute("SELECT id FROM users WHERE email = %s", (email,))
            if cursor.fetchone():
                return False, "This email is already registered"

            cursor.execute(
                "INSERT INTO users (username, password, email, phone) VALUES (%s, %s, %s, %s)",
                (username, password_hash, email, phone)
            )

            user_id = cursor.lastrowid

            cursor.execute(
                "INSERT INTO user_settings (user_id) VALUES (%s)",
                (user_id,)
            )

        conn.commit()
        return True, "Registration successful"
    except Exception as e:
        conn.rollback()
        print(f"Failed to register user: {e}")
        return False, f"Registration failed: {str(e)}"
    finally:
        conn.close()


def verify_user(username, password):
    """Verify user login"""
    conn = get_db_connection()
    if not conn:
        return False, "Database connection failed", None

    try:
        password_hash = hashlib.sha256(password.encode()).hexdigest()

        with conn.cursor() as cursor:
            cursor.execute("SELECT id, username, role FROM users WHERE username = %s AND password = %s",
                           (username, password_hash))
            user = cursor.fetchone()

            if not user:
                return False, "Incorrect username or password", None

            cursor.execute("UPDATE users SET last_login = NOW() WHERE id = %s", (user['id'],))

            cursor.execute("SELECT * FROM user_settings WHERE user_id = %s", (user['id'],))
            settings = cursor.fetchone()

            if not settings:
                cursor.execute("INSERT INTO user_settings (user_id) VALUES (%s)", (user['id'],))
                cursor.execute("SELECT * FROM user_settings WHERE user_id = %s", (user['id'],))
                settings = cursor.fetchone()

        conn.commit()

        user_config = {
            "id": user['id'],
            "username": user['username'],
            "role": user['role'],
            "gender": settings['gender'],
            "name": settings['name'],
            "age": settings['age'],
            "voice_speed": settings['voice_speed'],
            "voice_volume": settings['voice_volume'],
            "user_mode": settings['user_mode'],
            "encourage": settings['encourage']
        }

        return True, "Login successful", user_config
    except Exception as e:
        print(f"Failed to verify user: {e}")
        return False, f"Login failed: {str(e)}", None
    finally:
        conn.close()

def update_user_settings_in_db(user_id, settings):
    """Update user settings in database and sync role based on user_mode"""
    conn = get_db_connection()
    if not conn:
        return False, "Database connection failed"

    try:
        role = 'caretaker' if settings.get("user_mode") == "Family end" else 'user' #'Family end' are automatically assigned the 'caretaker' role

        with conn.cursor() as cursor:
            cursor.execute("""
                UPDATE user_settings 
                SET gender = %s, name = %s, age = %s, 
                    voice_speed = %s, voice_volume = %s, user_mode = %s, encourage = %s
                WHERE user_id = %s
                """,
                           (settings["gender"], settings["name"], settings["age"],
                            settings["voice_speed"], settings["voice_volume"], settings["user_mode"],
                            settings["encourage"],
                            user_id)
                           )

            cursor.execute(
                "UPDATE users SET role = %s WHERE id = %s",
                (role, user_id)
            )

        conn.commit()
        return True, "Settings updated successfully"
    except Exception as e:
        conn.rollback()
        print(f"Failed to update user settings: {e}")
        return False, f"Failed to update settings: {str(e)}"
    finally:
        conn.close()

def login_required(f):
    @functools.wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)

    return decorated_function


@app.route('/login', methods=['GET', 'POST'])
def login():
    """User login page"""
    error = None
    success = None

    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')

        current_time = datetime.now()

        if username not in login_attempts:
            login_attempts[username] = {
                "count": 0,
                "first_attempt_time": current_time,
                "lock_until": None
            }

        user_attempt = login_attempts[username]

        if user_attempt["lock_until"] and current_time < user_attempt["lock_until"]:
            error = "Too many failed attempts. Please try again later."
            return render_template('login.html', error=error, success=success)

        if not username or not password:
            error = "Please enter username and password"
        else:
            success_login, message, user_data = verify_user(username, password)
            if success_login:
                
                login_attempts[username] = { #Login with Rate Limiting 
                    "count": 0,
                    "first_attempt_time": current_time,
                    "lock_until": None
                }

                session['user_id'] = user_data['id']
                session['username'] = user_data['username']
                session['role'] = user_data['role']

                global user_settings
                user_settings = {
                    "gender": user_data['gender'],
                    "name": user_data['name'],
                    "age": user_data['age'],
                    "voice_speed": user_data['voice_speed'],
                    "voice_volume": user_data['voice_volume'],
                    "user_mode": user_data['user_mode'],
                    "encourage": user_data['encourage']
                }

                return redirect(url_for('index'))
            else:
                if current_time - user_attempt["first_attempt_time"] > timedelta(minutes=1):
                    user_attempt["count"] = 0
                    user_attempt["first_attempt_time"] = current_time

                user_attempt["count"] += 1

                if user_attempt["count"] >= 5:
                    user_attempt["lock_until"] = current_time + timedelta(minutes=2)
                    error = "Too many failed attempts. Account locked for 2 minutes."
                else:
                    error = f"{message} (Attempt {user_attempt['count']} of 5)"

    return render_template('login.html', error=error, success=success)


@app.route('/register', methods=['GET', 'POST'])
def register():
    """User registration page"""
    error = None

    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')
        email = request.form.get('email')
        verification_code = request.form.get('verification_code')
        phone = request.form.get('phone')

        if not username or not password or not email:
            error = "Username, password and email cannot be empty"
        elif password != confirm_password:
            error = "Passwords do not match"
        elif not verification_code:
            error = "Please enter verification code"
        else:
            success, message = register_user(username, password, email, verification_code, phone)
            if success:
                return redirect(url_for('login', success="Registration successful, please login"))
            else:
                error = message

    return render_template('register.html', error=error)


@app.route('/logout')
def logout():
    """User logout"""
    session.clear()
    return redirect(url_for('login'))


@app.route('/')
@login_required
def index():
    user = {
        'id': session.get('user_id'),
        'username': session.get('username', 'User')
    }
    return render_template('index.html', settings=user_settings, current_user=user)


@app.route('/update_settings', methods=['POST'])
@login_required
def update_settings():
    """Update user settings"""
    global user_settings

    data = request.get_json()
    if not data:
        return jsonify({"status": "error", "message": "No settings data received"}), 400

    for key in user_settings.keys():
        if key in data:
            user_settings[key] = data[key]

    try:
        user_id = session.get('user_id')
        success, message = update_user_settings_in_db(user_id, user_settings)
        if not success:
            return jsonify({"status": "error", "message": message}), 500
    except Exception as e:
        return jsonify({"status": "error", "message": f"Failed to save settings: {str(e)}"}), 500

    return jsonify({
        "status": "success",
        "message": "Settings updated",
        "settings": user_settings
    })


def get_prompt_template(): #Uses a local Ollama instance running the Qwen 2.5 3b model to generate natural, personalised guidance. 
    gender_term = ""
    age_term = ""

    if user_settings["gender"] == "Male": #A system prompt is constructed dynamically from the user's settings (name, gender, age, encouragement preference)
        gender_term = "Mr."
    elif user_settings["gender"] == "Female":
        gender_term = "Ms."

    if user_settings["age"] == "Elderly":
        age_term = "senior"
    elif user_settings["age"] == "Youth":
        age_term = "young"

    prompt = f'''
You are a voice navigation assistant serving blind people for walking.
Your user is {age_term} {user_settings["name"]} {gender_term}.
You guide blind people by informing them about the direction of the blind track, be sure to clearly state the turning direction of the blind track (left? right?), ensure the blind person always walks on the blind track, and give some care at appropriate times.
Note, blind people cannot see the road conditions, so they need your voice walking prompts.
Your tone should be gentle and energetic.
'''

    if user_settings["encourage"] == "On":
        prompt += '''
While guiding directions, please also give users warm encouragement and positive affirmations appropriately, such as praising them for walking well, making obvious progress, or encouraging them to remain confident, etc.
'''

    return prompt


right_turn_question = "Please use friendly and brief language to inform to turn right because the blind track turns right"
left_turn_question = "Please use friendly and brief language to inform to turn left because the blind track turns left"


def get_available_voices():
    global voices_cache
    if voices_cache is not None:
        return voices_cache

    engine = pyttsx3.init()
    voices = engine.getProperty('voices')
    available_voices = []

    for voice in voices:
        voice_info = {
            'id': voice.id,
            'name': voice.name,
            'gender': 'Female' if 'female' in voice.id.lower() or 'zira' in voice.name else 'Male'
        }
        available_voices.append(voice_info)

    voices_cache = available_voices
    return available_voices


def speak(text): #Initializes a new pyttsx3 engine instance per call 
    """Create new pyttsx3 instance in each thread for speech synthesis"""
    try:
        print(f"[Speech] Starting speech synthesis: '{text}'")
        local_engine = pyttsx3.init()

        voices = local_engine.getProperty('voices')
        print(f"[Speech] System available voice list:")
        for i, voice in enumerate(voices):
            print(f"  Voice {i + 1}: ID={voice.id}, Name={voice.name}")

        found_english_voice = False
        selected_voice = None

        for voice in voices:
            voice_name = voice.name.lower()
            if any(lang in voice_name for lang in ['english', 'en_us', 'en-gb', 'microsoft david', 'microsoft zira']):
                selected_voice = voice.id
                found_english_voice = True
                print(f"[Speech] Found English voice: {voice.name}")
                break

        if not found_english_voice and len(voices) > 0:
            selected_voice = voices[0].id
            print(f"[Speech] English voice not found, using first available voice: {voices[0].name}")

        if selected_voice:
            print(f"[Speech] Final voice ID used: {selected_voice}")
            local_engine.setProperty('voice', selected_voice)
        else:
            print("[Speech] Warning: No available voice found")

        if user_settings["voice_speed"] == "Slow": 
            local_engine.setProperty('rate', 150)
        elif user_settings["voice_speed"] == "Fast":
            local_engine.setProperty('rate', 250)
        else:  
            local_engine.setProperty('rate', 200)

        volume_mapping = {
            "Low": 0.5,
            "Medium": 0.8,
            "High": 1.0
        }
        volume = volume_mapping.get(user_settings["voice_volume"], 0.8)
        local_engine.setProperty('volume', volume)

        print(f"[Speech] Playing text: {text}")
        local_engine.say(text)

        print("[Speech] Starting runAndWait()...")
        local_engine.runAndWait()
        print("[Speech] Playback completed")
        return True
    except Exception as e:
        print(f"[Speech] Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def generate_frames():
    global last_call_time, current_speech_text, current_video_path, video_active

    user_id = current_user_id

    if not video_active or not current_video_path:
        current_speech_text = "Prompt: The system will analyze the blind track direction in real time, and automatically broadcast voice prompts when the direction changes."
        while not video_active or not current_video_path:
            wait_frame = create_info_frame("Please upload video file to start analysis")
            ret, buffer = cv2.imencode('.jpg', wait_frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            time.sleep(1)

    try:
        cap = cv2.VideoCapture(current_video_path)

        if not cap.isOpened():
            print(f"Cannot open video: {current_video_path}")
            cap = cv2.VideoCapture(current_video_path, cv2.CAP_FFMPEG)

            if not cap.isOpened():
                error_frame = create_error_frame(f"Cannot open video file: {os.path.basename(current_video_path)}")
                ret, buffer = cv2.imencode('.jpg', error_frame)
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
                video_active = False
                current_speech_text = "Video cannot be opened, please try uploading videos in other formats."
                return

        THRESHOLD_SLOPE = 0.2 #The slope of the fitted line is compared against a threshold
        frame_count = 0

        while cap.isOpened() and video_active:
            ret, frame = cap.read()
            frame_count += 1

            if not ret:
                if frame_count < 10:  
                    print(f"Cannot read video frames: {current_video_path}")
                    error_frame = create_error_frame("Video file corrupted or format not supported")
                    ret, buffer = cv2.imencode('.jpg', error_frame)
                    frame = buffer.tobytes()
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
                    video_active = False
                    current_speech_text = "Video file corrupted or format not supported, please try other videos."
                    break

                end_frame = create_info_frame(
                    "Video playback completed, \n"
                    "please upload new video"
                    )
                ret, buffer = cv2.imencode('.jpg', end_frame)
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
                video_active = False
                current_speech_text = "Video playback completed, please upload new video."
                break

            results = model(frame) #Real-Time Detection Pipeline
            centers = []  

            for result in results: #Frame-by-Frame YOLO Detection 
                boxes = result.boxes #Each frame is passed through the YOLO model, which returns bounding boxes for detected blind track tiles.
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0] #x1, y1 represent the top-left corner and x2, y2 represent the bottom-right corner of the bounding box
                    conf = box.conf[0]
                    cls = int(box.cls[0])
                    center_x = (x1 + x2) / 2 #The centre coordinates of all detected bounding boxes are collected.
                    center_y = (y1 + y2) / 2
                    centers.append((center_x, center_y))

                    class_names = model.names
                    label = f"{class_names[cls]}: {conf:.2f}"
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    cv2.putText(frame, label, (int(x1), int(y1) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            current_time = time.time()
            if len(centers) == 0 and current_time - last_call_time >= call_interval:
                no_track_message = "No blind tracks detected. Please slow down and move carefully until the blind track is found."
                current_speech_text = no_track_message
                threading.Thread(target=speak, args=(no_track_message,)).start()
                last_call_time = current_time

            elif len(centers) >= 2 and current_time - last_call_time >= call_interval: #Turn Direction Calculation via Linear Regression when 2 or more blind tracks are detected
                ys = np.array([c[1] for c in centers]) #represents the vertical coordinates of the detected center points
                xs = np.array([c[0] for c in centers]) #represents the horizontal coordinates
                slope, intercept = np.polyfit(ys, xs, 1) #YOLO slope detection

                print(f"[Blind Track Detection] Slope: {slope}, Intercept: {intercept}")

                if slope > THRESHOLD_SLOPE:
                    print("[Blind Track Detection] Detected left turn")
                    
                    try:
                        user_id = current_user_id
                        if user_id:
                            log_obstacle_detection(
                                user_id,
                                "left_turn",
                                float(abs(slope)),
                                current_video_path
                            )
                            print(f"[Logging] Logged left turn detection for user {user_id}")
                    except Exception as log_error:
                        print(f"[Logging Error] Failed to log left turn: {log_error}")
                    
                    response = ollama.chat(model="qwen2.5:3b", messages=[
                        {"role": "system", "content": get_prompt_template()},
                        {"role": "user", "content": left_turn_question}
                    ], stream=True)

                    answer_content = ""
                    for chunk in response:
                        content = chunk.get('message', {}).get('content', '')
                        if content:
                            answer_content += content

                    print(f"[Blind Track Detection] Generated left turn prompt: {answer_content}")

                    current_speech_text = answer_content
                    threading.Thread(target=speak, args=(answer_content,)).start()
                    last_call_time = current_time
                    print(f"[Blind Track Detection] Started left turn voice prompt")

                elif slope <- THRESHOLD_SLOPE:
                    print("[Blind Track Detection] Detected right turn")
                    
                    try:
                        user_id = current_user_id
                        if user_id:
                            log_obstacle_detection(
                                user_id,
                                "right_turn",
                                float(abs(slope)),
                                current_video_path
                            )
                            print(f"[Logging] Logged right turn detection for user {user_id}")
                    except Exception as log_error:
                        print(f"[Logging Error] Failed to log right turn: {log_error}")
                    
                    response = ollama.chat(model="qwen2.5:3b", messages=[
                        {"role": "system", "content": get_prompt_template()},
                        {"role": "user", "content": right_turn_question}
                    ], stream=True)

                    answer_content = ""
                    for chunk in response:
                        content = chunk.get('message', {}).get('content', '')
                        if content:
                            answer_content += content

                    print(f"[Blind Track Detection] Generated right turn prompt:: {answer_content}")

                    current_speech_text = answer_content
                    threading.Thread(target=speak, args=(answer_content,)).start()
                    last_call_time = current_time
                    print(f"[Blind Track Detection] Started right turn voice prompt")

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

        cap.release()

    except Exception as e:
        print(f"Video processing error: {e}")
        import traceback
        traceback.print_exc()
        error_frame = create_error_frame(f"Video processing error: {str(e)}")
        ret, buffer = cv2.imencode('.jpg', error_frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        video_active = False
        current_speech_text = "Video processing error, please try uploading other videos."


def create_error_frame(message):
    """Create error information frame - for English text"""
    img = Image.new('RGB', (640, 480), color=(0, 0, 0))
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype("arial.ttf", 30)
    except IOError:
        font = ImageFont.load_default()

    bbox = draw.textbbox((0, 0), message, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    position = ((640 - text_width) // 2, (480 - text_height) // 2)

    draw.text(position, message, font=font, fill=(255, 0, 0))

    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def create_info_frame(message):
    """Create information prompt frame - for English text"""
    img = Image.new('RGB', (640, 480), color=(41, 128, 185))
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype("arial.ttf", 30)
    except IOError:
        font = ImageFont.load_default()

    bbox = draw.textbbox((0, 0), message, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    position = ((640 - text_width) // 2, (480 - text_height) // 2)

    draw.text(position, message, font=font, fill=(255, 255, 255))

    try:
        small_font = ImageFont.truetype("DejaVuSans.ttf", 20)
    except IOError:
        small_font = ImageFont.load_default()

    help_text = "Supports mp4, avi, mov, mkv, webm formats"
    bbox = draw.textbbox((0, 0), help_text, font=small_font)
    help_width = bbox[2] - bbox[0]
    help_height = bbox[3] - bbox[1]
    help_position = ((640 - help_width) // 2, position[1] + text_height + 20)
    draw.text(help_position, help_text, font=small_font, fill=(200, 200, 200))

    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/stream_speech_text')
def stream_speech_text():
    def generate():
        global current_speech_text
        last_sent = ""

        if not current_speech_text:
            current_speech_text =  "Prompt: The system will analyze the blind track direction in real time, and automatically broadcast voice prompts when the direction changes."

        while True:
            if current_speech_text != last_sent:
                last_sent = current_speech_text
                yield f"{current_speech_text}\n\n"
            time.sleep(0.5)

    return Response(generate(), mimetype='text/event-stream')


@app.route('/send_message', methods=['POST'])
@login_required
def send_message():
    """Family member sends a message; saved to DB so the blind user's browser can poll and speak it."""
    sender_id = session.get('user_id')

    data = request.get_json()
    if not data:
        return jsonify({"status": "error", "message": "No data received"}), 400

    message = data.get('message', '').strip()
    recipient_id = data.get('recipient_id')  

    if not message:
        return jsonify({"status": "error", "message": "Message is empty"}), 400

    if not recipient_id:
        conn = get_db_connection()
        if conn:
            try:
                with conn.cursor() as cursor:
                    cursor.execute(
                        "SELECT user_id FROM user_settings WHERE user_mode = 'Blind end' LIMIT 1"
                    )
                    row = cursor.fetchone()
                    if row:
                        recipient_id = row['user_id']
            finally:
                conn.close()

    if not recipient_id:
        return jsonify({"status": "error", "message": "No blind user found to send the message to"}), 400

    try:
        conn = get_db_connection()
        if not conn:
            return jsonify({"status": "error", "message": "Database connection failed"}), 500

        with conn.cursor() as cursor:
            cursor.execute(
                "INSERT INTO family_messages (sender_id, recipient_id, message) VALUES (%s, %s, %s)",
                (sender_id, recipient_id, message)
            )
        conn.commit()
        conn.close()

        print(f"[Message] Family message saved for blind user {recipient_id}: {message}")
        return jsonify({"status": "success", "message": "Message sent successfully"})
    except Exception as e:
        print(f"[Message] Error saving message: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"status": "error", "message": f"Send failed: {str(e)}"}), 500


@app.route('/poll_messages', methods=['GET'])
@login_required
def poll_messages():
    """Blind user's browser polls this endpoint to fetch unread family messages."""
    user_id = session.get('user_id')

    conn = get_db_connection()
    if not conn:
        return jsonify({"status": "error", "messages": []}), 500

    try:
        with conn.cursor() as cursor:
            cursor.execute(
                "SELECT id, message FROM family_messages WHERE recipient_id = %s AND is_read = FALSE ORDER BY timestamp ASC",
                (user_id,)
            )
            rows = cursor.fetchall()

            if rows:
                ids = [str(row['id']) for row in rows]
                cursor.execute(
                    f"UPDATE family_messages SET is_read = TRUE WHERE id IN ({','.join(ids)})"
                )
            conn.commit()

        return jsonify({
            "status": "success",
            "messages": [row['message'] for row in rows]
        })
    except Exception as e:
        print(f"[Poll] Error fetching messages: {e}")
        return jsonify({"status": "error", "messages": []}), 500
    finally:
        conn.close()

current_user_id = None

@app.route('/upload_video', methods=['POST']) #Video Upload and Validation 
def upload_video():
    """Handle video upload"""
    global current_video_path, video_active, current_user_id

    user_id = session.get('user_id')
    current_user_id = user_id

    if 'video' not in request.files:
        return jsonify({"status": "error", "message": "No file uploaded"}), 400

    file = request.files['video']

    if file.filename == '':
        return jsonify({"status": "error", "message": "No file selected"}), 400

    if not allowed_file(file.filename):
        return jsonify(
            {"status": "error", "message": f"Unsupported file type, allowed types: {', '.join(ALLOWED_EXTENSIONS)}"}), 400

    try:
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

        timestamp = int(time.time())
        filename = f"{timestamp}_{secure_filename(file.filename)}"
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        test_cap = cv2.VideoCapture(file_path) #The video analysis engine processes uploaded videos frame by frame using the OpenCV library.
        if not test_cap.isOpened(): #Each frame extracted from the video is passed to the object detection model for obstacle identification.
            test_cap.release()
            if os.path.exists(file_path):
                os.remove(file_path)
            return jsonify({"status": "error", "message": "Cannot open video file, please check file format or try other videos"}), 400

        read_success = False
        for _ in range(5): 
            ret, _ = test_cap.read()
            if ret:
                read_success = True
                break

        fps = test_cap.get(cv2.CAP_PROP_FPS)
        frame_count = test_cap.get(cv2.CAP_PROP_FRAME_COUNT)

        test_cap.release()

        if not read_success:
            if os.path.exists(file_path):
                os.remove(file_path)
            return jsonify({"status": "error", "message": "Video file cannot read frames normally, please try other videos"}), 400
        
        if fps > 0 and frame_count > 0:
            duration_seconds = frame_count / fps
            if duration_seconds < 5:
                if os.path.exists(file_path):
                    os.remove(file_path)
                return jsonify({
                    "status": "error",
                    "message": f"Video is too short ({duration_seconds:.1f}s). Please upload a video that is at least 5 seconds long for meaningful analysis."
                }), 400

        if current_video_path and os.path.exists(current_video_path):
            try:
                os.remove(current_video_path)
            except Exception as e:
                print(f"Cannot delete old video file: {e}")

        current_video_path = file_path
        video_active = True
        print(f"Successfully uploaded video: {file_path}")

        return jsonify({
            "status": "success",
            "message": "Video uploaded successfully",
            "file_path": file_path
        })
    except Exception as e:
        print(f"Video upload error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"status": "error", "message": f"Upload failed: {str(e)}"}), 500

@app.route('/stop_video', methods=['POST'])
@login_required
def stop_video():
    """Stop all ongoing video processing and speech synthesis"""
    global video_active, current_speech_text

    video_active = False
    current_speech_text = "Operation cancelled."

    def announce_cancelled():
        speak("Operation cancelled.")

    threading.Thread(target=announce_cancelled).start()

    print("[Stop] Video processing halted by user command.")
    return jsonify({"status": "success", "message": "Processing halted. Operation cancelled."})

@app.route('/get_settings', methods=['GET'])
def get_settings():
    """Get current user settings"""
    return jsonify({
        "status": "success",
        "settings": user_settings
    })


@app.route('/get_available_voices', methods=['GET'])
def get_available_voices():
    """Get system available voice list"""
    try:
        voices = get_available_voices()

        return jsonify({
            "status": "success",
            "voices": voices
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Failed to get voice list: {str(e)}"
        }), 500


@app.route('/test_voice', methods=['POST'])
def voice_test():
    """Test voice settings, use same thread method as old version"""
    try:
        data = request.get_json()
        print(f"[Voice Test] Received request data: {data}")

        test_settings = {
            "voice_speed": data.get("voice_speed", user_settings["voice_speed"]),
            "voice_volume": data.get("voice_volume", user_settings["voice_volume"])
        }

        temp_settings = {
            "voice_speed": user_settings["voice_speed"],
            "voice_volume": user_settings["voice_volume"]
        }

        print(f"[Voice Test] Current settings: {temp_settings}")
        print(f"[Voice Test] Test settings: {test_settings}")

        user_settings.update(test_settings)

        test_text = data.get("test_text")

        if not test_text:
            encourage_status = "enabled" if user_settings.get("encourage") == "On" else "disabled"
            test_text = f"This is a test voice message to test the current voice settings effect. You have {encourage_status} the encouragement function."

        print(f"[Voice Test] Will play text: {test_text}")

        threading.Thread(target=speak, args=(test_text,)).start()

        def restore_settings():
            time.sleep(2)  
            user_settings.update(temp_settings)
            print("[Voice Test] Original settings restored")

        threading.Thread(target=restore_settings).start()

        print("[Voice Test] Voice test started")
        return jsonify({"status": "success", "message": "Voice test started"})
    except Exception as e:
        print(f"[Voice Test] Error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"status": "error", "message": f"Voice test failed: {str(e)}"})


@app.route('/send_verification_code', methods=['POST'])
def send_code():
    """Send email verification code"""
    email = request.form.get('email')
    purpose = request.form.get('purpose', 'register')  
    if not email:
        return jsonify({"status": "error", "message": "Email cannot be empty"}), 400

    if not is_valid_email(email):
        return jsonify({"status": "error", "message": "Email format is incorrect"}), 400

    if purpose == 'register':
        conn = get_db_connection()
        if conn:
            try:
                with conn.cursor() as cursor:
                    cursor.execute("SELECT id FROM users WHERE email = %s", (email,))
                    if cursor.fetchone():
                        return jsonify({"status": "error", "message": "This email is already registered"}), 400
            finally:
                conn.close()

    verification_code = generate_verification_code()
    success, message = send_verification_email(email, verification_code)

    if success:
        return jsonify({"status": "success", "message": "Verification code sent, please check your email"})
    else:
        return jsonify({"status": "error", "message": message}), 500


@app.route('/forget_password', methods=['GET', 'POST'])
def forget_password():
    """Forget password page"""
    error = None
    success = None

    if request.method == 'POST':
        email = request.form.get('email')
        verification_code = request.form.get('verification_code')
        new_password = request.form.get('new_password')
        confirm_password = request.form.get('confirm_password')

        if not email or not verification_code or not new_password:
            error = "All fields are required"
        elif new_password != confirm_password:
            error = "Passwords do not match"
        else:
            code_valid, message = verify_code(email, verification_code)
            if not code_valid:
                error = message
            else:
                conn = get_db_connection()
                if not conn:
                    error = "Database connection failed"
                else:
                    try:
                        password_hash = hashlib.sha256(new_password.encode()).hexdigest()
                        with conn.cursor() as cursor:
                            cursor.execute("SELECT id FROM users WHERE email = %s", (email,))
                            user = cursor.fetchone()

                            if not user:
                                error = "This email is not registered"
                            else:
                                cursor.execute(
                                    "UPDATE users SET password = %s WHERE email = %s",
                                    (password_hash, email)
                                )
                                conn.commit()
                                success = "Password reset successful, please login"
                    except Exception as e:
                        conn.rollback()
                        error = f"Password reset failed: {str(e)}"
                    finally:
                        conn.close()

    return render_template('forget_password.html', error=error, success=success)


@app.route('/update_location', methods=['POST'])
@login_required
def update_location():
    """Update user location"""
    user_id = session.get('user_id')
    data = request.get_json()

    if not data or 'lat' not in data or 'lng' not in data:
        return jsonify({"status": "error", "message": "Location data incomplete"}), 400

    user_locations[user_id] = {
        'lat': data['lat'],
        'lng': data['lng'],
        'timestamp': time.time()
    }

    return jsonify({
        "status": "success",
        "message": "Location updated"
    })


@app.route('/get_location/<int:user_id>', methods=['GET'])
@login_required
def get_location(user_id):
    """Get specified user's location"""
    current_user_id = session.get('user_id')

    if user_id in user_locations:
        if time.time() - user_locations[user_id]['timestamp'] > 300:
            return jsonify({
                "status": "warning",
                "message": "Location data has expired",
                "location": user_locations[user_id]
            })

        return jsonify({
            "status": "success",
            "location": user_locations[user_id]
        })
    else:
        return jsonify({
            "status": "error",
            "message": "User location data not found"
        }), 404


@app.route('/nearby_blindways', methods=['GET']) #Queries multiple government and GIS data sources in parallel using concurrent.futures.ThreadPoolExecutor with 4 workers
@login_required
def nearby_blindways():
    lat = request.args.get('lat', type=float)
    lng = request.args.get('lng', type=float)
    radius = request.args.get('radius', default=1000, type=int)  

    if lat is None or lng is None:
        return jsonify({"status": "error", "message": "Please provide location parameters"}), 400

    data_sources = [
        {
            "name": "ArcGIS Footpath",
            "url": "https://gisapps.nsdi.gov.lk/server/rest/services/SLNSDI/Transport/MapServer/12/query",
            "params": {
                "geometry": f"{lng},{lat}",
                "geometryType": "esriGeometryPoint",
                "inSR": 4326,
                "distance": radius,
                "outFields": "*",
                "f": "json"
            }
        },
        {
            "name": "Metadata Source 1",
            "url": "https://catalogv1.nsdi.gov.lk/rest/metadata/item/b8711178de104020bb66b3f884e5d0e6",
            "type": "metadata"
        },
        {
            "name": "Metadata Source 2",
            "url": "https://catalogv1.nsdi.gov.lk/rest/metadata/item/91390822bef54976b03e6ac75ce0f927",
            "type": "metadata"
        },
        {
            "name": "Metadata Source 3",
            "url": "https://catalogv1.nsdi.gov.lk/rest/metadata/item/682068ddf26d4c83a10de95ace830132",
            "type": "metadata"
        }
    ]

    all_blindways = []

    def fetch_arcgis_data(source):
        try:
            r = requests.get(source["url"], params=source["params"], timeout=10)
            data = r.json()
            features = data.get("features", [])
            
            blindways = []
            for feat in features:
                attrs = feat.get("attributes", {})
                geom = feat.get("geometry")
                if not geom:
                    continue

                coords = []
                if "paths" in geom:
                    for path in geom["paths"]:
                        for coord in path:
                            coords.append({"lat": coord[1], "lng": coord[0]})
                elif "rings" in geom:  
                    for ring in geom["rings"]:
                        for coord in ring:
                            coords.append({"lat": coord[1], "lng": coord[0]})

                if coords:
                    user_pt = (lat, lng) #calculates distances between the user’s current location and selected destinations using the Geopy library
                    distances = [geo_distance(user_pt, (p["lat"], p["lng"])).meters for p in coords] #Computes geographic distances based on latitude and longitude coordinates
                    min_dist = min(distances) if distances else None

                    blindways.append({
                        "name": attrs.get("NAME", f"Footpath - {source['name']}"),
                        "description": attrs.get("DESCRIPTION", ""),
                        "type": attrs.get("TYPE", "Footpath"),
                        "status": attrs.get("STATUS", "Unknown"),
                        "source": source["name"],
                        "points": coords,
                        "distance": round(min_dist, 1) if min_dist else None,
                        "bbox": calculate_bounding_box(coords),
                        "length_m": calculate_path_length(coords)
                    })
            return blindways
        except Exception as e:
            print(f"Error fetching {source['name']}: {e}")
            return []

    def fetch_metadata(source):
        try:
            r = requests.get(source["url"], timeout=10)
            data = r.json()
            
            coords = extract_coordinates_from_metadata(data)
            
            if coords:
                user_pt = (lat, lng)
                distances = [geo_distance(user_pt, (p["lat"], p["lng"])).meters for p in coords]
                min_dist = min(distances) if distances else None
                
                name = data.get("title", f"Blindway - {source['name']}")
                description = data.get("abstract", "")
                
                return [{
                    "name": name,
                    "description": description,
                    "type": "Metadata Source",
                    "status": "Unknown",
                    "source": source["name"],
                    "points": coords,
                    "distance": round(min_dist, 1) if min_dist else None,
                    "bbox": calculate_bounding_box(coords),
                    "length_m": calculate_path_length(coords),
                    "metadata": data 
                }]
        except Exception as e:
            print(f"Error fetching metadata {source['name']}: {e}")
        return []

    def calculate_bounding_box(points):
        if not points:
            return None
        lats = [p["lat"] for p in points]
        lngs = [p["lng"] for p in points]
        return {
            "min_lat": min(lats),
            "max_lat": max(lats),
            "min_lng": min(lngs),
            "max_lng": max(lngs)
        }

    def calculate_path_length(points):
        if len(points) < 2:
            return 0
        total = 0
        for i in range(len(points) - 1):
            p1 = points[i]
            p2 = points[i + 1]
            total += geo_distance((p1["lat"], p1["lng"]), (p2["lat"], p2["lng"])).meters
        return round(total, 1)

    def extract_coordinates_from_metadata(metadata):
        """Extract coordinates from metadata - adjust based on actual structure"""
        coords = []
        
        if "extent" in metadata and "coordinates" in metadata["extent"]:
            coords_list = metadata["extent"]["coordinates"]
            if isinstance(coords_list, list) and len(coords_list) > 0:
                for coord in coords_list:
                    if isinstance(coord, list) and len(coord) >= 2:
                        coords.append({"lng": coord[0], "lat": coord[1]})
        
        elif "boundingBox" in metadata:
            bbox = metadata["boundingBox"]
            if isinstance(bbox, list) and len(bbox) == 4:
                min_lng, min_lat, max_lng, max_lat = bbox
                coords = [
                    {"lng": min_lng, "lat": min_lat},
                    {"lng": max_lng, "lat": min_lat},
                    {"lng": max_lng, "lat": max_lat},
                    {"lng": min_lng, "lat": max_lat},
                    {"lng": min_lng, "lat": min_lat}  
                ]
        
        for key, value in metadata.items():
            if "geometry" in key.lower() and isinstance(value, dict):
                if "coordinates" in value:
                    coords_list = value["coordinates"]
                    if isinstance(coords_list, list):
                        flat_coords = []
                        def flatten_coords(arr):
                            for item in arr:
                                if isinstance(item, list):
                                    if isinstance(item[0], list):
                                        flatten_coords(item)
                                    elif len(item) >= 2:
                                        flat_coords.append({"lng": item[0], "lat": item[1]})
                        
                        flatten_coords(coords_list)
                        if flat_coords:
                            coords = flat_coords
                            break
        
        return coords

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = []
        for source in data_sources:
            if source.get("type") == "metadata":
                futures.append(executor.submit(fetch_metadata, source))
            else:
                futures.append(executor.submit(fetch_arcgis_data, source))
        
        for future in concurrent.futures.as_completed(futures):
            try:
                result = future.result()
                all_blindways.extend(result)
            except Exception as e:
                print(f"Error processing source: {e}")

    all_blindways.sort(key=lambda x: x["distance"] if x["distance"] is not None else float('inf'))

    unique_blindways = []
    seen_coords = set()
    
    for blindway in all_blindways:
        if blindway["points"]:
            coord_hash = hash(tuple((p["lat"], p["lng"]) for p in blindway["points"][:10]))
            if coord_hash not in seen_coords:
                seen_coords.add(coord_hash)
                unique_blindways.append(blindway)

    return jsonify({
        "status": "success",
        "blindways": unique_blindways[:50],  
        "total_found": len(unique_blindways),
        "sources_checked": len(data_sources),
        "user_location": {"lat": lat, "lng": lng, "radius": radius}
    })

@app.route('/get_user_details', methods=['GET'])
@login_required
def get_user_details():
    """Get current user's detailed information"""
    user_id = session.get('user_id')

    conn = get_db_connection()
    if not conn:
        return jsonify({"status": "error", "message": "Database connection failed"}), 500

    try:
        with conn.cursor() as cursor:
            cursor.execute("""
                SELECT username, email, phone, created_at, last_login
                FROM users
                WHERE id = %s
            """, (user_id,))

            user_info = cursor.fetchone()

            if not user_info:
                return jsonify({"status": "error", "message": "User information not found"}), 404

            created_at = user_info['created_at'].strftime('%Y-%m-%d %H:%M:%S') if user_info['created_at'] else "Unknown"
            last_login = user_info['last_login'].strftime('%Y-%m-%d %H:%M:%S') if user_info['last_login'] else "Unknown"

            return jsonify({
                "status": "success",
                "user_info": {
                    "username": user_info['username'],
                    "email": user_info['email'],
                    "phone": user_info['phone'] or "Not Set",
                    "created_at": created_at,
                    "last_login": last_login
                }
            })

    except Exception as e:
        print(f"Failed to get user information: {e}")
        return jsonify({"status": "error", "message": f"Failed to get user information: {str(e)}"}), 500
    finally:
        conn.close()

@app.route('/ocr', methods=['GET', 'POST'])
@login_required
def ocr_text():
    """OCR text recognition from images with comprehensive logging"""
    print("[DEBUG] /ocr route was called!")
    
    if request.method == 'GET':
        return jsonify({
            "status": "success", 
            "message": "OCR endpoint is working. Use POST to upload images."
        })
    
    try:
        if 'image' not in request.files:
            return jsonify({"status": "error", "error": "No image uploaded"}), 400

        file = request.files['image']
        if file.filename == '':
            return jsonify({"status": "error", "error": "No file selected"}), 400

        allowed_extensions = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff'}
        file_extension = file.filename.rsplit('.', 1)[1].lower() if '.' in file.filename else ''
        
        if not ('.' in file.filename and file_extension in allowed_extensions):
            log_ocr_usage(
                session.get('user_id'),
                0,
                False,
                f"Invalid file type: {file_extension}"
            )
            return jsonify({"status": "error", "error": "Invalid file type. Please upload an image (PNG, JPG, JPEG, GIF, BMP)."}), 400

        os.makedirs('uploads', exist_ok=True)
        
        filename = secure_filename(file.filename)
        timestamp = int(time.time())
        unique_filename = f"ocr_{timestamp}_{filename}"
        path = os.path.join('uploads', unique_filename)
        
        print(f"[OCR] Saving file to: {path}")
        file.save(path)

        try:
            print("[OCR] Starting OCR processing...")
            
            image = Image.open(path) #The image is saved with a unique timestamped filename to prevent collisions, opened with Pillow, 
            
            image_width, image_height = image.size
            print(f"[OCR] Image dimensions: {image_width}x{image_height}")
            
            if image.mode != 'RGB': #converted to RGB mode if needed, and processed through Tesseract
                image = image.convert('RGB')
            
            text = pytesseract.image_to_string(image) #OCR text extraction
            
            print(f"[OCR] OCR completed. Text found: {len(text.strip())} characters")
            print(f"[OCR] Extracted text: {text.strip()}")
            
            try:
                if os.path.exists(path):
                    os.remove(path)
                    print("[OCR] Temporary file cleaned up")
            except Exception as cleanup_error:
                print(f"[OCR] Warning: Could not clean up file: {cleanup_error}")
            
            cleaned_text = text.strip()
            
            if not cleaned_text:
                no_text_message = "No text could be detected in the image. Please try a clearer image with visible text."
                
                log_ocr_usage(
                    session.get('user_id'),
                    0,
                    False,
                    "No text detected in image"
                )
                
                return jsonify({
                    "status": "success", 
                    "text": no_text_message,
                    "spoken": False,
                    "characters": 0
                })
            
            log_ocr_usage(
                session.get('user_id'),
                len(cleaned_text),
                True,
                None
            )
            
            speech_text = cleaned_text
            if len(speech_text) > 500:
                speech_text = speech_text[:500] + "... (text truncated for speech)"
                print(f"[OCR] Text truncated from {len(cleaned_text)} to 500 characters for speech")
            
            threading.Thread(target=speak, args=(speech_text,)).start()
            print("[OCR] Text sent to speech synthesis")
            
            return jsonify({
                "status": "success", 
                "text": cleaned_text,
                "spoken": True,
                "characters": len(cleaned_text),
                "image_dimensions": f"{image_width}x{image_height}",
                "file_type": file_extension,
                "timestamp": datetime.now().isoformat()
            })
            
        except Exception as ocr_error:
            print(f"[OCR] OCR processing error: {str(ocr_error)}")
            import traceback
            traceback.print_exc()
            
            log_ocr_usage(
                session.get('user_id'),
                0,
                False,
                f"OCR processing failed: {str(ocr_error)}"
            )
            
            try:
                if os.path.exists(path):
                    os.remove(path)
            except:
                pass
            return jsonify({"status": "error", "error": f"OCR processing failed: {str(ocr_error)}"}), 500

    except Exception as e:
        print(f"[OCR] General error: {str(e)}")
        import traceback
        traceback.print_exc()
        
        log_ocr_usage(
            session.get('user_id'),
            0,
            False,
            f"General OCR error: {str(e)}"
        )
        
        return jsonify({"status": "error", "error": f"Upload failed: {str(e)}"}), 500
    
import requests
import json

@app.route('/environment', methods=['GET']) #Provides real-time outdoor safety assessment by fetching weather data from the Open-Meteo API 
@login_required
def environment_risk():
    """Environment risk prediction based on real weather data from Open-Meteo"""
    try:
        hour = datetime.now().hour
        
        user_id = session.get('user_id')
        lat, lng = get_user_location(user_id)
        
        weather_data = fetch_weather_data(lat, lng)
        
        if not weather_data:
            return fallback_environment_risk()
        
        temperature = weather_data.get('temperature', 20)
        weather_code = weather_data.get('weather_code', 0)
        is_day = weather_data.get('is_day', 1)
        precipitation = weather_data.get('precipitation', 0)
        cloud_cover = weather_data.get('cloud_cover', 50)
        
        weather_description = get_weather_description(weather_code)
        
        light_level = calculate_light_level(hour, is_day, cloud_cover)
        
        score = 100 #and computing a risk score between 0 and 100.
        
        if precipitation > 5:  
            score -= 30
        elif precipitation > 2:
            score -= 15
        elif precipitation > 0.5:
            score -= 10
            
        if weather_code in [71, 73, 75, 77, 85, 86]:  
            score -= 40
        elif weather_code in [51, 53, 55, 56, 57, 61, 63, 65, 66, 67, 80, 81, 82]: 
            score -= 25
        elif weather_code in [45, 48]:  
            score -= 30
        elif weather_code in [95, 96, 99]:  
            score -= 50
            
        if light_level < 30:
            score -= 40
        elif light_level < 60:
            score -= 20
            
        if hour < 6 or hour > 20:  
            score -= 30
        elif hour < 8 or hour > 18:  
            score -= 15
            
        score = max(0, min(100, score))
        
        if score > 75:
            level = "Safe"
            color = "green"
            advice = "Good conditions for navigation"
        elif score > 50:
            level = "Moderate"
            color = "orange"
            advice = "Proceed with caution"
        elif score > 25:
            level = "Risky"
            color = "red"
            advice = "High risk"
        else:
            level = "Dangerous"
            color = "darkred"
            advice = "Very dangerous conditions - avoid travel"

        msg = f"Environment safety check: {level}. Current conditions: {weather_description}, {temperature}°C, {light_level}% light level. {advice}"
        
        try:
            engine = pyttsx3.init()
            engine.say(msg)
            engine.runAndWait()
        except Exception as e:
            print(f"Speech synthesis error: {e}")

        log_environment_risk(
            user_id,
            level,
            score,
            weather_description,
            light_level,
            advice
        )

        return jsonify({
            "status": "success",
            "weather": weather_description,
            "temperature": temperature,
            "light_level": light_level,
            "precipitation": precipitation,
            "hour": hour,
            "risk_level": level,
            "risk_color": color,
            "score": score,
            "advice": advice,
            "message": msg,
            "is_real_data": True
        })
        
    except Exception as e:
        print(f"Environment risk prediction error: {e}")
        return fallback_environment_risk()

def get_user_location(user_id):
    """Get user location from session or use default"""
    if user_id and user_id in user_locations:
        location = user_locations[user_id]
        if time.time() - location['timestamp'] < 3600: 
            return location['lat'], location['lng']
    
    return 6.9271, 79.8612  

def fetch_weather_data(lat, lng): #Calls the Open-Meteo forecast endpoint with the user's GPS coordinates
    """Fetch weather data from Open-Meteo API"""
    try:
        url = f"https://api.open-meteo.com/v1/forecast" #Weather API call
        params = {
            'latitude': lat,
            'longitude': lng,
            'current': 'temperature_2m,relative_humidity_2m,precipitation,weather_code,cloud_cover,is_day',
            'timezone': 'auto'
        }
        
        response = requests.get(url, params=params, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            current = data.get('current', {})
            
            return {
                'temperature': current.get('temperature_2m', 20),
                'weather_code': current.get('weather_code', 0),
                'is_day': current.get('is_day', 1),
                'precipitation': current.get('precipitation', 0),
                'cloud_cover': current.get('cloud_cover', 50)
            }
        else:
            print(f"Open-Meteo API error: {response.status_code}")
            return None
            
    except Exception as e:
        print(f"Error fetching weather data: {e}")
        return None

def get_weather_description(weather_code):
    """Convert WMO weather code to human-readable description"""
    weather_descriptions = {
        0: "Clear Sky",
        1: "Mainly Clear", 
        2: "Partly Cloudy",
        3: "Overcast",
        45: "Fog",
        48: "Depositing Rime Fog",
        51: "Light Drizzle",
        53: "Moderate Drizzle",
        55: "Dense Drizzle",
        56: "Light Freezing Drizzle",
        57: "Dense Freezing Drizzle",
        61: "Slight Rain",
        63: "Moderate Rain",
        65: "Heavy Rain",
        66: "Light Freezing Rain",
        67: "Heavy Freezing Rain",
        71: "Slight Snow Fall",
        73: "Moderate Snow Fall",
        75: "Heavy Snow Fall",
        77: "Snow Grains",
        80: "Slight Rain Showers",
        81: "Moderate Rain Showers",
        82: "Violent Rain Showers",
        85: "Slight Snow Showers",
        86: "Heavy Snow Showers",
        95: "Thunderstorm",
        96: "Thunderstorm with Slight Hail",
        99: "Thunderstorm with Heavy Hail"
    }
    
    return weather_descriptions.get(weather_code, "Unknown Weather Conditions")

def calculate_light_level(hour, is_day, cloud_cover):
    """Calculate light level based on time of day and cloud cover"""
    if not is_day: 
        base_light = 10
    elif 6 <= hour <= 18:  
        base_light = 80
    else: 
        base_light = 40
    
    light_reduction = cloud_cover * 0.4  
    light_level = max(10, base_light - light_reduction)
    
    return round(light_level)

def fallback_environment_risk():
    """Fallback function with random data when API fails"""
    hour = datetime.now().hour
    weather = random.choice(["Clear", "Rain", "Fog", "Cloudy"])
    light_level = random.randint(10, 100)

    score = 100
    
    if weather == "Rain":
        score -= 25
    elif weather == "Fog":
        score -= 20
    elif weather == "Cloudy":
        score -= 10
        
    if light_level < 30:
        score -= 40
    elif light_level < 60:
        score -= 20
        
    if hour < 6 or hour > 20:
        score -= 30
    elif hour < 8 or hour > 18:
        score -= 15

    if score > 75: #Risk Score Calculation 
        level = "Safe"
        color = "green"
        advice = "Good conditions for navigation"
    elif score > 50:
        level = "Moderate"
        color = "orange"
        advice = "Proceed with caution"
    elif score > 25:
        level = "Risky"
        color = "red"
        advice = "High risk"
    else:
        level = "Dangerous"
        color = "darkred"
        advice = "Very dangerous conditions - avoid travel"

    msg = f"Environment safety check: {level}. Current conditions: {weather} weather, {light_level}% light level. {advice}"
    
    try:
        engine = pyttsx3.init()
        engine.say(msg)
        engine.runAndWait()
    except Exception as e:
        print(f"Speech synthesis error: {e}")

    log_environment_risk(
        session.get('user_id'),
        level,
        score,
        weather,
        light_level,
        advice
    )

    return jsonify({
        "status": "success",
        "weather": weather,
        "light_level": light_level,
        "hour": hour,
        "risk_level": level,
        "risk_color": color,
        "score": score,
        "advice": advice,
        "message": msg,
        "is_real_data": False
    })
    
@app.route('/caretaker_dashboard')
@login_required
@caretaker_required
def caretaker_dashboard():
    """Caretaker dashboard page"""
    user_locations_data = {}
    for user_id, location in user_locations.items():
        user_locations_data[user_id] = {
            'user_id': user_id,
            'lat': location['lat'],
            'lng': location['lng'],
            'timestamp': location['timestamp']
        }

    conn = get_db_connection()
    if conn:
        try:
            with conn.cursor() as cursor:
                cursor.execute("SELECT id, username, email FROM users WHERE role = 'user'")
                users = cursor.fetchall()
                
                for user in users:
                    user_id = user['id']
                    if user_id in user_locations:
                        location = user_locations[user_id]
                        user_locations_data[user_id] = {
                            'lat': location['lat'],
                            'lng': location['lng'],
                            'timestamp': location['timestamp'],
                            'username': user['username'],
                            'email': user['email']
                        }
        finally:
            conn.close()

    user_locations_json = json.dumps(user_locations_data)
    
    return render_template('caretaker_dashboard.html', 
                         user_locations_json=user_locations_json)

@app.route('/api/caretaker/analytics') #Data from all activity log tables over the past 7 days using SQL GROUP BY queries
@login_required
@caretaker_required
def caretaker_analytics():
    """Get analytics data for caretaker dashboard"""
    try:
        conn = get_db_connection()
        if not conn:
            return jsonify({"status": "error", "message": "Database connection failed"}), 500

        with conn.cursor() as cursor:
            cursor.execute("SELECT COUNT(*) as total_users FROM users WHERE role = 'user'")
            total_users = cursor.fetchone()['total_users']

            cursor.execute("SELECT COUNT(*) as active_today FROM users WHERE last_login >= CURDATE()")
            active_today = cursor.fetchone()['active_today']

            cursor.execute("""
                SELECT detection_type, COUNT(*) as count, 
                       AVG(confidence) as avg_confidence 
                FROM obstacle_detection_logs 
                WHERE timestamp >= DATE_SUB(NOW(), INTERVAL 7 DAY)
                GROUP BY detection_type
            """)
            obstacle_stats = cursor.fetchall()

            cursor.execute("""
                SELECT risk_level, COUNT(*) as count, 
                       AVG(risk_score) as avg_score 
                FROM environment_risk_logs 
                WHERE timestamp >= DATE_SUB(NOW(), INTERVAL 7 DAY)
                GROUP BY risk_level
            """)
            environment_stats = cursor.fetchall()

            cursor.execute("""
                SELECT DATE(timestamp) as date, 
                       COUNT(*) as usage_count,
                       AVG(characters_extracted) as avg_characters
                FROM ocr_usage_logs 
                WHERE timestamp >= DATE_SUB(NOW(), INTERVAL 7 DAY)
                GROUP BY DATE(timestamp)
                ORDER BY date
            """)
            ocr_stats = cursor.fetchall()

            cursor.execute("""
                SELECT DATE(timestamp) as date,
                       COUNT(DISTINCT user_id) as active_users,
                       COUNT(*) as total_events
                FROM (
                    SELECT user_id, timestamp FROM obstacle_detection_logs 
                    UNION ALL 
                    SELECT user_id, timestamp FROM environment_risk_logs 
                    UNION ALL 
                    SELECT user_id, timestamp FROM ocr_usage_logs
                ) AS combined_events
                WHERE timestamp >= DATE_SUB(NOW(), INTERVAL 7 DAY)
                GROUP BY DATE(timestamp)
                ORDER BY date
            """)
            daily_activity = cursor.fetchall()

        return jsonify({
            "status": "success",
            "analytics": {
                "total_users": total_users,
                "active_today": active_today,
                "obstacle_stats": obstacle_stats,
                "environment_stats": environment_stats,
                "ocr_stats": ocr_stats,
                "daily_activity": daily_activity
            }
        })

    except Exception as e:
        print(f"Analytics error: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500
    finally:
        conn.close()

@app.route('/api/caretaker/navigation_analytics')
@login_required
@caretaker_required
def navigation_analytics():
    """Get navigation analytics for caretaker dashboard"""
    try:
        conn = get_db_connection()
        if not conn:
            return jsonify({"status": "error", "message": "Database connection failed"}), 500

        with conn.cursor() as cursor:
            cursor.execute("""
                SELECT 
                    COUNT(*) as total_navigations,
                    SUM(completed) as completed_navigations,
                    AVG(safety_score) as avg_safety_score,
                    AVG(distance_km) as avg_distance
                FROM navigation_logs 
                WHERE timestamp >= DATE_SUB(NOW(), INTERVAL 7 DAY)
            """)
            nav_stats = cursor.fetchone()

            cursor.execute("""
                SELECT 
                    CASE 
                        WHEN safety_score >= 75 THEN 'Safe'
                        WHEN safety_score >= 50 THEN 'Moderate' 
                        WHEN safety_score >= 25 THEN 'Risky'
                        ELSE 'Dangerous'
                    END as safety_level,
                    COUNT(*) as count,
                    AVG(distance_km) as avg_distance,
                    AVG(duration_minutes) as avg_duration
                FROM navigation_logs 
                WHERE timestamp >= DATE_SUB(NOW(), INTERVAL 7 DAY)
                GROUP BY safety_level
                ORDER BY 
                    CASE safety_level
                        WHEN 'Safe' THEN 1
                        WHEN 'Moderate' THEN 2
                        WHEN 'Risky' THEN 3
                        ELSE 4
                    END
            """)
            safety_distribution = cursor.fetchall()

            cursor.execute("""
                SELECT 
                    DATE(timestamp) as date,
                    COUNT(*) as navigation_count,
                    AVG(safety_score) as avg_safety_score,
                    AVG(distance_km) as avg_distance
                FROM navigation_logs 
                WHERE timestamp >= DATE_SUB(NOW(), INTERVAL 7 DAY)
                GROUP BY DATE(timestamp)
                ORDER BY date
            """)
            daily_trends = cursor.fetchall()

            cursor.execute("""
                SELECT u.username, nl.safety_score, nl.distance_km, 
                       nl.timestamp, nl.start_lat, nl.start_lng,
                       nl.end_lat, nl.end_lng
                FROM navigation_logs nl
                JOIN users u ON nl.user_id = u.id
                WHERE nl.safety_score < 50 
                AND nl.timestamp >= DATE_SUB(NOW(), INTERVAL 24 HOUR)
                ORDER BY nl.safety_score ASC
                LIMIT 10
            """)
            high_risk_navigations = cursor.fetchall()

        return jsonify({
            "status": "success",
            "navigation_analytics": {
                "total_navigations": nav_stats['total_navigations'],
                "completion_rate": round((nav_stats['completed_navigations'] / nav_stats['total_navigations']) * 100, 1) if nav_stats['total_navigations'] > 0 else 0,
                "avg_safety_score": round(nav_stats['avg_safety_score'], 1) if nav_stats['avg_safety_score'] else 0,
                "avg_distance": round(nav_stats['avg_distance'], 2) if nav_stats['avg_distance'] else 0,
                "safety_distribution": safety_distribution,
                "daily_trends": daily_trends,
                "high_risk_navigations": high_risk_navigations
            }
        })

    except Exception as e:
        print(f"Navigation analytics error: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500
    finally:
        conn.close()

@app.route('/api/caretaker/user_activity')
@login_required
@caretaker_required
def user_activity():
    """Get detailed user activity data"""
    try:
        conn = get_db_connection()
        if not conn:
            return jsonify({"status": "error", "message": "Database connection failed"}), 500

        with conn.cursor() as cursor:
            cursor.execute("""
                SELECT u.username, u.email, u.last_login,
                       COUNT(DISTINCT odl.id) as obstacle_detections,
                       COUNT(DISTINCT erl.id) as risk_checks,
                       COUNT(DISTINCT oul.id) as ocr_uses,
                       MAX(COALESCE(odl.timestamp, erl.timestamp, oul.timestamp)) as last_activity
                FROM users u
                LEFT JOIN obstacle_detection_logs odl ON u.id = odl.user_id
                LEFT JOIN environment_risk_logs erl ON u.id = erl.user_id
                LEFT JOIN ocr_usage_logs oul ON u.id = oul.user_id
                WHERE u.role = 'user'
                GROUP BY u.id, u.username, u.email, u.last_login
                ORDER BY last_activity DESC
            """)
            user_activity = cursor.fetchall()

        return jsonify({
            "status": "success",
            "user_activity": user_activity
        })

    except Exception as e:
        print(f"User activity error: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500
    finally:
        conn.close()

@app.route('/api/caretaker/risk_alerts')
@login_required
@caretaker_required
def risk_alerts():
    """Get high-risk alerts"""
    try:
        conn = get_db_connection()
        if not conn:
            return jsonify({"status": "error", "message": "Database connection failed"}), 500

        with conn.cursor() as cursor:
            cursor.execute("""
                SELECT u.username, erl.risk_level, erl.risk_score, erl.weather, 
                       erl.light_level, erl.advice, erl.timestamp
                FROM environment_risk_logs erl
                JOIN users u ON erl.user_id = u.id
                WHERE erl.risk_level IN ('Risky', 'Dangerous')
                AND erl.timestamp >= DATE_SUB(NOW(), INTERVAL 24 HOUR)
                ORDER BY erl.timestamp DESC
            """)
            risk_alerts = cursor.fetchall()

        return jsonify({
            "status": "success",
            "risk_alerts": risk_alerts
        })

    except Exception as e:
        print(f"Risk alerts error: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500
    finally:
        conn.close()
    
def calculate_route_safety_score(start_lat, start_lng, end_lat, end_lng, weather_data, hour_of_day):
    """Calculate comprehensive route safety score (0-100)"""
    
    score = 100
    
    distance_km = calculate_distance_km(start_lat, start_lng, end_lat, end_lng)
    if distance_km > 10:
        score -= 30
    elif distance_km > 5:
        score -= 20
    elif distance_km > 2:
        score -= 10
    elif distance_km > 1:
        score -= 5
    
    if hour_of_day < 6 or hour_of_day > 20:  
        score -= 25
    elif hour_of_day < 8 or hour_of_day > 18:  
        score -= 15
    
    if weather_data.get('precipitation', 0) > 5:
        score -= 20
    elif weather_data.get('precipitation', 0) > 2:
        score -= 10
    
    weather_code = weather_data.get('weather_code', 0)
    if weather_code in [71, 73, 75, 77, 85, 86]:  
        score -= 25
    elif weather_code in [51, 53, 55, 61, 63, 65, 80, 81, 82]: 
        score -= 15
    elif weather_code in [45, 48]:  
        score -= 20
    
    route_complexity = estimate_route_complexity(start_lat, start_lng, end_lat, end_lng)
    score -= route_complexity * 5
    
    return max(0, min(100, score))

def calculate_distance_km(lat1, lon1, lat2, lon2):
    """Calculate distance between two points in kilometers"""
    R = 6371  
    
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1) #Haversine formula:to estimate distance, then applies deductions for long distances, time of day, weather conditions, and estimated route complexity
    
    a = (math.sin(dlat/2) * math.sin(dlat/2) + 
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * 
         math.sin(dlon/2) * math.sin(dlon/2))
    
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    return R * c

def estimate_route_complexity(lat1, lon1, lat2, lon2):
    """Estimate route complexity based on coordinates (simulated)"""

    distance = calculate_distance_km(lat1, lon1, lat2, lon2)
    
    if distance < 1:
        return 1  
    elif distance < 3:
        return 2  
    else:
        return 3  

def estimate_navigation_time(distance_km, weather_condition):
    """Estimate navigation time in minutes"""
    base_speed = 5
    
    if weather_condition in ["Rain", "Snow", "Fog"]:
        base_speed *= 0.7  
    elif weather_condition in ["Clear", "Mainly Clear"]:
        base_speed *= 1.1  
    
    time_hours = distance_km / base_speed
    return int(time_hours * 60)     

@app.route('/start_navigation', methods=['POST'])
@login_required
def start_navigation():
    """Start a navigation session with safety prediction"""
    try:
        user_id = session.get('user_id')
        data = request.get_json()

        print(f"Received navigation data: {data}")
        
        if not data or 'start_lat' not in data or 'start_lng' not in data or 'end_lat' not in data or 'end_lng' not in data:
            return jsonify({"status": "error", "message": "Missing navigation data"}), 400
        
        try:
            start_lat = float(data['start_lat'])
            start_lng = float(data['start_lng'])
            end_lat = float(data['end_lat'])
            end_lng = float(data['end_lng'])
        except ValueError as e:
            return jsonify({"status": "error", "message": f"Invalid coordinates: {str(e)}"}), 400
        
        weather_data = fetch_weather_data(start_lat, start_lng)
        if not weather_data:
            weather_data = {
                'temperature': 20,
                'weather_code': 0,
                'precipitation': 0,
                'weather_description': 'clear sky'
            }
        
        hour_of_day = datetime.now().hour
        safety_score = calculate_route_safety_score(
            start_lat, start_lng,
            end_lat, end_lng,
            weather_data, hour_of_day
        )
        
        distance_km = float(data.get('distance_km') or calculate_distance_km(start_lat, start_lng, end_lat, end_lng))

        LONG_DISTANCE_THRESHOLD_KM = 100
        long_distance_warning = None
        if distance_km > LONG_DISTANCE_THRESHOLD_KM:
            long_distance_warning = (
                f"Warning: Your destination is {round(distance_km, 1)} km away. "
                "This is an extremely long distance for a walking navigation route. "
                "Please ensure you have planned transportation, sufficient rest stops, "
                "and that this journey is safe and intentional."
            )
        
        estimated_time = estimate_navigation_time(distance_km, weather_data.get('weather_description', 'clear'))
        
        nav_id = log_navigation(
            user_id=user_id,
            start_lat=start_lat,
            start_lng=start_lng,
            end_lat=end_lat, 
            end_lng=end_lng,
            distance_km=distance_km,
            duration_minutes=estimated_time,
            safety_score=safety_score,
            completed=False
        )

        if not nav_id:
            return jsonify({"status": "error", "message": "Failed to save navigation to database"}), 500
        
        if safety_score > 75:
            safety_level = "Safe"
            advice = "Good conditions for navigation"
        elif safety_score > 50:
            safety_level = "Moderate"
            advice = "Proceed with caution"
        elif safety_score > 25:
            safety_level = "Risky"
            advice = "High risk"
        else:
            safety_level = "Dangerous"
            advice = "Very dangerous conditions - avoid travel"
        
        response_data = {
            "status": "success",
            "safety_score": safety_score,
            "safety_level": safety_level,
            "distance_km": round(distance_km, 2),
            "estimated_time_minutes": estimated_time,
            "advice": advice,
            "weather_condition": weather_data.get('weather_description', 'Unknown'),
            "navigation_id": nav_id,
            "is_long_distance": bool(long_distance_warning)
        }

        if long_distance_warning:
            response_data["long_distance_warning"] = long_distance_warning

        return jsonify(response_data)
        
    except Exception as e:
        print(f"Navigation start error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)}), 500
    
@app.route('/debug_navigation', methods=['GET'])
@login_required
def debug_navigation():
    """Debug endpoint to check navigation logs"""
    user_id = session.get('user_id')
    conn = get_db_connection()
    
    if not conn:
        return jsonify({"status": "error", "message": "Database connection failed"}), 500
    
    try:
        with conn.cursor() as cursor:
            cursor.execute("SHOW TABLES LIKE 'navigation_logs'")
            table_exists = cursor.fetchone()
            
            if not table_exists:
                return jsonify({"status": "error", "message": "navigation_logs table does not exist"})
            
            cursor.execute("SELECT COUNT(*) as count FROM navigation_logs WHERE user_id = %s", (user_id,))
            count_result = cursor.fetchone()
            
            cursor.execute("SELECT * FROM navigation_logs WHERE user_id = %s ORDER BY timestamp DESC LIMIT 5", (user_id,))
            recent_records = cursor.fetchall()
            
            return jsonify({
                "status": "success",
                "table_exists": True,
                "total_records": count_result['count'],
                "recent_records": recent_records
            })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500
    finally:
        conn.close()

@app.route('/complete_navigation/<int:nav_id>', methods=['POST'])
@login_required
def complete_navigation(nav_id):
    """Mark a navigation session as completed"""
    try:
        user_id = session.get('user_id')
        conn = get_db_connection()
        
        if not conn:
            return jsonify({"status": "error", "message": "Database connection failed"}), 500
        
        with conn.cursor() as cursor:
            cursor.execute(
                "UPDATE navigation_logs SET completed = TRUE WHERE id = %s AND user_id = %s",
                (nav_id, user_id)
            )
            conn.commit()
            
        return jsonify({"status": "success", "message": "Navigation completed"})
        
    except Exception as e:
        print(f"Navigation completion error: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500
    finally:
        if conn:
            conn.close()

@app.route('/get_active_navigations', methods=['GET'])
@login_required
def get_active_navigations():
    """Get active navigation sessions for the user"""
    try:
        user_id = session.get('user_id')
        conn = get_db_connection()
        
        if not conn:
            return jsonify({"status": "error", "message": "Database connection failed"}), 500
        
        with conn.cursor() as cursor:
            cursor.execute("""
                SELECT id, start_lat, start_lng, end_lat, end_lng, 
                       distance_km, safety_score, timestamp
                FROM navigation_logs 
                WHERE user_id = %s AND completed = FALSE
                ORDER BY timestamp DESC
            """, (user_id,))
            
            active_navigations = cursor.fetchall()
            
        return jsonify({
            "status": "success",
            "active_navigations": active_navigations
        })
        
    except Exception as e:
        print(f"Get active navigations error: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500
    finally:
        if conn:
            conn.close()

def get_last_navigation_id(user_id):
    """Get the ID of the most recent navigation session for a user"""
    conn = get_db_connection()
    if not conn:
        return None
    
    try:
        with conn.cursor() as cursor:
            cursor.execute("""
                SELECT id FROM navigation_logs 
                WHERE user_id = %s 
                ORDER BY timestamp DESC 
                LIMIT 1
            """, (user_id,))
            
            result = cursor.fetchone()
            return result['id'] if result else None
            
    except Exception as e:
        print(f"Get last navigation ID error: {e}")
        return None
    finally:
        conn.close()

@app.route('/api/caretaker/user_routes/<int:user_id>', methods=['GET'])
@login_required
@caretaker_required
def user_routes(user_id):
    """Get navigation routes for a specific user"""
    try:
        conn = get_db_connection()
        if not conn:
            return jsonify({"status": "error", "message": "Database connection failed"}), 500

        with conn.cursor() as cursor:
            cursor.execute("""
                SELECT id, start_lat, start_lng, end_lat, end_lng, 
                       distance_km, safety_score, completed, timestamp
                FROM navigation_logs 
                WHERE user_id = %s
                ORDER BY timestamp DESC
                LIMIT 20
            """, (user_id,))
            
            routes = cursor.fetchall()
            
            for route in routes:
                for key in ['start_lat', 'start_lng', 'end_lat', 'end_lng', 'distance_km', 'safety_score']:
                    if route[key] is not None:
                        route[key] = float(route[key])
            
        return jsonify({
            "status": "success",
            "routes": routes
        })
        
    except Exception as e:
        print(f"Error getting user routes: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500
    finally:
        conn.close()

@app.route('/api/caretaker/user_locations', methods=['GET'])
@login_required
@caretaker_required
def get_user_locations_api():
    """API endpoint to get all user locations"""
    try:
        conn = get_db_connection()
        locations = []
        
        if conn:
            with conn.cursor() as cursor:
                cursor.execute("SELECT id, username, email FROM users WHERE role = 'user'")
                users = cursor.fetchall()
                
                for user in users:
                    user_id = user['id']
                    if user_id in user_locations:
                        location = user_locations[user_id]
                        is_active = (time.time() - location['timestamp']) < 300 
                        
                        locations.append({
                            'user_id': user_id,
                            'username': user['username'],
                            'email': user['email'],
                            'latitude': location['lat'],
                            'longitude': location['lng'],
                            'timestamp': location['timestamp'],
                            'is_active': is_active
                        })
        
        return jsonify({
            "status": "success",
            "locations": locations
        })
        
    except Exception as e:
        print(f"Error getting user locations: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/caretaker/user_locations')
@login_required
@caretaker_required
def get_user_locations():
    """Get current locations of all users from memory"""
    try:
        conn = get_db_connection()
        if not conn:
            return jsonify({"status": "error", "message": "Database connection failed"}), 500

        with conn.cursor() as cursor:
            cursor.execute("""
                SELECT id, username, email, last_login
                FROM users 
                WHERE role = 'user'
            """)
            
            users = cursor.fetchall()
            
        locations = []
        for user in users:
            user_id = user['id']
            if user_id in user_locations:
                loc = user_locations[user_id]
                locations.append({
                    'user_id': user_id,
                    'username': user['username'],
                    'email': user['email'],
                    'latitude': loc['lat'],
                    'longitude': loc['lng'],
                    'timestamp': loc['timestamp'],
                    'is_active': (time.time() - loc['timestamp']) < 300 
                })
        
        return jsonify({
            "status": "success",
            "locations": locations
        })
        
    except Exception as e:
        print(f"Error getting user locations: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500
    finally:
        conn.close()

if __name__ == '__main__':
    init_database()
    app.run(debug=True)

