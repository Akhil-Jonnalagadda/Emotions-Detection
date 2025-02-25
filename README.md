# Emotions Detection 

## Overview
This project is a Django-based web application for emotion detection using image processing and machine learning techniques. The system uses a webcam to capture real-time facial expressions and analyzes them to detect emotions. Based on the detected emotion, it plays a corresponding song.

## Features
- Uses webcam to capture real-time facial expressions
- AI-based facial expression recognition
- Web-based interface using Django
- Plays a song based on detected emotion
- Stores results in an SQLite database

## Project Structure
```
/emotionsDetection
|-- db.sqlite3               # Database file
|-- manage.py                # Django project manager
|-- emotionsDetection/        # Main Django app
|-- player/                   # Secondary Django app/module
|-- static/                   # Static files (CSS, JS, images)
|-- templates/                # HTML templates for UI
|-- Train/                    # Training data for emotion recognition
```

## How to Set Up
### 1. Install Dependencies
Make sure you have Python 3.x installed. Then, install the required dependencies:
```bash
pip install -r requirements.txt
```
(If `requirements.txt` is missing, manually install Django using `pip install django`)

### 2. Set Up the Database
Run the following command to apply migrations:
```bash
python manage.py migrate
```

### 3. Run the Web Application
Start the Django development server:
```bash
python manage.py runserver
```
## Testing Emotion Detection
1. Open the web application in your browser.
2. Allow camera access for real-time facial expression detection.
3. The system will analyze your facial expression and play a song accordingly.



