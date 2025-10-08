#imports

import os
import random
import logging
from datetime import datetime, timedelta, timezone
from typing import Dict, Any
import json
import uuid

from flask import Flask, render_template, request, session, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_socketio import SocketIO, emit, join_room, leave_room
from werkzeug.middleware.proxy_fix import ProxyFix
import firebase_admin
from firebase_admin import auth as fb_auth, credentials as fb_credentials
# from transformers import pipeline # Import pipeline # Removed transformers pipeline
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer # Import VADER
# from predict_stress import predict_stress # Removed
import cv2
import numpy as np


# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s" 
)
logger=logging.getLogger(__name__)

class Config:
    SECRET_KEY=os.environ.get("SECRET_KEY") or os.urandom(24)
    DEBUG=os.environ.get("FLASK_DEBUG","False").lower() in ('true','1','t')
    CORS_ORIGINS=os.environ.get("CORS_ORIGINS", "*")
    SQLALCHEMY_DATABASE_URI=os.environ.get("DATABASE_URL") or f"sqlite:///{os.path.join(os.path.dirname(__file__), 'app.db')}"
    SQLALCHEMY_TRACK_MODIFICATIONS=False

    #chatrooms
    CHAT_ROOMS={
        "🧠 StressScope",
        "⚙️ NeuroGauge",
        "📊 MindMetric",
        "💡 StressSense"
    }

app=Flask(__name__)
app.config.from_object(Config)

#Handle Reverse Proxy
app.wsgi_app=ProxyFix(app.wsgi_app, x_proto=1, x_host=1)

# Database
db=SQLAlchemy(app)

#Set up the Socket
socketio=SocketIO(app, cors_allowed_origins=app.config["CORS_ORIGINS"],logger=True,engine_io_logger=True)

# Models
class User(db.Model):
    id=db.Column(db.Integer, primary_key=True)
    firebase_uid=db.Column(db.String(128), unique=True, index=True)
    username=db.Column(db.String(80), index=True)
    created_at=db.Column(db.DateTime, default=datetime.utcnow)

class Chat(db.Model):
    id=db.Column(db.Integer, primary_key=True)
    user_id=db.Column(db.Integer, db.ForeignKey('user.id'), nullable=True)
    username=db.Column(db.String(80))
    room=db.Column(db.String(120), index=True)
    message=db.Column(db.Text, nullable=False)
    timestamp=db.Column(db.DateTime, default=datetime.utcnow, index=True)

class VideoFrame(db.Model):
    id=db.Column(db.Integer, primary_key=True)
    user_id=db.Column(db.Integer, db.ForeignKey('user.id'), nullable=True)
    frame_path=db.Column(db.String(512), nullable=False)
    captured_at=db.Column(db.DateTime, default=datetime.utcnow, index=True)

class Analysis(db.Model):
    id=db.Column(db.Integer, primary_key=True)
    user_id=db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    computed_at=db.Column(db.DateTime, default=datetime.utcnow, index=True)
    chat_score=db.Column(db.Float)
    video_score=db.Column(db.Float)
    combined_score=db.Column(db.Float)
    details_json=db.Column(db.Text)  # serialized breakdown for charts

# Firebase Admin init (optional; only if service account is provided)
FIREBASE_CREDS_PATH=os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
if FIREBASE_CREDS_PATH and os.path.exists(FIREBASE_CREDS_PATH):
    try:
        firebase_admin.initialize_app(fb_credentials.Certificate(FIREBASE_CREDS_PATH))
        logger.info("Firebase Admin initialized with service account")
    except Exception as init_err:
        logger.error(f"Firebase Admin init failed: {str(init_err)}")
else:
    logger.warning("Firebase Admin not initialized (no GOOGLE_APPLICATION_CREDENTIALS)")

#Make a database
active_users: Dict[str, dict]={}
ANALYSIS_PROGRESS: Dict[str, int] = {}

#Make a user
def generate_guest_username() -> str:
    timestamp=datetime.now().strftime("%H%M")
    return f"Guest{timestamp}{random.randint(1000,9999)}"

#Home Route
@app.route("/")
def index():
    # Do not auto-generate a guest username; ask user on UI first
    current_username=session.get("username", "")
    if 'client_id' not in session:
        session['client_id'] = str(uuid.uuid4())
    return render_template("chat.html", username=current_username, rooms=app.config["CHAT_ROOMS"])

@app.post('/upload_frame')
def upload_frame():
    try:
        if 'frame' not in request.files:
            return jsonify({"error":"missing frame"}), 400
        file = request.files['frame']
        if not file.filename:
            return jsonify({"error":"no filename"}), 400

        # resolve user for storage path
        uid = session.get('firebase_uid') or 'guest'
        frames_dir = os.path.join(os.path.dirname(__file__), 'static', 'frames', uid)
        os.makedirs(frames_dir, exist_ok=True)
        ts = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        filename = f"{ts}.jpg"
        save_path = os.path.join(frames_dir, filename)
        file.save(save_path)

        # record in DB
        user_id=None
        if 'firebase_uid' in session:
            user=db.session.execute(db.select(User).filter_by(firebase_uid=session['firebase_uid'])).scalar_one_or_none()
            if user:
                user_id=user.id
        vf = VideoFrame(user_id=user_id, frame_path=os.path.relpath(save_path, os.path.dirname(__file__)))
        db.session.add(vf)
        db.session.commit()
        return jsonify({"ok":True})
    except Exception as e:
        db.session.rollback()
        logger.error(f"upload_frame error: {str(e)}")
        return jsonify({"error":"server error"}), 500

def _map_emotion_to_stress(emotion: str, confidence: float = 0.5) -> int:
    """Map emotions to stress scores (0-100, higher = more stressed) with confidence adjustment"""
    base_mapping = {
        'happy': 5,      # Very low stress
        'neutral': 25,   # Low stress
        'surprise': 35,  # Mild stress
        'sad': 65,       # High stress
        'disgust': 70,   # High stress
        'fear': 80,      # Very high stress
        'angry': 90,     # Very high stress
    }
    
    base_score = base_mapping.get((emotion or '').lower(), 50)  # Default to moderate stress
    
    # Adjust score based on confidence
    # Low confidence = more uncertainty = slightly higher stress
    confidence_adjustment = (1 - confidence) * 10
    adjusted_score = base_score + confidence_adjustment
    
    return int(max(0, min(100, adjusted_score)))

def _analyze_emotion_opencv(image_path: str) -> Dict[str, Any]:
    """Analyze emotion using OpenCV Haar Cascades for face detection and basic analysis"""
    try:
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            return {'emotion': 'neutral', 'confidence': 0.0, 'stress_score': 50}
        
        # Convert to grayscale for face detection
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Load Haar cascade for face detection
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        
        if len(faces) == 0:
            return {'emotion': 'neutral', 'confidence': 0.0, 'stress_score': 50}
        
        # Get the largest face
        face = max(faces, key=lambda x: x[2] * x[3])
        x, y, w, h = face
        
        # Extract face region
        face_roi = gray[y:y+h, x:x+w]
        
        # Analyze facial features
        emotion, confidence = _analyze_face_features(face_roi)
        stress_score = _map_emotion_to_stress(emotion, confidence)
        
        return {
            'emotion': emotion,
            'confidence': confidence,
            'stress_score': stress_score
        }
        
    except Exception as e:
        logger.error(f"OpenCV emotion analysis error: {str(e)}")
        return {'emotion': 'neutral', 'confidence': 0.0, 'stress_score': 50}

def _analyze_face_features(face_roi):
    """Analyze facial features using OpenCV to determine emotion"""
    try:
        # Load eye and mouth cascades
        eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        mouth_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')
        
        # Detect eyes
        eyes = eye_cascade.detectMultiScale(face_roi, 1.1, 3)
        
        # Detect mouth/smile
        smiles = mouth_cascade.detectMultiScale(face_roi, 1.1, 3)
        
        # Analyze based on detected features
        if len(smiles) > 0:
            # Smile detected - likely happy
            return 'happy', 0.7
        elif len(eyes) >= 2:
            # Eyes detected, no smile - likely neutral or concerned
            # Check eye openness
            eye_openness = _calculate_eye_openness_opencv(eyes, face_roi)
            if eye_openness < 0.3:
                return 'fear', 0.6  # Squinting/concerned
            else:
                return 'neutral', 0.5
        else:
            # No clear features detected
            return 'neutral', 0.3
            
    except Exception as e:
        logger.error(f"Face feature analysis error: {str(e)}")
        return 'neutral', 0.3

def _calculate_eye_openness_opencv(eyes, face_roi):
    """Calculate eye openness from detected eyes"""
    if len(eyes) < 2:
        return 0.5
    
    # Get the two largest eyes
    eyes = sorted(eyes, key=lambda x: x[2] * x[3], reverse=True)[:2]
    
    total_openness = 0
    for (ex, ey, ew, eh) in eyes:
        # Extract eye region
        eye_roi = face_roi[ey:ey+eh, ex:ex+ew]
        
        # Calculate vertical gradient to estimate openness
        if eye_roi.size > 0:
            # Simple heuristic: more vertical edges = more open
            sobel_y = cv2.Sobel(eye_roi, cv2.CV_64F, 0, 1, ksize=3)
            openness = np.mean(np.abs(sobel_y)) / 255.0
            total_openness += openness
    
    return total_openness / len(eyes) if len(eyes) > 0 else 0.5

# Ensure VADER lexicon is downloaded
nltk.download('vader_lexicon', quiet=True)

# Initialize VADER sentiment analyzer
sentiment_analyzer = SentimentIntensityAnalyzer()

@app.post('/analysis')
def run_analysis():
    try:
        logger.info("Starting analysis...")
        data = request.get_json(silent=True) or {}
        room_filter = (data.get('room') or '').strip() or None
        since_iso = (data.get('since') or '').strip() or None
        since_dt = None
        if since_iso:
            try:
                since_dt = datetime.fromisoformat(since_iso.replace('Z','+00:00'))
            except Exception:
                logger.warning(f"Invalid since param: {since_iso}")
        client_id = session.get('client_id')
        if client_id:
            ANALYSIS_PROGRESS[client_id] = 0
        # Resolve user
        user_id=None
        if 'firebase_uid' in session:
            user=db.session.execute(db.select(User).filter_by(firebase_uid=session['firebase_uid'])).scalar_one_or_none()
            if user:
                user_id=user.id
        username=session.get('username')
        logger.info(f"Analysis for user: {username}, user_id: {user_id}")

        # Load chats for user with optional room and since filters
        if user_id is not None:
            stmt = db.select(Chat).filter_by(user_id=user_id)
        else:
            stmt = db.select(Chat).filter_by(username=username)
        if room_filter:
            stmt = stmt.filter(Chat.room == room_filter)
        if since_dt:
            stmt = stmt.filter(Chat.timestamp >= since_dt)
        chats = db.session.execute(stmt).scalars().all()
        
        logger.info(f"Found {len(chats)} chat messages for user {username} (user_id: {user_id}), room={room_filter}, since={since_dt}")
        
        # Debug: Log some sample messages
        for i, chat in enumerate(chats[:3]):  # Log first 3 messages
            logger.info(f"Sample message {i+1}: {chat.message[:50]}... (timestamp: {chat.timestamp})")

        # Determine chat session start and end times
        chat_start_time = since_dt or (min(c.timestamp for c in chats if c.timestamp) if chats else datetime.utcnow())
        chat_end_time = max(c.timestamp for c in chats if c.timestamp) if chats else datetime.utcnow()

        total_units = max(1, len(chats) + 1)  # +1 to avoid div by zero
        completed_units = 0

        chat_series=[]
        for i, c in enumerate(chats):
            try:
                # Use VADER sentiment analyzer
                logger.info(f"Analyzing message {i+1}/{len(chats)}: {c.message[:50]}...")
                sentiment_scores = sentiment_analyzer.polarity_scores(c.message)
                compound_score = sentiment_scores['compound']
                logger.info(f"Sentiment scores: {sentiment_scores}")

                # Enhanced VADER sentiment to stress mapping
                # Consider all sentiment components for more accurate stress detection
                pos_score = sentiment_scores['pos']
                neg_score = sentiment_scores['neg']
                neu_score = sentiment_scores['neu']
                
                # More nuanced stress calculation
                # Refined thresholds: clearly positive -> 0% stress
                if compound_score >= 0.20:  # confidently positive
                    score = 0
                elif compound_score >= 0.05: # mildly positive
                    base_score = 25 - (compound_score - 0.05) / 0.15 * 25  # map 0.05..0.20 -> 25..0
                    confidence_factor = pos_score * 0.2
                    score = int(max(0, base_score - confidence_factor * 10))
                elif compound_score <= -0.05: # Negative sentiment
                    # Scale negative sentiment (-1 to -0.05) to high stress (100 to 50)
                    base_score = 50 + (abs(compound_score) - 0.05) / 0.95 * 50
                    # Adjust based on confidence in negative sentiment
                    confidence_factor = neg_score * 0.4
                    score = int(min(100, base_score + confidence_factor * 15))
                else: # Neutral sentiment (-0.05 to 0.05)
                    # Neutral sentiment with slight bias toward stress if uncertainty
                    uncertainty_factor = 1 - neu_score
                    score = int(50 + uncertainty_factor * 10)

                chat_series.append({
                    't': (c.timestamp.replace(tzinfo=timezone.utc).isoformat() if c.timestamp else datetime.utcnow().replace(tzinfo=timezone.utc).isoformat()),
                    'score': score,
                    'text': c.message[:200]
                })
                logger.info(f"Added chat data point: score={score}, timestamp={c.timestamp}")
            except Exception as e:
                logger.error(f"VADER sentiment analysis error for message {i+1}: {str(e)}")
            finally:
                completed_units += 1
                if client_id:
                    ANALYSIS_PROGRESS[client_id] = int((completed_units/total_units)*100)

        # Load frames
        if user_id is not None:
            frames=db.session.execute(db.select(VideoFrame).filter_by(user_id=user_id)).scalars().all()
        else:
            frames=db.session.execute(db.select(VideoFrame).filter_by(user_id=None)).scalars().all()

        # Filter frames to be within chat session times
        frames = [f for f in frames if f.captured_at and chat_start_time <= f.captured_at <= chat_end_time]

        # Optionally sample frames to limit runtime (more aggressive for speed)
        max_frames=60
        if len(frames) > max_frames:
            frames = frames[:: max(1, len(frames)//max_frames) ]

        video_series=[]
        base_dir=os.path.dirname(__file__)
        # add frames to total units
        total_units = max(1, completed_units + len(frames))

        # Get session start time for proper timestamping (already determined as chat_start_time)
        session_start = chat_start_time

        for i, f in enumerate(frames):
            try:
                frame_path=os.path.join(base_dir, f.frame_path)

                # Try OpenCV first (better compatibility)
                try:
                    result = _analyze_emotion_opencv(frame_path)
                    emotion = result['emotion']
                    score = result['stress_score']
                    confidence = result['confidence']
                except Exception as cv_error:
                    logger.warning(f"OpenCV failed, falling back to DeepFace: {cv_error}")
                    # Fallback to DeepFace
                    result=DeepFace.analyze(
                        img_path=frame_path,
                        actions=['emotion'],
                        enforce_detection=False,
                        detector_backend='mediapipe'
                    )
                    r = result[0] if isinstance(result, list) else result
                    emotion=r.get('dominant_emotion', 'neutral')
                    # Get confidence from emotion scores if available
                    emotion_scores = r.get('emotion', {})
                    if emotion_scores:
                        confidence = emotion_scores.get(emotion, 0.5)
                    else:
                        confidence = 0.5
                    score=_map_emotion_to_stress(emotion, confidence)

                # Use actual frame timestamp, already filtered to be within chat session
                frame_time = (f.captured_at.replace(tzinfo=timezone.utc).isoformat() if f.captured_at else datetime.utcnow().replace(tzinfo=timezone.utc).isoformat())

                video_series.append({
                    't': frame_time,
                    'score': score,
                    'emotion': emotion,
                    'confidence': confidence,
                    'path': f.frame_path
                })
            except Exception as e:
                logger.error(f"CV error: {str(e)}")
            finally:
                completed_units += 1
                if client_id:
                    ANALYSIS_PROGRESS[client_id] = int((completed_units/total_units)*100)

        # Aggregates
        chat_avg = (sum(p['score'] for p in chat_series)/len(chat_series)) if chat_series else None
        video_avg = (sum(p['score'] for p in video_series)/len(video_series)) if video_series else None
        if chat_avg is not None and video_avg is not None:
            combined = (chat_avg + video_avg)/2
        elif chat_avg is not None:
            combined = chat_avg
        else:
            combined = video_avg

        # Categorical labels for combined stress level
        stress_level_map = {
            0: "no_stress",
            1: "mild",
            2: "moderate",
            3: "high",
            4: "severe"
        }

        # Recommendations (3 succinct items + valid links)
        def recs(score):
            if score is None:
                return [
                    { 'title': 'How to get started with mindfulness (Headspace blog)', 'link': 'https://www.headspace.com/mindfulness' },
                    { 'title': 'Small habits that help (Sleep Foundation)', 'link': 'https://www.sleepfoundation.org/sleep-hygiene/healthy-sleep-tips' },
                    { 'title': 'Beginner breathing guide', 'link': 'https://www.healthline.com/health/box-breathing' },
                ]
            v = float(score)
            if v < 20:
                return [
                    { 'title': '2-minute calm breathing', 'link': 'https://www.healthline.com/health/box-breathing' },
                    { 'title': 'Short mindful break', 'link': 'https://www.mindful.org/meditation/mindfulness-getting-started/' },
                    { 'title': 'Stay consistent with sleep', 'link': 'https://www.sleepfoundation.org/sleep-hygiene/healthy-sleep-tips' },
                ]
            if v < 40:
                return [
                    { 'title': '5-minute guided meditation', 'link': 'https://www.youtube.com/watch?v=inpok4MKVLM' },
                    { 'title': '10-min walk ideas (CDC)', 'link': 'https://www.cdc.gov/physicalactivity/basics/adults/index.htm' },
                    { 'title': 'Quick read: Managing mild stress', 'link': 'https://www.apa.org/topics/stress/tips' },
                ]
            if v < 60:
                return [
                    { 'title': 'Guided body scan', 'link': 'https://www.youtube.com/watch?v=ihO02wUzgkc' },
                    { 'title': 'Article: Cognitive coping strategies', 'link': 'https://www.health.harvard.edu/mind-and-mood/coping-with-stress' },
                    { 'title': 'Book: Why Zebras Don\'t Get Ulcers (Robert Sapolsky)', 'link': 'https://www.goodreads.com/book/show/86197.Why_Zebras_Don_t_Get_Ulcers' },
                ]
            if v < 80:
                return [
                    { 'title': '10-minute breathing for anxiety', 'link': 'https://www.youtube.com/watch?v=7-8n4fQ3b2s' },
                    { 'title': 'Stress management tips (APA)', 'link': 'https://www.apa.org/topics/stress' },
                    { 'title': 'Book: The Relaxation Response (Benson)', 'link': 'https://www.goodreads.com/book/show/84737.The_Relaxation_Response' },
                ]
            return [
                { 'title': 'Find a therapist directory', 'link': 'https://www.psychologytoday.com/us/therapists' },
                { 'title': 'Crisis resources (worldwide)', 'link': 'https://www.opencounseling.com/suicide-hotlines' },
                { 'title': 'Understanding chronic stress (Harvard Health)', 'link': 'https://www.health.harvard.edu/staying-healthy/understanding-the-stress-response' },
            ]

        recommendations = recs(combined)

        # Derive richer metrics (cap extremes for stability)
        peak_chat = max(chat_series, key=lambda p: p['score']) if chat_series else None
        peak_video = max(video_series, key=lambda p: p['score']) if video_series else None
        stress_spikes = sum(1 for p in video_series if p['score'] >= 75)
        dominant_emotion = None
        if video_series:
            counts = {}
            for p in video_series:
                e = (p.get('emotion') or '').lower()
                counts[e] = counts.get(e, 0) + 1
            dominant_emotion = max(counts, key=counts.get)

        # Categorical combined level (0..4)
        def combined_category(val):
            if val is None:
                return None
            v = max(0, min(100, float(val)))
            if v < 20: return 0
            if v < 40: return 1
            if v < 60: return 2
            if v < 80: return 3
            return 4

        combined_level = combined_category(combined)
        combined_level_text = stress_level_map.get(combined_level, "unknown")

        payload = {
            'chat_series': chat_series,
            'video_series': video_series,
            'chat_avg': chat_avg,
            'video_avg': video_avg,
            'combined': None if combined is None else max(0, min(100, combined)),
            'combined_level': combined_level,
            'combined_level_text': combined_level_text, # Add categorical text
            'chat_time': {
                'start': (chat_start_time.replace(tzinfo=timezone.utc).isoformat() if chat_start_time else None),
                'end': (chat_end_time.replace(tzinfo=timezone.utc).isoformat() if chat_end_time else None)
            },
            'chat_metrics': {
                'avg': None if chat_avg is None else max(0, min(100, chat_avg)),
                'count_messages': len(chat_series),  # Use filtered, analyzed series count
                'peak': peak_chat,
            },
            'video_metrics': {
                'avg': None if video_avg is None else max(0, min(100, video_avg)),
                'count_frames': len(video_series),
                'spikes': stress_spikes,
                'dominant_emotion': dominant_emotion,
            },
            'recommendations': recommendations,
        }

        # Persist analysis
        if user_id is not None:
            analysis = Analysis(
                user_id=user_id,
                chat_score=chat_avg,
                video_score=video_avg,
                combined_score=combined,
                details_json=json.dumps(payload)
            )
            db.session.add(analysis)
            db.session.commit()

        if client_id:
            ANALYSIS_PROGRESS[client_id] = 100
        
        # Log final analysis results
        logger.info(f"Analysis completed successfully. Payload keys: {list(payload.keys())}")
        logger.info(f"Chat series length: {len(payload.get('chat_series', []))}")
        logger.info(f"Video series length: {len(payload.get('video_series', []))}")
        logger.info(f"Message count: {payload.get('chat_metrics', {}).get('count_messages', 0)}")
        logger.info(f"Combined score: {payload.get('combined', 'N/A')}")
        
        return jsonify(payload)
    except Exception as e:
        db.session.rollback()
        logger.error(f"analysis error: {str(e)}")
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")
        return jsonify({"error":"server error"}), 500

@app.get('/analysis/progress')
def analysis_progress():
    client_id = session.get('client_id')
    pct = ANALYSIS_PROGRESS.get(client_id, 0) if client_id else 0
    return jsonify({ 'percent': int(pct) })

# --- Simple username auth ---
@app.post('/login')
def simple_login():
    data=request.get_json(force=True, silent=True) or {}
    username=data.get('username', '').strip()
    if not username:
        return jsonify({"error":"missing username"}), 400
    session['username']=username
    # ensure user exists (without firebase)
    user=db.session.execute(db.select(User).filter_by(username=username)).scalar_one_or_none()
    if not user:
        user=User(username=username)
        db.session.add(user)
        db.session.commit()
    return jsonify({"ok":True, "username":username})

@app.post('/logout')
def simple_logout():
    session.pop('firebase_uid', None)
    session.pop('username', None)
    return jsonify({"ok":True})

@socketio.event
def connect():
    try:
        if 'username' not in session or not session['username']:
            logger.info("Rejecting socket connection without username set")
            return False
        
        active_users[request.sid]={
            'username': session['username'],
            'connected_at': datetime.now().isoformat()
        }

        emit('active_users', {'users': [user['username'] for user in active_users.values()]},broadcast=True)
        logger.info(f"User {session['username']} connected")

    except Exception as e:
        logger.error(f"Error connecting user: {str(e)}")
        return False

@socketio.event
def disconnect():
    try:
        if request.sid in active_users:
            username=active_users[request.sid]['username']
            del active_users[request.sid]
            emit('active_users', {'users': [user['username'] for user in active_users.values()]},broadcast=True)
            logger.info(f"User {username} disconnected")

    except Exception as e:
        logger.error(f"Error disconnecting user: {str(e)}")
    
@socketio.on('join')
def on_join(data:dict):
    try:
        username=session['username']
        room=data['room']

        if room not in app.config['CHAT_ROOMS']:
            logger.warning(f"No room available")
            return
        
        join_room(room)
        active_users[request.sid]['room']=room

        emit('status', {'msg': f"{username} has joined the room",'type': 'join','timestamp': datetime.now().isoformat()}, room=room)

        logger.info(f"{username} has joined")
    
    except Exception as e:
        logger.error(f"Error joining room: {str(e)}")

@socketio.on('leave')
def on_leave(data:dict):
    try:
        username=session['username']
        room=data['room']

        leave_room(room)
        if request.sid in active_users:
            active_users[request.sid].pop('room',None)
        
        emit('status', {'msg': f"{username} has left the room",'type': 'leave','timestamp': datetime.now().isoformat()}, room=room)

        logger.info(f"User {username} has left the room")
    
    except Exception as e:
        logger.error(f"Error leaving the room: {str(e)}")

@socketio.on('message')
def handle_message(data: dict):
    try:
        username = session['username']
        room = data.get('room', 'General')
        msg_type = data.get('type', 'message')
        message = data.get('msg', '').strip()
        
        if not message:
            return
        
        timestamp = datetime.now().isoformat()
        
        if msg_type == 'private':
            # Handle private messages
            target_user = data.get('target')
            if not target_user:
                return
                
            for sid, user_data in active_users.items():
                if user_data['username'] == target_user:
                    emit('private_message', {
                        'msg': message,
                        'from': username,
                        'to': target_user,
                        'timestamp': timestamp
                    }, room=sid)
                    logger.info(f"Private message sent: {username} -> {target_user}")
                    return
                    
            logger.warning(f"Private message failed - user not found: {target_user}")
        
        else:
            # Regular room message
            if room not in app.config['CHAT_ROOMS']:
                logger.warning(f"Message to invalid room: {room}")
                return
                
            # Persist chat to DB (no heavy analysis here)
            try:
                user_id=None
                if 'firebase_uid' in session:
                    user=db.session.execute(db.select(User).filter_by(firebase_uid=session['firebase_uid'])).scalar_one_or_none()
                    if user:
                        user_id=user.id
                chat_row=Chat(
                    user_id=user_id,
                    username=username,
                    room=room,
                    message=message,
                )
                db.session.add(chat_row)
                db.session.commit()
            except Exception as db_err:
                db.session.rollback()
                logger.error(f"DB error saving chat: {str(db_err)}")
                
            emit('message', {
                'msg': message,
                'username': username,
                'room': room,
                'timestamp': timestamp
            }, room=room)
            
            logger.info(f"Message sent in {room} by {username}")
    
    except Exception as e:
        logger.error(f"Message handling error: {str(e)}")

        
        



if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    # Ensure tables exist before starting the server
    with app.app_context():
        db.create_all()
    socketio.run(
        app,
        host='0.0.0.0',
        port=port,
        debug=app.config['DEBUG'],
        use_reloader=app.config['DEBUG']
    )