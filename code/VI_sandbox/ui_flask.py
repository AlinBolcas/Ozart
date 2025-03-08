from flask import Flask, render_template, request, jsonify, url_for, redirect, session, send_from_directory
import os
import json
import time
import sys
import threading
import queue
from PIL import Image
from datetime import datetime
import secrets
import base64
from io import BytesIO
import logging
import requests

# Add parent directory to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import Ozart functionality
from III_ozart.music_analysis import MusicAnalyzer
from III_ozart.ozart_simple import OzartAgent

# Define paths
OUTPUT_DIR = "output"
IMG_DIR = os.path.join(OUTPUT_DIR, "images")
DESC_DIR = os.path.join(OUTPUT_DIR, "descriptions")
AUDIO_DIR = os.path.join(OUTPUT_DIR, "music_analysis")
STATIC_DIR = os.path.join("code", "IV_ui", "static")
TEMPLATE_DIR = os.path.join("code", "IV_ui", "templates")

# Create directories
os.makedirs(IMG_DIR, exist_ok=True)
os.makedirs(DESC_DIR, exist_ok=True)
os.makedirs(AUDIO_DIR, exist_ok=True)
os.makedirs(STATIC_DIR, exist_ok=True)
os.makedirs(TEMPLATE_DIR, exist_ok=True)

# Add this debugging section right before app creation
print("\n=== Template and Static File Debug Info ===")
print(f"TEMPLATE_DIR absolute path: {os.path.abspath(TEMPLATE_DIR)}")
print(f"STATIC_DIR absolute path: {os.path.abspath(STATIC_DIR)}")
print(f"Template index.html exists: {os.path.exists(os.path.join(TEMPLATE_DIR, 'index.html'))}")
print(f"Static styles.css exists: {os.path.exists(os.path.join(STATIC_DIR, 'styles.css'))}")
print(f"Static script.js exists: {os.path.exists(os.path.join(STATIC_DIR, 'script.js'))}")
print("="*50 + "\n")

# Create Flask app
app = Flask(__name__, 
           static_folder=os.path.abspath(STATIC_DIR),
           template_folder=os.path.abspath(TEMPLATE_DIR))
app.secret_key = secrets.token_hex(16)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global objects
agent = OzartAgent(debug=False, images_per_song=1)
process_queue = queue.Queue()
result_cache = {}
processing_status = {}

# Process tracking
active_processes = {}

# Utility functions
def get_all_artworks():
    """Get all artworks with metadata"""
    artworks = []
    
    if not os.path.exists(DESC_DIR):
        return []
    
    # Get description files
    desc_files = [f for f in os.listdir(DESC_DIR) if f.endswith(".json")]
    desc_files.sort(reverse=True)  # Newest first
    
    for desc_file in desc_files:
        try:
            with open(os.path.join(DESC_DIR, desc_file), 'r') as f:
                metadata = json.load(f)
            
            # Try to find the image
            image_path = metadata.get("image_path", "")
            if not os.path.exists(image_path):
                # Try alternative path
                image_filename = os.path.basename(image_path)
                alt_path = os.path.join(IMG_DIR, image_filename)
                if os.path.exists(alt_path):
                    image_path = alt_path
                else:
                    continue  # Skip if image not found
            
            # Load image and convert to base64 for web display
            image = Image.open(image_path)
            buffered = BytesIO()
            image.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
            
            # Add to results
            artworks.append({
                "metadata": metadata,
                "image": img_str,
                "image_path": image_path
            })
        except Exception as e:
            logger.error(f"Error loading artwork {desc_file}: {e}")
    
    return artworks

def process_song_direct(song_input, job_id):
    """Process a song directly using MusicAnalyzer and image generation"""
    try:
        # Update status
        processing_status[job_id] = {
            "status": "processing",
            "message": f"Analyzing song: {song_input}",
            "progress": 10
        }
        
        logger.info(f"🎵 Analyzing song: {song_input}")
        
        # Initialize music analyzer if not already done
        music_analyzer = MusicAnalyzer()
        
        # Analyze the song
        analysis_results = music_analyzer.process_input(song_input)
        
        if "error" in analysis_results:
            raise Exception(analysis_results["error"])
        
        # Extract information from analysis
        song_title = analysis_results.get("title", song_input)
        artist = analysis_results.get("artist", "Unknown Artist")
        audio_path = analysis_results.get("audio_path", "")
        description = analysis_results.get("interpretation", "")
        
        # Update status
        processing_status[job_id] = {
            "status": "processing",
            "message": f"Creating artwork for: {song_title}",
            "progress": 50
        }
        
        # Create prompt for image generation
        prompt = f"Create a visual representation of the song '{song_title}' by {artist}. {description[:300]}"
        
        # Generate image using Replicate API
        replicate_api = None
        try:
            from I_integrations.replicate_API import ReplicateAPI
            replicate_api = ReplicateAPI()
        except Exception as e:
            logger.error(f"Failed to load Replicate API: {e}")
            raise Exception("Image generation service unavailable")
        
        # Generate image
        image_url = replicate_api.generate_image(
            prompt=prompt,
            aspect_ratio="1:1"
        )
        
        if not image_url:
            raise Exception("Failed to generate image")
            
        # Download image
        response = requests.get(image_url)
        if response.status_code != 200:
            raise Exception(f"Failed to download image: {response.status_code}")
            
        # Save image
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        image_path = os.path.join(IMG_DIR, f"{timestamp}_{song_title.replace(' ', '_')}.png")
        with open(image_path, "wb") as f:
            f.write(response.content)
        
        # Save metadata
        metadata = {
            "song_title": song_title,
            "artist": artist,
            "theme": "Musical Interpretation",
            "prompt": prompt,
            "description": description,
            "timestamp": datetime.now().isoformat(),
            "human_time": datetime.now().strftime("%B %d, %Y %H:%M"),
            "image_path": image_path,
            "audio_path": audio_path
        }
        
        # Save to descriptions folder
        desc_path = os.path.join(DESC_DIR, f"{timestamp}_{song_title.replace(' ', '_')}.json")
        with open(desc_path, "w") as f:
            json.dump(metadata, f, indent=2)
        
        # Update status to complete
        processing_status[job_id] = {
            "status": "complete",
            "message": f"Completed: {song_title}",
            "progress": 100
        }
        
        logger.info(f"✓ Generated artwork for '{song_title}' by {artist}")
        
    except Exception as e:
        logger.error(f"❌ Error processing song: {str(e)}")
        processing_status[job_id] = {
            "status": "error",
            "message": f"Error: {str(e)}",
            "progress": 0
        }

def process_thread(songs):
    """Background thread for processing multiple songs"""
    job_ids = []
    
    for song in songs:
        if song.strip():
            # Create a unique job ID
            job_id = f"job_{datetime.now().strftime('%Y%m%d%H%M%S')}_{secrets.token_hex(4)}"
            job_ids.append(job_id)
            
            # Create initial status
            processing_status[job_id] = {
                "status": "pending",
                "message": f"Queued: {song}",
                "progress": 0
            }
            
            # Start processing in a separate thread
            thread = threading.Thread(
                target=process_song_direct, 
                args=(song, job_id)
            )
            thread.daemon = True
            thread.start()
            
            # Store active process
            active_processes[job_id] = {
                "thread": thread,
                "song": song,
                "start_time": datetime.now()
            }
    
    return job_ids

# Flask routes
@app.route('/')
def index():
    # Get all artworks
    artworks = get_all_artworks()
    
    # Get profile image
    profile_image = None
    profile_paths = ["profile_02.jpg", "data/profile_02.jpg", "../data/profile_02.jpg"]
    for path in profile_paths:
        if os.path.exists(path):
            try:
                with open(path, "rb") as img_file:
                    profile_image = base64.b64encode(img_file.read()).decode('utf-8')
                break
            except Exception as e:
                logger.error(f"Failed to load profile image from {path}: {e}")
    
    # Always use the cover image as header, never use artworks
    header_image = None
    header_paths = ["cover_02.png", "data/cover_02.png", "../data/cover_02.png"]
    for path in header_paths:
        if os.path.exists(path):
            try:
                with open(path, "rb") as img_file:
                    header_image = base64.b64encode(img_file.read()).decode('utf-8')
                break
            except Exception as e:
                logger.error(f"Failed to load header image from {path}: {e}")
    
    if header_image is None:
        logger.warning("Cover image not found. Please check paths.")
    
    # Debug log for images
    logger.info(f"Profile image loaded: {profile_image is not None}")
    logger.info(f"Header image loaded: {header_image is not None}")
    
    # Get artwork count
    artwork_count = len([f for f in os.listdir(DESC_DIR) if f.endswith('.json')]) if os.path.exists(DESC_DIR) else 0
    
    # Get recent song
    recent_song = "None yet"
    if artworks:
        recent_metadata = artworks[0]["metadata"]
        recent_song = recent_metadata.get("song_title", "None yet")
    
    # Check for messages in session
    messages = []
    if 'messages' in session:
        messages = session['messages']
        session.pop('messages', None)
    
    # Check if any processes are running
    is_processing = any(proc["thread"].is_alive() for proc in active_processes.values())
    
    return render_template(
        'index.html',
        artworks=artworks,
        profile_image=profile_image,
        header_image=header_image,
        artwork_count=artwork_count,
        recent_song=recent_song,
        messages=messages,
        is_processing=is_processing
    )

@app.route('/process', methods=['POST'])
def process():
    song_input = request.form.get('song_input', '')
    
    if not song_input:
        # Add error message to session
        if 'messages' not in session:
            session['messages'] = []
        session['messages'].append({
            'type': 'error',
            'text': 'Please enter a song name or URL'
        })
        return redirect(url_for('index'))
    
    # Split by commas
    songs = [s.strip() for s in song_input.split(',') if s.strip()]
    
    # Process in background
    job_ids = process_thread(songs)
    
    # Add success message
    if 'messages' not in session:
        session['messages'] = []
    session['messages'].append({
        'type': 'success',
        'text': f'Started processing: {song_input}'
    })
    
    return redirect(url_for('index'))

@app.route('/status')
def status():
    """Get current processing status"""
    # Clean up completed processes
    for job_id in list(active_processes.keys()):
        if not active_processes[job_id]["thread"].is_alive():
            # Keep status but remove thread reference
            active_processes.pop(job_id, None)
    
    # Return all statuses
    return jsonify(processing_status)

@app.route('/audio/<path:filename>')
def serve_audio(filename):
    """Serve audio files"""
    # This needs security improvements for production!
    return send_from_directory(AUDIO_DIR, filename)

if __name__ == "__main__":
    # Run the Flask app
    app.run(debug=True, port=8080) 