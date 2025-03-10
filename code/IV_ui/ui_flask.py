import sys
import os
from pathlib import Path

# Add parent directory to Python path and set ROOT_DIR
ROOT_DIR = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT_DIR))

# First import Flask and extensions
from flask import Flask, render_template, request, jsonify, send_from_directory, abort, session, redirect, url_for
from flask_cors import CORS
from flask_login import LoginManager, login_user, logout_user, login_required, current_user
from oauthlib.oauth2 import WebApplicationClient

# Then import our local modules
from code.IV_ui.models import db, User
from code.III_ozart.ozart_simple import OzartAgent

# Rest of the imports
from PIL import Image
import logging
import socket
import requests
import threading
import queue
import json
from datetime import datetime

# Allow OAuth over HTTP for development
os.environ['OAUTHLIB_INSECURE_TRANSPORT'] = '1'

# Function to print server URLs
def print_server_urls(host, port):
    print("\n" + "="*50)
    print(f"🚀 Ozart server running at:")
    print(f"   http://localhost:{port}")
    print(f"   http://{socket.gethostbyname(socket.gethostname())}:{port}")
    print("="*50 + "\n")

app = Flask(__name__, 
           static_url_path='/static',
           static_folder='static')
CORS(app)

# Configure Flask with session secret
app.config.update(
    DEBUG=os.getenv('FLASK_DEBUG', '1') == '1',  # Default to debug mode in development
    TEMPLATES_AUTO_RELOAD=True,
    SEND_FILE_MAX_AGE_DEFAULT=0,
    SECRET_KEY=os.getenv('SECRET_KEY', os.urandom(24)),
    SQLALCHEMY_DATABASE_URI=f'sqlite:///{os.path.join(ROOT_DIR, "instance", "ozart.db")}',
    SQLALCHEMY_TRACK_MODIFICATIONS=False,
    GOOGLE_CLIENT_ID=os.getenv('GOOGLE_CLIENT_ID'),
    GOOGLE_CLIENT_SECRET=os.getenv('GOOGLE_CLIENT_SECRET')
)

# Configure logging to be less verbose
log = logging.getLogger('werkzeug')
log.setLevel(logging.WARNING)  # Only show warnings and errors

# Create storage directories
OUTPUT_DIR = os.path.abspath("output")
IMG_DIR = os.path.join(OUTPUT_DIR, "images")
DESC_DIR = os.path.join(OUTPUT_DIR, "descriptions")
AUDIO_DIR = os.path.join(OUTPUT_DIR, "music_analysis")

os.makedirs(IMG_DIR, exist_ok=True)
os.makedirs(DESC_DIR, exist_ok=True)
os.makedirs(AUDIO_DIR, exist_ok=True)

# Create static folders and fallback image
STATIC_DIR = os.path.join(os.path.dirname(__file__), 'static')
STATIC_DATA_DIR = os.path.join(STATIC_DIR, 'data')
os.makedirs(STATIC_DATA_DIR, exist_ok=True)

FALLBACK_PATH = os.path.join(STATIC_DATA_DIR, 'fallback.png')
if not os.path.exists(FALLBACK_PATH):
    img = Image.new('RGB', (100, 100), color='black')
    img.save(FALLBACK_PATH)

# Create favicon
FAVICON_PATH = os.path.join(STATIC_DIR, 'images')
os.makedirs(FAVICON_PATH, exist_ok=True)
if not os.path.exists(os.path.join(FAVICON_PATH, 'favicon.png')):
    img = Image.new('RGB', (32, 32), color='black')
    img.save(os.path.join(FAVICON_PATH, 'favicon.png'))

# Initialize agent without keys initially
ozart_agent = OzartAgent(debug=False)

# Thread-safe queues
processing_queue = queue.Queue()
error_queue = queue.Queue()

# Initialize extensions
db.init_app(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

google_client = WebApplicationClient(app.config['GOOGLE_CLIENT_ID'])

# Only allow OAuth over HTTP in development
if os.getenv('FLASK_ENV') == 'development':
    os.environ['OAUTHLIB_INSECURE_TRANSPORT'] = '1'

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

@app.route('/')
def index():
    artworks = ozart_agent.get_all_artworks()
    artwork_count = len(artworks)
    recent_title = "None yet"
    
    if artworks:
        metadata = artworks[0].get("metadata", {})
        song_data = metadata.get("song", {})
        recent_title = song_data.get("title", "None yet")
    
    return render_template('index.html', 
                         artwork_count=artwork_count,
                         recent_title=recent_title)

@app.route('/process', methods=['POST'])
def process_song():
    """Process song but check for API keys first"""
    # Check if we have keys - either from session or from logged in user
    if current_user.is_authenticated:
        has_openai = bool(current_user.api_keys.get('openai_key') or os.getenv("OPENAI_API_KEY"))
        has_replicate = bool(current_user.api_keys.get('replicate_key') or os.getenv("REPLICATE_API_TOKEN"))
    else:
        has_openai = bool(session.get('openai_key') or os.getenv("OPENAI_API_KEY"))
        has_replicate = bool(session.get('replicate_key') or os.getenv("REPLICATE_API_TOKEN"))
    
    if not (has_openai and has_replicate):
        return jsonify({
            'error': 'API keys required',
            'has_openai': has_openai,
            'has_replicate': has_replicate
        }), 403
    
    # Continue with song processing
    song_input = request.json.get('song')
    if not song_input:
        return jsonify({'error': 'No song provided'})
    
    # Create agent with session keys if needed
    global ozart_agent
    if session.get('openai_key') or session.get('replicate_key'):
        ozart_agent = OzartAgent(
            debug=False,
            openai_api_key=session.get('openai_key'),
            replicate_api_token=session.get('replicate_key')
        )
    
    thread = threading.Thread(target=process_song_thread, args=(song_input,))
    thread.daemon = True
    thread.start()
    
    return jsonify({'status': 'processing'})

@app.route('/status')
def get_status():
    try:
        errors = []
        while not error_queue.empty():
            try:
                errors.append(error_queue.get_nowait())
            except queue.Empty:
                break
        
        artworks = ozart_agent.get_all_artworks()
        formatted_artworks = [format_artwork(artwork) for artwork in artworks]
        
        return jsonify({
            'processing': not processing_queue.empty(),
            'errors': errors,
            'artwork_count': len(artworks),
            'artworks': formatted_artworks
        })
    except Exception as e:
        return jsonify({
            'processing': False,
            'errors': [str(e)],
            'artwork_count': 0,
            'artworks': []
        })

@app.route('/output/<path:filename>')
def serve_file(filename):
    """Serve files from output directory"""
    try:
        requested_path = os.path.abspath(os.path.join(OUTPUT_DIR, filename))
        if not requested_path.startswith(OUTPUT_DIR):
            abort(403)
        
        directory = os.path.dirname(requested_path)
        file_name = os.path.basename(requested_path)
        
        return send_from_directory(directory, file_name)
    except Exception:
        abort(404)

def format_artwork(artwork):
    """Format artwork data for JSON response"""
    metadata = artwork.get("metadata", {})
    image_path = metadata.get("image", {}).get("path", "")
    song_data = metadata.get("song", {})
    
    # Get audio path from session path
    audio_path = None
    if "session_path" in song_data:
        session_dir = song_data["session_path"]
        if os.path.exists(session_dir):
            for file in os.listdir(session_dir):
                if file.endswith('.wav'):
                    audio_path = os.path.join(session_dir, file)
                    audio_path = os.path.relpath(audio_path, OUTPUT_DIR)
                    break
    
    return {
        'image_path': image_path,
        'song_title': song_data.get('title', 'Unknown'),
        'artist': song_data.get('artist', 'Unknown'),
        'interpretation': song_data.get('description', ''),
        'prompt': metadata.get("image", {}).get("prompt", ""),
        'human_time': metadata.get("human_time", "Unknown time"),
        'audio_path': audio_path
    }

def process_song_thread(song_input):
    """Process song in background thread"""
    try:
        processing_queue.put(song_input)
        result = ozart_agent.process_single_song(song_input)
        
        if 'error' in result:
            error_queue.put(result['error'])
    except Exception as e:
        error_queue.put(str(e))
    finally:
        try:
            processing_queue.get_nowait()
        except queue.Empty:
            pass

@app.route('/save-keys', methods=['POST'])
def save_api_keys():
    """Save API keys to user profile or session"""
    data = request.json
    keys = {
        'openai_key': data.get('openai_key'),
        'replicate_key': data.get('replicate_key')
    }
    
    if current_user.is_authenticated:
        current_user.api_keys = keys
        db.session.commit()
    else:
        session['api_keys'] = keys
    
    return jsonify({'success': True})

@app.route('/check-keys')
def check_keys():
    """Check if API keys are available"""
    has_openai = bool(session.get('openai_key') or os.getenv("OPENAI_API_KEY"))
    has_replicate = bool(session.get('replicate_key') or os.getenv("REPLICATE_API_TOKEN"))
    
    return jsonify({
        'has_openai': has_openai,
        'has_replicate': has_replicate,
        'keys_required': not (has_openai and has_replicate)
    })

@app.route('/static/data/<path:filename>')
def serve_data_file(filename):
    """Serve files from data directory"""
    data_dir = Path(__file__).parent.parent.parent / 'data'
    return send_from_directory(data_dir, filename)

@app.route('/debug-artworks')
def debug_artworks():
    """Debug endpoint to see raw artwork data"""
    try:
        artworks = ozart_agent.get_all_artworks()
        artwork_details = []
        for i, artwork in enumerate(artworks[:3]):
            metadata = artwork.get("metadata", {})
            image_path = metadata.get("image", {}).get("path", "")
            full_path = os.path.join("output", image_path) if image_path else ""
            exists = os.path.exists(full_path) if full_path else False
            
            artwork_details.append({
                "index": i,
                "keys": list(artwork.keys()),
                "metadata_keys": list(metadata.keys()),
                "image_path": image_path,
                "full_path": full_path,
                "image_exists": exists,
                "song_title": artwork.get("song", {}).get("title", "Unknown"),
            })
        
        formatted = [format_artwork(artwork) for artwork in artworks]
        
        return jsonify({
            'raw_count': len(artworks),
            'artwork_details': artwork_details,
            'formatted_count': len(formatted),
            'formatted_sample': formatted[:3] if formatted else []
        })
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/debug-json-files')
def debug_json_files():
    """Debug endpoint to examine JSON files directly"""
    try:
        results = []
        
        if os.path.exists(DESC_DIR):
            for filename in os.listdir(DESC_DIR):
                if filename.endswith('.json'):
                    file_path = os.path.join(DESC_DIR, filename)
                    
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                    
                    image_path = data.get("image", {}).get("path", "")
                    song_title = data.get("song", {}).get("title", "Unknown")
                    
                    full_path = os.path.join("output", image_path) if image_path else ""
                    exists = os.path.exists(full_path) if full_path else False
                    
                    results.append({
                        "filename": filename,
                        "keys": list(data.keys()),
                        "image_path": image_path,
                        "image_exists": exists,
                        "song_title": song_title
                    })
        
        return jsonify({
            'file_count': len(results),
            'files': results
        })
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/save-prompts', methods=['POST'])
def save_prompts():
    """Save custom prompts to user profile or session"""
    data = request.json
    prompts = {
        'music_analysis_system_prompt': data.get('music_analysis_system_prompt', ''),
        'music_analysis_user_prompt': data.get('music_analysis_user_prompt', ''),
        'image_prompt_system_prompt': data.get('image_prompt_system_prompt', ''),
        'image_prompt_user_prompt': data.get('image_prompt_user_prompt', '')
    }
    
    if current_user.is_authenticated:
        current_user.custom_prompts = prompts
        db.session.commit()
    else:
        session['custom_prompts'] = prompts
    
    return jsonify({'success': True})

@app.route('/get-prompts')
def get_prompts():
    """Get current prompts (custom or default)"""
    try:
        # Return custom prompts from session if available
        if 'custom_prompts' in session:
            return jsonify({
                'success': True, 
                'prompts': session['custom_prompts'],
                'is_custom': True
            })
            
        # Otherwise load default prompts
        prompts_path = Path("data/default_prompts.json")
        if prompts_path.exists():
            with open(prompts_path, 'r') as f:
                default_prompts = json.load(f)
                return jsonify({
                    'success': True, 
                    'prompts': default_prompts,
                    'is_custom': False
                })
                
        return jsonify({'success': False, 'error': 'Default prompts not found'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        user = User.query.filter_by(email=email).first()
        
        if user and user.check_password(password):
            login_user(user)
            return redirect(url_for('index'))
            
        return render_template('login.html', error="Invalid credentials")
        
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        confirm = request.form.get('confirm_password')
        
        if password != confirm:
            return render_template('register.html', error="Passwords don't match")
            
        if User.query.filter_by(email=email).first():
            return render_template('register.html', error="Email already registered")
            
        user = User(email=email)
        user.set_password(password)
        db.session.add(user)
        db.session.commit()
        
        login_user(user)
        return redirect(url_for('index'))
        
    return render_template('register.html')

@app.route('/login/google')
def google_login():
    google_provider_cfg = requests.get('https://accounts.google.com/.well-known/openid-configuration').json()
    authorization_endpoint = google_provider_cfg["authorization_endpoint"]
    
    request_uri = google_client.prepare_request_uri(
        authorization_endpoint,
        redirect_uri=request.base_url + "/callback",
        scope=["openid", "email", "profile"],
    )
    return redirect(request_uri)

@app.route('/login/google/callback')
def google_callback():
    try:
        # Handle Google OAuth callback
        code = request.args.get("code")
        google_provider_cfg = requests.get('https://accounts.google.com/.well-known/openid-configuration').json()
        token_endpoint = google_provider_cfg["token_endpoint"]
        
        # Get tokens
        token_url, headers, body = google_client.prepare_token_request(
            token_endpoint,
            authorization_response=request.url,
            redirect_url=request.base_url,
            code=code
        )
        token_response = requests.post(
            token_url,
            headers=headers,
            data=body,
            auth=(app.config['GOOGLE_CLIENT_ID'], app.config['GOOGLE_CLIENT_SECRET']),
        )
        
        google_client.parse_request_body_response(token_response.text)
        
        # Get user info
        userinfo_endpoint = google_provider_cfg["userinfo_endpoint"]
        uri, headers, body = google_client.add_token(userinfo_endpoint)
        userinfo_response = requests.get(uri, headers=headers, data=body)
        
        if userinfo_response.json().get("email_verified"):
            google_id = userinfo_response.json()["sub"]
            email = userinfo_response.json()["email"]
            name = userinfo_response.json().get("name", email.split('@')[0])
            
            # Create or get user
            user = User.query.filter_by(google_id=google_id).first()
            if not user:
                user = User(google_id=google_id, email=email, name=name)
                db.session.add(user)
                db.session.commit()
                
            login_user(user)
            return redirect(url_for('index'))
        
        return render_template('login.html', error="Could not verify Google account.")
    except Exception as e:
        print(f"Google login error: {str(e)}")
        return render_template('login.html', error="An error occurred during Google login. Please try again.")

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('index'))

@app.route('/get-user-data')
def get_user_data():
    """Get user's saved data (API keys and prompts)"""
    if current_user.is_authenticated:
        return jsonify({
            'api_keys': current_user.api_keys,
            'custom_prompts': current_user.custom_prompts,
            'email': current_user.email
        })
    else:
        return jsonify({
            'api_keys': session.get('api_keys', {}),
            'custom_prompts': session.get('custom_prompts', {}),
            'email': None
        })

if __name__ == '__main__':
    host = '0.0.0.0'
    port = 5001
    
    # Print URLs before starting the server
    print_server_urls(host, port)
    
    app.run(host=host, port=port) 