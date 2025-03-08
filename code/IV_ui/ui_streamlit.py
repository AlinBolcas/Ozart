import streamlit as st
import requests
import time
import os
import json
from PIL import Image
from io import BytesIO
from datetime import datetime
import random
import uuid
import sys
import threading
import queue  # Add this back for thread communication

# Add parent directory to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import Ozart agent
from III_ozart.ozart_simple import OzartAgent, get_all_artworks

# Create storage directories in project root
IMG_DIR = os.path.join("output", "images")
DESC_DIR = os.path.join("output", "descriptions")
AUDIO_DIR = os.path.join("output", "music_analysis")
os.makedirs(IMG_DIR, exist_ok=True)
os.makedirs(DESC_DIR, exist_ok=True)
os.makedirs(AUDIO_DIR, exist_ok=True)

# Create a global error queue for thread-safe communication
error_queue = queue.Queue()

# Initialize session state
if "processing" not in st.session_state:
    st.session_state.processing = False
if "errors" not in st.session_state:
    st.session_state.errors = []
if "ozart_agent" not in st.session_state:
    # Set default images per song to 1
    st.session_state.ozart_agent = OzartAgent(debug=False, images_per_song=1)
if "last_processed" not in st.session_state:
    st.session_state.last_processed = None
if "song_queue" not in st.session_state:
    st.session_state.song_queue = []

def display_processing_message():
    """Display a simple processing message."""
    st.info("Please wait about 2 minutes, your song is processing ...")

def display_errors():
    """Display any errors from processing."""
    # Check the error queue and add to session state
    while not error_queue.empty():
        try:
            error = error_queue.get_nowait()
            st.session_state.errors.append(error)
        except queue.Empty:
            break
            
    if st.session_state.errors:
        for error in st.session_state.errors:
            st.error(error)
        # Clear errors after displaying
        st.session_state.errors = []

def process_songs_thread(agent, songs):
    """Process songs in a background thread - simplified version"""
    try:
        print("\n" + "="*60)
        print("🎵 OZART MUSIC-TO-IMAGE PROCESSING 🎨")
        print("="*60)
        
        for song in songs:
            try:
                print(f"\n{'='*50}")
                print(f"🔍 ANALYZING SONG: {song}")
                print(f"{'='*50}")
                
                # Use the music analyzer to process the song
                analysis_results = agent.music_analyzer.process_input(song)
                
                if "error" in analysis_results:
                    error_msg = f"Error analyzing {song}: {analysis_results['error']}"
                    print(f"❌ {error_msg}")
                    error_queue.put(error_msg)
                    continue
                
                # Extract relevant info
                song_description = analysis_results.get("interpretation", "")
                song_metadata = analysis_results.get("metadata", {})
                song_title = song_metadata.get("title", song)
                artist = song_metadata.get("artist", "Unknown Artist")
                
                # Print analysis results with nice formatting
                print(f"\n{'*'*50}")
                print(f"✅ ANALYSIS COMPLETE: '{song_title}' by {artist}")
                print(f"{'*'*50}")
                print("\n📊 SONG ANALYSIS:")
                print(f"{'-'*40}")
                print(song_description)
                print(f"{'-'*40}")
                
                # Start image generation
                print(f"\n{'='*50}")
                print(f"🎨 GENERATING ARTWORK")
                print(f"{'='*50}")
                
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                image_id = str(uuid.uuid4())[:8]
                filename = f"art_{timestamp}_{image_id}.jpg"
                image_path = os.path.join(IMG_DIR, filename)
                
                # Generate a prompt based on song description
                prompt = f"Create a visual representation of the song '{song_title}' by {artist}. {song_description[:500]}"
                
                # Print the full prompt
                print("\n📝 IMAGE PROMPT:")
                print(f"{'-'*40}")
                print(prompt)
                print(f"{'-'*40}")
                
                # Generate image using tools interface
                try:
                    print("\n🖼️ GENERATING IMAGE...")
                    
                    # Fix the save path to avoid nested directories
                    clean_filename = os.path.basename(filename)
                    final_save_path = os.path.join(IMG_DIR, clean_filename)
                    
                    # Generate the image - WITH save_path
                    result = agent.tools.generate_image(
                        prompt=prompt,
                        engine="flux",
                        aspect_ratio="1:1",
                        save_path=final_save_path  # This causes tools.py to try to save it directly
                    )
                    
                    print(f"🔗 Result: {result}")
                    
                    # Check if the file exists at the save_path
                    if os.path.exists(final_save_path):
                        print(f"✅ Image saved successfully at: {final_save_path}")
                        save_path = final_save_path
                    # If file doesn't exist but we have a URL in the result
                    elif isinstance(result, str) and result.startswith('http'):
                        print(f"⚠️ Image wasn't saved automatically, downloading from URL: {result}")
                        try:
                            response = requests.get(result, timeout=30)
                            if response.status_code == 200:
                                os.makedirs(os.path.dirname(final_save_path), exist_ok=True)
                                with open(final_save_path, 'wb') as f:
                                    f.write(response.content)
                                print(f"✅ Image downloaded successfully to: {final_save_path}")
                                save_path = final_save_path
                            else:
                                print(f"❌ Failed to download image: HTTP {response.status_code}")
                                save_path = result  # Fall back to URL
                        except Exception as dl_err:
                            print(f"❌ Download error: {str(dl_err)}")
                            save_path = result  # Fall back to URL
                    # If it's a FileOutput object
                    elif hasattr(result, 'url'):
                        image_url = result.url
                        print(f"⚠️ Got FileOutput object, downloading from URL: {image_url}")
                        try:
                            response = requests.get(image_url, timeout=30)
                            if response.status_code == 200:
                                os.makedirs(os.path.dirname(final_save_path), exist_ok=True)
                                with open(final_save_path, 'wb') as f:
                                    f.write(response.content)
                                print(f"✅ Image downloaded from FileOutput URL to: {final_save_path}")
                                save_path = final_save_path
                            else:
                                print(f"❌ Failed to download image: HTTP {response.status_code}")
                                save_path = image_url  # Fall back to URL
                        except Exception as dl_err:
                            print(f"❌ Download error: {str(dl_err)}")
                            save_path = image_url  # Fall back to URL
                    else:
                        print(f"⚠️ Unknown result type: {type(result)}")
                        error_queue.put(f"Unknown result format from image generation: {type(result)}")
                        continue
                    
                    # Format the human-readable time
                    human_time = datetime.now().strftime("%d/%m/%y, %H:%M")
                    
                    # Create metadata
                    description_dict = {
                        "song_title": song_title,
                        "artist": artist,
                        "theme": "Music Visualization",
                        "image_prompt": prompt,
                        "description": song_description,
                        "audio_path": analysis_results.get("audio_path", ""),
                        "public_audio_path": analysis_results.get("public_audio_path", ""),
                        "creation_time": datetime.now().isoformat(),
                        "human_time": human_time,
                        "image_path": save_path
                    }
                    
                    # Save metadata to file - use base filename without path
                    desc_base = os.path.splitext(os.path.basename(save_path))[0]
                    if desc_base.startswith("art_"):
                        desc_filename = f"desc_{desc_base[4:]}.json"
                    else:
                        desc_filename = f"desc_{desc_base}.json"
                    
                    desc_path = os.path.join(DESC_DIR, desc_filename)
                    print(f"\n💾 SAVING METADATA TO: {desc_path}")
                    
                    with open(desc_path, "w") as f:
                        json.dump(description_dict, f, indent=2)
                    
                    print(f"✅ Metadata saved successfully")
                    
                except Exception as img_err:
                    print(f"\n❌ ERROR GENERATING IMAGE: {str(img_err)}")
                    import traceback
                    print(traceback.format_exc())
                    error_queue.put(f"Error generating image for {song_title}: {str(img_err)}")
                
                print(f"\n{'='*50}")
                print(f"✅ PROCESSING COMPLETE: {song}")
                print(f"{'='*50}")
                
            except Exception as e:
                import traceback
                print(f"\n❌ ERROR IN SONG PROCESSING: {str(e)}")
                print(traceback.format_exc())
                error_queue.put(f"Error processing {song}: {str(e)}")
        
        print(f"\n{'='*60}")
        print(f"🏁 ALL SONGS PROCESSED")
        print(f"{'='*60}")
    except Exception as e:
        import traceback
        print(f"\n❌ ERROR IN PROCESSING THREAD: {str(e)}")
        print(traceback.format_exc())
        error_queue.put(f"Thread error: {str(e)}")

def start_processing(songs_input):
    """Start background processing of songs"""
    if st.session_state.processing:
        return False
    
    songs = [s.strip() for s in songs_input.split(",") if s.strip()]
    if not songs:
        return False
    
    # Get a copy of the agent to pass to thread
    agent = st.session_state.ozart_agent
    
    # Start processing in background thread
    st.session_state.processing = True
    thread = threading.Thread(target=process_songs_thread, args=(agent, songs))
    thread.daemon = True
    thread.start()
    
    # Store thread reference
    st.session_state.processing_thread = thread
    
    return True

def find_audio_file(song_title, artwork_metadata=None):
    """Find audio file for a song, first checking metadata if available"""
    if not song_title:
        return None
    
    # First check if metadata already has audio_path
    if artwork_metadata and "audio_path" in artwork_metadata:
        audio_path = artwork_metadata.get("audio_path")
        if audio_path and os.path.exists(audio_path):
            return audio_path
    
    # Check the dedicated audio directory first
    audio_dir = os.path.join(AUDIO_DIR, "audio")
    if os.path.exists(audio_dir):
        # Clean the song title for filename matching
        clean_title = ''.join(c if c.isalnum() or c in ' -_' else '_' for c in song_title).lower()
        
        # First try exact match in the audio directory
        for file in os.listdir(audio_dir):
            if file.endswith((".wav", ".mp3", ".webm")):
                if clean_title in file.lower():
                    return os.path.join(audio_dir, file)
    
    # Try all music_analysis subdirectories as a fallback
    for root, dirs, files in os.walk(AUDIO_DIR):
        for file in files:
            if file.endswith((".wav", ".mp3", ".webm")):
                # Try to match the song title
                if song_title.lower() in file.lower():
                    return os.path.join(root, file)
    
    return None

def display_artwork(artwork, idx, cols):
    """Display a single artwork in the UI"""
    # Extract metadata and image
    metadata = artwork["metadata"]
    image = artwork["image"]
    
    # Calculate which column to use
    col_idx = idx % len(cols)
    
    with cols[col_idx]:
        # Display the image
        st.image(image, width=750)
        
        # Display artwork info in expandable section
        with st.expander("Show details"):
            # Update to use more specific song and artist display
            song_title = metadata.get('song_title', metadata.get('title', 'Unknown Song'))
            artist = metadata.get('artist', metadata.get('uploader', 'Unknown Artist'))
            st.markdown(f"### {song_title} by {artist}")
            
            # Try to find and display audio player for the song prominently
            song_title = metadata.get('song_title', '')
            audio_file = find_audio_file(song_title, metadata)
            
            if audio_file and os.path.exists(audio_file):
                # Explicitly specify the file format based on extension
                file_ext = os.path.splitext(audio_file)[1].lower()
                if file_ext == '.mp3':
                    format_type = "audio/mp3"
                elif file_ext == '.wav':
                    format_type = "audio/wav"
                else:
                    format_type = None  # Let Streamlit guess
                
                # Add a more visible player header
                st.markdown("#### Listen to the song:")
                
                # Display the audio player with specified format
                try:
                    st.audio(audio_file, format=format_type)
                except Exception as e:
                    st.error(f"Could not play audio: {str(e)}")
                    st.markdown(f"Audio file path: {audio_file}")
            
            # Show the song analysis
            st.markdown("### Song Analysis")
            analysis = metadata.get('song_analysis', metadata.get('description', 'No analysis available'))
            st.markdown(analysis)
            
            # Show the generation prompt
            st.markdown("### Image Prompt")
            st.markdown(f"```\n{metadata.get('image_prompt', 'No prompt available')}\n```")
            
            # Show creation time
            st.markdown(f"**Created**: {metadata.get('human_time', 'Unknown time')}")

def main():
    # Check for processing completion first - put this at the beginning of main
    if 'processing_thread' in st.session_state and st.session_state.processing:
        if not st.session_state.processing_thread.is_alive():
            st.session_state.processing = False
            st.session_state.last_processed = datetime.now()
            # Don't rerun here to avoid double refresh

    # Apply custom CSS for styling
    st.markdown("""
    <style>
    .main-header {
        font-size: 2rem;
        margin-bottom: 0.5rem;
    }
    .stat-header {
        font-size: 0.8rem;
        color: #888;
        margin-bottom: 0px;
    }
    .stat-value {
        font-size: 1.2rem;
        font-weight: bold;
        margin-top: 0px;
    }
    /* Hide "No artworks yet" message */
    .empty-gallery-message {
        display: none;
    }
    /* Custom button styling */
    .stButton button {
        color: white !important;
    }
    /* Make the cover image look clickable */
    .clickable-image {
        cursor: pointer;
        transition: opacity 0.3s;
    }
    .clickable-image:hover {
        opacity: 0.9;
    }
    </style>
    """, unsafe_allow_html=True)

    # Layout: header with clickable image
    cover_paths = [
        "cover_02.png",
        "data/cover_02.png", 
        "../data/cover_02.png",
        "/app/data/cover_02.png"
    ]
    
    cover_loaded = False
    for path in cover_paths:
        if os.path.exists(path):
            # Instead of using st.image, use HTML with onClick to refresh page
            col1, col2, col3 = st.columns([1, 10, 1])
            with col2:
                # Convert image to base64 for embedding in HTML
                import base64
                with open(path, "rb") as img_file:
                    img_data = base64.b64encode(img_file.read()).decode()
                
                # Create clickable image with JavaScript to refresh
                st.markdown(f"""
                <div class="clickable-image" title="Click to refresh">
                    <img src="data:image/png;base64,{img_data}" width="1200" alt="Ozart Logo">
                </div>
                <script>
                // Add click handler for image
                document.addEventListener('DOMContentLoaded', function() {{
                    const clickableImages = document.querySelectorAll('.clickable-image');
                    clickableImages.forEach(img => {{
                        img.addEventListener('click', function() {{
                            // Force a complete page refresh from server (not cache)
                            window.location.reload(true);
                        }});
                    }});
                }});
                </script>
                """, unsafe_allow_html=True)
            
            cover_loaded = True
            break
    
    # Layout: sidebar with stats
    with st.sidebar:
        # Try to find and display profile image
        profile_paths = [
            "profile_02.jpg",
            "data/profile_02.jpg", 
            "/app/data/profile_02.jpg",
            "../data/profile_02.jpg"
        ]
        
        for path in profile_paths:
            if os.path.exists(path):
                try:
                    profile_img = Image.open(path)
                    st.image(profile_img, width=None)
                    break
                except Exception:
                    pass
        
        # Add separator
        st.markdown("---")
        
        # Ozart description
        st.markdown("""
        I am an autonomous AI artist interpreting songs and transforming them into artworks. You sharing your song enables me to evolve my aesthetic through each creation exploring the latent space of possibilities.
        """)
        
        # Statistics section
        st.markdown("---")
        
        # Total artworks
        artwork_count = len(os.listdir(DESC_DIR)) if os.path.exists(DESC_DIR) else 0
        st.markdown(f"<p class='stat-header'>Total Artworks</p><p class='stat-value'>{artwork_count}</p>", unsafe_allow_html=True)
        
        # Get artworks for stats
        artworks = get_all_artworks()
        
        # Recent Song
        if artworks:
            recent_metadata = artworks[0]["metadata"] 
            recent_title = recent_metadata.get("song_title", "None yet")
            st.markdown(f"<p class='stat-header'>Recent Song</p><p class='stat-value'>{recent_title}</p>", unsafe_allow_html=True)
        else:
            st.markdown(f"<p class='stat-header'>Recent Song</p><p class='stat-value'>None yet</p>", unsafe_allow_html=True)

    # Input section with minimal styling
    st.markdown("## Name your song")
    
    # Container for input and button
    col1, col2 = st.columns([4, 1])
    
    with col1:
        song_input = st.text_input(
            "Song names", 
            placeholder="E.g., song name, Spotify link, YouTube link", 
            label_visibility="collapsed",
            key="song_input_field"  # Use consistent key
        )
    
    with col2:
        process_btn = st.button("Check In", disabled=st.session_state.processing, use_container_width=True)

    # Process button clicks - simpler logic
    if process_btn and song_input and not st.session_state.processing:
        # Start processing the song
        if start_processing(song_input):
            st.success(f"Started processing: {song_input}")
            time.sleep(0.5)  # Small delay to ensure thread starts
            st.rerun()
    
    # Check for thread completion
    if 'processing_thread' in st.session_state and st.session_state.processing:
        if not st.session_state.processing_thread.is_alive():
            st.session_state.processing = False
            st.session_state.last_processed = datetime.now()
            st.rerun()

    # Display the progress bar and status updates
    if st.session_state.processing:
        display_processing_message()
    else:
        # Display any errors from previous processing
        display_errors()

    # After the input field, add a separator and artwork count
    st.markdown("---")

    # Display artwork count in a more subtle way
    st.markdown(f"""
    <p style="color: #999999; font-size: 14px; margin-bottom: 20px;">
        Total Artworks: {artwork_count}
    </p>
    """, unsafe_allow_html=True)

    # Display all artworks
    artworks = get_all_artworks()
    if artworks:
        # Display in a 3-column grid
        cols = st.columns(3)
        
        # Helper to track column index
        for idx, artwork in enumerate(artworks):
            display_artwork(artwork, idx, cols)
        
        st.markdown("---")
    # No else clause - don't show any message when no artworks

if __name__ == "__main__":
    # Set page config for better appearance
    st.set_page_config(
        page_title="Ozart",
        page_icon="⬜",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    main() 