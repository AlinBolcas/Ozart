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

# Create storage directories in project root
IMG_DIR = os.path.join("output", "images")
DESC_DIR = os.path.join("output", "descriptions")

# Ensure the directories exist
os.makedirs(IMG_DIR, exist_ok=True)
os.makedirs(DESC_DIR, exist_ok=True)

def get_ai_response(song_input):
    """Generate an artistic response based on song input"""
    if not song_input:
        return "Waiting for your song inspiration..."
    
    # Placeholder response logic
    return f"Creating artistic interpretations inspired by: {song_input}"

def generate_artwork():
    """Generate a random artwork (placeholder function)"""
    # Add timestamp to avoid caching
    timestamp = int(datetime.now().timestamp() * 1000)
    # Using cat image as placeholder for now
    image_url = f"https://cataas.com/cat/cute?t={timestamp}"
    
    try:
        # Try to load the image
        response = requests.get(image_url)
        if response.status_code == 200:
            return Image.open(BytesIO(response.content))
        else:
            # Fallback to another reliable source if the first fails
            fallback_url = f"https://placekitten.com/400/300?t={timestamp}"
            fallback_response = requests.get(fallback_url)
            if fallback_response.status_code == 200:
                return Image.open(BytesIO(fallback_response.content))
            else:
                return None
    except Exception as e:
        st.error(f"Error generating artwork: {str(e)}")
        return None

def generate_artwork_description():
    """Generate an artistic description of the artwork (placeholder)"""
    descriptions = [
        "An exploration of rhythm through visual metaphor, expressing the song's underlying emotional current.",
        "This piece captures the crescendo moment, translating auditory peaks into visual intensity.",
        "A visual interpretation of the song's melodic structure, with each color representing tonal variations.",
        "The artwork embodies the lyrical narrative through symbolic representation and textural contrast.",
        "Dynamic movement patterns inspired by the song's tempo changes and emotional progression.",
        "A composition that translates the song's harmonic structure into visual depth and spatial relationships.",
        "This generation captures the essence of the musical atmosphere through color temperature and form.",
        "An abstract representation of the song's emotional journey, emphasized through contrast and flow.",
        "The piece visualizes sound waves as organic forms, creating a synesthetic experience of the music.",
        "A textural exploration inspired by the song's instrumental layers and timbral qualities."
    ]
    return random.choice(descriptions)

def save_new_artwork():
    """Create new artwork, save to disk, and return metadata"""
    image = generate_artwork()
    if not image:
        return None
    
    # Generate unique ID and timestamp
    artwork_id = str(uuid.uuid4())[:8]
    timestamp = datetime.now()
    timestamp_str = timestamp.strftime("%Y%m%d_%H%M%S")
    
    # Generate description
    description = generate_artwork_description()
    
    # Create filenames
    img_filename = f"art_{timestamp_str}_{artwork_id}.jpg"
    desc_filename = f"desc_{timestamp_str}_{artwork_id}.json"
    
    # Save image - convert to RGB first to avoid mode P error
    img_path = os.path.join(IMG_DIR, img_filename)
    if image.mode in ('RGBA', 'LA', 'P'):
        image = image.convert('RGB')
    image.save(img_path, "JPEG")
    
    # Save description
    metadata = {
        "id": artwork_id,
        "timestamp": timestamp.isoformat(),
        "human_time": timestamp.strftime("%d/%m/%y, %H:%M"),
        "description": description,
        "image_path": img_path
    }
    
    with open(os.path.join(DESC_DIR, desc_filename), 'w') as f:
        json.dump(metadata, f)
    
    return metadata

def get_all_artworks(limit=20):
    """Get all artworks, newest first, up to limit"""
    # Get all description files
    desc_files = [f for f in os.listdir(DESC_DIR) if f.startswith("desc_") and f.endswith(".json")]
    
    # Sort by timestamp (filenames start with the timestamp)
    desc_files.sort(reverse=True)
    
    # Load each description and corresponding image
    artworks = []
    for i, desc_file in enumerate(desc_files[:limit]):
        try:
            with open(os.path.join(DESC_DIR, desc_file), 'r') as f:
                metadata = json.load(f)
            
            # Extract the image filename
            img_path = metadata.get("image_path")
            if not img_path:
                # Fallback to extracting from description filename
                img_file = desc_file.replace('desc_', 'art_').replace('.json', '.jpg')
                img_path = os.path.join(IMG_DIR, img_file)
            
            # Only add if the image file exists
            if os.path.exists(img_path):
                # Load the image
                image = Image.open(img_path)
                artworks.append({
                    "metadata": metadata,
                    "image": image
                })
        except Exception as e:
            print(f"Error loading artwork data: {e}")
    
    return artworks

def make_black_transparent(img, threshold=20):
    """Convert black/dark pixels to transparent"""
    # Create a copy and ensure it's RGBA (with alpha channel)
    img = img.convert("RGBA")
    datas = img.getdata()
    
    newData = []
    for item in datas:
        # If pixel is dark (close to black), make it transparent
        if item[0] <= threshold and item[1] <= threshold and item[2] <= threshold:
            newData.append((0, 0, 0, 0))  # Transparent
        else:
            newData.append(item)  # Keep as is
            
    img.putdata(newData)
    return img

def main():
    # Move the CSS styling to the top - unchanged
    st.markdown("""
    <style>
        .stTextInput input {
            background-color: #0A0A0A !important;
            color: #999999 !important;
            border: 1px solid #333333 !important;
        }
        .stButton button {
            background-color: #111111 !important;
            color: white !important;
            border: 1px solid #333333 !important;
        }
        .stApp {
            background-color: #000000 !important;
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Look for cover image in common locations
    
    # Try specific image paths
    cover_paths = [
        "cover_02.png",
        "data/cover_02.png", 
        "/app/data/cover_02.png",
        "../data/cover_02.png"
    ]
    
    profile_paths = [
        "profile_02.jpg",
        "data/profile_02.jpg", 
        "/app/data/profile_02.jpg",
        "../data/profile_02.jpg"
    ]
    
    # Check each cover path
    for path in cover_paths:
        if os.path.exists(path):
            cover_img = Image.open(path)
            col1, col2, col3 = st.columns([1, 10, 1])
            with col2:
                # Use a larger width, up to 1200px
                st.image(cover_img, width=1200)
            break
    
    # Minimal sidebar menu
    with st.sidebar:
        # Profile image - make it full width
        for path in profile_paths:
            if os.path.exists(path):
                try:
                    profile_img = Image.open(path)
                    # Use full width of sidebar for profile
                    st.image(profile_img, width=None, use_container_width=True)
                    break
                except Exception:
                    pass
        
        # Just one separator at the top - removed header and buttons
        st.markdown("---")
        
        # Add project description - removed title, single paragraph, first person
        st.markdown("""
        I am Ozart, an autonomous digital artist born at the intersection of music and imagery. 
        I translate the essence of songs into visual expressions, evolving my aesthetic through each creation. 
        When you share your music with me, I decode its emotional blueprint, explore the latent space of possibilities, 
        and materialize what I hear in ways the human eye has never seen before.
        """)
        
        # Statistics section with smaller font
        st.markdown("---")
        
        # Custom CSS for smaller, gray statistics
        st.markdown("""
        <style>
        .stat-header {
            font-size: 14px;
            font-weight: bold;
            margin-bottom: 0px;
        }
        .stat-value {
            font-size: 16px;
            color: #BBBBBB;
            margin-top: 0px;
        }
        </style>
        """, unsafe_allow_html=True)
        
        # Total artworks (using custom HTML for smaller size)
        artwork_count = len(os.listdir(DESC_DIR)) if os.path.exists(DESC_DIR) else 0
        st.markdown(f"<p class='stat-header'>Total Artworks</p><p class='stat-value'>{artwork_count}</p>", unsafe_allow_html=True)
        
        # Current genre preference (placeholder)
        current_genres = ["Ambient", "Experimental", "Neo-Classical", "Electronic"]
        current_genre = random.choice(current_genres) if artwork_count > 0 else "Undetermined"
        st.markdown(f"<p class='stat-header'>Genre Preference</p><p class='stat-value'>{current_genre}</p>", unsafe_allow_html=True)
        
        # Current theme (placeholder)
        current_themes = ["Geometric Abstraction", "Chromatic Harmony", "Textural Depth", "Organic Forms"]
        current_theme = random.choice(current_themes) if artwork_count > 0 else "Exploring"
        st.markdown(f"<p class='stat-header'>Current Theme</p><p class='stat-value'>{current_theme}</p>", unsafe_allow_html=True)
    
    # Initialize session state with new timer mechanism
    if 'last_generation_time' not in st.session_state:
        st.session_state.last_generation_time = time.time()
    if 'last_check_time' not in st.session_state:
        st.session_state.last_check_time = time.time()
    if 'should_display_artworks' not in st.session_state:
        st.session_state.should_display_artworks = False
    if 'current_song_input' not in st.session_state:
        st.session_state.current_song_input = ""
    
    # Input field and check-in button
    col1, col2 = st.columns([4, 1])
    with col1:
        song_input = st.text_input("Name your songs:", key="song_input", 
                                 help="Enter song names separated by commas",
                                 label_visibility="visible")
        # Store the input for the timer-based generator
        if song_input:
            st.session_state.current_song_input = song_input
    
    with col2:
        # Add space to move button down
        st.write("")
        st.write("")  # Each write adds a bit more space
        check_in = st.button("Check In", use_container_width=True)
        
        # Check In button only refreshes the UI, doesn't create artworks
        if check_in:
            st.session_state.should_display_artworks = True
            st.session_state.last_check_time = time.time()
    
    # After the separator line, add the artwork count display
    st.markdown("---")

    # Show total artworks count on main panel too
    artwork_count = len(os.listdir(DESC_DIR)) if os.path.exists(DESC_DIR) else 0
    st.markdown(f"<p style='text-align: left; color: #999999; font-size: 14px;'>Total artworks: {artwork_count}</p>", unsafe_allow_html=True)
    
    # Automatic artwork generation on timer (every 2 seconds)
    current_time = time.time()
    generation_elapsed = current_time - st.session_state.last_generation_time
    
    # Only generate if there's input and 2 seconds have passed
    if st.session_state.current_song_input and generation_elapsed >= 2:
        save_new_artwork()
        st.session_state.last_generation_time = current_time
        st.session_state.should_display_artworks = True
    
    # Display all artworks if we have input or Check In was pressed
    if st.session_state.should_display_artworks:
        artworks = get_all_artworks()
        
        if not artworks:
            st.info("No artworks yet. Generation will begin as soon as you enter songs.")
        else:
            # Display all artworks, newest first
            for artwork in artworks:
                # First line: date/time
                st.markdown(f"**{artwork['metadata'].get('human_time', '')}**")
                # Second line: caption/description
                st.markdown(f"{artwork['metadata'].get('description', '')}")
                # Then the image
                st.image(artwork["image"], width=750)
                st.markdown("---")
    else:
        # Show welcome message when we haven't shown artworks yet
        st.info("Enter song names above to begin the artistic interpretation process.")

if __name__ == "__main__":
    # Set page config for better appearance
    st.set_page_config(
        page_title="Ozart",
        page_icon="🎨",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    main() 