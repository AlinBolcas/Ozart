import yt_dlp
from pydub import AudioSegment
import opensmile
import os
import json
import uuid
import re
from datetime import datetime
from dotenv import load_dotenv
from openai import OpenAI
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import requests
from urllib.parse import urlparse, parse_qs, quote_plus
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import subprocess
import tempfile
import time

# Load environment variables from .env file
load_dotenv()

# Initialize OpenAI client with API key from environment
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Initialize Spotify client if credentials are available
try:
    spotify_client_id = os.getenv("SPOTIFY_CLIENT_ID")
    spotify_client_secret = os.getenv("SPOTIFY_CLIENT_SECRET")
    
    if not spotify_client_id or not spotify_client_secret:
        print("Spotify credentials missing. Check your .env file for SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET")
        spotify = None
    else:
        print(f"Initializing Spotify client with ID: {spotify_client_id[:5]}...")
        client_credentials_manager = SpotifyClientCredentials(
            client_id=spotify_client_id,
            client_secret=spotify_client_secret
        )
        spotify = spotipy.Spotify(client_credentials_manager=client_credentials_manager)
        
        # Test the connection with a public endpoint instead of user data
        try:
            # Simple request to verify credentials - search is a public endpoint
            test_results = spotify.search("test", limit=1)
            print("Spotify connection successful!")
        except spotipy.exceptions.SpotifyException as e:
            print(f"Spotify authentication test failed: {e}")
            spotify = None
            
except Exception as e:
    print(f"Error initializing Spotify: {e}")
    spotify = None

# Create output directories
OUTPUT_DIR = os.path.join("output", "user_songs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def create_session_folder():
    """Create a uniquely named folder for this processing session"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_id = str(uuid.uuid4())[:8]
    folder_name = f"{timestamp}_{session_id}"
    session_path = os.path.join(OUTPUT_DIR, folder_name)
    os.makedirs(session_path, exist_ok=True)
    return session_path

#################################
# SPOTIFY PIPELINE FUNCTIONS
#################################

def extract_spotify_id(spotify_url):
    """Extract track ID from Spotify URL"""
    parsed_url = urlparse(spotify_url)
    if 'open.spotify.com' not in parsed_url.netloc:
        return None
    
    path_parts = parsed_url.path.split('/')
    if 'track' in path_parts:
        track_index = path_parts.index('track')
        if track_index + 1 < len(path_parts):
            return path_parts[track_index + 1]
    return None

def get_spotify_track_info(track_id):
    """Get basic track info from Spotify API - simplified version"""
    if not spotify:
        print("Spotify API credentials not configured")
        return None
    
    try:
        print(f"Getting basic track info for: {track_id}")
        track_info = spotify.track(track_id)
        
        # Return only basic track details
        return {
            "id": track_id,
            "name": track_info['name'],
            "artist": track_info['artists'][0]['name'],
            "album": track_info.get('album', {}).get('name', "Unknown"),
            "duration_ms": track_info.get('duration_ms', 0),
            "popularity": track_info.get('popularity', 0),
            "url": track_info.get('external_urls', {}).get('spotify', "")
        }
    except Exception as e:
        print(f"Error getting Spotify track info: {e}")
        return None

#################################
# YOUTUBE/OPENSMILE PIPELINE
#################################

def download_youtube_audio(url, output_path):
    """Download audio from YouTube URL and extract metadata"""
    # First extract metadata
    ydl_opts_info = {
        'quiet': True,
        'no_warnings': True,
        'extract_flat': True,
    }
    
    # Get video metadata first
    with yt_dlp.YoutubeDL(ydl_opts_info) as ydl:
        info = ydl.extract_info(url, download=False)
        
    # Store metadata
    metadata = {
        'title': info.get('title', 'Unknown'),
        'uploader': info.get('uploader', 'Unknown'),
        'description': info.get('description', ''),
        'duration': info.get('duration', 0),
        'view_count': info.get('view_count', 0),
        'upload_date': info.get('upload_date', '')
    }
    
    # Truncate description if it's too long
    if len(metadata['description']) > 500:
        metadata['description'] = metadata['description'][:500] + "..."
    
    print(f"Video title: {metadata['title']}")
    print(f"Uploader: {metadata['uploader']}")
    print(f"Duration: {metadata['duration']} seconds")
    
    # Now download the audio
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': output_path + '.%(ext)s',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
            'preferredquality': '192',
        }],
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

    wav_file = output_path + '.wav'
    return wav_file, metadata

def extract_audio_features(wav_file):
    """Extract audio features using openSMILE"""
    smile = opensmile.Smile(
        feature_set=opensmile.FeatureSet.eGeMAPSv02,
        feature_level=opensmile.FeatureLevel.Functionals,
    )
    features = smile.process_file(wav_file)
    return features

def visualize_opensmile_features(features, output_path):
    """Create visualizations of openSMILE features"""
    features_dict = features.iloc[0].to_dict()
    
    # Group features by category
    categories = {
        "Pitch": [k for k in features_dict.keys() if any(x in k.lower() for x in ["f0", "pitch", "semitone"])],
        "Energy": [k for k in features_dict.keys() if any(x in k.lower() for x in ["loudness", "energy", "shimmer", "amplitude"])],
        "Spectral": [k for k in features_dict.keys() if any(x in k.lower() for x in ["spectral", "mfcc", "harmonic"])],
        "Voice": [k for k in features_dict.keys() if any(x in k.lower() for x in ["jitter", "formant"])]
    }
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for i, (category, features_list) in enumerate(categories.items()):
        if not features_list:
            continue
            
        # Get values, limiting to 10 features max per category for readability
        if len(features_list) > 10:
            features_list = features_list[:10]
            
        values = [features_dict[f] for f in features_list]
        
        # Normalize values for better visualization
        min_val = min(values)
        max_val = max(values)
        if max_val != min_val:
            values = [(v - min_val) / (max_val - min_val) for v in values]
        else:
            values = [0.5 for _ in values]
            
        # Shorten feature names for display
        labels = [f.split('_')[0][:12] for f in features_list]
        
        # Plot
        axes[i].bar(labels, values, color='teal')
        axes[i].set_title(f"{category} Features")
        axes[i].set_ylim(0, 1)
        axes[i].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    
    return output_path

def display_features(features):
    """Print a concise summary of the extracted features"""
    features_dict = features.iloc[0].to_dict()
    
    # Organize features by category
    categories = {
        "Pitch": ["F0", "pitch", "semitone"],
        "Energy/Loudness": ["loudness", "energy", "shimmer", "amplitude"],
        "Spectral": ["spectral", "mfcc", "harmonic", "alpha"],
        "Voice Quality": ["jitter", "harmonics", "formant"]
    }
    
    print("\n=== AUDIO FEATURES SUMMARY ===")
    
    # Print only category counts and a few representative metrics
    for category, keywords in categories.items():
        # Count features in this category
        category_features = [k for k, v in features_dict.items() 
                            if any(keyword.lower() in k.lower() for keyword in keywords)]
        
        if category_features:
            # Get at most 3 representative features to display
            sample_features = category_features[:3]
            values = [features_dict[k] for k in sample_features]
            
            print(f"\n--- {category}: {len(category_features)} metrics extracted ---")
            print(f"Example metrics: {', '.join([f'{k.split('_')[0]}: {features_dict[k]:.2f}' for k in sample_features])}")
        else:
            print(f"\n--- {category}: No features found ---")
    
    # Print total number of features
    print(f"\nTotal features extracted: {len(features_dict)}")
    
    return features_dict

def generate_description_from_opensmile(features, metadata=None):
    """Generate interpretation focused on audio characteristics based on openSMILE features"""
    features_dict = features.iloc[0].to_dict()
    
    # Create feature groups for better interpretation
    feature_groups = {
        "Pitch and Melody": {
            "Average Pitch": features_dict.get("F0semitoneFrom27.5Hz_sma3nz_amean", 0),
            "Pitch Variation": features_dict.get("F0semitoneFrom27.5Hz_sma3nz_stddevNorm", 0),
            "Pitch Range": features_dict.get("F0semitoneFrom27.5Hz_sma3nz_range", 0)
        },
        "Dynamics and Energy": {
            "Average Loudness": features_dict.get("loudness_sma3_amean", 0),
            "Loudness Variation": features_dict.get("loudness_sma3_stddevNorm", 0), 
            "Energy Fluctuation": features_dict.get("spectralFlux_sma3_amean", 0)
        },
        "Timbre and Texture": {
            "Brightness": features_dict.get("mfcc1_sma3_amean", 0),
            "Spectral Centroid": features_dict.get("spectralCentroid_sma3_amean", 0),
            "Harmonic Richness": features_dict.get("spectralHarmonicity_sma3_amean", 0)
        },
        "Rhythmic Elements": {
            "Temporal Variation": features_dict.get("jitterLocal_sma3nz_amean", 0),
            "Amplitude Modulation": features_dict.get("shimmerLocal_sma3nz_amean", 0)
        }
    }
    
    # Convert grouped features to JSON
    features_json = json.dumps(feature_groups, indent=2)
    
    # Create context from metadata if available
    context = ""
    if metadata:
        context = f"""
Title: {metadata['title']}
Uploader: {metadata['uploader']}
Duration: {metadata['duration']} seconds
Description: {metadata['description']}
"""
    
    # Create prompt that focuses on substantive analysis with concise language
    prompt = (
        "You're a music critic with a deep understanding of how sound affects emotion. "
        "I've analyzed a song and will share technical details, but I need your response "
        "to sound like a natural, focused reaction.\n\n"
        f"{context}\n"
        f"The analysis shows data about pitch, dynamics, timbre and rhythm: {features_json}\n\n"
        "Write a focused impression of this song that:\n"
        "- Describes the song's specific emotional qualities and mood\n"
        "- Identifies distinctive elements of the performance or composition\n"
        "- Explains how the song progresses from beginning to end\n"
        "- Notes any shifts in energy, intensity, or emotional quality\n\n"
        "IMPORTANT GUIDELINES:\n"
        "1. Be SPECIFIC and SUBSTANTIVE - avoid generic observations that could apply to any song\n"
        "2. Be CONCISE - every sentence should provide meaningful insight\n" 
        "3. Focus ONLY on what you can reasonably infer from the data and song knowledge\n"
        "4. DON'T use filler text or meaningless statements\n"
        "5. DON'T mention technical metrics or numbers - translate them to listener observations\n"
        "6. DON'T write like an essay - write like a focused listening experience\n"
    )

    # Generate interpretation with more focused, substantive tone
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a precise music analyst who focuses on substantive observations. You avoid filler text and vague statements. You translate technical insights into specific, meaningful observations about the music without mentioning the technical data itself."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=500,
        temperature=0.7,  # Slightly reduced for more precision
    )

    interpretation = response.choices[0].message.content.strip()
    return interpretation

def text_to_speech(text, output_path, voice=None):
    """Convert text to speech using OpenAI's API and play it on macOS
    
    Available voices:
    - alloy: Neutral, balanced voice
    - echo: Versatile, expressive voice
    - fable: Expressive, youthful voice
    - onyx: Deep, authoritative voice
    - nova: Warm, clear voice
    - shimmer: Bright, optimistic voice
    - sage: Gentle, measured voice
    - ash: Patient, thoughtful voice
    - coral: Friendly, enthusiastic voice
    """
    print("Converting interpretation to speech...")
    
    # Valid voice options
    valid_voices = ["alloy", "echo", "fable", "onyx", "nova", "shimmer", "sage", "ash", "coral"]
    
    # If no voice specified or invalid voice, use sage as default
    if not voice or voice not in valid_voices:
        voice = "sage"
    
    try:
        # Generate speech using OpenAI's API
        response = client.audio.speech.create(
            model="tts-1",
            voice=voice,  # Use dynamically selected voice
            input=text
        )
        
        # Save the audio to a file
        response.stream_to_file(output_path)
        print(f"Speech saved to: {output_path} using '{voice}' voice")
        
        # Play the audio on macOS
        print("Playing audio interpretation...")
        subprocess.run(["afplay", output_path], check=True)
        
        return True
    except Exception as e:
        print(f"Error generating speech: {e}")
        return False

def select_voice_for_song(interpretation, metadata=None):
    """
    Use LLM to select the most appropriate voice for the song based on interpretation
    
    Available voices:
    - alloy: Neutral, balanced voice
    - echo: Versatile, expressive voice
    - fable: Expressive, youthful voice
    - onyx: Deep, authoritative voice
    - nova: Warm, clear voice
    - shimmer: Bright, optimistic voice
    - sage: Gentle, measured voice
    - ash: Patient, thoughtful voice
    - coral: Friendly, enthusiastic voice
    
    Returns a valid voice name string, or "sage" as default if any issues occur
    """
    # Define default voice as fallback
    default_voice = "sage"
    
    # Define available voices with their characteristics
    available_voices = {
        "alloy": "Neutral, balanced voice best for factual, balanced content",
        "echo": "Versatile, expressive voice good for emotional range and dynamic content",
        "fable": "Expressive, youthful voice ideal for upbeat, energetic, or playful content",
        "onyx": "Deep, authoritative voice suited for serious, powerful, or dramatic content",
        "nova": "Warm, clear voice great for friendly, sincere, or heartfelt content",
        "shimmer": "Bright, optimistic voice perfect for uplifting, positive, or joyful content",
        "sage": "Gentle, measured voice ideal for thoughtful, contemplative, or calming content",
        "ash": "Patient, thoughtful voice good for introspective, reflective, or subtle content",
        "coral": "Friendly, enthusiastic voice suited for engaging, conversational, or energetic content"
    }
    
    try:
        # Create context from metadata if available
        context = ""
        if metadata:
            context = f"Title: {metadata.get('title', 'Unknown')}\n"
            context += f"Artist: {metadata.get('uploader', 'Unknown')}\n"
        
        # Prepare prompt for the LLM
        prompt = (
            f"Based on this interpretation of a song, select the most appropriate voice for text-to-speech narration:\n\n"
            f"{context}\n"
            f"Interpretation: {interpretation}\n\n"
            "Available voices:\n"
        )
        
        # Add voice descriptions to prompt
        for voice, description in available_voices.items():
            prompt += f"- {voice}: {description}\n"
        
        prompt += (
            "\nConsider the mood, emotion, and energy of the song. "
            "Choose ONE voice from the list above that would best convey the feeling of this interpretation. "
            "Respond with ONLY the voice name (e.g., 'alloy', 'echo', etc.) - no other text or explanation."
        )
        
        # Make API call to get voice recommendation
        response = client.chat.completions.create(
            model="gpt-4o-mini",  # Using a smaller, faster model for this simple task
            messages=[
                {"role": "system", "content": "You are a helpful assistant that chooses the most appropriate text-to-speech voice based on song content. Respond with ONLY the single voice name."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=50,  # Very short response needed
            temperature=0.3,  # Lower temperature for more predictable results
        )
        
        # Extract voice from response
        selected_voice = response.choices[0].message.content.strip().lower()
        
        # Clean up response (remove any punctuation, quotes etc.)
        selected_voice = re.sub(r'[^\w]', '', selected_voice)
        
        # Validate selected voice is in available list
        if selected_voice in available_voices:
            print(f"Selected voice '{selected_voice}' based on song interpretation")
            return selected_voice
        else:
            print(f"LLM returned invalid voice '{selected_voice}', using default '{default_voice}'")
            return default_voice
            
    except Exception as e:
        print(f"Error selecting voice: {e}")
        print(f"Using default voice '{default_voice}'")
        return default_voice

def process_youtube_pipeline(youtube_url, session_path):
    """Complete YouTube/openSMILE pipeline with context extraction and focused interpretation"""
    # Sanitize URL to create a valid filename
    safe_url = re.sub(r'[^\w\-_]', '_', youtube_url)
    if len(safe_url) > 100:  # Limit length
        safe_url = safe_url[:100]
    
    # Create output path
    output_path = os.path.join(session_path, safe_url)
    
    print("Downloading audio and extracting metadata from YouTube...")
    wav_file, metadata = download_youtube_audio(youtube_url, output_path)
    
    print("Extracting audio features with openSMILE...")
    features = extract_audio_features(wav_file)
    
    print("Analyzing features...")
    features_summary = display_features(features)
    
    # Create visualization
    viz_path = os.path.join(session_path, "opensmile_features.png")
    visualize_opensmile_features(features, viz_path)
    print(f"Visualization saved to: {viz_path}")
    
    print(f"Generating interpretation for: {metadata['title']}...")
    interpretation = generate_description_from_opensmile(features, metadata)
    
    # Save results to JSON
    results = {
        "url": youtube_url,
        "processed_time": datetime.now().isoformat(),
        "wav_file": wav_file,
        "metadata": metadata,
        "features_summary": features_summary,
        "interpretation": interpretation,
        "visualization_path": viz_path
    }
    
    with open(os.path.join(session_path, "opensmile_analysis.json"), "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    print("\nGenerated Interpretation:")
    print(interpretation)
    
    # Select appropriate voice for the song based on interpretation and metadata
    selected_voice = select_voice_for_song(interpretation, metadata)
    
    # Generate and play speech with selected voice
    speech_path = os.path.join(session_path, "interpretation.mp3")
    text_to_speech(interpretation, speech_path, voice=selected_voice)
    results["speech_path"] = speech_path
    results["tts_voice"] = selected_voice
    
    print(f"\nFiles saved to: {session_path}")
    return results

#################################
# MAIN PROCESS FUNCTION
#################################

def search_youtube(query):
    """Search YouTube for a query and return the first result URL"""
    print(f"Searching YouTube for: '{query}'")
    
    # URL encode the query
    encoded_query = quote_plus(query)
    search_url = f"https://www.youtube.com/results?search_query={encoded_query}"
    
    try:
        # Make a request to the search URL
        response = requests.get(search_url)
        if response.status_code != 200:
            print(f"Failed to search YouTube: Status code {response.status_code}")
            return None
        
        # Extract video IDs using regex
        # YouTube search results contain videoId in the page
        video_ids = re.findall(r"watch\?v=(\S{11})", response.text)
        
        if not video_ids:
            print("No YouTube videos found for this query")
            return None
        
        # Get the first unique video ID
        unique_ids = []
        for vid in video_ids:
            if vid not in unique_ids:
                unique_ids.append(vid)
                
        # Create YouTube URL
        first_video_url = f"https://www.youtube.com/watch?v={unique_ids[0]}"
        print(f"Found YouTube video: {first_video_url}")
        return first_video_url
    
    except Exception as e:
        print(f"Error searching YouTube: {e}")
        return None

def process_audio_url(url):
    """Process audio from either YouTube or Spotify URL using appropriate pipeline"""
    # Create a session folder
    session_path = create_session_folder()
    
    if 'spotify.com' in url and 'track' in url:
        print("\n=== DETECTED SPOTIFY URL ===")
        # Extract track ID and get basic info for search
        track_id = extract_spotify_id(url)
        if track_id:
            try:
                # Get simplified track info
                track_info = get_spotify_track_info(track_id)
                if not track_info:
                    raise Exception("Failed to get track info")
                    
                track_name = track_info['name']
                artist_name = track_info['artist']
                
                print(f"Found track: {track_name} by {artist_name}")
                print("Searching YouTube for this track automatically...")
                
                # Create search query and find YouTube video
                search_query = f"{track_name} {artist_name} official audio"
                youtube_url = search_youtube(search_query)
                
                if not youtube_url:
                    print("Automatic YouTube search failed. Try a different search query:")
                    search_query = input("Enter search query (or direct YouTube URL): ")
                    
                    # Check if the input is a URL or a search query
                    if 'youtube.com' in search_query or 'youtu.be' in search_query:
                        youtube_url = search_query
                    else:
                        youtube_url = search_youtube(search_query)
                        
                    if not youtube_url:
                        print("Could not find a YouTube video. Exiting.")
                        return None
                
                print("\n=== USING YOUTUBE/OPENSMILE PIPELINE ===")
                
                # Add Spotify metadata to results
                results = process_youtube_pipeline(youtube_url, session_path)
                if results:
                    results["spotify_info"] = {
                        "track_id": track_id,
                        "track_name": track_name,
                        "artist_name": artist_name,
                        "album_name": track_info['album'] if 'album' in track_info else "Unknown"
                    }
                    # Update the saved JSON
                    with open(os.path.join(session_path, "opensmile_analysis.json"), "w") as f:
                        json.dump(results, f, indent=2, default=str)
                
                return results
            except Exception as e:
                print(f"Error processing Spotify URL: {e}")
                print("Falling back to direct YouTube URL input:")
                youtube_url = input("> ")
                return process_youtube_pipeline(youtube_url, session_path)
        
    elif 'youtube.com' in url or 'youtu.be' in url:
        print("\n=== USING YOUTUBE/OPENSMILE PIPELINE ===")
        return process_youtube_pipeline(url, session_path)
        
    else:
        print("\nUnsupported URL format. Please provide a YouTube or Spotify track URL.")
        return None

#################################
# MAIN EXECUTION
#################################

if __name__ == "__main__":
    # Check if API key is properly loaded
    if not client.api_key:
        print("Error: OpenAI API key not found in .env file")
        exit(1)
    
    while True:
        # Print instructions
        print("\n=== DUAL AUDIO ANALYSIS SYSTEM ===")
        print("This tool analyzes music using two different pipelines:")
        print("1. Spotify API pipeline - for Spotify track URLs")
        print("2. YouTube + openSMILE pipeline - for YouTube URLs")
        print("\nType 'exit' or 'quit' to end the program")
        
        if not spotify:
            print("\nNOTE: For Spotify functionality, add these to your .env file:")
            print("SPOTIFY_CLIENT_ID=your_spotify_client_id")
            print("SPOTIFY_CLIENT_SECRET=your_spotify_client_secret")
        
        # Get input URL
        print("\nEnter a Spotify track URL or YouTube URL:")
        url = input("> ")
        
        # Check if user wants to exit
        if url.lower() in ['exit', 'quit', 'q']:
            print("Exiting program. Goodbye!")
            break
        
        # Process using appropriate pipeline
        process_audio_url(url)
        
        print("\n" + "="*50)
        print("Analysis complete! Ready for the next song.")
        time.sleep(1)  # Small pause before looping