import os
import json
import uuid
import re
import tempfile
import time
import subprocess
import numpy as np
import pandas as pd
from typing import Dict, Optional, Union, List, Tuple, Any
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse, parse_qs, quote_plus
import sys

# Third-party imports
import yt_dlp
import opensmile
import requests
from dotenv import load_dotenv

# Add path to I_integrations
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "I_integrations")))

# Import our OpenAI API wrapper
try:
    from openai_API import OpenAIAPI
except ImportError:
    print("Error: Could not import OpenAIAPI. Make sure openai_API.py is in the I_integrations directory.")
    OpenAIAPI = None

# Optional Spotify integration
try:
    import spotipy
    from spotipy.oauth2 import SpotifyClientCredentials
    SPOTIFY_AVAILABLE = True
except ImportError:
    SPOTIFY_AVAILABLE = False
    print("Spotify integration not available. Install spotipy with: pip install spotipy")


class MusicAnalyzer:
    """
    A focused music analysis tool that:
    1. Downloads audio from YouTube based on various input types (song name, Spotify URL, YouTube URL)
    2. Extracts audio features using openSMILE
    3. Provides AI-powered interpretation of the audio characteristics using OpenAI API
    """
    
    def __init__(self, output_base_dir: str = "output"):
        """
        Initialize the MusicAnalyzer with necessary components and configurations.
        
        Args:
            output_base_dir: Base directory for storing all outputs
        """
        # Load environment variables
        load_dotenv()
        
        # Set up OpenAI API
        if OpenAIAPI:
            self.openai = OpenAIAPI()
        else:
            self.openai = None
            print("Warning: OpenAI API integration is not available.")
        
        # Set up Spotify if available
        self.spotify = self._initialize_spotify()
        
        # Set up output directories
        self.output_base_dir = Path(output_base_dir)
        self.output_dir = self.output_base_dir / "music_analysis"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize openSMILE
        self.smile = opensmile.Smile(
            feature_set=opensmile.FeatureSet.eGeMAPSv02,
            feature_level=opensmile.FeatureLevel.Functionals,
        )
    
    def _initialize_spotify(self):
        """Initialize Spotify client if credentials are available."""
        if not SPOTIFY_AVAILABLE:
            return None
            
        spotify_client_id = os.getenv("SPOTIFY_CLIENT_ID")
        spotify_client_secret = os.getenv("SPOTIFY_CLIENT_SECRET")
        
        if not spotify_client_id or not spotify_client_secret:
            return None
        
        try:
            client_credentials_manager = SpotifyClientCredentials(
                client_id=spotify_client_id,
                client_secret=spotify_client_secret
            )
            spotify = spotipy.Spotify(client_credentials_manager=client_credentials_manager)
            
            # Test the connection with a public endpoint
            test_results = spotify.search("test", limit=1)
            return spotify
            
        except Exception as e:
            print(f"Error initializing Spotify: {e}")
            return None
    
    def create_session_folder(self) -> Path:
        """Create a uniquely named folder for this processing session."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_id = str(uuid.uuid4())[:8]
        folder_name = f"{timestamp}_{session_id}"
        session_path = self.output_dir / folder_name
        session_path.mkdir(exist_ok=True)
        return session_path
    
    def process_input(self, input_text: str) -> Dict[str, Any]:
        """
        Process input text which could be a URL, song name, or file path.
        Returns a dictionary with analysis results.
        """
        if not input_text:
            return {"error": "Empty input provided"}
        
        try:
            # Create output directory with timestamp and unique ID
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            session_id = str(uuid.uuid4())[:8]  # Use first 8 chars of UUID
            
            # Create a directory name based on cleaned song name if possible
            cleaned_input = ''.join(c if c.isalnum() or c in ' -_' else '_' for c in input_text[:30])
            cleaned_input = cleaned_input.strip().replace(' ', '_')
            
            output_dir = os.path.join(self.output_dir, f"{timestamp}_{session_id}_{cleaned_input}")
            os.makedirs(output_dir, exist_ok=True)
            
            # Log the starting of analysis
            print(f"\nAnalyzing: {input_text}")
            
            # Step 1: Get audio file (download or locate)
            audio_path, song_metadata = self._get_audio_file(input_text, output_dir)
            
            if not audio_path or not os.path.exists(audio_path):
                return {"error": f"Failed to obtain audio for {input_text}"}
            
            # Extract proper song title and metadata
            if 'name' in song_metadata:
                # This is Spotify metadata
                song_title = song_metadata.get('name', input_text)
                artist_name = song_metadata.get('artist', 'Unknown Artist')
                description = f"Song from Spotify. Album: {song_metadata.get('album', 'Unknown')}"
                duration = song_metadata.get('duration_ms', 0) / 1000 if 'duration_ms' in song_metadata else 0
                uploader = artist_name
            else:
                # This is YouTube metadata or direct input
                song_title = song_metadata.get('title', input_text)
                artist_name = song_metadata.get('uploader', 'Unknown Artist')
                description = song_metadata.get('description', '')
                duration = song_metadata.get('duration', 0)
                uploader = song_metadata.get('uploader', 'Unknown')
            
            print(f"Using song title: {song_title} by {artist_name}")
            
            # Generate a more consistent filename for the audio
            audio_ext = os.path.splitext(audio_path)[1]
            clean_song_name = ''.join(c if c.isalnum() or c in ' -_' else '_' for c in input_text)
            clean_song_name = clean_song_name.strip().replace(' ', '_')
            new_audio_name = f"song_{timestamp}_{clean_song_name}{audio_ext}"
            new_audio_path = os.path.join(output_dir, new_audio_name)
            
            # Rename the audio file for consistency
            try:
                os.rename(audio_path, new_audio_path)
                audio_path = new_audio_path  # Update the path
            except:
                pass  # If rename fails, continue with original path
            
            # Add processed audio file to a public "audio" folder for easy access
            public_audio_dir = os.path.join(self.output_dir, "audio")
            os.makedirs(public_audio_dir, exist_ok=True)
            public_audio_path = os.path.join(public_audio_dir, new_audio_name)
            
            # Copy the file to public audio directory
            try:
                import shutil
                shutil.copy2(audio_path, public_audio_path)
            except:
                pass  # If copy fails, continue
            
            # Extract features
            print("Extracting audio features...")
            features = self._extract_audio_features(audio_path)
            
            # Display features summary
            print("Analyzing features...")
            features_summary = self._display_features(features)
            
            # Generate interpretation
            print("Generating AI interpretation...")
            interpretation = self._generate_interpretation(features, song_metadata)
            
            # Save interpretation to text file
            interpretation_path = os.path.join(output_dir, "interpretation.txt")
            with open(interpretation_path, "w") as f:
                f.write(interpretation)
            
            # Compile results
            result = {
                "url": input_text,
                "processed_time": datetime.now().isoformat(),
                "audio_path": audio_path,
                "public_audio_path": public_audio_path,
                "features_summary": features_summary,
                "interpretation": interpretation,
                "interpretation_path": interpretation_path,
                "session_path": output_dir,
                "metadata": {
                    "title": song_title,
                    "artist": artist_name,
                    "uploader": uploader,
                    "duration": duration,
                    "description": description,
                    "view_count": song_metadata.get('view_count', 0),
                    "upload_date": song_metadata.get('upload_date', '')
                }
            }
            
            # Save results
            with open(os.path.join(output_dir, "analysis_results.json"), "w") as f:
                json.dump(result, f, indent=2, default=str)
            
            print(f"\nAnalysis complete! Results saved to: {output_dir}")
            return result
            
        except Exception as e:
            print(f"Error processing input: {e}")
            return {"error": str(e)}
    
    def _get_audio_file(self, input_text: str, output_dir: str) -> Tuple[Optional[str], Dict[str, Any]]:
        """
        Get audio file from input (could be URL, song name, local file).
        Returns tuple (audio_path, song_metadata).
        """
        # Check if input is a Spotify URL
        if 'spotify.com' in input_text and 'track' in input_text:
            print("Detected Spotify URL")
            # Extract track ID
            track_id = self._extract_spotify_id(input_text)
            if not track_id:
                print("Invalid Spotify URL format")
                return None, {}
            
            # Get track info
            track_info = self._get_spotify_track_info(track_id)
            if not track_info:
                print("Failed to get Spotify track info")
                return None, {}
            
            # Save metadata
            with open(os.path.join(output_dir, "spotify_metadata.json"), "w") as f:
                json.dump(track_info, f, indent=2)
            
            # Create search query and find YouTube video
            search_query = f"{track_info['name']} {track_info['artist']} official audio"
            print(f"Searching YouTube for: {search_query}")
            youtube_url = self._search_youtube(search_query)
            
            if not youtube_url:
                print("Failed to find YouTube video for this Spotify track")
                return None, {}
            
            # Download from YouTube and merge metadata
            audio_path, youtube_metadata = self._download_from_youtube(youtube_url, output_dir)
            # Keep Spotify's track_info as primary, but fill in missing details from YouTube
            return audio_path, track_info
        
        # Check if input is a YouTube URL
        elif 'youtube.com' in input_text or 'youtu.be' in input_text:
            print("Detected YouTube URL")
            return self._download_from_youtube(input_text, output_dir)
        
        # Check if input is a local file
        elif os.path.exists(input_text) and input_text.endswith(('.mp3', '.wav', '.ogg', '.flac')):
            print("Detected local audio file")
            # Copy file to output directory
            import shutil
            dest_path = os.path.join(output_dir, os.path.basename(input_text))
            shutil.copy2(input_text, dest_path)
            
            # Extract title from filename
            filename = os.path.basename(input_text)
            title = os.path.splitext(filename)[0]
            return dest_path, {"title": title, "uploader": "Local File"}
        
        # Treat as song name - search on YouTube
        else:
            print(f"Treating as song name: {input_text}")
            youtube_url = self._search_youtube(input_text + " official audio")
            if not youtube_url:
                print(f"Could not find YouTube video for: {input_text}")
                return None, {}
            
            audio_path, metadata = self._download_from_youtube(youtube_url, output_dir)
            # For direct song name input, use the input as the song title if we couldn't parse it
            if metadata.get('title') == 'unknown':
                metadata['title'] = input_text
            return audio_path, metadata

    def _download_from_youtube(self, youtube_url: str, output_dir: str) -> Tuple[Optional[str], Dict[str, Any]]:
        """Download audio from a YouTube URL. Returns (audio_path, metadata)"""
        try:
            import yt_dlp as youtube_dl
        except ImportError:
            try:
                import youtube_dl
            except ImportError:
                print("Error: Neither yt-dlp nor youtube-dl is installed")
                return None, {}
        
        # Configure youtube-dl options
        ydl_opts = {
            'format': 'bestaudio/best',
            'outtmpl': os.path.join(output_dir, '%(title)s.%(ext)s'),
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'wav',
                'preferredquality': '192',
            }],
            'quiet': False,
            'no_warnings': False,
            # Add options to help with frequent YouTube blocks
            'geo_bypass': True,
            'geo_bypass_country': 'US',
            'socket_timeout': 15,
            'retries': 5
        }
        
        try:
            # Download audio and extract metadata
            with youtube_dl.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(youtube_url, download=True)
                title = info.get('title', 'unknown')
                
                # Extract artist and title from the video title
                artist = "Unknown Artist"
                song_title = title
                
                # Try to parse artist from title if it contains a hyphen (common format: "Artist - Song Title")
                if " - " in title:
                    parts = title.split(" - ", 1)
                    artist = parts[0].strip()
                    song_title = parts[1].strip()
                    # Remove things like "(Official Audio)" from song_title
                    song_title = re.sub(r'\(.*?\)|\[.*?\]', '', song_title).strip()
                
                # Create metadata dictionary from YouTube info
                metadata = {
                    'title': song_title,
                    'artist': artist,
                    'uploader': info.get('uploader', artist),
                    'duration': info.get('duration', 0),
                    'description': info.get('description', ''),
                    'view_count': info.get('view_count', 0),
                    'upload_date': info.get('upload_date', '')
                }
                
                # Look for the downloaded file
                audio_path = None
                for file in os.listdir(output_dir):
                    if file.endswith('.wav'):
                        audio_path = os.path.join(output_dir, file)
                        break
                
                # If we can't find the wav file, try finding any audio file with the title
                if not audio_path:
                    clean_title = ''.join(c if c.isalnum() or c in ' -_' else '_' for c in title)
                    for file in os.listdir(output_dir):
                        if clean_title.lower() in file.lower() and file.endswith(('.mp3', '.wav', '.m4a')):
                            audio_path = os.path.join(output_dir, file)
                            break
                
                if not audio_path:
                    print(f"⚠️ Warning: Downloaded audio file not found in {output_dir}")
                    return None, {}
                    
                return audio_path, metadata
            
        except Exception as e:
            print(f"Error downloading from YouTube: {e}")
            
            # Create a placeholder audio metadata file so UI doesn't break completely
            placeholder_path = os.path.join(output_dir, "download_failed.txt")
            with open(placeholder_path, 'w') as f:
                f.write(f"Failed to download: {youtube_url}\nError: {str(e)}")
            
            print(f"✏️ Created placeholder file: {placeholder_path}")
            print(f"⚠️ Try manually downloading this video and placing the audio in: {output_dir}")
            return None, {}

    def _extract_audio_features(self, wav_file: Path) -> pd.DataFrame:
        """Extract audio features using openSMILE."""
        features = self.smile.process_file(str(wav_file))
        return features
    
    def _display_features(self, features: pd.DataFrame) -> Dict[str, float]:
        """Print a concise summary of the extracted features."""
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
                # Create the metrics string separately for clarity
                metrics = []
                for k in sample_features:
                    feature_name = k.split("_")[0]
                    feature_value = features_dict[k]
                    metrics.append(f"{feature_name}: {feature_value:.2f}")
                metrics_str = ", ".join(metrics)
                print(f"Example metrics: {metrics_str}")
            else:
                print(f"\n--- {category}: No features found ---")
        
        # Print total number of features
        print(f"\nTotal features extracted: {len(features_dict)}")
        
        return features_dict
    
    def _generate_interpretation(self, features: pd.DataFrame, metadata: Dict[str, Any]) -> str:
        """Generate AI interpretation of audio features."""
        if not self.openai:
            return "OpenAI API not available for generating interpretation."
        
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
        
        # Extract song title safely from metadata - check if it exists in different formats
        song_title = metadata.get('title', 'Unknown Song')
        artist = metadata.get('artist', metadata.get('uploader', 'Unknown Artist'))
        
        # Create context from metadata
        context = f"""
Song Title: {song_title}
Artist/Uploader: {artist}
Duration: {metadata.get('duration', 0)} seconds
"""
        if 'description' in metadata and metadata['description']:
            # Truncate long descriptions
            desc = metadata['description']
            if len(desc) > 200:
                desc = desc[:197] + "..."
            context += f"Description: {desc}\n"
        
        # Create prompt for OpenAI
        prompt = (
            "You're a music critic analyzing a specific song. I'll provide the song details and audio analysis data. "
            "Your response should be a concise, insightful interpretation of the music.\n\n"
            f"{context}\n"
            f"The analysis shows data about pitch, dynamics, timbre and rhythm: {features_json}\n\n"
            "Write a focused, concise impression of this specific song that:\n"
            f"- Directly references '{song_title}' by {artist} in your analysis\n"
            "- Describes the song's emotional qualities and mood in 2-3 sentences\n"
            "- Identifies 1-2 distinctive elements of the performance\n"
            "- Notes the most significant energy or intensity shifts\n\n"
            "IMPORTANT GUIDELINES:\n"
            "1. Be SPECIFIC to this song - avoid generic observations\n"
            "2. Keep your entire response under 150 words\n" 
            "3. Make every sentence provide meaningful insight\n"
            "4. DON'T mention technical metrics - translate them to listener observations\n"
        )

        # Generate interpretation using OpenAI API
        interpretation = self.openai.chat_completion(
            user_prompt=prompt,
            system_prompt="You are a precise music analyst who focuses on substantive observations. Keep your analysis under 150 words and specific to the song mentioned.",
            temperature=0.7,
            max_tokens=200  # Limit token count to ensure brevity
        )

        return interpretation
    
    def _extract_spotify_id(self, spotify_url: str) -> Optional[str]:
        """Extract track ID from Spotify URL."""
        parsed_url = urlparse(spotify_url)
        if 'open.spotify.com' not in parsed_url.netloc:
            return None
        
        path_parts = parsed_url.path.split('/')
        if 'track' in path_parts:
            track_index = path_parts.index('track')
            if track_index + 1 < len(path_parts):
                return path_parts[track_index + 1]
        return None
    
    def _get_spotify_track_info(self, track_id: str) -> Optional[Dict[str, Any]]:
        """Get track info from Spotify API."""
        if not self.spotify:
            print("Spotify API not available")
            return None
        
        try:
            print(f"Getting track info for: {track_id}")
            track_info = self.spotify.track(track_id)
            
            # Extract relevant details
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
    
    def _search_youtube(self, query: str) -> Optional[str]:
        """Search YouTube for a query and return the first result URL."""
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


# Example usage
if __name__ == "__main__":
    analyzer = MusicAnalyzer()
    
    print("\n=== MUSIC ANALYZER ===")
    print("This tool analyzes music using openSMILE and provides AI-powered interpretation.")
    print("You can enter:")
    print("1. A song name (e.g., 'Bohemian Rhapsody Queen')")
    print("2. A Spotify track URL (e.g., 'https://open.spotify.com/track/4u7EnebtmKWzUH433cf5Qv')")
    print("3. A YouTube URL (e.g., 'https://www.youtube.com/watch?v=fJ9rUzIMcZQ')")
    print("\nType 'exit' or 'quit' to end the program")
    
    while True:
        print("\nEnter a song name, Spotify URL, or YouTube URL:")
        input_str = input("> ")
        
        if input_str.lower() in ['exit', 'quit', 'q']:
            print("Exiting program. Goodbye!")
            break
        
        # Process input
        results = analyzer.process_input(input_str)
        
        if results and "error" not in results:
            # Print interpretation
            print("\n=== MUSIC INTERPRETATION ===")
            print(results["interpretation"])
            print(f"\nFull analysis saved to: {results['session_path']}")
        else:
            print("\n❌ Analysis failed. Please try again with a different input.")
        
        print("\n" + "="*50)

