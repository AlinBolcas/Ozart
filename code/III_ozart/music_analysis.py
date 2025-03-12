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
        
        # Load default prompts
        self.load_prompts()
    
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
    
    def process_input(self, input_str: str) -> Dict[str, Any]:
        """Process input text which could be a URL, song name, or file path."""
        try:
            print("\n🎵 Starting music analysis...")
            
            # Create unique session ID and directory
            session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
            session_id += f"_{str(uuid.uuid4())[:8]}"
            session_dir = os.path.join(self.output_dir, session_id)
            os.makedirs(session_dir, exist_ok=True)
            
            # Step 1: Get audio file
            print("\n📥 Obtaining audio...")
            if input_str.startswith(("https://open.spotify.com/", "spotify:")):
                print("   • Source: Spotify")
                # Handle Spotify URL
                track_info = self._get_spotify_track_info(self._extract_spotify_id(input_str))
                if not track_info:
                    return {"error": "Failed to get Spotify track info"}
                    
                youtube_url = self._search_youtube(f"{track_info['name']} {track_info['artist']}")
                if not youtube_url:
                    return {"error": "Failed to find YouTube video"}
                    
                download_result = self._download_youtube(youtube_url, session_dir)
                audio_path, yt_metadata = download_result
                if not audio_path:
                    return {"error": "Failed to download audio"}
                    
                # Combine metadata - prioritize Spotify info
                metadata = {
                    "source": "spotify",
                    "title": track_info["name"],
                    "artist": track_info["artist"],
                    "album": track_info.get("album", "Unknown"),
                    "spotify_data": track_info,
                    "youtube_data": yt_metadata,
                    "audio_path": audio_path
                }
                
            else:
                if input_str.startswith(("https://www.youtube.com/", "https://youtu.be/")):
                    print("   • Source: YouTube")
                    youtube_url = input_str
                else:
                    print("   • Source: Search query")
                    youtube_url = self._search_youtube(input_str)
                    if not youtube_url:
                        return {"error": "Failed to find YouTube video"}
                
                download_result = self._download_youtube(youtube_url, session_dir)
                audio_path, yt_metadata = download_result
                if not audio_path:
                    return {"error": "Failed to download audio"}
                    
                # Use YouTube metadata
                metadata = {
                    "source": "youtube",
                    "title": yt_metadata.get("title", "Unknown Song"),
                    "artist": yt_metadata.get("artist", "Unknown Artist"),
                    "youtube_data": yt_metadata,
                    "audio_path": audio_path
                }
            
            # Step 2: Extract features
            print("\n🔍 Extracting audio features...")
            features = self._extract_audio_features(Path(audio_path))
            if features is None:
                return {"error": "Failed to extract audio features"}
            
            # Display feature analysis
            feature_summary = self._display_features(features)
            
            # Step 3: Generate interpretation
            print("\n🤖 Generating AI interpretation...")
            interpretation = self._generate_interpretation(features, metadata)
            if not interpretation:
                return {"error": "Failed to generate interpretation"}
            
            print(f"\n📝 === SONG INTERPRETATION === 📝")
            print(f"'{metadata['title']}' by {metadata['artist']}")
            print("-" * 40)
            print(interpretation)
            print("=" * 40)
            
            # Step 4: Save everything in one consolidated JSON
            analysis_data = {
                "session_id": session_id,
                "timestamp": datetime.now().isoformat(),
                "metadata": metadata,
                "features": features.to_dict(orient='records')[0],
                "feature_summary": feature_summary,
                "interpretation": interpretation,
                "paths": {
                    "session_dir": session_dir,
                    "audio_file": audio_path
                }
            }
            
            # Save consolidated JSON
            analysis_path = os.path.join(session_dir, "analysis.json")
            with open(analysis_path, 'w', encoding='utf-8') as f:
                json.dump(analysis_data, f, ensure_ascii=False, indent=2)
            
            print(f"\n✅ Analysis complete! Results saved to: {session_dir}")
            
            # Return complete analysis results
            return {
                "session_path": session_dir,
                "audio_path": audio_path,
                "analysis_path": analysis_path,
                "interpretation": interpretation,
                "metadata": metadata,
                "features": features
            }
            
        except Exception as e:
            print(f"❌ Error processing input: {e}")
            return {"error": f"Failed to process input: {str(e)}"}
    
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
            audio_path, youtube_metadata = self._download_youtube(youtube_url, output_dir)
            # Keep Spotify's track_info as primary, but fill in missing details from YouTube
            return audio_path, track_info
        
        # Check if input is a YouTube URL
        elif 'youtube.com' in input_text or 'youtu.be' in input_text:
            print("Detected YouTube URL")
            return self._download_youtube(input_text, output_dir)
        
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
            
            audio_path, metadata = self._download_youtube(youtube_url, output_dir)
            # For direct song name input, use the input as the song title if we couldn't parse it
            if metadata.get('title') == 'unknown':
                metadata['title'] = input_text
            return audio_path, metadata

    def _download_youtube(self, youtube_url: str, output_dir: str) -> Tuple[Optional[str], Dict[str, Any]]:
        """Download audio from a YouTube URL using yt-dlp."""
        
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
    
    def _display_features(self, features: pd.DataFrame) -> Dict[str, Any]:
        """Print a concise summary of the extracted features."""
        try:
            features_dict = features.iloc[0].to_dict()
            
            # Group features by type
            feature_groups = {
                "Pitch & Melody": [],
                "Energy & Dynamics": [],
                "Timbre & Texture": [],
                "Voice & Rhythm": []
            }
            
            # Categorize features
            for key, value in features_dict.items():
                if "F0" in key:
                    feature_groups["Pitch & Melody"].append((key, value))
                elif "loudness" in key or "shimmer" in key:
                    feature_groups["Energy & Dynamics"].append((key, value))
                elif "spectral" in key or "mfcc" in key:
                    feature_groups["Timbre & Texture"].append((key, value))
                elif "jitter" in key or "voice" in key:
                    feature_groups["Voice & Rhythm"].append((key, value))
            
            # Print beautiful summary
            print("\n🎵 === AUDIO ANALYSIS SUMMARY === 🎵")
            for category, features in feature_groups.items():
                if features:
                    print(f"\n📊 {category}")
                    print("   " + "─" * 40)
                    # Print first 3 examples with cleaned up names
                    for key, value in features[:3]:
                        clean_name = key.split('_')[0].replace('F0', 'Pitch').title()
                        print(f"   • {clean_name}: {value:.2f}")
            
            total = sum(len(group) for group in feature_groups.values())
            print(f"\n📈 Total features analyzed: {total}")
            
            return features_dict
            
        except Exception as e:
            print(f"❌ Error displaying features: {e}")
            return {}

    def analyze_song(self, title, artist, features=None):
        """
        Analyze a song using AI and provide an interpretation.
        
        Args:
            title: Song title
            artist: Song artist
            features: Optional audio features dictionary
            
        Returns:
            Dictionary with interpretation and metadata
        """
        try:
            # Format the user prompt with song info
            formatted_user_prompt = self.user_prompt.format(
                title=title,
                artist=artist
            )
            
            # Add audio features to the prompt if available
            if features:
                feature_text = "\n\nAudio features:\n"
                for key, value in features.items():
                    feature_text += f"- {key}: {value}\n"
                formatted_user_prompt += feature_text
            
            # Generate interpretation using OpenAI
            interpretation = self.openai.chat_completion(
                user_prompt=formatted_user_prompt,
                system_prompt=self.system_prompt
            )
            
            return interpretation
            
        except Exception as e:
            print(f"Error analyzing song: {e}")
            return "Could not analyze the song due to an error."

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
        
        # Extract song information
        song_title = metadata.get('title', 'Unknown Song')
        artist = metadata.get('artist', metadata.get('uploader', 'Unknown Artist'))
        
        # Use loaded prompts rather than hardcoded ones
        try:
            # Format the user prompt with song info
            formatted_user_prompt = self.user_prompt.format(
                title=song_title,
                artist=artist
            )
            
            # Add audio features
            feature_text = "\n\nAudio features:\n" + json.dumps(feature_groups, indent=2)
            formatted_user_prompt += feature_text
            
            # Generate interpretation using OpenAI
            interpretation = self.openai.chat_completion(
                user_prompt=formatted_user_prompt,
                system_prompt=self.system_prompt,
                temperature=0.7,
                max_tokens=400
            )
            
            return interpretation
        except Exception as e:
            print(f"Error using custom prompts: {e}")
            
            # Fallback to original implementation if custom prompts fail
            context = f"Song Title: {song_title}\nArtist/Uploader: {artist}\n"
            features_json = json.dumps(feature_groups, indent=2)
            
            fallback_prompt = (
                "You're a music critic analyzing a specific song. I'll provide the song details and audio analysis data. "
                f"{context}\n"
                f"The analysis shows data about pitch, dynamics, timbre and rhythm: {features_json}\n\n"
                "Write a focused, concise impression of this specific song that captures its emotional qualities."
            )
            
            return self.openai.chat_completion(
                user_prompt=fallback_prompt,
                system_prompt="You are a precise music analyst focusing on substantive observations.",
                temperature=0.7,
                max_tokens=400
            )
    
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

    def load_prompts(self, custom_prompts=None):
        """Load prompts from default file or custom dict"""
        try:
            if custom_prompts:
                self.system_prompt = custom_prompts.get('music_analysis_system_prompt', '')
                self.user_prompt = custom_prompts.get('music_analysis_user_prompt', '')
                return
                
            # Try to load from file
            prompts_path = Path("data/default_prompts.json")
            if prompts_path.exists():
                with open(prompts_path, 'r') as f:
                    prompts = json.load(f)
                    self.system_prompt = prompts.get('music_analysis_system_prompt', '')
                    self.user_prompt = prompts.get('music_analysis_user_prompt', '')
            else:
                # Fallback to defaults
                self.system_prompt = "You are a professional music analyst with expertise in different music genres."
                self.user_prompt = "Please analyze this song and provide an interpretation."
                
        except Exception as e:
            print(f"Error loading prompts: {e}")
            # Set defaults if loading fails
            self.system_prompt = "You are a professional music analyst with expertise in different music genres."
            self.user_prompt = "Please analyze this song and provide an interpretation."


# Example usage
if __name__ == "__main__":
    analyzer = MusicAnalyzer()
    
    print("\n" + "="*60)
    print("🎵 MUSIC ANALYZER - TEST MODE 🎵")
    print("="*60 + "\n")
    
    print("You can enter:")
    print("1. A song name (e.g., 'Bohemian Rhapsody Queen')")
    print("2. A Spotify track URL")
    print("3. A YouTube URL")
    print("\nType 'exit' to quit\n")
    
    while True:
        print("\nEnter a song name, Spotify URL, or YouTube URL:")
        input_str = input("> ")
        
        if input_str.lower() in ['exit', 'quit', 'q']:
            print("Exiting program. Goodbye!")
            break
        
        # Process input
        results = analyzer.process_input(input_str)
        
        if "error" in results:
            print(f"\n❌ Analysis failed: {results['error']}")
        else:
            # The process_input function now handles all the printing:
            # - Audio feature analysis
            # - Song interpretation with title/artist
            # - Save location
            print("\n✨ Analysis complete!")
            print(f"📁 Full analysis saved to: {results['session_path']}")
            print(f"🎵 Audio file: {results['audio_path']}")
            print(f"📊 Analysis data: {results['analysis_path']}")
        
        print("\n" + "="*60)
