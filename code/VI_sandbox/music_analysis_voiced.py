import os
import json
import uuid
import re
import tempfile
import time
import subprocess
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Optional, Union, List, Tuple, Any
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse, parse_qs, quote_plus
import sys

# Third-party imports
import yt_dlp
from pydub import AudioSegment
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
    A comprehensive music analysis tool that:
    1. Downloads audio from YouTube based on various input types (song name, Spotify URL, YouTube URL)
    2. Extracts audio features using openSMILE
    3. Generates visualizations of audio features
    4. Provides AI-powered interpretation using OpenAI API
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
        
        print(f"MusicAnalyzer initialized. Outputs will be stored in: {self.output_dir}")
        
    def _initialize_spotify(self):
        """Initialize Spotify client if credentials are available."""
        if not SPOTIFY_AVAILABLE:
            return None
            
        spotify_client_id = os.getenv("SPOTIFY_CLIENT_ID")
        spotify_client_secret = os.getenv("SPOTIFY_CLIENT_SECRET")
        
        if not spotify_client_id or not spotify_client_secret:
            print("Spotify credentials missing. Check your .env file for SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET")
            return None
        
        try:
            print(f"Initializing Spotify client with ID: {spotify_client_id[:4]}...")
            client_credentials_manager = SpotifyClientCredentials(
                client_id=spotify_client_id,
                client_secret=spotify_client_secret
            )
            spotify = spotipy.Spotify(client_credentials_manager=client_credentials_manager)
            
            # Test the connection with a public endpoint
            test_results = spotify.search("test", limit=1)
            print("Spotify connection successful!")
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
        """
        Process different types of music inputs (song name, Spotify URL, YouTube URL).
        
        Args:
            input_str: Either a song name, Spotify URL, or YouTube URL
            
        Returns:
            Dictionary with analysis results
        """
        session_path = self.create_session_folder()
        print(f"Created session folder: {session_path}")
        
        # Determine input type and process accordingly
        if 'spotify.com' in input_str and 'track' in input_str:
            return self._process_spotify_url(input_str, session_path)
        elif 'youtube.com' in input_str or 'youtu.be' in input_str:
            return self._process_youtube_url(input_str, session_path)
        else:
            # Treat as song name - search YouTube
            return self._process_song_name(input_str, session_path)
    
    def _process_spotify_url(self, spotify_url: str, session_path: Path) -> Dict[str, Any]:
        """Process a Spotify track URL."""
        print("\n=== Processing Spotify URL ===")
        
        # Extract track ID
        track_id = self._extract_spotify_id(spotify_url)
        if not track_id:
            print("Invalid Spotify URL format")
            return {"error": "Invalid Spotify URL format"}
        
        # Get track info
        track_info = self._get_spotify_track_info(track_id)
        if not track_info:
            print("Failed to get Spotify track info")
            return {"error": "Failed to get Spotify track info"}
        
        # Create search query and find YouTube video
        search_query = f"{track_info['name']} {track_info['artist']} official audio"
        print(f"Searching YouTube for: {search_query}")
        youtube_url = self._search_youtube(search_query)
        
        if not youtube_url:
            print("Failed to find YouTube video for this Spotify track")
            return {"error": "Failed to find YouTube video for this Spotify track"}
        
        # Process YouTube URL
        results = self._process_youtube_url(youtube_url, session_path)
        
        # Add Spotify metadata
        if "error" not in results:
            results["spotify_info"] = track_info
            
            # Save updated results
            with open(session_path / "analysis_results.json", "w") as f:
                json.dump(results, f, indent=2, default=str)
        
        return results
    
    def _process_youtube_url(self, youtube_url: str, session_path: Path) -> Dict[str, Any]:
        """Process a YouTube URL."""
        print(f"\n=== Processing YouTube URL: {youtube_url} ===")
        
        try:
            # Download audio
            print("Downloading audio...")
            wav_file, metadata = self._download_youtube_audio(youtube_url, session_path)
            
            # Extract features
            print("Extracting audio features...")
            features = self._extract_audio_features(wav_file)
            
            # Display features summary
            print("Analyzing features...")
            features_summary = self._display_features(features)
            
            # Create visualization
            print("Creating visualization...")
            viz_path = session_path / "feature_visualization.png"
            self._visualize_features(features, viz_path)
            
            # Generate interpretation
            print("Generating AI interpretation...")
            interpretation = self._generate_interpretation(features, metadata)
            
            # Save interpretation to text file
            interpretation_path = session_path / "interpretation.txt"
            with open(interpretation_path, "w") as f:
                f.write(interpretation)
            
            # Generate audio version of interpretation
            print("Converting interpretation to speech...")
            voice = self._select_voice_for_interpretation(interpretation, metadata)
            speech_path = session_path / "interpretation.mp3"
            self._text_to_speech(interpretation, speech_path, voice)
            
            # Compile results
            results = {
                "url": youtube_url,
                "processed_time": datetime.now().isoformat(),
                "wav_file": str(wav_file),
                "metadata": metadata,
                "features_summary": features_summary,
                "interpretation": interpretation,
                "visualization_path": str(viz_path),
                "speech_path": str(speech_path),
                "tts_voice": voice,
                "session_path": str(session_path)
            }
            
            # Save results
            with open(session_path / "analysis_results.json", "w") as f:
                json.dump(results, f, indent=2, default=str)
            
            print(f"\nAnalysis complete! Results saved to: {session_path}")
            return results
            
        except Exception as e:
            print(f"Error processing YouTube URL: {e}")
            return {"error": str(e)}
    
    def _process_song_name(self, song_name: str, session_path: Path) -> Dict[str, Any]:
        """Process a song name by searching YouTube."""
        print(f"\n=== Processing song: {song_name} ===")
        
        # Search YouTube
        youtube_url = self._search_youtube(song_name)
        if not youtube_url:
            print("Failed to find YouTube video for this song")
            return {"error": "Failed to find YouTube video for this song"}
        
        # Process YouTube URL
        return self._process_youtube_url(youtube_url, session_path)
    
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
    
    def _download_youtube_audio(self, url: str, session_path: Path) -> Tuple[Path, Dict[str, Any]]:
        """Download audio from YouTube URL and extract metadata."""
        # Create output file path
        safe_url = re.sub(r'[^\w\-_]', '_', url)
        if len(safe_url) > 100:  # Limit length
            safe_url = safe_url[:100]
        
        output_path = session_path / safe_url
        
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
        if metadata['description'] and len(metadata['description']) > 500:
            metadata['description'] = metadata['description'][:500] + "..."
        
        print(f"Video title: {metadata['title']}")
        print(f"Uploader: {metadata['uploader']}")
        print(f"Duration: {metadata['duration']} seconds")
        
        # Now download the audio
        ydl_opts = {
            'format': 'bestaudio/best',
            'outtmpl': str(output_path) + '.%(ext)s',
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'wav',
                'preferredquality': '192',
            }],
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])

        wav_file = output_path.with_suffix('.wav')
        return wav_file, metadata
    
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
                print(f"Example metrics: {', '.join([f'{k.split('_')[0]}: {features_dict[k]:.2f}' for k in sample_features])}")
            else:
                print(f"\n--- {category}: No features found ---")
        
        # Print total number of features
        print(f"\nTotal features extracted: {len(features_dict)}")
        
        return features_dict
    
    def _visualize_features(self, features: pd.DataFrame, output_path: Path) -> Path:
        """Create visualizations of audio features."""
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
        
        # Create context from metadata
        context = ""
        if metadata:
            context = f"""
Title: {metadata['title']}
Uploader: {metadata['uploader']}
Duration: {metadata['duration']} seconds
Description: {metadata['description']}
"""
        
        # Create prompt for OpenAI
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

        # Generate interpretation using OpenAI API
        interpretation = self.openai.chat_completion(
            user_prompt=prompt,
            system_prompt="You are a precise music analyst who focuses on substantive observations. You avoid filler text and vague statements. You translate technical insights into specific, meaningful observations about the music without mentioning the technical data itself.",
            temperature=0.7
        )

        return interpretation
    
    def _select_voice_for_interpretation(self, interpretation: str, metadata: Dict[str, Any]) -> str:
        """Select the most appropriate voice for text-to-speech based on the interpretation."""
        if not self.openai:
            return "nova"  # Default voice if OpenAI API is not available
        
        # Available voices and their characteristics
        available_voices = {
            "alloy": "Neutral, balanced voice best for factual, balanced content",
            "echo": "Versatile, expressive voice good for emotional range and dynamic content",
            "fable": "Expressive, youthful voice ideal for upbeat, energetic, or playful content",
            "onyx": "Deep, authoritative voice suited for serious, powerful, or dramatic content",
            "nova": "Warm, clear voice great for friendly, sincere, or heartfelt content",
            "shimmer": "Bright, optimistic voice perfect for uplifting, positive, or joyful content"
        }
        
        # Create context from metadata
        context = ""
        if metadata:
            context = f"Title: {metadata.get('title', 'Unknown')}\n"
            context += f"Artist: {metadata.get('uploader', 'Unknown')}\n"
        
        # Prepare prompt for OpenAI
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
        
        # Get voice recommendation from OpenAI
        response = self.openai.chat_completion(
            user_prompt=prompt,
            system_prompt="You are a helpful assistant that chooses the most appropriate text-to-speech voice based on song content. Respond with ONLY the single voice name.",
            temperature=0.3,
            max_tokens=50
        )
        
        # Extract and validate voice name
        selected_voice = response.strip().lower()
        selected_voice = re.sub(r'[^\w]', '', selected_voice)
        
        if selected_voice in available_voices:
            print(f"Selected voice '{selected_voice}' based on song interpretation")
            return selected_voice
        else:
            print(f"Invalid voice '{selected_voice}', using default 'nova'")
            return "nova"
    
    def _text_to_speech(self, text: str, output_path: Path, voice: str = "nova") -> bool:
        """Convert text to speech using OpenAI's API."""
        if not self.openai:
            print("OpenAI API not available for text-to-speech")
            return False
        
        print(f"Converting interpretation to speech using '{voice}' voice...")
        
        try:
            # Generate speech
            speech_path = self.openai.text_to_speech(
                text=text,
                voice=voice,
                output_path=str(output_path)
            )
            
            print(f"Speech saved to: {output_path}")
            return True
            
        except Exception as e:
            print(f"Error generating speech: {e}")
            return False
    
    def play_audio(self, audio_path: Union[str, Path]) -> bool:
        """Play an audio file using the appropriate system command."""
        audio_path = str(audio_path)
        print(f"Playing audio: {audio_path}")
        
        try:
            if sys.platform == "darwin":  # macOS
                subprocess.run(["afplay", audio_path])
            elif sys.platform == "win32":  # Windows
                os.startfile(audio_path)
            else:  # Linux
                subprocess.run(["aplay", audio_path])
            return True
        except Exception as e:
            print(f"Error playing audio: {e}")
            return False
    
    def display_image(self, image_path: Union[str, Path]) -> bool:
        """Display an image using the appropriate system command."""
        image_path = str(image_path)
        print(f"Displaying image: {image_path}")
        
        try:
            if sys.platform == "darwin":  # macOS
                subprocess.run(["open", image_path])
            elif sys.platform == "win32":  # Windows
                os.startfile(image_path)
            else:  # Linux
                subprocess.run(["xdg-open", image_path])
            return True
        except Exception as e:
            print(f"Error displaying image: {e}")
            return False


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
            print("\n=== INTERPRETATION ===")
            print(results["interpretation"])
            
            # Play speech
            if "speech_path" in results:
                print("\n=== PLAYING AUDIO INTERPRETATION ===")
                analyzer.play_audio(results["speech_path"])
            
            # Show visualization
            if "visualization_path" in results:
                print("\n=== SHOWING VISUALIZATION ===")
                analyzer.display_image(results["visualization_path"])
        else:
            print("\n❌ Analysis failed. Please try again with a different input.")
        
        print("\n" + "="*50)
