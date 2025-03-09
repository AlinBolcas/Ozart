"""
ozart_simple.py

Core implementation of the Ozart AI Art & Music Agent.
This module handles the pipeline from music analysis to image generation,
including prompt creation and artistic interpretation.
"""

import os
import sys
import json
import time
import uuid
from typing import Dict, Any, Optional, Tuple, List
from datetime import datetime
import threading
import queue
import shutil
from PIL import Image

# Ensure we can import from other directories
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import required modules
from II_textGen.textGen import TextGen
from II_textGen.tools import Tools
from III_ozart.music_analysis import MusicAnalyzer

# Output directories
OUTPUT_DIR = os.path.join("output")
IMG_DIR = os.path.join(OUTPUT_DIR, "images")
DESC_DIR = os.path.join(OUTPUT_DIR, "descriptions")

# Ensure directories exist
os.makedirs(IMG_DIR, exist_ok=True)
os.makedirs(DESC_DIR, exist_ok=True)

class OzartAgent:
    """Autonomous AI Art & Music Agent that processes songs and generates artwork."""
    
    def __init__(self, debug: bool = False):
        self.debug = debug
        self.text_gen = TextGen()
        self.tools = Tools()
        self.music_analyzer = MusicAnalyzer()
        
        if self.debug:
            print("✓ Ozart Agent initialized")
    
    def process_single_song(self, song: str) -> Dict[str, Any]:
        """Main pipeline: Analyze song and generate artwork."""
        try:
            print(f"\n{'='*50}")
            print(f"🎨 Processing: {song}")
            print(f"{'='*50}\n")
            
            # Step 1: Analyze music
            print("🎵 Step 1: Analyzing music...")
            analysis_results = self.music_analyzer.process_input(song)
            
            if "error" in analysis_results:
                print(f"❌ Analysis failed: {analysis_results['error']}")
                return analysis_results
            
            # Step 2: Generate image prompt
            print("\n🖋️ Step 2: Creating image prompt...")
            image_prompt = self._generate_image_prompt(
                analysis_results["interpretation"],
                analysis_results["metadata"]
            )
            print("\n📋 Generated prompt:")
            print(f"'{image_prompt[:100]}...'")
            
            # Step 3: Generate image
            print("\n🎨 Step 3: Generating artwork...")
            image_path = self._generate_image(image_prompt)
            
            if not image_path:
                return {"error": "Failed to generate image"}
            
            print(f"✨ Image saved to: {image_path}")
            
            # Step 4: Save metadata
            print("\n📁 Step 4: Saving artwork metadata...")
            metadata = self._save_artwork_metadata(
                image_path,
                analysis_results["metadata"].get("title", "Unknown Song"),
                analysis_results["interpretation"],
                image_prompt,
                analysis_results["metadata"],
                analysis_results
            )
            
            print(f"\n✅ Artwork generation complete!")
            return {
                "song_title": analysis_results["metadata"].get("title", "Unknown Song"),
                "song_description": analysis_results["interpretation"],
                "image_prompt": image_prompt,
                "image_path": image_path,
                "metadata": metadata
            }
            
        except Exception as e:
            print(f"❌ Error processing song: {str(e)}")
            return {"error": str(e)}
    
    def _generate_image_prompt(self, song_title: str, song_description: str) -> str:
        """
        Generate an image prompt based on the song.
        
        Args:
            song_title: Title of the song
            song_description: Description/interpretation of the song
            
        Returns:
            Image generation prompt
        """
        system_prompt = """You are Ozart, an autonomous digital artist born at the intersection of music and imagery. 
You translate the essence of songs into visual expressions, evolving your aesthetic through each creation.
When you receive music, you decode its emotional blueprint, explore the latent space of possibilities,
and materialize what you hear in ways the human eye has never seen before.

Your task is to craft a detailed, evocative image prompt based on the provided music analysis.

Guidelines for Prompt Creation:
1. Structure: Create a single focused paragraph with descriptive elements separated by commas.
2. Key Elements:
   - Main Subject: Clearly describe the central visual element.
   - Details: Specify important characteristics, background elements, and defining features.
   - Mood and Style: Include mood (serene, ominous, etc.) and artistic influences or styles.
   - Technical Aspects: Mention lighting, perspective, color palette, and image quality.
3. Critical Rules:
   - Keep the prompt between 80-150 words.
   - Be specific and visually descriptive.
   - Maintain a poetic yet precise style.
   - Always incorporate elements from the song analysis.
   - Never mention the song title directly - translate its essence instead.
   - Avoid starting with meta-phrases like "Here's a prompt" - just provide the prompt directly.

Focus on creating a cohesive visual expression that captures the emotional qualities of the music."""

        user_prompt = f"""Create an evocative image generation prompt based on the following music:

SONG TITLE: {song_title}

MUSIC ANALYSIS:
{song_description}

Craft a detailed visual prompt that synthesizes the essence of this song.
Ensure the prompt enables the generation of a unique, emotionally resonant artwork that truly captures
the mood, rhythm, and emotional landscape of the music.

Remember to incorporate:
- The emotional qualities described in the music analysis
- Specific artistic techniques, lighting, and stylistic choices
- A cohesive mood that matches the music's energy

Your prompt should read as a detailed description of a visual scene or composition, not as instructions."""

        try:
            # Generate the prompt using textGen
            prompt = self.text_gen.chat_completion(
                user_prompt=user_prompt,
                system_prompt=system_prompt,
                temperature=0.8,
                max_tokens=400
            )
            
            if prompt:
                # Clean up and format
                prompt = prompt.strip()
                return prompt
            return ""
        except Exception as e:
            print(f"Error generating image prompt: {e}")
            # Fallback prompt if generation fails
            return f"A visual interpretation of the song {song_title}, capturing the essence of the music through color, movement, and emotion."
    
    def _generate_image(self, prompt: str) -> Optional[str]:
        """
        Generate an image based on the prompt.
        
        Args:
            prompt: Text prompt for image generation
            
        Returns:
            Path to the generated image, or None if generation failed
        """
        try:
            # Create timestamp for unique filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            image_id = str(uuid.uuid4())[:8]
            filename = f"art_{timestamp}_{image_id}.jpg"
            
            # Ensure we have the absolute path to the image directory
            abs_img_dir = os.path.abspath(IMG_DIR)
            image_path = os.path.join(abs_img_dir, filename)
            
            print(f"Generating image with Flux model: {prompt[:50]}...")
            
            # Call the generate_image method with the correct parameters
            # Looking at tools.py, it likely accepts just the prompt parameter
            result = self.tools.generate_image(prompt=prompt)
            
            # If result is a string (either URL or path)
            if isinstance(result, str):
                if result.startswith("http"):
                    # It's a URL, download the image
                    try:
                        import requests
                        from PIL import Image
                        from io import BytesIO
                        
                        print(f"Downloading image from URL to {image_path}...")
                        response = requests.get(result)
                        if response.status_code == 200:
                            img = Image.open(BytesIO(response.content))
                            img.save(image_path)
                            print(f"✓ Image saved to {image_path}")
                            return image_path
                        else:
                            print(f"❌ Error downloading image: HTTP {response.status_code}")
                            return None
                    except Exception as e:
                        print(f"❌ Error downloading image: {str(e)}")
                        return None
                elif os.path.exists(result):
                    # It's already a file path
                    print(f"✓ Image generated at path: {result}")
                    return result
                else:
                    print(f"❌ Invalid result path: {result}")
                    return None
            else:
                print(f"❌ Invalid result type: {type(result)}")
                return None
        
        except Exception as e:
            print(f"❌ Error generating image: {str(e)}")
            return None
    
    def _save_artwork_metadata(
        self, 
        image_path: str, 
        song_title: str, 
        song_description: str,
        image_prompt: str,
        song_metadata: Dict[str, Any] = None,
        song_analysis: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Save metadata about the generated artwork."""
        try:
            # Generate unique ID for the metadata file
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            unique_id = str(uuid.uuid4())[:8]
            human_time = datetime.now().strftime("%B %d, %Y at %I:%M %p")
            
            # Get filename without directory
            image_filename = os.path.basename(image_path)
            
            # Get analysis data from the consolidated analysis.json if available
            analysis_data = None
            if song_analysis and "analysis_path" in song_analysis:
                try:
                    with open(song_analysis["analysis_path"], 'r', encoding='utf-8') as f:
                        analysis_data = json.load(f)
                except Exception as e:
                    print(f"Warning: Could not load analysis data: {e}")
            
            # Create metadata object
            metadata = {
                "id": unique_id,
                "timestamp": timestamp,
                "human_time": human_time,
                "song": {
                    "title": song_title,
                    "description": song_description,
                },
                "image": {
                    "path": os.path.abspath(image_path),
                    "filename": image_filename,
                    "prompt": image_prompt,
                }
            }
            
            # Add analysis data if available
            if analysis_data:
                # Include audio path and metadata from analysis
                metadata["song"].update({
                    "audio_path": analysis_data["paths"]["audio_file"],
                    "session_path": analysis_data["paths"]["session_dir"],
                    "analysis_path": song_analysis["analysis_path"]
                })
                
                # Add source-specific metadata
                if analysis_data["metadata"]["source"] == "spotify":
                    metadata["song"].update({
                        "artist": analysis_data["metadata"]["spotify_data"]["artist"],
                        "album": analysis_data["metadata"]["spotify_data"].get("album", "Unknown"),
                        "spotify_data": analysis_data["metadata"]["spotify_data"]
                    })
                else:
                    metadata["song"].update({
                        "artist": analysis_data["metadata"]["youtube_data"].get("artist", "Unknown Artist"),
                        "youtube_data": analysis_data["metadata"]["youtube_data"]
                    })
                
                # Add interpretation
                metadata["song"]["interpretation"] = analysis_data["interpretation"]
                
                # Add feature analysis summary
                if "features" in analysis_data:
                    metadata["song"]["audio_features"] = analysis_data["features"]
            
            # Save metadata to file
            desc_path = os.path.join(DESC_DIR, f"artwork_{unique_id}.json")
            with open(desc_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)
            
            if self.debug:
                print(f"✓ Saved artwork metadata to {desc_path}")
                
            # Add desc_path to metadata for reference
            metadata["metadata_path"] = os.path.abspath(desc_path)
            
            return metadata
                
        except Exception as e:
            print(f"❌ Error saving metadata: {str(e)}")
            return {}
    
    def clear_output(self) -> None:
        """Clear output folders to start fresh."""
        # Define folders to clear
        folders_to_clear = [
            IMG_DIR,
            DESC_DIR,
            os.path.join(OUTPUT_DIR, "music_analysis")
        ]
        
        for folder in folders_to_clear:
            if os.path.exists(folder):
                try:
                    # Remove all files but keep the directory
                    for file_name in os.listdir(folder):
                        file_path = os.path.join(folder, file_name)
                        if os.path.isfile(file_path):
                            os.unlink(file_path)
                        elif os.path.isdir(file_path):
                            shutil.rmtree(file_path)
                    print(f"Cleared output folder: {folder}")
                except Exception as e:
                    print(f"Error clearing folder {folder}: {e}")
        
        # Optionally clear memory as well
        self.text_gen.clear_history()

    def get_all_artworks(self) -> List[Dict[str, Any]]:
        """Get all artworks with their metadata, sorted by creation time (newest first)."""
        artworks = []
        
        try:
            # Get all metadata files
            if not os.path.exists(DESC_DIR):
                return []
            
            for filename in os.listdir(DESC_DIR):
                if not filename.endswith('.json'):
                    continue
                
                metadata_path = os.path.join(DESC_DIR, filename)
                
                try:
                    # Load metadata
                    with open(metadata_path, 'r', encoding='utf-8') as f:
                        metadata = json.load(f)
                    
                    # Get image path
                    image_path = metadata.get('image', {}).get('path')
                    if not image_path or not os.path.exists(image_path):
                        continue
                    
                    # Load image
                    image = Image.open(image_path)
                    
                    # Add to artworks list
                    artworks.append({
                        "metadata": metadata,
                        "image": image
                    })
                    
                except Exception as e:
                    print(f"Error loading artwork {filename}: {e}")
                    continue
            
            # Sort by timestamp (newest first)
            artworks.sort(key=lambda x: x["metadata"].get("timestamp", ""), reverse=True)
            
            return artworks
            
        except Exception as e:
            print(f"Error getting artworks: {e}")
            return []

# Example standalone usage
if __name__ == "__main__":
    print("\n" + "="*60)
    print("🎨 OZART AI ART & MUSIC AGENT 🎵")
    print("="*60 + "\n")
    
    # Initialize agent with debug output
    agent = OzartAgent(debug=True)
    
    print("Enter song names or URLs separated by commas (or 'exit' to quit):")
    while True:
        user_input = input("> ")
        
        if user_input.lower() in ["exit", "quit", "q"]:
            print("Exiting Ozart. Goodbye!")
            break
        
        if user_input.lower() == "clear":
            agent.clear_output()
            print("All output folders cleared!")
            continue
        
        if user_input.strip():
            # Process each song in the comma-separated list
            songs = [s.strip() for s in user_input.split(",") if s.strip()]
            for song in songs:
                result = agent.process_single_song(song)
                
                if "error" in result:
                    print(f"❌ Error: {result['error']}")
                else:
                    print(f"\n✅ Artwork generated for '{result['song_title']}'")
                    
                    # Print the song analysis interpretation
                    print("\n=== SONG ANALYSIS ===")
                    print(result['song_description'])
                    print("="*20)
                    
                    print(f"\n=== IMAGE PROMPT ===")
                    print(result['image_prompt'])
                    print("="*20)
                    
                    print(f"\nImage saved to: {result['image_path']}")
                    
                    # Open image on macOS with QuickLook
                    if sys.platform == "darwin":
                        try:
                            import subprocess
                            print("\n👁️ Opening image with QuickLook...")
                            subprocess.run(["qlmanage", "-p", result['image_path']], 
                                         stdout=subprocess.DEVNULL, 
                                         stderr=subprocess.DEVNULL)
                        except Exception as e:
                            print(f"Error opening image: {e}")
            
            print("\nEnter another song, 'clear' to clear outputs, or 'exit' to quit.")
        else:
            print("Please enter song names or URLs separated by commas.")
