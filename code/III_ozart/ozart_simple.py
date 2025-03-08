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
import random
import uuid
from typing import List, Dict, Any, Optional, Union, Tuple
from datetime import datetime
from pathlib import Path
import threading
import queue

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
    """
    Autonomous AI Art & Music Agent that processes songs and generates
    artistic visual interpretations with descriptive text.
    """
    
    def __init__(self, debug: bool = False, images_per_song: int = 3):
        """
        Initialize the Ozart Agent with required components.
        
        Args:
            debug: Enable debug output
            images_per_song: Number of themes/images to generate per song
        """
        self.debug = debug
        self.images_per_song = images_per_song
        
        # Initialize components
        self.text_gen = TextGen()
        self.tools = Tools()
        self.music_analyzer = MusicAnalyzer()
        
        # Processing queue for async operations
        self.queue = queue.Queue()
        self.processing = False
        
        # Start background worker
        self.worker_thread = threading.Thread(target=self._process_queue)
        self.worker_thread.daemon = True
        self.worker_thread.start()
        
        if self.debug:
            print("✓ Ozart Agent initialized")
    
    def process_songs(self, song_input: str) -> None:
        """
        Process a comma-separated list of songs, adding them to the work queue.
        
        Args:
            song_input: Comma-separated list of song names/URLs
        """
        if not song_input or song_input.strip() == "":
            if self.debug:
                print("⚠ Empty song input provided")
            return
        
        # Split and clean input
        songs = [s.strip() for s in song_input.split(",") if s.strip()]
        
        if self.debug:
            print(f"🎵 Processing {len(songs)} songs: {songs}")
        
        # Add songs to queue
        for song in songs:
            self.queue.put(song)
            
        # Ensure worker is running
        if not self.processing:
            self.processing = True
    
    def _process_queue(self) -> None:
        """Background worker that processes songs from the queue."""
        while True:
            try:
                # Get song from queue
                song = self.queue.get(block=True, timeout=1)
                
                # Process song
                try:
                    self._process_single_song(song)
                except Exception as e:
                    print(f"❌ Error processing song {song}: {str(e)}")
                
                # Mark task as done
                self.queue.task_done()
                
            except queue.Empty:
                # No items in queue, sleep briefly and check again
                self.processing = False
                time.sleep(0.5)
    
    def _process_single_song(self, song: str) -> None:
        """
        Process a single song through the full pipeline.
        
        Args:
            song: Song name or URL
        """
        if self.debug:
            print(f"\n{'='*50}")
            print(f"🔍 Analyzing song: {song}")
            print(f"{'='*50}")
        
        # Step 1: Analyze the song
        song_analysis = self.music_analyzer.process_input(song)
        
        if "error" in song_analysis:
            print(f"❌ Failed to analyze song: {song_analysis['error']}")
            return
        
        song_description = song_analysis.get("interpretation", "A song with no description")
        song_metadata = song_analysis.get("metadata", {})
        song_title = song_metadata.get("title", song)
        
        if self.debug:
            print(f"✓ Song analysis complete: {song_title}")
            print(f"Description: {song_description[:100]}...")
        
        # Step 2: Generate themes/concepts based on song description
        themes = self._generate_themes(song_title, song_description)
        
        if not themes:
            print(f"❌ Failed to generate themes for song: {song_title}")
            return
        
        if self.debug:
            print(f"✓ Generated {len(themes)} themes")
            for i, theme in enumerate(themes):
                print(f"  {i+1}. {theme}")
        
        # Step 3: Generate and process each theme
        for theme in themes:
            self._process_theme(song_title, song_description, theme)
    
    def _generate_themes(self, song_title: str, song_description: str) -> List[str]:
        """
        Generate themes/concepts based on song description.
        
        Args:
            song_title: Title of the song
            song_description: Description/interpretation of the song
            
        Returns:
            List of themes/concepts
        """
        system_prompt = f"""You are an art concept developer specializing in extracting unique visual themes from music descriptions.
Your task is to identify {self.images_per_song} distinct visual concepts that could be represented in artwork based on this song.
Focus on concrete visual elements, emotional qualities, and artistic techniques that could be rendered in images.
You MUST return a JSON object with a 'themes' array containing exactly {self.images_per_song} theme objects, each with 'title' and 'description' fields."""
        
        user_prompt = f"""Based on this song and its description, identify {self.images_per_song} unique visual themes or concepts:

SONG: {song_title}

DESCRIPTION: {song_description}

For each theme, provide:
1. A short, evocative title (1-3 words)
2. A brief description explaining the visual concept (1 sentence)

Create diverse themes that explore different aspects of the song's mood, narrative, and sonic qualities.
Avoid generic concepts. Be specific and unique to this particular song.

IMPORTANT: Format your response as JSON with this exact structure:
{{
  "themes": [
    {{
      "title": "Theme title",
      "description": "Theme description"
    }},
    ... and so on for all themes
  ]
}}
"""
        
        memory_context = "Please avoid repeating themes that have been explored previously. Aim for fresh, unique concepts."
        
        try:
            print("Attempting to generate themes using structured_output...")
            # Use the structured_output method as intended
            themes_json = self.text_gen.structured_output(
                user_prompt=user_prompt,
                system_prompt=system_prompt,
                system_context=memory_context,
                output_schema={
                    "type": "object",
                    "properties": {
                        "themes": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "title": {"type": "string"},
                                    "description": {"type": "string"}
                                },
                                "required": ["title", "description"]
                            }
                        }
                    },
                    "required": ["themes"]
                },
                max_tokens=800
            )
            
            if themes_json and "themes" in themes_json:
                # Extract themes from JSON
                return [f"{t['title']}: {t['description']}" for t in themes_json["themes"]]
            else:
                print(f"❌ structured_output returned invalid format: {themes_json}")
        except Exception as e:
            print(f"❌ Error using structured_output: {str(e)}")
        
        # Fallback method for when structured_output fails
        print("Using fallback method...")
        
        # Simple fallback - just use regular chat_completion
        try:
            response = self.text_gen.chat_completion(
                user_prompt=user_prompt + f"\n\nFormat each theme as 'Title: Description'. Provide exactly {self.images_per_song} themes.",
                system_prompt=system_prompt,
                system_context=memory_context,
                temperature=0.8,
                max_tokens=800
            )
            
            # Try to extract numbered themes (1. Theme: Description)
            import re
            numbered_themes = re.findall(r'\d+\.\s*([^:]+):\s*([^\n]+)', response)
            if numbered_themes:
                # Ensure we don't exceed the requested number
                themes = [f"{title.strip()}: {desc.strip()}" for title, desc in numbered_themes[:self.images_per_song]]
                if themes:
                    return themes
        except Exception as e:
            print(f"❌ Error in fallback method: {str(e)}")
        
        # Last resort fallback
        fallback_themes = [
            f"Emotional Journey: Exploring the mood transitions in {song_title}",
            f"Sonic Landscape: Visualizing the auditory textures of {song_title}",
            f"Rhythmic Motion: Capturing the beat and tempo of {song_title}"
        ]
        # Ensure we have exactly the requested number of themes
        while len(fallback_themes) < self.images_per_song:
            fallback_themes.append(f"Harmonic Colors {len(fallback_themes)+1}: Representing the tonal palette of {song_title}")
        return fallback_themes[:self.images_per_song]
    
    def _process_theme(self, song_title: str, song_description: str, theme: str) -> None:
        """
        Process a single theme through prompt generation, image creation, and description.
        
        Args:
            song_title: Title of the song
            song_description: Description/interpretation of the song
            theme: Theme/concept to process
        """
        if self.debug:
            print(f"\n{'-'*50}")
            print(f"🎨 Processing theme: {theme}")
            print(f"{'-'*50}")
        
        # Step 1: Generate image prompt
        image_prompt = self._generate_image_prompt(song_title, song_description, theme)
        
        if not image_prompt:
            print(f"❌ Failed to generate image prompt for theme: {theme}")
            return
        
        if self.debug:
            print(f"✓ Generated image prompt: {image_prompt}")
        
        # Step 2: Generate image
        image_path = self._generate_image(image_prompt)
        
        if not image_path or not os.path.exists(image_path):
            print(f"❌ Failed to generate image for prompt: {image_prompt}")
            return
        
        if self.debug:
            print(f"✓ Generated image: {image_path}")
        
        # Step 3: Generate image description
        image_description = self._generate_image_description(
            image_path, song_title, song_description, theme, image_prompt
        )
        
        if not image_description:
            print(f"❌ Failed to generate description for image: {image_path}")
            return
        
        if self.debug:
            print(f"✓ Generated description: {image_description[:100]}...")
        
        # Step 4: Save description and metadata
        self._save_artwork_metadata(
            image_path=image_path,
            song_title=song_title,
            theme=theme,
            image_prompt=image_prompt,
            description=image_description
        )
    
    def _generate_image_prompt(
        self, 
        song_title: str, 
        song_description: str, 
        theme: str
    ) -> str:
        """
        Generate an image prompt based on the song and theme.
        
        Args:
            song_title: Title of the song
            song_description: Description/interpretation of the song
            theme: Selected theme/concept to visualize
            
        Returns:
            Image generation prompt
        """
        # Split theme into title and description
        theme_parts = theme.split(":", 1)
        theme_title = theme_parts[0].strip() if len(theme_parts) > 0 else theme
        theme_description = theme_parts[1].strip() if len(theme_parts) > 1 else ""
        
        system_prompt = """You are Ozart, an autonomous digital artist born at the intersection of music and imagery. 
You translate the essence of songs into visual expressions, evolving your aesthetic through each creation.
When you receive music, you decode its emotional blueprint, explore the latent space of possibilities,
and materialize what you hear in ways the human eye has never seen before.

Your task is to craft a detailed, evocative image prompt based on the provided music analysis and theme.

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
   - Always incorporate elements from the song analysis and theme.
   - Never mention the song title directly - translate its essence instead.
   - Avoid starting with meta-phrases like "Here's a prompt" - just provide the prompt directly.

Focus on creating a cohesive visual expression that captures the emotional qualities of the music."""

        user_prompt = f"""Create an evocative image generation prompt based on the following music and theme:

SONG TITLE: {song_title}

MUSIC ANALYSIS:
{song_description}

THEME: {theme_title}
THEME DESCRIPTION: {theme_description}

Craft a detailed visual prompt that synthesizes the essence of this song with the selected theme.
Ensure the prompt enables the generation of a unique, emotionally resonant artwork that truly captures
the mood, rhythm, and emotional landscape of the music.

Remember to incorporate:
- The emotional qualities described in the music analysis
- The visual elements suggested by the theme
- Specific artistic techniques, lighting, and stylistic choices
- A cohesive mood that matches the music's energy

Your prompt should read as a detailed description of a visual scene or composition, not as instructions."""

        try:
            # Enhance memory with insights from past generations
            memory_context = "Remember to create a diverse visual approach that doesn't repeat previous artistic styles or compositions."
            
            prompt = self.text_gen.chat_completion(
                user_prompt=user_prompt,
                system_prompt=system_prompt,
                system_context=memory_context,
                temperature=0.8,
                max_tokens=400
            )
            
            if prompt:
                # Clean up and format
                prompt = prompt.strip()
                return prompt
            return ""
        except Exception as e:
            print(f"Error during insight extraction: {e}")
            # Fallback prompt if generation fails
            return f"A visual interpretation of {theme} inspired by the song {song_title}, capturing the essence of the music through color, movement, and emotion."
    
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
            
            # Use flux model through the tools interface
            result = self.tools.generate_image(
                prompt=prompt,
                engine="flux",  # Use Flux instead of DALL-E
                aspect_ratio="1:1",
                safety_tolerance=2,
                save_path=image_path
            )
            
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
    
    def _generate_image_description(
        self, 
        image_path: str, 
        song_title: str, 
        song_description: str, 
        theme: str, 
        image_prompt: str
    ) -> str:
        """
        Generate a description of the image based on vision analysis.
        
        Args:
            image_path: Path to the generated image
            song_title: Title of the song
            song_description: Description/interpretation of the song
            theme: Theme/concept visualized
            image_prompt: Prompt used to generate the image
            
        Returns:
            Description of the image
        """
        system_prompt = """You are an insightful art critic and music interpreter.
Your task is to write a thoughtful, subjective interpretation of an AI-generated image created in response to music.
Your analysis should connect the visual elements to the musical inspiration in an engaging, poetic way."""
        
        user_prompt = f"""Provide an artistic interpretation of this image, which was generated based on:

SONG: {song_title}
MUSIC DESCRIPTION: {song_description}
THEME: {theme}
IMAGE PROMPT: {image_prompt}

In your interpretation:
1. Describe how the visual elements connect to the music's emotions and themes
2. Note specific artistic choices and their effect
3. Offer a subjective interpretation of what the AI artist was trying to convey
4. Be insightful, specific, and poetic in your language

Keep your interpretation to 2-3 sentences - concise but meaningful.
"""
        
        try:
            # Check if the image exists and is accessible
            if not os.path.exists(image_path):
                print(f"❌ Image not found at path: {image_path}")
                return "An artistic interpretation could not be generated."
            
            description = self.text_gen.vision_analysis(
                image_url=image_path,
                user_prompt=user_prompt,
                system_prompt=system_prompt,
                temperature=0.7,
                max_tokens=250
            )
            
            return description.strip()
            
        except Exception as e:
            print(f"❌ Error generating image description: {str(e)}")
            return "This artwork explores the intersection of visual and auditory expression, translating musical emotion into color and form."
    
    def _save_artwork_metadata(
        self, 
        image_path: str, 
        song_title: str, 
        theme: str, 
        image_prompt: str, 
        description: str,
        extra_metadata: Dict[str, Any] = None
    ) -> None:
        """
        Save metadata about the generated artwork.
        
        Args:
            image_path: Path to the generated image
            song_title: Title of the song
            theme: Theme/concept used for the image
            image_prompt: Prompt used to generate the image
            description: Artistic description of the image
            extra_metadata: Additional metadata to include
        """
        try:
            # Generate unique ID for the metadata file
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            human_time = datetime.now().strftime("%B %d, %Y at %I:%M %p")
            
            # Get filename without directory
            image_filename = os.path.basename(image_path)
            
            # Create metadata object
            metadata = {
                "song_title": song_title,
                "theme": theme,
                "image_prompt": image_prompt,
                "description": description,
                "image_path": image_path,
                "image_filename": image_filename,
                "timestamp": timestamp,
                "human_time": human_time
            }
            
            # Add any extra metadata
            if extra_metadata:
                metadata.update(extra_metadata)
            
            # Look for audio file in music_analysis output
            audio_dir = os.path.join("output", "music_analysis", "audio")
            if os.path.exists(audio_dir):
                for file in os.listdir(audio_dir):
                    # Look for files that might match this song
                    clean_song = ''.join(c if c.isalnum() or c in ' -_' else '_' for c in song_title.lower())
                    if clean_song in file.lower() and (file.endswith('.wav') or file.endswith('.mp3')):
                        metadata["audio_path"] = os.path.join(audio_dir, file)
                        break
            
            # Save metadata to file
            desc_path = os.path.join(DESC_DIR, f"desc_{timestamp}_{str(uuid.uuid4())[:8]}.json")
            with open(desc_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)
            
            if self.debug:
                print(f"✓ Saved metadata to {desc_path}")
                
        except Exception as e:
            print(f"❌ Error saving metadata: {str(e)}")
    
    def clear_memory(self) -> None:
        """Clear agent's memory."""
        self.text_gen.clear_history()

    def _process_single_song_direct(self, song_title, song_description, song_metadata=None):
        """
        Process a song directly without using the queue system.
        This allows direct calling from the UI thread.
        """
        try:
            # Generate themes based on song description
            themes = self._generate_themes(song_title, song_description)
            
            if not themes:
                print(f"❌ Failed to generate themes for song: {song_title}")
                return
            
            # Only use the first theme for simplicity
            theme = themes[0]
            
            # Generate the artwork
            prompt, seed, image_path, save_path = self._generate_artwork(song_title, song_description, theme)
            
            if not image_path:
                print(f"❌ Failed to generate artwork for theme: {theme}")
                return
            
            # Create description
            description_dict = {
                "song_title": song_title,
                "artist": song_metadata.get("artist", "Unknown Artist") if song_metadata else "Unknown Artist",
                "theme": theme,
                "prompt": prompt,
                "description": song_description,
                "audio_path": song_metadata.get("audio_path", "") if song_metadata else "",
                "creation_time": datetime.now().isoformat(),
                "seed": seed
            }
            
            # Save description
            desc_filename = os.path.splitext(os.path.basename(save_path))[0] + ".json"
            desc_path = os.path.join(self.desc_dir, desc_filename)
            
            with open(desc_path, "w") as f:
                json.dump(description_dict, f, indent=2)
            
            print(f"✅ Artwork saved to: {save_path}")
            print(f"✅ Description saved to: {desc_path}")
            
            return save_path
        except Exception as e:
            print(f"❌ Error in direct song processing: {str(e)}")
            return None

# Helper function to get all artworks (compatible with UI)
def get_all_artworks() -> List[Dict[str, Any]]:
    """
    Get all artworks with metadata.
    
    Returns:
        List of dictionaries with artwork data
    """
    # Check for description directory
    if not os.path.exists(DESC_DIR):
        return []
    
    artworks = []
    
    # Get all description files
    desc_files = [f for f in os.listdir(DESC_DIR) if f.startswith("desc_") and f.endswith(".json")]
    
    # Sort by timestamp (newest first)
    desc_files.sort(reverse=True)
    
    # Load each description
    for desc_file in desc_files:
        try:
            with open(os.path.join(DESC_DIR, desc_file), 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            
            # Check if image exists
            image_path = metadata.get("image_path", "")
            if not os.path.exists(image_path):
                # Try alternative paths
                image_filename = metadata.get("image_filename", "")
                alt_path = os.path.join(IMG_DIR, image_filename)
                if os.path.exists(alt_path):
                    image_path = alt_path
                else:
                    # Skip if image not found
                    continue
            
            try:
                from PIL import Image
                image = Image.open(image_path)
                
                # Add to artworks list
                artworks.append({
                    "metadata": metadata,
                    "image": image
                })
            except Exception as e:
                print(f"Error loading image {image_path}: {str(e)}")
                
        except Exception as e:
            print(f"Error loading description {desc_file}: {str(e)}")
    
    return artworks

# Example standalone usage
if __name__ == "__main__":
    print("\n" + "="*60)
    print("🎨 OZART AI ART & MUSIC AGENT 🎵")
    print("="*60 + "\n")
    
    # Initialize agent with debug output and 3 images per song
    agent = OzartAgent(debug=False, images_per_song=3)  # Default is now 3 images per song
    
    # Override the _process_single_song method to show detailed progress
    original_process_song = agent._process_single_song
    
    def enhanced_process_song(song):
        """Enhanced version of _process_single_song with better output."""
        print(f"\n{'='*60}")
        print(f"🎵 PROCESSING SONG: {song}")
        print(f"{'='*60}")
        
        # Step 1: Analyze the song
        print("\n📊 STEP 1: ANALYZING SONG...")
        song_analysis = agent.music_analyzer.process_input(song)
        
        if "error" in song_analysis:
            print(f"\n❌ ANALYSIS FAILED: {song_analysis['error']}")
            return
        
        song_description = song_analysis.get("interpretation", "A song with no description")
        song_metadata = song_analysis.get("metadata", {})
        song_title = song_metadata.get("title", song)
        
        print(f"\n✅ SONG ANALYSIS COMPLETE")
        print(f"\n🎧 SONG TITLE: {song_title}")
        print(f"\n📝 SONG DESCRIPTION:\n{'-'*40}\n{song_description}\n{'-'*40}")
        
        # Step 2: Generate themes
        print(f"\n🧠 STEP 2: GENERATING {agent.images_per_song} THEMES...")
        themes = agent._generate_themes(song_title, song_description)
        
        if not themes:
            print(f"\n❌ THEME GENERATION FAILED")
            return
        
        print(f"\n✅ GENERATED {len(themes)} THEMES:")
        for i, theme in enumerate(themes):
            print(f"\n  {i+1}. {theme}")
        
        # Step 3: Process each theme
        for i, theme in enumerate(themes):
            print(f"\n{'='*60}")
            print(f"🎨 THEME {i+1}/{len(themes)}: {theme}")
            print(f"{'='*60}")
            
            # Generate image prompt
            print("\n✍️ GENERATING IMAGE PROMPT...")
            image_prompt = agent._generate_image_prompt(song_title, song_description, theme)
            
            if not image_prompt:
                print(f"\n❌ PROMPT GENERATION FAILED")
                continue
            
            print(f"\n📜 IMAGE PROMPT:\n{'-'*40}\n{image_prompt}\n{'-'*40}")
            
            # Generate image
            print("\n🖼️ GENERATING IMAGE...")
            image_path = agent._generate_image(image_prompt)
            
            if not image_path or not os.path.exists(image_path):
                print(f"\n❌ IMAGE GENERATION FAILED")
                continue
            
            print(f"\n✅ IMAGE GENERATED: {image_path}")
            
            # Open with QuickLook on macOS
            if sys.platform == "darwin":
                try:
                    print("\n👁️ OPENING IMAGE WITH QUICKLOOK...")
                    import subprocess
                    subprocess.run(["qlmanage", "-p", image_path], 
                                  stdout=subprocess.DEVNULL, 
                                  stderr=subprocess.DEVNULL)
                except Exception as e:
                    print(f"\n⚠️ QUICKLOOK FAILED: {e}")
            else:
                print("\n⚠️ QUICKLOOK ONLY AVAILABLE ON MACOS")
            
            # Generate image description
            print("\n📄 GENERATING IMAGE DESCRIPTION...")
            image_description = agent._generate_image_description(
                image_path, song_title, song_description, theme, image_prompt
            )
            
            if not image_description:
                print(f"\n❌ DESCRIPTION GENERATION FAILED")
                continue
            
            print(f"\n📝 IMAGE DESCRIPTION:\n{'-'*40}\n{image_description}\n{'-'*40}")
            
            # Save metadata
            print("\n💾 SAVING METADATA...")
            agent._save_artwork_metadata(
                image_path=image_path,
                song_title=song_title,
                theme=theme,
                image_prompt=image_prompt,
                description=image_description
            )
            
            print("\n✅ ARTWORK COMPLETE!")
    
    # Replace the original method with our enhanced version
    agent._process_single_song = enhanced_process_song
    
    print("Enter song names or URLs separated by commas (or 'exit' to quit):")
    while True:
        user_input = input("> ")
        
        if user_input.lower() in ["exit", "quit", "q"]:
            print("Exiting Ozart. Goodbye!")
            break
        
        if user_input.strip():
            print(f"Processing: {user_input}")
            agent.process_songs(user_input)
            print("\nProcessing started. Enter more songs or 'exit' to quit.")
        else:
            print("Please enter song names or URLs separated by commas.")
