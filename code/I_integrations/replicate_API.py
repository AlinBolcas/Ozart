import os
import sys
import time
import replicate
import requests
import tempfile
import subprocess
import concurrent.futures
from typing import Dict, List, Optional, Union, Any, Tuple
from dotenv import load_dotenv
from pathlib import Path

class ReplicateAPI:
    """
    Integrated Replicate API wrapper with image, video, and music generation capabilities.
    Focuses on returning URLs for generated content rather than managing files directly.
    """

    def __init__(self, api_token: Optional[str] = None):
        """Initialize Replicate API with optional API token override"""
        # Load from .env file if exists
        load_dotenv()
        
        # Use provided API token if available, otherwise use environment variable
        self.api_token = api_token or os.getenv("REPLICATE_API_TOKEN")
        
        if not self.api_token:
            print("Warning: No Replicate API token provided or found in environment")
        
        # Will be used in each method
        self.client = None
        if self.api_token:
            os.environ["REPLICATE_API_TOKEN"] = self.api_token

    def run_model(
        self,
        model_path: str,
        input_data: Dict,
        version: Optional[str] = None
    ) -> Any:
        """
        Run any Replicate model with given inputs and return results.
        
        Args:
            model_path: The model identifier (e.g., 'owner/model-name')
            input_data: Dictionary of input parameters for the model
            version: Optional specific model version
            
        Returns:
            The model's output (often URLs to generated content)
        """
        try:
            # Pre-process any image inputs
            for key, value in input_data.items():
                if isinstance(value, str) and (
                    key in ['image', 'image_path', 'init_image'] or 'image' in key
                ) and not value.startswith(('http://', 'https://')):
                    input_data[key] = self.prepare_image_input(value)
            
            # Set the complete model path with version if provided
            if version:
                model = f"{model_path}:{version}"
            else:
                model = model_path
                
            # Run the model
            output = replicate.run(model, input=input_data)
            
            # Handle different output formats consistently
            if isinstance(output, list) and output:
                # Most media generation models return a list with the first item being the URL
                return output[0]
            
            return output
            
        except Exception as e:
            print(f"Error running model: {e}")
            return None

    def prepare_image_input(self, image_path: str) -> Optional[Union[str, bytes]]:
        """
        Prepare image input for Replicate API.
        
        Args:
            image_path: Path to local image file or URL
            
        Returns:
            URL string for remote files or file object for local files
        """
        try:
            # If already a URL, return as is
            if image_path.startswith(('http://', 'https://')):
                return image_path
                
            # If it's a local file, return file object
            if os.path.exists(image_path):
                return open(image_path, "rb")
                
            raise ValueError(f"Invalid image path: {image_path}")
            
        except Exception as e:
            print(f"Error preparing image input: {e}")
            return None

    def generate_image(
        self,
        prompt: str,
        negative_prompt: Optional[str] = None,
        aspect_ratio: str = "3:2",
        output_format: str = "jpg",
        raw: bool = False,
        safety_tolerance: int = 2,  # Range 0-6, default 2
        image_prompt_strength: float = 0.1,
    ) -> Optional[str]:
        """
        Generate image using Flux Pro Ultra model.
        
        Args:
            prompt: Text description of the desired image
            negative_prompt: What to avoid (optional)
            aspect_ratio: Image aspect ratio (default: "3:2")
            output_format: Output file format (default: "jpg")
            raw: Whether to use raw mode (default: False)
            safety_tolerance: Safety filter level (range 0-6, default: 2)
            image_prompt_strength: Strength of image prompt (default: 0.1)
            
        Returns:
            URL to the generated image or None if generation failed
        """
        try:
            input_data = {
                "prompt": prompt,
                "aspect_ratio": aspect_ratio,
                "output_format": output_format,
                "raw": raw,
                "safety_tolerance": safety_tolerance,
                "image_prompt_strength": image_prompt_strength
            }
            
            # Only add negative_prompt if provided
            if negative_prompt:
                input_data["negative_prompt"] = negative_prompt

            output = self.run_model(
                "black-forest-labs/flux-1.1-pro-ultra",
                input_data=input_data
            )
            
            print(f"Image generated successfully: {prompt[:30]}...")
            return output
            
        except Exception as e:
            print(f"Error generating image: {e}")
            return None

    def generate_video(
        self,
        image_url: str,
        prompt: str,
        max_area: str = "720x1280",
        fast_mode: str = "Balanced",
        num_frames: int = 81,  # Minimum required by model
        sample_shift: int = 5,
        sample_steps: int = 30,
        frames_per_second: int = 16,
        sample_guide_scale: float = 5.0
    ) -> Optional[str]:
        """
        Generate video from an image using WAN 2.1 I2V model.
        
        Args:
            image_url: URL of the source image
            prompt: Text description of the desired video motion
            max_area: Maximum resolution in format WIDTHxHEIGHT (default: "720x1280")
            fast_mode: Generation speed mode (default: "Balanced")
            num_frames: Total number of frames (minimum 81 required by model)
            sample_shift: Motion intensity (default: 5)
            sample_steps: Number of diffusion steps (default: 30)
            frames_per_second: FPS of output video (default: 16)
            sample_guide_scale: Guidance scale (default: 5.0)
            
        Returns:
            URL to the generated video or None if generation failed
        """
        try:
            # Convert FileOutput object to string URL if needed
            if hasattr(image_url, 'url'):
                image_url = image_url.url
            
            # Ensure we have a valid URL
            if not isinstance(image_url, str) or not image_url.startswith(('http://', 'https://')):
                raise ValueError(f"Invalid image URL: {image_url}. Must be a URL string.")
            
            print(f"Using image URL for video generation: {image_url[:50]}...")
            
            output = self.run_model(
                "wavespeedai/wan-2.1-i2v-720p",
                input_data={
                    "image": image_url,
                    "prompt": prompt,
                    "max_area": max_area,
                    "fast_mode": fast_mode,
                    "num_frames": num_frames,
                    "sample_shift": sample_shift,
                    "sample_steps": sample_steps,
                    "frames_per_second": frames_per_second,
                    "sample_guide_scale": sample_guide_scale
                }
            )
            
            print(f"Video generated successfully from image")
            return output
            
        except Exception as e:
            print(f"Error generating video: {type(e).__name__}: {e}")
            return None

    def generate_music(
        self,
        prompt: str,
        duration: int = 8,
        model_version: str = "stereo-large",
        top_k: int = 250,
        top_p: float = 0.0,
        temperature: float = 1.0,
        continuation: bool = False,
        output_format: str = "mp3",
        continuation_start: int = 0,
        multi_band_diffusion: bool = False,
        normalization_strategy: str = "peak",
        classifier_free_guidance: float = 3.0
    ) -> Optional[str]:
        """
        Generate music using Meta's MusicGen model.
        
        Args:
            prompt: Text description of desired music
            duration: Length in seconds (default: 8)
            model_version: Model version to use (default: "stereo-large")
            top_k: Top-k sampling parameter (default: 250)
            top_p: Top-p sampling parameter (default: 0.0)
            temperature: Generation temperature (default: 1.0)
            continuation: Whether to continue from previous (default: False)
            output_format: Output audio format (default: "mp3")
            continuation_start: Start time for continuation (default: 0)
            multi_band_diffusion: Use multi-band diffusion (default: False)
            normalization_strategy: Audio normalization (default: "peak")
            classifier_free_guidance: Guidance scale (default: 3.0)
            
        Returns:
            URL to the generated audio or None if generation failed
        """
        try:
            output = self.run_model(
                "meta/musicgen",
                input_data={
                    "prompt": prompt,
                    "duration": duration,
                    "model_version": model_version,
                    "top_k": top_k,
                    "top_p": top_p,
                    "temperature": temperature,
                    "continuation": continuation,
                    "output_format": output_format,
                    "continuation_start": continuation_start,
                    "multi_band_diffusion": multi_band_diffusion,
                    "normalization_strategy": normalization_strategy,
                    "classifier_free_guidance": classifier_free_guidance
                },
                version="671ac645ce5e552cc63a54a2bbff63fcf798043055d2dac5fc9e36a837eedcfb"
            )
            
            print(f"Music generated successfully: {prompt[:30]}...")
            return output
            
        except Exception as e:
            print(f"Error generating music: {e}")
            return None

# Helper functions for downloading and displaying media
def download_file(url: str, output_dir: Optional[str] = None, filename: Optional[str] = None) -> Optional[str]:
    """
    Download a file from a URL to a specific output directory.
    
    Args:
        url: URL of the file to download
        output_dir: Directory to save the file (defaults to temp directory)
        filename: Optional filename (generated from timestamp if not provided)
        
    Returns:
        Path to the downloaded file
    """
    try:
        # Create output directory if provided
        if output_dir:
            # Create base output directory
            base_dir = Path("output")
            if not base_dir.exists():
                base_dir.mkdir(parents=True, exist_ok=True)
                
            # Create specific media directory
            media_dir = base_dir / output_dir
            if not media_dir.exists():
                media_dir.mkdir(parents=True, exist_ok=True)
                
            # Generate filename if not provided
            if not filename:
                extension = url.split('.')[-1] if '.' in url.split('/')[-1] else 'tmp'
                timestamp = time.strftime("%Y%m%d-%H%M%S")
                filename = f"{output_dir}_{timestamp}.{extension}"
                
            output_path = media_dir / filename
        else:
            # Use temp directory if no output directory specified
            extension = url.split('.')[-1] if '.' in url.split('/')[-1] else 'tmp'
            fd, output_path = tempfile.mkstemp(suffix=f'.{extension}')
            os.close(fd)
            output_path = Path(output_path)
        
        # Download the file
        print(f"Downloading to {output_path}...")
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                
        print(f"Downloaded successfully ({os.path.getsize(output_path) / 1024:.1f} KB)")
        return str(output_path)
        
    except Exception as e:
        print(f"Error downloading file: {e}")
        return None

def display_media(file_path: str, media_type: str = "image"):
    """Display media using appropriate system tools."""
    try:
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            return False
            
        if sys.platform == "darwin":  # macOS
            if media_type == "image":
                # Try QuickLook first
                print("Opening image with QuickLook...")
                subprocess.run(["qlmanage", "-p", file_path], 
                              stdout=subprocess.DEVNULL, 
                              stderr=subprocess.DEVNULL)
            elif media_type == "video":
                # For videos, use QuickTime Player instead of QuickLook for better playback
                print("Opening video with QuickTime Player...")
                subprocess.run(["open", "-a", "QuickTime Player", file_path])
            elif media_type == "audio":
                # Use afplay for audio
                print("Playing audio...")
                subprocess.run(["afplay", file_path])
                
        elif sys.platform == "win32":  # Windows
            # Use the default application
            os.startfile(file_path)
            
        else:  # Linux
            try:
                subprocess.run(["xdg-open", file_path])
            except:
                print(f"Could not open file: {file_path}")
                return False
                
        return True
        
    except Exception as e:
        print(f"Error displaying media: {e}")
        return False

def merge_video_audio(video_path: str, audio_path: str, filename: Optional[str] = None) -> Optional[str]:
    """Merge video and audio into a single file using ffmpeg."""
    try:
        # Create output directory
        output_dir = Path("output/videos")
        if not output_dir.exists():
            output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate output filename if not provided
        if not filename:
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            filename = f"merged_{timestamp}.mp4"
            
        output_path = output_dir / filename
        
        print(f"Merging video and audio to {output_path}...")
        
        # Use ffmpeg to merge video and audio
        ffmpeg_cmd = [
            "ffmpeg", "-y",  # Overwrite output file if exists
            "-i", video_path,  # Video input
            "-i", audio_path,  # Audio input
            "-map", "0:v",  # Use video from first input
            "-map", "1:a",  # Use audio from second input
            "-c:v", "copy",  # Copy video codec
            "-shortest",  # Make output duration same as shortest input
            str(output_path)
        ]
        
        # Check if ffmpeg is available
        try:
            subprocess.run(["ffmpeg", "-version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        except (subprocess.SubprocessError, FileNotFoundError):
            print("Error: ffmpeg not found. Please install ffmpeg to merge video and audio.")
            return None
            
        # Run ffmpeg command
        process = subprocess.run(ffmpeg_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        if process.returncode != 0:
            print(f"Error merging files: {process.stderr.decode()}")
            return None
            
        print(f"Successfully merged video and audio to {output_path}")
        return str(output_path)
        
    except Exception as e:
        print(f"Error merging video and audio: {e}")
        return None

# Interactive demo with parallel processing
def run_interactive_demo():
    print("\n==== REPLICATE API INTERACTIVE DEMO ====\n")
    
    # Initialize API
    try:
        api = ReplicateAPI()
    except ValueError as e:
        print(f"Error: {e}")
        print("Make sure to set the REPLICATE_API_TOKEN environment variable.")
        sys.exit(1)
    
    # Creative prompt for all generations
    creative_prompt = "A generative audiovisual landscape where music waveforms morph into flowing geometric patterns, featuring reactive particle systems that respond to audio frequencies, with layered motion graphics transitioning between organic and digital aesthetics, all rendered in a real-time visual style"
    
    # Image generation and interaction loop
    image_url = None
    image_path = None
    retry_image = True
    
    while retry_image:
        # 1. Generate Image
        print("\n=== Generating Image ===")
        image_url = api.generate_image(
            prompt=creative_prompt,
            aspect_ratio="1:1",  # Square aspect ratio
            safety_tolerance=6    # Maximum allowed is 6
        )
        
        if not image_url:
            print("Image generation failed.")
            choice = input("\nDo you want to retry? (y/n): ").lower()
            if choice != 'y':
                print("Exiting demo.")
                return
            continue
            
        print(f"Image URL: {image_url}")
        
        # Download and display image
        image_path = download_file(image_url, output_dir="images", filename="demo_image.jpg")
        if image_path:
            display_media(image_path, "image")
            
            # Ask user for next action
            print("\nOptions:")
            print("1. Exit demo")
            print("2. Retry image generation")
            print("3. Proceed with video generation from this image")
            
            choice = input("Enter your choice (1-3): ")
            
            if choice == '1':
                print("Exiting demo.")
                return
            elif choice == '2':
                print("\nRetrying image generation...")
                continue
            else:
                # User chose to proceed with this image
                retry_image = False
        else:
            print("Failed to download and display image.")
            choice = input("\nDo you want to retry? (y/n): ").lower()
            if choice != 'y':
                print("Exiting demo.")
                return
    
    # At this point, we have a good image and we'll proceed with parallel video/music generation
    if image_url and image_path:
        print("\n=== Starting Parallel Generation ===")
        video_prompt = f"Camera slowly panning around {creative_prompt}, revealing more of the landscape"
        music_prompt = f"Epic orchestral soundtrack with magical elements for {creative_prompt}"
        
        # Create variables for results
        video_url = None
        music_url = None
        video_path = None
        music_path = None
        
        # Define worker functions for concurrent execution
        def generate_video_worker():
            print("\n🎬 Generating Video...")
            nonlocal video_url, video_path
            
            # Make sure we're using the URL string
            if hasattr(image_url, 'url'):
                image_url_str = image_url.url
            else:
                image_url_str = str(image_url)
                
            # Generate video
            video_url = api.generate_video(
                image_url=image_url_str,
                prompt=video_prompt,
                num_frames=81,  # Minimum required by the model
                sample_steps=20  # Faster for demo
            )
            
            if video_url:
                print(f"Video generation complete: {video_url}")
                # Download video
                video_path = download_file(video_url, output_dir="videos", filename="demo_video.mp4")
                return True
            else:
                print("Video generation failed.")
                return False
        
        def generate_music_worker():
            print("\n🎵 Generating Music...")
            nonlocal music_url, music_path
            
            # Generate music
            music_url = api.generate_music(
                prompt=music_prompt,
                duration=5,  # Short duration to match video
                model_version="stereo-large"
            )
            
            if music_url:
                print(f"Music generation complete: {music_url}")
                # Download music
                music_path = download_file(music_url, output_dir="music", filename="demo_music.mp3")
                return True
            else:
                print("Music generation failed.")
                return False
        
        # Use concurrent futures to run tasks in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            # Submit tasks
            video_future = executor.submit(generate_video_worker)
            music_future = executor.submit(generate_music_worker)
            
            # Display progress
            print("Waiting for generation tasks to complete...")
            pending = {video_future, music_future}
            done = set()
            start_time = time.time()
            last_update_time = 0  # Track when we last printed an update
            
            while pending:
                # Poll for completion less frequently (every 2 seconds)
                just_done, pending = concurrent.futures.wait(
                    pending, 
                    timeout=2.0,  # Increased from 0.5 to 2.0 seconds
                    return_when=concurrent.futures.FIRST_COMPLETED
                )
                done |= just_done
                
                # Only print updates every 2+ seconds
                current_time = time.time()
                if current_time - last_update_time >= 2.0 or just_done:
                    elapsed = current_time - start_time
                    print(f"⏳ Waiting for tasks... ({len(done)}/{len(done) + len(pending)} complete, {elapsed:.1f}s elapsed)")
                    last_update_time = current_time
        
        print("\n=== Generation Tasks Complete ===")
        
        # Check results
        video_success = video_future.result() and video_path
        music_success = music_future.result() and music_path
        
        if video_success and music_success:
            print("\n=== Merging Video and Audio ===")
            merged_path = merge_video_audio(video_path, music_path, filename="demo_merged.mp4")
            
            if merged_path:
                print("\n=== Playing Merged Video with Audio ===")
                display_media(merged_path, "video")
            else:
                # If merging failed, play them separately
                print("\n=== Playing Video and Audio Separately ===")
                display_media(video_path, "video")
                time.sleep(1)  # Give a second to start the video
                display_media(music_path, "audio")
        else:
            # Play what we have
            if video_success:
                print("\n=== Playing Video (without audio) ===")
                display_media(video_path, "video")
                
            if music_success:
                print("\n=== Playing Audio (without video) ===")
                display_media(music_path, "audio")
    
    print("\n==== DEMO COMPLETE ====")

# Entry point
if __name__ == "__main__":
    run_interactive_demo() 