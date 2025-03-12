# OZART: AI Music-to-Art Generator

![Ozart Logo](data/cover_02.png)

## Overview

Ozart is an autonomous AI artist that interprets songs and transforms them into unique visual artworks. By analyzing the emotional and musical characteristics of songs, Ozart creates evocative images that reflect the essence of the music.

**Current Version:** MVP (Music Visualization Pipeline)

## Features

- **Song Analysis:** Analyzes songs via name, Spotify URL, or YouTube link
- **AI Interpretation:** Generates detailed musical interpretation of songs' emotional qualities
- **Artistic Visualization:** Creates unique artwork based on the musical analysis
- **Streamlit Interface:** Simple and elegant user experience for submitting songs and viewing gallery

## How It Works

Ozart follows a sophisticated pipeline to transform music into visual art:

1. **Music Input & Analysis:** Enter a song name or URL to analyze its musical characteristics
2. **AI Interpretation:** The system generates a detailed interpretation of the song's emotional qualities
3. **Image Prompt Creation:** Based on the analysis, an optimized image generation prompt is crafted
4. **Artwork Generation:** The prompt is used to create a unique visual representation
5. **Gallery Display:** All created artworks are displayed in a beautiful gallery interface

![Ozart Pipeline](data/ozart_flowchart_01.png)

## Technical Architecture

Ozart is built with a modular architecture that integrates several AI systems:

- **Frontend:** Streamlit-based UI for user interaction and gallery display
- **Song Analysis:** Uses advanced audio processing and OpenAI's language models
- **Image Generation:** Leverages Replicate and Flux image generation technologies
- **Metadata Management:** Sophisticated tracking of artwork/song relationships

## Technology Stack

- **AI/ML:** OpenAI API for text analysis and prompt generation
- **Image Generation:** Replicate and Flux for high-quality image creation
- **Frontend:** Streamlit for intuitive user interface
- **Backend:** Python with threading for efficient parallel processing
- **Data Storage:** JSON-based metadata and file system for images and analyses

## Project Roadmap

Ozart is being developed in phases:

### Phase 1: MVP Implementation (Current)
- Music analysis from song titles/URLs
- Basic text-to-image pipeline
- Storage of generations for evaluation

### Phase 2: Enhanced Music Processing
- Incorporate lightweight models for raw audio analysis
- Improve image refinement with deeper feedback
- Expand memory functions for iterative learning

### Phase 3: Full Autonomy & Scaling
- AI-driven critique and scoring system
- Multi-agent collaboration for sophisticated iterations
- Expanded API support for deeper music-image correlations

## Getting Started

1. **Clone and Setup**
```bash
# Clone the repository
git clone https://github.com/AlinBolcas/Ozart.git
cd Ozart

# Create and activate virtual environment
python -m venv venv

# On Windows:
venv\Scripts\activate

# On Mac/Linux:
source venv/bin/activate

# Install requirements
pip install -r requirements.txt
```

2. **API Keys Setup**
Your friend will need to get their own API keys:
- OpenAI API key from: https://platform.openai.com/api-keys
- Replicate API key from: https://replicate.com/account/api-tokens
- Spotify API keys:
  1. Go to https://developer.spotify.com/dashboard
  2. Log in or create a Spotify account
  3. Create a new app to get Client ID and Client Secret

They can enter these keys in two ways:
- Through the web interface once the app is running
- Or create a `.env` file in the root directory:
```
OPENAI_API_KEY=sk-...
REPLICATE_API_TOKEN=r8_...
SPOTIFY_CLIENT_ID=2f...
SPOTIFY_CLIENT_SECRET=017..
```

3. **Run the Application**
```bash
# Start the Flask server
python app.py
```

The website will be available at:
- http://localhost:5001 (or next available port if 5001 is taken)

4. **Troubleshooting Common Issues**
If they encounter any errors:
- Make sure all dependencies are installed: `pip install -r requirements.txt`
- Check if Python version is 3.8 or higher: `python --version`
- Ensure API keys are properly set
- Make sure no other application is using port 5001
- Check if all required output directories exist (they should be created automatically)

5. **Optional: Development Mode**
If they want to make changes:
```bash
# Set Flask to development mode
export FLASK_ENV=development  # On Mac/Linux
set FLASK_ENV=development    # On Windows
```

## Examples

Ozart can create varied artistic interpretations based on different musical styles:

- Classical compositions become elegant, structured visual pieces
- Hip-hop tracks transform into vibrant, dynamic artwork
- Ambient music generates atmospheric, ethereal visuals
- Rock songs produce bold, energetic compositions

## Limitations

The current MVP version has some limitations:

- Relies primarily on song metadata rather than direct audio analysis
- Single image generation per song (future versions will explore multiple interpretations)
- Limited refinement capabilities (iterative improvement coming in future versions)

## Contributing

Contributions to Ozart are welcome! Whether you're interested in improving the music analysis, enhancing the image generation, or refining the user interface, please feel free to submit pull requests.

## License

MIT

## Acknowledgments

- OpenAI for powerful language models
- Replicate and Flux for image generation capabilities
- OpenSmile for audio analysis
- The broader AI art and music community for inspiration

---

*Ozart: Where music becomes visual art through the lens of artificial intelligence* 