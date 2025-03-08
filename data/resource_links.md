# Build once
docker build -t ozart .

# Run with your code directory mounted
docker run --name ozart-container -p 8501:8501 \
  -v $(pwd):/app \
  ozart


---

# OpenSMILE
https://audeering.github.io/opensmile-python/usage.html
https://en.wikipedia.org/wiki/OpenSMILE
https://github.com/audeering/opensmile-python/

# OpenFace
https://github.com/TadasBaltrusaitis/OpenFace

# MediaPipe
https://github.com/google/mediapipe

# SAM2
https://github.com/facebookresearch/sam2

# Faiss
https://github.com/facebookresearch/faiss