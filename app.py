import os
import sys

# Add the code directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import and run streamlit
if __name__ == "__main__":
    import subprocess
    import streamlit.web.cli as stcli

    # Run streamlit with our UI file
    sys.argv = [
        "streamlit",
        "run",
        "code/IV_ui/ui_ozart.py",
        "--server.port=8501",
        "--server.address=0.0.0.0"  # Required for Docker
    ]
    stcli.main()
