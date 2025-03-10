import os
import secrets
from pathlib import Path

def generate_secret_key():
    """Generate a secure secret key"""
    return secrets.token_hex(32)

def create_env_file():
    """Create production .env file"""
    env_template = """
# Flask settings
FLASK_DEBUG=0  # Production mode
SECRET_KEY={secret_key}

# OpenAI API
OPENAI_API_KEY={openai_key}

# Replicate API
REPLICATE_API_TOKEN={replicate_key}

# Google OAuth
GOOGLE_CLIENT_ID={google_id}
GOOGLE_CLIENT_SECRET={google_secret}

# Spotify API
SPOTIFY_CLIENT_ID={spotify_id}
SPOTIFY_CLIENT_SECRET={spotify_secret}

# Email settings
EMAIL_USER={email_user}
EMAIL_PASS={email_pass}

# Other APIs
OPENWEATHERMAP_API_KEY={weather_key}
TRIPO_API_KEY={tripo_key}
""".strip()

    # Generate new secret key
    secret_key = generate_secret_key()
    
    # Read existing keys from development .env
    current_env = {}
    if Path('.env').exists():
        with open('.env', 'r') as f:
            for line in f:
                if '=' in line:
                    key, value = line.strip().split('=', 1)
                    current_env[key.strip()] = value.strip()
    
    # Create production .env
    with open('.env.production', 'w') as f:
        f.write(env_template.format(
            secret_key=secret_key,
            openai_key=current_env.get('OPENAI_API_KEY', 'your-openai-key'),
            replicate_key=current_env.get('REPLICATE_API_TOKEN', 'your-replicate-key'),
            google_id=current_env.get('GOOGLE_CLIENT_ID', 'your-google-client-id'),
            google_secret=current_env.get('GOOGLE_CLIENT_SECRET', 'your-google-client-secret'),
            spotify_id=current_env.get('SPOTIFY_CLIENT_ID', 'your-spotify-client-id'),
            spotify_secret=current_env.get('SPOTIFY_CLIENT_SECRET', 'your-spotify-client-secret'),
            email_user=current_env.get('EMAIL_USER', 'your-email'),
            email_pass=current_env.get('EMAIL_PASS', 'your-email-password'),
            weather_key=current_env.get('OPENWEATHERMAP_API_KEY', 'your-weather-key'),
            tripo_key=current_env.get('TRIPO_API_KEY', 'your-tripo-key')
        ))
    print("✅ Created production .env file")

def check_required_keys():
    """Check if all required API keys are present"""
    required_keys = [
        'OPENAI_API_KEY',
        'REPLICATE_API_TOKEN',
        'GOOGLE_CLIENT_ID',
        'GOOGLE_CLIENT_SECRET',
        'SPOTIFY_CLIENT_ID',
        'SPOTIFY_CLIENT_SECRET'
    ]
    
    missing = []
    with open('.env.production', 'r') as f:
        content = f.read()
        for key in required_keys:
            if f"{key}=your-" in content:
                missing.append(key)
    
    if missing:
        print("\n⚠️  Missing required API keys:")
        for key in missing:
            print(f"  - {key}")
        return False
    return True

if __name__ == "__main__":
    print("\n🚀 Preparing Ozart for production...\n")
    
    # Create production environment file
    create_env_file()
    
    # Check for required keys
    if check_required_keys():
        print("\n✅ All required API keys are present")
        print("\n✨ Production environment ready!")
    else:
        print("\n⚠️  Please update missing API keys in .env.production before deploying")
    
    print("\n📝 Next steps:")
    print("1. Review .env.production")
    print("2. Update any missing API keys")
    print("3. Rename .env.production to .env in your production environment")
    print("4. Run python code/IV_ui/init_db.py to initialize the database")
    print("5. Start the server with python app.py") 