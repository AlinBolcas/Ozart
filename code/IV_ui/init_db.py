import os
import sys
from pathlib import Path

# Add the project root directory to Python path
ROOT_DIR = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT_DIR))

# Now we can import our modules
from code.IV_ui.ui_flask import app, db
from code.IV_ui.models import User

def init_database():
    """Initialize the database with proper permissions"""
    with app.app_context():
        # Make sure the instance directory exists and is writable
        db_dir = os.path.join(ROOT_DIR, "instance")
        os.makedirs(db_dir, exist_ok=True)
        
        # Set proper permissions for the directory
        os.chmod(db_dir, 0o777)
        
        # Set the database path explicitly
        db_path = os.path.join(db_dir, "ozart.db")
        
        # Override the database URI in app config
        app.config['SQLALCHEMY_DATABASE_URI'] = f'sqlite:///{db_path}'
        
        # Create all database tables
        db.create_all()
        
        # Set proper permissions for the database file
        if os.path.exists(db_path):
            os.chmod(db_path, 0o666)
            
        print(f"✅ Database created at: {db_path}")

def check_environment():
    """Check and validate environment setup"""
    required_vars = [
        'GOOGLE_CLIENT_ID',
        'GOOGLE_CLIENT_SECRET',
        'SECRET_KEY'
    ]
    
    missing = [var for var in required_vars if not os.getenv(var)]
    
    if missing:
        print("❌ Missing environment variables:")
        for var in missing:
            print(f"  - {var}")
        return False
    
    print("✅ Environment variables validated!")
    return True

def create_directories():
    """Create necessary directories with proper permissions"""
    dirs = [
        os.path.join(ROOT_DIR, "output/images"),
        os.path.join(ROOT_DIR, "output/descriptions"),
        os.path.join(ROOT_DIR, "output/music_analysis"),
        os.path.join(ROOT_DIR, "instance"),  # for SQLite database
    ]
    
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        # Set directory permissions
        os.chmod(dir_path, 0o777)
        print(f"✅ Created directory with write permissions: {dir_path}")

if __name__ == "__main__":
    print("\n🚀 Initializing Ozart...\n")
    
    # Check environment
    env_ok = check_environment()
    
    # Create directories
    create_directories()
    
    try:
        # Initialize database
        init_database()
        print("\n✅ Database initialized successfully!")
    except Exception as e:
        print(f"\n❌ Database initialization failed: {str(e)}")
        env_ok = False
    
    if env_ok:
        print("\n✨ Ozart initialized successfully!")
    else:
        print("\n⚠️  Ozart initialized with warnings. Please check the logs above.") 