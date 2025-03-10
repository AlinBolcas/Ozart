from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin
from werkzeug.security import generate_password_hash, check_password_hash
from cryptography.fernet import Fernet
import json
import os
import base64
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

db = SQLAlchemy()

class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(120), unique=True)
    password_hash = db.Column(db.String(128))
    google_id = db.Column(db.String(128), unique=True, nullable=True)
    name = db.Column(db.String(128))
    
    # Add user preferences
    api_keys = db.Column(db.JSON, default={})  # Store API keys
    custom_prompts = db.Column(db.JSON, default={})  # Store custom prompts
    
    def __init__(self, **kwargs):
        super(User, self).__init__(**kwargs)
        # Generate a unique salt for this user
        self.salt = base64.b64encode(os.urandom(32)).decode('utf-8')
    
    def _get_encryption_key(self):
        """Generate encryption key from app secret and user salt"""
        # Use PBKDF2 to derive a key from the app secret and user salt
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=base64.b64decode(self.salt),
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(
            kdf.derive(os.environ.get('SECRET_KEY', 'fallback-secret-key').encode())
        )
        return Fernet(key)
    
    def set_password(self, password):
        if password:  # Only set if password provided
            self.password_hash = generate_password_hash(password)
    
    def check_password(self, password):
        if self.password_hash:
            return check_password_hash(self.password_hash, password)
        return False
    
    def set_prompts(self, prompts_dict):
        """Store custom prompts as JSON string"""
        self.custom_prompts = json.dumps(prompts_dict)
        
    def get_prompts(self):
        """Retrieve custom prompts"""
        return json.loads(self.custom_prompts) if self.custom_prompts else None
    
    def set_api_keys(self, keys_dict):
        """Encrypt and store API keys"""
        if not keys_dict:
            self.api_keys = None
            return
            
        # Convert dict to JSON string
        json_data = json.dumps(keys_dict)
        
        # Encrypt the JSON string
        f = self._get_encryption_key()
        encrypted_data = f.encrypt(json_data.encode())
        
        # Store as base64 string
        self.api_keys = base64.b64encode(encrypted_data).decode('utf-8')
        
    def get_api_keys(self):
        """Decrypt and return API keys"""
        if not self.api_keys:
            return None
            
        try:
            # Decode from base64
            encrypted_data = base64.b64decode(self.api_keys)
            
            # Decrypt the data
            f = self._get_encryption_key()
            decrypted_data = f.decrypt(encrypted_data)
            
            # Parse JSON
            return json.loads(decrypted_data)
        except Exception as e:
            print(f"Error decrypting API keys: {e}")
            return None 