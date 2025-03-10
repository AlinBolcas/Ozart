#!/bin/bash

# AWS instance details
AWS_HOST="your-ec2-instance-ip"
PEM_FILE="path/to/your-key.pem"
APP_DIR="/home/ubuntu/ozart"

# Make sure pem file has correct permissions
chmod 400 $PEM_FILE

# Create required directories on remote server
ssh -i $PEM_FILE ubuntu@$AWS_HOST "mkdir -p $APP_DIR"

# Copy application files
scp -i $PEM_FILE -r \
    code \
    static \
    templates \
    app.py \
    requirements.txt \
    .env \
    ubuntu@$AWS_HOST:$APP_DIR/

# Install dependencies and set up application
ssh -i $PEM_FILE ubuntu@$AWS_HOST "bash -s" << 'ENDSSH'
    # Update system
    sudo apt-get update
    sudo apt-get install -y python3-pip python3-venv nginx

    # Create and activate virtual environment
    cd /home/ubuntu/ozart
    python3 -m venv .venv
    source .venv/bin/activate

    # Install dependencies
    pip install -r requirements.txt
    pip install gunicorn

    # Initialize database
    python code/IV_ui/init_db.py

    # Create systemd service
    sudo tee /etc/systemd/system/ozart.service << EOF
[Unit]
Description=Ozart Web Application
After=network.target

[Service]
User=ubuntu
WorkingDirectory=/home/ubuntu/ozart
Environment="PATH=/home/ubuntu/ozart/.venv/bin"
ExecStart=/home/ubuntu/ozart/.venv/bin/gunicorn -w 4 -b 127.0.0.1:5001 app:app
Restart=always

[Install]
WantedBy=multi-user.target
EOF

    # Configure Nginx
    sudo tee /etc/nginx/sites-available/ozart << EOF
server {
    listen 80;
    server_name your-domain.com;  # Replace with your domain

    location / {
        proxy_pass http://127.0.0.1:5001;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
    }
}
EOF

    # Enable site and restart services
    sudo ln -s /etc/nginx/sites-available/ozart /etc/nginx/sites-enabled/
    sudo rm /etc/nginx/sites-enabled/default
    sudo systemctl daemon-reload
    sudo systemctl start ozart
    sudo systemctl enable ozart
    sudo systemctl restart nginx
ENDSSH 