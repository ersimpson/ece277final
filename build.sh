if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
    echo "Created virtual environment."
fi
source venv/bin/activate
pip install pip-tools

echo "Building requirements..."
pip-compile requirements.in
echo "Built requirements."

scp -r src/ requirements.txt install.sh ersimpson@dsmlp-login.ucsd.edu:~/ECE277/final/
ssh ersimpson@dsmlp-login.ucsd.edu "chmod +x ~/ECE277/final/install.sh"
