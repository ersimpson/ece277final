cd ~/ECE277/final

if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
    echo "Created virtual environment."
fi
source venv/bin/activate
pip install pip-tools

echo "Installing requirements..."
pip-sync requirements.txt
echo "Installed requirements."