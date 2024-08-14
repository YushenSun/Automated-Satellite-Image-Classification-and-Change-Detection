import os

def ensure_directory_exists(directory):
    """Ensure the directory exists; if not, create it."""
    if not os.path.exists(directory):
        os.makedirs(directory)
