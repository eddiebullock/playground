"""
Runner script for the dashboard.
"""

import os
import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Run the Streamlit app
if __name__ == "__main__":
    os.system(f"streamlit run {project_root}/src/visualization/dashboard.py") 