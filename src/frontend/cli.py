import subprocess
import sys
from pathlib import Path


def main() -> None:
    """Launch the Streamlit app."""
    app_path = Path(__file__).parent / "app.py"
    subprocess.run(
        [sys.executable, "-m", "streamlit", "run", str(app_path)],
        check=False,
    )
