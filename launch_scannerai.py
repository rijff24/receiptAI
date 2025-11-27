"""Launcher script for running ScannerAI in local (desktop) mode."""

from __future__ import annotations

import os
import sys
import threading
import time
from pathlib import Path

# Unbuffer stdout to ensure immediate output
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(line_buffering=True)

from streamlit.web import cli as stcli


class LoadingIndicator:
    """Simple loading indicator with throbber and milestone tracking."""
    
    def __init__(self, total_steps: int = 4):
        self.total_steps = total_steps
        self.current_step = 0
        self.current_message = ""
        self.running = False
        self.thread = None
        self.spinner_chars = "|/-\\"
        self.spinner_index = 0
    
    def _spin(self):
        """Internal spinner loop."""
        while self.running:
            char = self.spinner_chars[self.spinner_index % len(self.spinner_chars)]
            message = f"{self.current_message} {char} (Step {self.current_step}/{self.total_steps})"
            sys.stdout.write(f"\r{message}")
            sys.stdout.flush()
            time.sleep(0.1)
            self.spinner_index += 1
    
    def start(self):
        """Start the spinner."""
        self.running = True
        self.thread = threading.Thread(target=self._spin, daemon=True)
        self.thread.start()
    
    def update(self, step: int, message: str):
        """Update the current step and message."""
        self.current_step = step
        self.current_message = message
    
    def stop(self):
        """Stop the spinner and clear the line."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=0.5)
        sys.stdout.write("\r" + " " * 80 + "\r")
        sys.stdout.flush()


def main() -> int:
    # Show immediate feedback - this appears as soon as Python code starts running
    # (Note: There may be a 10+ second delay before this due to PyInstaller extraction)
    # Force immediate output
    sys.stdout.buffer.write(b"Initializing ScannerAI...\r\n")
    sys.stdout.buffer.flush()
    sys.stdout.write("Initializing ScannerAI...\n")
    sys.stdout.flush()
    
    # Initialize loading indicator with immediate generic message
    loader = LoadingIndicator(total_steps=4)
    loader.update(0, "Initializing")
    loader.start()
    
    try:
        # Small delay to ensure message is visible
        time.sleep(0.1)
        
        # Step 1: Setting up paths
        loader.update(1, "Setting up paths")
        time.sleep(0.2)  # Brief pause to show the message
        
        base_path = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))
        app_script = base_path / "scripts" / "lcf_receipt_entry_streamlit.py"

        if not app_script.exists():
            loader.stop()
            print(f"Unable to locate Streamlit app at: {app_script}", file=sys.stderr)
            return 1

        os.chdir(base_path)

        # Step 2: Configuring environment
        loader.update(2, "Configuring environment")
        time.sleep(0.2)
        
        os.environ.setdefault("SCANNERAI_HOSTED_MODE", "0")
        os.environ.setdefault("SCANNERAI_LOCAL_LAUNCHER", "1")
        os.environ.setdefault("STREAMLIT_BROWSER_GATHER_USAGE_STATS", "0")
        os.environ["STREAMLIT_GLOBAL_DEVELOPMENT_MODE"] = "false"

        # Step 3: Preparing Streamlit
        loader.update(3, "Preparing Streamlit")
        time.sleep(0.2)
        
        sys.argv = [
            "streamlit",
            "run",
            str(app_script),
            "--server.port",
            "8501",
            "--browser.serverAddress",
            "localhost",
        ]
        
        # Step 4: Starting server
        loader.update(4, "Starting server")
        time.sleep(0.3)
        
        # Stop loader before Streamlit takes over
        loader.stop()
        print("\nScannerAI is ready!")
        print("The application will open in your browser shortly.\n")
        
        stcli.main()
        return 0
    except KeyboardInterrupt:
        loader.stop()
        print("\nApplication interrupted by user.")
        return 1
    except Exception as e:
        loader.stop()
        print(f"\nError starting application: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

