"""
main.py
-------
Application entry point.

Run with:
    python main.py
"""

import sys
import os

# Ensure the src package is on the path regardless of how the script is invoked
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from gui import main

if __name__ == "__main__":
    main()
