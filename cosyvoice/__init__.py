import os
import sys

# Automatically add third_party/Matcha-TTS to sys.path so matcha can be imported
_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_MATCHA_DIR = os.path.join(_ROOT_DIR, "third_party", "Matcha-TTS")
if os.path.exists(_MATCHA_DIR) and _MATCHA_DIR not in sys.path:
    sys.path.insert(0, _MATCHA_DIR)
