import os

# ---------------------------------------------------------------------------
# Model IDs
# ---------------------------------------------------------------------------
# Free tier, fast — used for planning and reflection
MODEL_PLANNER = "gemma-4-26b-a4b-it"

# Best reasoning — used for literature, hypothesis, analysis
MODEL_RESEARCH = "gemini-3.1-pro-preview"

# Cheaper/faster — used for final reply formatting
MODEL_REPLY = "gemini-3.1-flash-lite-preview"

# ---------------------------------------------------------------------------
# Google Drive paths (Colab mounts Drive at /content/drive)
# ---------------------------------------------------------------------------
DRIVE_ROOT = os.environ.get("BIORESEARCH_DRIVE_ROOT", "/content/drive/MyDrive/BioResearch")
SESSIONS_DIR = os.path.join(DRIVE_ROOT, "sessions")
UPLOADS_DIR = os.path.join(DRIVE_ROOT, "uploads")

# ---------------------------------------------------------------------------
# Research settings
# ---------------------------------------------------------------------------
MAX_ITERATIONS = int(os.environ.get("MAX_ITERATIONS", "5"))

# ---------------------------------------------------------------------------
# API key (set via Colab Secrets or environment variable)
# ---------------------------------------------------------------------------
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "")
