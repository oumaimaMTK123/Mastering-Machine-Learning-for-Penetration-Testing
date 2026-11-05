# Access and local setup

This project was created in the chat cloud workspace at /workspace/video-ssl.
The project is also included in the selected GitHub repository under video-ssl/.
Cloud changes only appear on GitHub after an explicit commit and push.

Cloud access: return to the same chat/environment and ask the assistant to read,
edit or run files in /workspace/video-ssl. A terminal/file browser is not exposed
in every client; no specific UI button is assumed. Publishing a cloud environment
preserves its prepared state for reuse; it does not publish a GitHub repository.
If this workspace is unavailable later, restore this ZIP into another environment.

Local setup (Python 3.12, Linux/macOS terminal or Windows PowerShell):

    cd video-ssl
    python -m venv .venv

Activate on Linux/macOS:

    source .venv/bin/activate

Activate on Windows PowerShell:

    .venv\Scripts\Activate.ps1

Then:

    python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cpu
    python -m pip install -r requirements.lock
    python selftest.py

The dependency versions were validated on Linux/Python 3.12; other platforms
have not been tested. install.sh is specific to the original cloud workspace.
The main README documents scientific protocol and remaining blockers. Relative
paths in config.json work when commands run from the project directory.
No videos, pretrained weights or Python environment are included. The selftest
regenerates synthetic videos and a tiny random model. No scientific benchmark
results have been produced. reports/ contains only the engineering validation
records copied from this run; their original absolute paths refer to the cloud.
