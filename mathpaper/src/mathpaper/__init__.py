"""Mathematics-paper retrieval and agent tools."""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path.cwd() / ".env", override=False)

# Full process-lifetime opt-out must precede importing ONNX Runtime.
# See microsoft/onnxruntime/docs/Privacy.md.
os.environ["ORT_DISABLE_TELEMETRY"] = "1"
os.environ.setdefault("FASTMCP_CHECK_FOR_UPDATES", "off")
os.environ.setdefault("FASTMCP_TELEMETRY_MODE", "off")

__version__ = "0.1.0"
