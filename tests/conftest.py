from __future__ import annotations

import sys
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# Unit tests should validate local behavior without trying to send remote
# traces from restricted CI/sandbox networks. Runtime CLI still honors .env.
os.environ["LANGSMITH_TRACING"] = "false"
os.environ["PART_FINDER_TRACE_LANGFUSE"] = "0"
os.environ["PART_FINDER_USE_LLM"] = "0"
