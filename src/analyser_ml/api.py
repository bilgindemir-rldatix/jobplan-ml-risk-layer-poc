"""
API alias for analyser-ml.

Allows running:
python -m uvicorn analyser_ml.api:app --app-dir src
"""

from jobplan_risk.api import app

__all__ = ["app"]
