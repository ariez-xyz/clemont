"""Clemont - Monitoring AI models for fairness and robustness."""

__version__ = "1.0.0"
__author__ = "David Pape"

# Always available backends
from .backends.faiss import BruteForce
from .backends.kdtree import KdTree

# Optional backends
try:
    from .backends.snn import Snn
    _HAS_SNN = True
except ImportError:
    _HAS_SNN = False
    Snn = None

# BDD backend only if dependencies available
try:
    from .backends.bdd import BDD
    _HAS_BDD = True
except ImportError:
    _HAS_BDD = False
    BDD = None

__all__ = ["BruteForce", "KdTree"]
if _HAS_SNN:
    __all__.append("Snn")
if _HAS_BDD:
    __all__.append("BDD")

def list_available_backends():
    """List all available backends."""
    backends = ["BruteForce", "KdTree"]
    if _HAS_SNN:
        backends.append("Snn")
    else:
        backends.append("Snn (UNAVAILABLE - requires manual installaion, see readme)")
    if _HAS_BDD:
        backends.append("BDD")
    else:
        backends.append("BDD (UNAVAILABLE - requires manual installation, see readme)")
    return backends
