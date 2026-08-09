"""Shared research library spine.

Library code is print-free, argparse-free, and importable. All CLI surfaces live
under ``scripts/``. Package contents are copied verbatim into each project's
``src/<package>/`` tree by the scaffold.
"""

from __future__ import annotations

__all__ = ["__version__"]
__version__ = "0.1.0"
