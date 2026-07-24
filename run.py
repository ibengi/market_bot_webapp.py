#!/usr/bin/env python3
"""Point d'entree Atlas Decision Engine.
Exemples :
  python run.py --demo --loop --shadow
  python run.py --scan-only
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "src", "utils"))
from paths import add_src_to_path  # noqa: E402
add_src_to_path()

from kalshi_alpha_bot import main  # noqa: E402

if __name__ == "__main__":
    main()
