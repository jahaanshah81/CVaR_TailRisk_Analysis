"""
Shared pytest configuration: makes `src/` importable as top-level modules
(matching the convention every module in this project already uses, e.g.
`from black_scholes import bs_price`), without needing an installed
package or a `src` prefix on every import in the test suite.
"""

import os
import sys

SRC_DIR = os.path.join(os.path.dirname(__file__), "..", "src")
sys.path.insert(0, os.path.abspath(SRC_DIR))
