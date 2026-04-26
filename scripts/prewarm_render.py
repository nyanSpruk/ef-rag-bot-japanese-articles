from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app import CHUNKING_STRATEGIES, DEFAULT_CHUNKING_STRATEGY, get_cached_vector_store


def main() -> None:
    strategy = CHUNKING_STRATEGIES[DEFAULT_CHUNKING_STRATEGY]
    get_cached_vector_store(
        chunk_size=int(strategy["chunk_size"]),
        chunk_overlap=int(strategy["chunk_overlap"]),
    )
    print(
        "Prewarmed Render search assets for strategy "
        f"{DEFAULT_CHUNKING_STRATEGY} ({strategy['chunk_size']}/{strategy['chunk_overlap']})."
    )


if __name__ == "__main__":
    main()
