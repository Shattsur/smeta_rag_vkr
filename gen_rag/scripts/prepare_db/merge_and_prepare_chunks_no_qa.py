from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from shared.prepare_db.merge_and_prepare_chunks_no_qa import main


if __name__ == "__main__":
    raise SystemExit(main())