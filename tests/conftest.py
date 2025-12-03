"""テスト実行時にsrcをパスへ追加してパッケージを解決する。"""

import sys
from pathlib import Path


def pytest_configure() -> None:
    """srcディレクトリをimportパスに追加する。"""
    root = Path(__file__).resolve().parents[1]
    src_path = root / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))
