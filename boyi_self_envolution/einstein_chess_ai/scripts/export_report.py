"""Export report.md to PDF when pandoc is available."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    md = root / "docs" / "report.md"
    pdf = root / "docs" / "report.pdf"
    if not md.exists():
        raise FileNotFoundError(md)
    if shutil.which("pandoc") is None:
        print(f"pandoc not found. Markdown report is ready at {md}")
        return
    subprocess.run(["pandoc", str(md), "-o", str(pdf)], cwd=root, check=True)
    print(f"Exported {pdf}")


if __name__ == "__main__":
    main()

