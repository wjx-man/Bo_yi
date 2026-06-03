"""测试导入路径设置，确保测试可以从项目根目录导入 src 包。"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    # 将 einstein_chess_ai 加入模块搜索路径，测试中即可使用 from src... 导入。
    sys.path.insert(0, str(ROOT))

