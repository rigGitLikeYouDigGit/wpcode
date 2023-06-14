
from __future__ import annotations
import sys, os
"""TEST setting up specific environment variables and path stuff"""

usdPyPath = r"F:\all_projects_desktop\common\edCode\external\usd.py37.windows-x86_64.release@0.22.11\lib\python"

# if usdPyPath not in sys.path:
# 	sys.path.append(usdPyPath)

from pathlib import Path


externalRootPath = Path(r"F:\wp\code\python\external")
dirName = f"py{sys.version_info.major}-{sys.version_info.minor}"
sys.path.append(str(externalRootPath / dirName))

# ensure we pull from the right pc install
installDirName = Path(r"C:\Python{}{}".format(sys.version_info.major, sys.version_info.minor))

sys.path.append(str(installDirName / "Lib" / "site-packages"))










