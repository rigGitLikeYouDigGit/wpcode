
from __future__ import annotations

from pathlib import Path

import sys, os, json

"""run this script instead of blender's native launcher to load wp code"""

configPath = Path(__file__).parents[3] / "config.json"

configMap = json.load(open(configPath, "r"))

print("configMap", configMap)

blenderExePath = Path(configMap["blenderExePath"])
blenderLaunchFolder = blenderExePath.parent
blenderStartupScriptPath = (Path(__file__).parent / "startup.py").as_posix()
print("blenderStartupScriptPath", blenderStartupScriptPath)

os.environ["BLENDER_VAR"] = "adkjhkjakfagaskjd"
os.chdir(blenderLaunchFolder)
os.system(f"""{blenderExePath} -P {blenderStartupScriptPath} --log-level -1""")


