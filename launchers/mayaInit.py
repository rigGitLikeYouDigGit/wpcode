
# launcher script for maya
# find WEPRESENT_ROOT on system, add WePresent packages to env

"""we SHOULD have a full package management system, where for each domain
(maya, houdini, pure python, etc) we can specify a different interpreter
and package set
very primitive for now"""


import sys, os, pprint, json
from pathlib import Path
import forbiddenfruit, deepdiff
WPROOT_KEY = "WEPRESENT_ROOT"
wpRoot = Path(os.getenv(WPROOT_KEY))
pyRoot = wpRoot / "code" / "python"

pluginRoot = wpRoot / "code" / "maya" / "plugin"

# nothing yet for real package management, different interpreters, software etc
# everything runs on the same wp
#sys.path.insert(0, str(pyRoot))
#os.environ["PYTHONPATH"] = str(pyRoot) + os.pathsep + os.getenv("PYTHONPATH", "")

# load wp config dict
# maybe move this to top level
configPath = wpRoot / "code" / "config.json"
with open(configPath, "r") as f:
	config = json.load(f)

# add any path variables to env
envOverrides = config["maya"]["env"]
for key, paths in envOverrides.items():
	print("RAW ENV", key, os.environ.get(key, ""))
	if isinstance(paths, (tuple, list)):
		for path in reversed(paths):
			expandPath = os.path.expandvars(path)
			expandPath = expandPath.replace("\\", "/").replace("//", "/").replace("\\/", "/")
			os.environ[key] = expandPath + ";" + os.environ.get(key, "")
	else: # set int values or single strings
		os.environ[key] = paths

	print("ENV OVERRIDE", key, os.environ[key])

# get path to maya exe
exe = config["maya"]["exe"]

def enclose(s, char):
	return char + s + char

# run maya
os.system(enclose(exe, '"'))




