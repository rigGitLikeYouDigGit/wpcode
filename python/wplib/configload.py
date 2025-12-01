from __future__ import annotations
import types, typing as T
import pprint, json
from wplib import log
from wplib.override import override
from pathlib import Path


"""for real though how DO you handle configs
and settings exposed to the user?

- don't want to edit files directly versioned in the repo
- need to expose it in a consistent place for user to change locally


- repo code defines structure, defaults
- local folders in documents or appdata define user overrides
"""


def loadConfig(defaultRepoPath, userConfigPaths=()):

	data = json.load(Path(defaultRepoPath).open("r"))

	for i in userConfigPaths:
		userConfig = json.load(Path(defaultRepoPath).open("r"))
		data = override(data, userConfig)


