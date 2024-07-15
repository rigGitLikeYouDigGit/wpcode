
from __future__ import annotations
import typing as T


""" extract any static data we need from the api, write it out to
a config file

	TODO: later on, unite this with the node gathering
"""

import json, sys, os
from pathlib import Path
from typing import TypedDict
from collections import defaultdict
from dataclasses import dataclass

import orjson

from wptree import Tree

from wpm import WN, om, cmds, oma

TARGET_NODE_DATA_PATH = Path(__file__).parent / "constant.json"



