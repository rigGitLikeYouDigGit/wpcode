
from __future__ import annotations
import typing as T

from wplib import inheritance, CodeRef
from wplib.serial.constant import FORMAT_DATA_KEY, SERIAL_TYPE_KEY

def getFormatData(data:dict):
	return data.get(FORMAT_DATA_KEY, None)

def getDataCodeRefStr(data:dict)->str:
	"""Get the code ref from the given data dict.
	"""
	return getFormatData(data).get(SERIAL_TYPE_KEY, None)

