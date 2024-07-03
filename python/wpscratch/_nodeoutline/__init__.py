
from __future__ import annotations
import typing as T
import os, shutil, types
from pathlib import Path


from wpscratch._nodeoutline.node import WN, retriever


# Transform = retriever.getNodeCls("Transform")
# print(Transform)
# print(Transform.dpath)
#
#
# Transform = retriever.getNodeCls("Transform")
# print("second transform get")
# # d = WN.Transform.dpath


if __name__ == '__main__':
	print(WN.Transform.dpath)
	print(WN.Dag.dpath)

	print(issubclass(WN.Transform, WN))
	pass
