from __future__ import annotations
from enum import Enum
from PySide2 import QtCore, QtGui

# custom qt data roles
# no idea how to prevent these clashing with other custom classes -
# hopefully it won't ever come up
addressRole = QtCore.Qt.UserRole + 1 # return address sequence to tree
relAddressRole = QtCore.Qt.UserRole + 2 # retrieve relative address in ui
childBoundsRole = QtCore.Qt.UserRole + 3 # return bounds of entire subtree in ui
treeObjRole = QtCore.Qt.UserRole + 4 # return full tree object
rowHeight = 16

# colour constants

createColour = (160, 255, 160)
modifyColour = (100, 160, 255)
deleteColour = (255, 100, 100)

errorColour = QtGui.QColor(250, 150, 150)


