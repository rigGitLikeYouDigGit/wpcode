

from PySide2 import QtCore, QtWidgets, QtGui
from wpui.widget.canvas import *

app = QtWidgets.QApplication()

scene = WpCanvasScene()
w = WpCanvasView(parent=None, scene=scene,
                 )
# item = WpCanvasItem(parent=None,
#                     )
item = QtWidgets.QGraphicsRectItem(QtCore.QRectF(0, 0, 100, 100))
item.setBrush(QtGui.QBrush(QtCore.Qt.red))
item.setFlag(QtWidgets.QGraphicsItem.ItemIsSelectable)
scene.addItem(item)
scene.centreItem(item)
w.show()
app.exec_()
