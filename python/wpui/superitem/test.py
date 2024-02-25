



import sys, pprint

from PySide2 import QtCore, QtGui, QtWidgets


from wpui.layout import genAutoLayout
from wpui.superitem import SuperItem, SuperWidget

class DemoWidget(QtWidgets.QWidget):

	def __init__(self, baseData, parent=None):
		super().__init__(parent=parent)
		self.baseData = baseData
		self.btn = QtWidgets.QPushButton("result", parent=self)
		self.btn.clicked.connect(self.onBtnClicked)

		# widget to display item
		self.rootItem = SuperItem.forData(baseData)
		#view = self.rootItem.getNewWidget()
		#view.setParent(self)
		w = SuperWidget(parent=self)
		w.setTopItem(self.rootItem)


		self.layout = genAutoLayout(self)
		self.setLayout(self.layout)

	def onBtnClicked(self):
		"""display the result in output"""
		print("base:")
		pprint.pprint(self.baseData)
		print("result:")
		pprint.pprint(self.rootItem.wpResultObj())



if __name__ == '__main__':
	import sys
	import qt_material
	app = QtWidgets.QApplication(sys.argv)
	#qt_material.apply_stylesheet(app, theme='dark_blue.xml')

	#data = {"a": 1, "b": [6, {"b": "d"}], "c": 3}
	data = [1, 2, 3, 4]
	widget = DemoWidget(data)
	widget.show()

	sys.exit(app.exec_())
