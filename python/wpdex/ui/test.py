



import pprint

from PySide2 import QtWidgets


from wpui.layout import genAutoLayout
from wpdex.ui import WpDexWidget, WpDexWindow

class DemoWidget(QtWidgets.QWidget):

	def __init__(self, baseData, parent=None):
		super().__init__(parent=parent)
		self.baseData = baseData
		self.btn = QtWidgets.QPushButton("result", parent=self)
		self.btn.clicked.connect(self.onBtnClicked)

		self.w = WpDexWindow(parent=self)
		self.w.setRootObj(baseData)
		#w.setFixedSize(500, 500)

		# layout = QtWidgets.QVBoxLayout()
		# layout.addWidget(self.btn)
		# layout.addWidget(w)
		# self.setLayout(layout)


		layout = QtWidgets.QVBoxLayout()
		layout.addWidget(self.btn)
		layout.addWidget(self.w)
		self.setLayout(layout)

		self.resize(400, 400)

	def onBtnClicked(self):
		"""display the result in output"""
		print("base:")
		pprint.pprint(self.baseData)
		print("result:")
		pprint.pprint(self.w.dex().obj)



if __name__ == '__main__':
	import sys

	app = QtWidgets.QApplication(sys.argv)
	#qt_material.apply_stylesheet(app, theme='dark_blue.xml')

	#data = {"a": 1, "b": [6, {"b": "d"}], "c": 3}
	data = [1, 2, 3, 4]
	#data = 9
	widget = DemoWidget(data)
	widget.show()

	sys.exit(app.exec_())
