

if __name__ == '__main__':
	import sys

	structure = [
		"a",
		[2, 3, 4],
		{"a": 1, "listKey" : ["F", "F", "afsdahskjdh"],
		          "b": 2},
		"b",
	]
	app = QtWidgets.QApplication()
	structure = SuperItemBase.forValue(structure)
	print(structure)

	view = structure.getNewView()
	view.show()
	sys.exit(app.exec_())





