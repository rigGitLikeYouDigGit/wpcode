
from __future__ import annotations

import typing as T

from PySide2 import QtWidgets, QtCore, QtGui

from tree.lib.object import TypeNamespace

from wp.constant import CurrentAssetProject


"""simple selector to choose between asset projects"""


class ProjectSelectorWidget(OptionComboWidget):
	pass

	@classmethod
	def create(cls)->ProjectSelectorWidget:
		params = AtomicWidgetParams(
			type=AtomicWidgetType.OptionMenu,
			name="Project",
			tooltip="Project to work with",
			options=CurrentAssetProject,
			optionType=AtomicWidgetParams.OptionType.Menu
		)

		return cls(value=CurrentAssetProject.Test,
		           params=params
		           )

if __name__ == '__main__':
	import sys
	from PySide2.QtWidgets import QApplication

	app = QApplication(sys.argv)
	widget = ProjectSelectorWidget.create()
	widget.show()
	sys.exit(app.exec_())


