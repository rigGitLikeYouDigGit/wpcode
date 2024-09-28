
from __future__ import  annotations
import typing as T

import param
from param import rx

from wpdex import WpDex, WpDexProxy

from PySide2 import QtCore, QtWidgets, QtGui

"""
tests for proper reactive widgets in Qt visualising arbitrary structures -
this is the second half of the work, now that the proxy system works

pyreaqtive is quite comprehensive but only works with their custom data structures
 as model - I couldn't immediately see where to modify it, but could return to it
 if needed
 
 
need single holder widget class to dynamic-dispatch into whatever wpdex type is shown


break up ui more - for smallest atomic widget go back to a consistent
interface to get/set value
consider how we might expose "plugs" on ui elements - for a text box, we pass in the
current value, but also the options? or a callable to return the options?

"""

class WxBox(QtWidgets.QCheckBox):
	""" where ref is something like

	dataRoot.ref("branch", "leaf", "myBoolKey")

	seems pointless to go through models for all of these, we still
	need to set up the dependency stuff on either end
	"""


	def __init__(self, ref, parent=None):
		super().__init__(parent=parent)
		# self.dataRoot = dataRoot
		# self.path = path
		ref.sync(self.setChecked)
		# in this extreme simple case, we want something like this
		# we could capture setChecked() being called to update the ref,
		# and somehow set the ref to the same thing?

		# for properly formatting text, we might want a proper chain like
		textRef.upper().join(",").output( self.setText)
		textRef.setFrom( self.text, when=self.textChanged ) # trigger when a qt signal goes off?
		# we're never going to hit all of this first try, this is a good start



def makeUi():
	data = {"root" : True}
	rdata = rx(data)
	rdata.rx.watch(lambda *args : print(f"changed {args} {data}"))

	rdata.rx.value["root"] = False

	parentWidget = QtWidgets.QWidget()
	checkBoxWidgetA : QtWidgets.QCheckBox = rx(QtWidgets.QCheckBox(parent=parentWidget))
	checkBoxWidgetB : QtWidgets.QCheckBox = rx(QtWidgets.QCheckBox(parent=parentWidget))
	checkBoxWidgetA.setChecked(rdata["root"])
	checkBoxWidgetB.setChecked(rdata["root"])
	layout = QtWidgets.QVBoxLayout()
	layout.addWidget(checkBoxWidgetA.rx.value)
	layout.addWidget(checkBoxWidgetB.rx.value)
	parentWidget.setLayout(layout)

	return parentWidget




if __name__ == '__main__':
	data = {"root" : True}
	rdata = rx(data)
	rdata.rx.watch(lambda *args : print(f"changed {args} {rdata.rx.value}"))

	rdata.rx.value["root"] = True

	# app = QtWidgets.QApplication()
	# w = makeUi()
	# w.show()
	#
	# app.exec_()


