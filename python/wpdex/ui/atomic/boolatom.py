
from __future__ import annotations
import typing as T

from PySide2 import QtCore, QtWidgets, QtGui

import reactivex as rx

from wplib import log
from wplib.object import Signal

from wpdex import WpDexProxy, Reference
#from wpdex.ui.react import ReactiveWidget
from wpdex.ui.atomic.base import AtomicWidget
"""a widget might have multiple 'slots' to link - 
think of option box, not just the value
some of this could be bidirectional through refs

availableSlots = CheckBox.availableSlots()
w = widget(parent)
instanceSlots = w.slots() ?
w.slots["value"].connect(ref)?

w = widget(parent, slots={"value" : ref})

a slot should have some direct connection to ui itself,
so value is Slot(uiGet="setChecked", uiSet="checked")

also need to tell which qt signals flag the ui has changed -
connect them all to a uiEdited method to dispatch to slots?
	
	Not sure if we should put the link system in this or just the
	uniform interface
"""

class BoolAtomicWidget(AtomicWidget):
	"""expect to inherit from qCheckBox or similar"""
	def _uiChangeQtSignals(self) ->list[QtCore.Signal]:
		return [self.stateChanged]
	def getValue(self)->bool:
		return self.isChecked()
	def setValue(self, value, **kwargs):
		self.setChecked(value)


class AtomCheckBox(QtWidgets.QCheckBox, BoolAtomicWidget):
	"""pressed fires when the button is held down, regardless of whether 
	committed - consider later highlighting the effects of the button on 
	press, before activating them when the state changes.
	"""
	def __init__(self, parent=None, **kwargs,
	             ):
		super().__init__(parent)
		BoolAtomicWidget.__init__(self, **kwargs)


class AtomRadioToggle(QtWidgets.QRadioButton, BoolAtomicWidget):
	def __init__(self, parent=None, **kwargs,
	             ):
		"""single slot of value"""
		super().__init__(parent)
		BoolAtomicWidget.__init__(self, **kwargs)

class AtomStickyButton(QtWidgets.QPushButton, BoolAtomicWidget):
	"""button like this should also be able to receive live text"""
	def __init__(self, parent=None, **kwargs,
	             ):
		super().__init__(parent)
		BoolAtomicWidget.__init__(self, **kwargs)
		self.setCheckable(True)

def makeUi():
	w = QtWidgets.QWidget()
	layout = QtWidgets.QVBoxLayout()
	w.setLayout(layout)

	# create 2 views on the same value
	boxA = AtomCheckBox(parent=w, name="boxA")
	layout.addWidget(boxA)
	boxB = AtomCheckBox(parent=w, name="boxB")
	layout.addWidget(boxB)

	# create "model" with a consistent bool held in a dict
	# this can be ANY SHAPE OF DATA in python
	model = {"root" : True}
	# wrap in proxy
	wxmodel = WpDexProxy(model)

	# link displayed values to REFERENCES into the SAME PATH
	boxA.VALUE.link(wxmodel.ref("root"))
	boxB.VALUE.link(wxmodel.ref("root"))
	return w


if __name__ == '__main__':
	app = QtWidgets.QApplication()
	# w = AtomStickyButton()
	# w.setText("test")
	w = makeUi()
	w.setFixedSize(200, 200)
	w.show()
	app.exec_()


	# d = {"root" : True}
	# w = WpDexProxy(d)
	# log(w.dex().root)
	# log(w.dex().branchMap())
	# ref = w.ref("root")
	# log("ref", ref)
	# log(ref.dex())
	# log("ROOT", ref.root)
	# log(ref.root.dex())
	# log(ref.root.dex().branchMap())
	#log(ref.root.dex().access())


