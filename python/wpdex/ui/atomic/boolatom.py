
from __future__ import annotations
import typing as T

from PySide2 import QtCore, QtWidgets, QtGui

from wplib import log

from wpdex import WpDexProxy
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

def slots(self, w:QtWidgets.QWidget):
	return {
		"value" : ()
	}


class WidgetDescriptor:

	def __init__(self, w:QtWidgets.QWidget,
	             getFn:callable=None,
	             setFn:callable=None,
	             ):
		self.w = w
		self.getFn = getFn
		self.setFn = setFn

	def link(self, ):


class AtomCheckBox(QtWidgets.QCheckBox, AtomicWidget):

	_uiChangeSignals = ["stateChanged",
	                    #"pressed"
	                    ]
	"""pressed fires when the button is held down, regardless of whether 
	committed - consider later highlighting the effects of the button on 
	press, before activating them when the state changes.
	"""

	def __init__(self, parent=None,
	             #value=None
	             ):
		"""single slot of value"""
		super().__init__(parent)
		self.setTristate(False)

		for i in self._uiChangeSignals:
			getattr(self, i).connect(self.uiChanged)



	def getValue(self)->bool:
		return self.isChecked()
	def setValue(self, value, **kwargs):
		self.setChecked(value)

	def uiChanged(self, *args, **kwargs):
		"""top level trigger for any ui change -
		this will trigger regardless of source, so either filter
		or mute	signals outside this scope where needed
		"""
		pass
		# log("ui changed", args, kwargs)
		# log(self.sender(), self.senderSignalIndex())

class AtomRadioToggle(QtWidgets.QRadioButton, AtomicWidget):
	def getValue(self):
		return self.isChecked()
	def setValue(self, value, **kwargs):
		self.setChecked(value)

class AtomStickyButton(QtWidgets.QPushButton, AtomicWidget):
	"""button like this should also be able to receive live text"""

	def __init__(self, parent=None):
		super().__init__(parent)
		self.setCheckable(True)
	def getValue(self):
		return self.isChecked()
	def setValue(self, value, **kwargs):
		self.setChecked(value)

def makeUi():
	w = QtWidgets.QWidget()
	layout = QtWidgets.QVBoxLayout()
	w.setLayout(layout)

	# create "model" with a consistent bool held in a dict
	model = {"root" : True}
	# wrap in proxy
	wxmodel = WpDexProxy(model)

	# create 2 views on the same value
	boxA = AtomCheckBox(parent=w)
	layout.addWidget(boxA)
	boxB = AtomCheckBox(parent=w)
	layout.addWidget(boxB)



if __name__ == '__main__':
	app = QtWidgets.QApplication()
	w = AtomStickyButton()
	w.setText("test")
	w.show()
	app.exec_()

