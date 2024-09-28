
from __future__ import annotations
import typing as T

from PySide2 import QtCore, QtWidgets, QtGui

import reactivex as rx

from wplib import log
from wplib.object import Signal

from wpdex import WpDexProxy, Reference
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

"""
could have these descriptors for show(), enabled() etc
"""
class WidgetHook:

	def __init__(self, w:QtWidgets.QWidget,
	             getFn:callable=None,
	             setFn:callable=None,
	             ):
		self.w = w
		self.getFn = getFn
		self.setFn = setFn
		self.uiChangedSignal = Signal(name="uiChanged")
		self.uiChangedSignal.connect(self.onUiChanged)

	def __repr__(self):
		return f"Hook({self.getFn}, {self.setFn}, {self.w})"

	def onUiChanged(self, *args, **kwargs):
		"""triggered whenever the ui changes -
		both on direct edit and on sync"""
		#log("onUIChanged", self, args, kwargs)

	def link(self, ref:Reference):
		"""set up bidirectional dependence between this
		ui field and the given reference -
		we can't insert any transformation here (unless it's reversible)
		"""
		# set to ref value first
		log("setting first")
		self.setFn( ref() )
		def onRefChanged(*args, **kwargs):
			log("on ref changed", self, args, kwargs)
			log("new value", ref() )
			self.setFn( ref() )
		# ref.dex().root.getEventSignal("main").connect(
		# 	lambda event : self.setFn( ref() ))
		ref.dex().root.getEventSignal("main").connect(onRefChanged)
		self.uiChangedSignal.connect(
			lambda *args : ref.dex().write( self.getFn() )
		                       )

	# def connections(self)->dict[str, list[callable]]:
	#


base = object
if T.TYPE_CHECKING:
	base = QtWidgets.QWidget
class ReactiveWidget(base):


	def __init__(self, name:str=""):
		self.setObjectName(name)
		self.ENABLED = WidgetHook(
			self, getFn=self.isEnabled, setFn=self.setEnabled	)
		self.VISIBLE = WidgetHook(
			self, getFn=self.isVisible, setFn=self.setVisible		)
		self.VALUE = WidgetHook(
			self, getFn=self.getValue, setFn=self.setValue)

		# connect signals to drive
		for i in self._uiChangeQtSignals():
			i.connect(self.onUiChanged)
		#TODO: the old agony of Qt widget init sequencing might strike here -
		#  it shouldn't matter when hooks are created though

	def __repr__(self):
		return f"{type(self)}({self.objectName()})"

	def _uiChangeQtSignals(self)->list[QtCore.Signal]:
		"""return the qt signals of this widget to connect to the
		onUiChanged() manager function"""
		raise NotImplementedError

	def hooks(self)->dict[str, WidgetHook]:
		return {k : v for k, v in self.__dict__.items() if isinstance(v, WidgetHook)}

	def onUiChanged(self, *args, **kwargs):
		"""top level trigger for any ui change -
		this will trigger regardless of source, so either filter
		or mute	signals outside this scope where needed
		"""
		if self.sender() is self:
			log("skipping box", self)
		for hookName, hook in self.hooks().items():
			hook.uiChangedSignal(*args, **kwargs)

	# uniform interface for getting and setting values on common widgets
	def getValue(self):
		raise NotImplementedError
	def setValue(self, value, **kwargs):
		raise NotImplementedError

	def getOptions(self):
		raise NotImplementedError
	def setOptions(self, value, **kwargs):
		raise NotImplementedError

class AtomicWidget(ReactiveWidget):
	"""MAYBE it's worth turning fields like value, options etc
	into generated descriptors or something

	i say "etc" but I can't think of any real uses outside those
	"""


class AtomCheckBox(QtWidgets.QCheckBox, AtomicWidget):


	"""pressed fires when the button is held down, regardless of whether 
	committed - consider later highlighting the effects of the button on 
	press, before activating them when the state changes.
	"""

	def __init__(self, parent=None, name="",
	             #value=None
	             ):
		"""single slot of value"""
		super().__init__(parent)
		AtomicWidget.__init__(self, name)
		# self.setTristate(False)
		#
		# for i in self._uiChangeSignals:
		# 	getattr(self, i).connect(self.uiChanged)

	def _uiChangeQtSignals(self) ->list[QtCore.Signal]:
		return [self.stateChanged]

	def getValue(self)->bool:
		return self.isChecked()
	def setValue(self, value, **kwargs):
		self.setChecked(value)



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
	boxA = AtomCheckBox(parent=w, name="boxA")
	layout.addWidget(boxA)
	boxB = AtomCheckBox(parent=w, name="boxB")
	layout.addWidget(boxB)

	boxA.VALUE.link(wxmodel.ref("root"))
	boxB.VALUE.link(wxmodel.ref("root"))
	return w


if __name__ == '__main__':
	app = QtWidgets.QApplication()
	# w = AtomStickyButton()
	# w.setText("test")
	w = makeUi()
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


