
from __future__ import annotations

from PySide2 import QtCore, QtWidgets, QtGui

from wplib import log
from wplib.inheritance import MetaResolver
from wpdex import WpDex, SeqDex, WpDexProxy, DictDex
from wpdex.ui.atomic import AtomicWidgetOld, AtomicView, AtomicStandardItemModel, AtomStandardItem

# class DictKeyStandardModel(AtomicStandardItemModel):
#
# 	def _processValueForUi(self, rawValue):
# 		assert isinstance(rawValue, str) # internal key of the dex (for the value that holds the actual dict key) should always be a string
# 		if rawValue.startswith("key:"):
# 			rawValue = rawValue.split("key:", 1)[0]
# 		return rawValue
#
# 	def


class DictDexModel(AtomicStandardItemModel):
	forTypes = (DictDex, )
	def _buildItems(self):
		self.clear()
		items = tuple(self.dex().branchMap().items())
		keyTies = [i for i in items if "key:" in str(i[0])]
		valueTies = [i for i in items if "key:" not in str(i[0])]
		for i, ((keyKey, keyDex),
		     (valueKey, valueDex)) in enumerate(zip(keyTies, valueTies)):
			keyItemType = AtomStandardItem.adaptorForObject(keyDex)
			assert keyItemType
			valueItemType = AtomStandardItem.adaptorForObject(valueDex)
			assert valueItemType
			keyItem = keyItemType(value=keyDex)
			valueItem = valueItemType(value=valueDex)
			self.appendRow([keyItem, valueItem])


class DictDexView(AtomicView
                 ):
	"""view for dict
	"""
	forTypes = (DictDex,)


if __name__ == '__main__':


	from wpdex.ui import AtomicWindow
	d = {"strkey" : "value",
	     4 : 5}
	p = WpDexProxy(d)
	dex = p.dex()
	log(dex, dex.branchMap())


	ref = p.ref()
	log("ref", ref, "ref val", ref.rx.value)

	app = QtWidgets.QApplication()
	w = AtomicWindow(parent=None,
	                 value=ref)
	w.show()
	app.exec_()


