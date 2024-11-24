
from __future__ import annotations

from PySide2 import QtCore, QtWidgets, QtGui

from wplib import log
from wplib.inheritance import MetaResolver
from wpdex import WpDex, SeqDex, WpDexProxy
from wpdex.ui.atomic import AtomicWidgetOld, AtomicView, AtomicStandardItemModel, AtomStandardItem





class SeqDexItem(AtomStandardItem):
	forTypes = (SeqDex, )

class SeqDexModel(AtomicStandardItemModel):
	forTypes = (SeqDex, )

	def _buildItems(self):
		self.clear()
		for k, dex in self.dex().branchMap().items():
			itemType = AtomStandardItem.adaptorForObject(dex)
			item = itemType(value=dex)
			self.appendRow([item])

#class SeqDexView(AtomicWidget, QtWidgets.QTreeView):
class SeqDexView(AtomicView
                 ):
	"""view for a list

	drag behaviour -
	drop directly on top of an item: override or swap?
	drop between items: insert

	or...
	ignore drag/drop for now, emulate something like a cursor in vim

	click an item to select it
	press enter to edit it
	ctl-v overwrites a selected item
	press alt to switch to cursor between items?

	work with selection spans / cursor spans - each has a start and end index


	"""
	forTypes = (SeqDex,)




if __name__ == '__main__':

	from wpdex.ui.base import WpDexWindow
	from wpdex.ui import AtomicWindow
	data = [1, 2, 3,
	            [4, 5],
	        6,
	        [ 4, 5,
	          [ 6, 7, ],
	          8 ]
	        ]

	p = WpDexProxy(data)
	#ref = p.ref(3)
	ref = p.ref()
	log("ref", ref, "ref val", ref.rx.value)

	app = QtWidgets.QApplication()
	# w = WpDexWindow(parent=None,
	#                 value=ref)
	w = AtomicWindow(parent=None,
	                 value=ref)
	w.show()
	app.exec_()


