
from __future__ import annotations
import typing as T

from tree import Signal
from wp.constant import Status

if T.TYPE_CHECKING:
	from wplib.errorreport import ErrorReport

class StatusObject:
	"""object holding a Status constant, providing
	methods to check it and set error message
	"""

	def __init__(self):
		self.status = Status.Success
		self.errorReport : ErrorReport = None
		self.statusChanged = Signal()

	def check(self):
		"""override - here implement some abstract check to
		see if status object has issues"""

	def setStatus(self, status:Status.T(), emit=True):
		"""set status, optionally emit signal"""
		oldStatus = self.status
		self.status = status
		if oldStatus != status:
			self.onStatusChanged()
			if emit:
				self.statusChanged.emit()

	def onStatusChanged(self):
		"""override - here implement any logic to run when status changes
		fires before signal is emitted"""