
from edRig import cmds, om
from .main import WN


class RampPlug(object):
	"""don't you love underscores and capital letters?
	called thus:
	ramp = RampPlug(myAnnoyingString)
	setAttr(ramp.point(0).value, 5.0)
	to be fair, if these are special-cased in maya, why not here"""

	class _Point(object):
		def __init__(self, root, index):
			self.root = root
			self.index = index

		@property
		def _base(self):
			return "{}[{}].{}_".format(
				self.root, self.index, self.root	)

		@property
		def value(self):
			return self._base + "FloatValue"
		@property
		def position(self):
			return self._base + "Position"
		@property
		def interpolation(self):
			return self._base + "Interp"

	def __init__(self, rootPlug):
		"""root is remapValue"""
		self.root = rootPlug
		self.points = {}

	def point(self, id):
		""":rtype : RampPlug._Point"""
		return self._Point(self.root, id)


class RemapValue(WN):
	"""wrapper for rempValue nodes"""
	nodeType = "remapValue"

	attrs = ["inputMin", "inputMax", "inputValue", "outValue"]

	@property
	def ramp(self):
		""":returns rampPlug object
		:rtype : RampPlug"""
		#return RampPlug(self + ".value")
		return self._ramp

	@classmethod
	def create(cls, name=None, n=None, *args, **kwargs):
		remap = super(RemapValue, cls).create(name=None, n=None,
		                                      *args, **kwargs)
		remap._ramp = RampPlug(remap + ".value")


	def getRampInstance(self, name="rampInstance"):
		"""creates new ramp exactly mirroring master and connects it by message
		:rtype RemapValue"""
		newRemap = ECA("remapValue", n=name)
		for i in range(cmds.getAttr(self+".value", size=True)):
			attr.con(self + ".value[{}]".format(i),
					 newRemap +".value[{}]".format(i))
		attr.makeMutualConnection(self, newRemap, attrType="message",
								  startName="instances", endName="master")
		return newRemap

	@property
	def instances(self):
		"""look up ramps connected to master by string"""
		return attr.getImmediateFuture(self + ".instances")
