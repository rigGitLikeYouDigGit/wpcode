""" create live networks of connected wheels and cables under tension """
from edRig import cmds

from edRig import EdNode, ECA, plug, transform, curve


class PulleySystem(object):
	""" trash name
	class for connected systems of lines under tension,
	threaded around wheels
	"""
	def __init__(self, name="pulleySystem"):
		self.name = name
		self.wheels = []
		self.outputCurve = None

	def addWheel(self, name="newWheel"):
		""" assumed to be added in order """
		wheel = Wheel(name=name)
		self.wheels.append(wheel)
		return wheel

	def orderWheels(self):
		nWheels = len(self.wheels)
		#self.wheels = [i for i in reversed(self.wheels)]
		for i, val in enumerate((self.wheels)):
		#for i, val in enumerate((self.wheels)):
			nextIndex = (i + 1) % nWheels
			prevIndex = (i - 1 + nWheels) % nWheels
			print(("index {} next {} prev {}".format(i, nextIndex, prevIndex)))
			val.next = self.wheels[ nextIndex ]
			val.prev = self.wheels[ prevIndex ]

	def build(self):
		self.orderWheels()
		self.outputCurve = ECA("nurbsCurve", n=self.name + "outputCurve")
		for i in self.wheels:
			i.build()

	def link(self):
		attach = ECA("attachCurve", n=self.name + "finalAttach")
		index = 0
		for i in self.wheels:
			i.link()
			# for n in i.curves:
			# 	n.con(n + ".local", attach + ".inputCurves[{}]".format(index))
			# 	index += 1
		#attach.con("outputCurve", self.outputCurve + ".create")




class Wheel(object):
	""" individual wheel in pulley system
	search "homothetic circles" and it should point
	the right way"""

	def __init__(self, basePoint=None, name="wheel"):
		""":type basePoint : EdNode
		:param basePoint : EdNode
		central point from which main vector will compute

		each wheel deals with itself and the relationship to its next

		"""
		self.point = basePoint
		self.name = name
		self.next = None # next wheel
		self.prev = None # previous wheel

		self.outputNext = None
		self.outputPrev = None

		self.group = None
		self.pin = None

		self.curves = []

		# interface parametres
		self.pos = None
		self.radius = None
		self.prevRotMat = None
		self.prevRotVec = None
		self.line = None

		pass


	#@EdNode.withPrefix(self.name)
	def build(self):

		""" creates base node network """

		# main group for everything
		self.group = ECA("transform", n=self.name + "_main")

		# main pin input
		self.pin = ECA("locator", n=self.name + "pointInput")
		self.pin.addAttr(attrName="radius", min=0, dv=1.0)
		#self.pin.con( self.pin + ".scaleX", self.pin + ".radius") # for now
		self.pin.parentTo(self.group)
		self.uRadius = self.pin + ".radius"

		# main switch to say if this wheel is external or internal
		self.pin.addAttr(attrName="external", attrType="bool")


		# radius flipping setup
		self.pin.addAttr(attrName="flip", attrType="bool")
		choice = ECA("choice", n=self.name + "radiusSwitch")
		flipMdl = ECA("multDoubleLinear", n=self.name + "radiusFlip")
		flipMdl.set("input1", 1)
		flipMdl.set("input2", -1)
		#flipped by default
		choice.con(flipMdl + ".input2", "input[0]")
		choice.con(flipMdl + ".input1", "input[1]")

		choice.con(self.pin + ".flip", "selector")
		# 1 / -1 used in other places in setup
		self.radius = plug.multPlugs(self.uRadius, choice + ".output")
		self.radiusChoice = choice


		# testing switches
		self.pin.addAttr(attrName="flipMidPoint", attrType="bool")
		self.midSwitch = ECA("choice", n=self.name + "midpointSwitch")
		self.midSwitch.con(self.pin + ".flipMidPoint", "selector")

		self.pin.addAttr(attrName="flipDegrees", attrType="bool")
		self.degreeSwitch = ECA("choice", n=self.name + "degreeSwitch")
		self.degreeSwitch.con(self.pin + ".flipDegrees", "selector")

		self.pin.addAttr(attrName="flipPrevMat", attrType="bool")
		self.prevMatSwitch = ECA("choice", n=self.name + "prevMatSwitch")
		self.prevMatSwitch.con(self.pin + ".flipPrevMat", "selector")

		self.nextMatSwitch = ECA("choice", n=self.name + ".nextMatSwitch")


		# base orient node to track cross product between input vectors
		orient = ECA("transform", n=self.name + "vectorOrient")
		orient.parentTo(self.pin)
		self.orient = orient

		# connection points
		#self.prevGrp = ECA("transform", n=self.name + "prev_grp", parent=self.orient)
		#self.prevPoint = ECA("locator", n=self.name + "prev_point", parent=self.prevGrp)
		self.prevPoint = ECA("locator", n=self.name + "prev_point", parent=self.pin)
		self.prevPoint.setColour((0.2, 0.2, 1.0))
		#self.nextGrp = ECA("transform", n=self.name + "next_grp", parent=self.orient)
		#self.nextPoint = ECA("locator", n=self.name + "next_point", parent=self.nextGrp)
		self.nextPoint = ECA("locator", n=self.name + "next_point", parent=self.pin)
		self.nextPoint.setColour((1.0, 0.2, 0.2))

		# arc midpoint
		self.midpoint = ECA("locator", n=self.name + "arcMidpoint")
		self.midpoint.con(self.radius, "translateX")
		#self.midpoint.parentTo(self.orient)
		self.midpoint.parentTo(self.pin)
		self.midpoint.setColour((1,0,0))

		# proxy and temp control, canted rotation not implemented yet
		self.proxy = self.makeProxy()
		self.proxy.parentTo(orient)

		# curve setup
		self.arc = ECA("makeThreePointCircularArc", n=self.name + "Arc")
		#self.arc = ECA("makeTwoPointCircularArc", n=self.name + "Arc")
		#self.arc.con(self.uRadius, "radius")

		self.pos = self.pin + ".translate"

		self.circumference = ECA("nurbsCurve", n=self.name + "curve")
		self.arc.con("outputCurve", self.circumference + ".create")
		self.circumference.parentTo(self.pin)
		self.curves.append(self.circumference)
		#self.line = ECA("nurbsCurve", n=self.name + "line", parent=self.pin)
		#self.curves.append(self.line)

		# dangling hook for adjacent wheel to fill in
		#self.prevRotMat = ECA("composeMatrix", n=self.name + "prevRotMat")
		self.prevRotMat = ECA("passMatrix", n=self.name + "prevRotMat")
		self.prevRotVec = ECA("multiplyDivide", n=self.name + "prevRotVector")


	def link(self):
		"""
		connects wheel to pulley system
		before and after can be the same wheel
		for now we only support two linked wheels - more can be achieved
		more easily with hacks if necessary
		there are probably inefficiencies in here among the linear stuff,
		but at least it's all pretty parallel
		"""

		EdNode.prefixStack.append(self.name)

		flipXOR = plug.xorPlugs(self.pin + ".flip",
		                          self.next.pin + ".flip")
		radiusMult = plug.normaliseBoolPlug(flipXOR)
		#self.radius = plug.multLinearPlugs(self.uRadius, radiusMult)


		# main vector inputs
		# to centres of adjacent wheels
		vecToPrev = plug.vecFromTo(self.pos,
		                           self.prev.pos,
		                           name="from{}_to{}".format(self.name,
		                                                     self.prev.name)
		                           )
		vecToNext = plug.vecFromTo(self.pos,
		                           self.next.pos,
		                           name="from{}_to{}".format(self.name,
		                                                     self.next.name)
		                           )

		# vecWithout = plug.vecFromTo(self.prev.pos,
		#                             self.next.pos)

		# normal between points
		inNormal = plug.crossProduct(vecToPrev, vecToNext, normalise=True,
		                             name=self.name + "_normalCross")
		vecToPrevN = plug.normalisePlug(vecToPrev)
		vecToNextN = plug.normalisePlug(vecToNext)


		nextOrientMat = transform.buildTangentMatrix(
			(0, 0, 0), vecToNextN, inNormal)

		# test a weird shear idea
		shearMat = transform.fourByFourFromCompoundPlugs(
			xPlug=vecToNextN, yPlug=inNormal, zPlug=vecToPrevN,
			posPlug=None, name=self.name+"ShearAimMat"
		)


		transform.decomposeMatrixPlug(nextOrientMat, target=self.orient )

		# midpoint setup
		midDir = plug.vectorMatrixMultiply((-1, 0, -1), shearMat, normalise=False)
		midDir = plug.normalisePlug(midDir)
		midPos = plug.multPlugs(self.uRadius, midDir)
		# we do still need normalisation, shear distorts vector length

		self.midpoint.con(midPos, "translate")


		# find homothetic centres
		""" equation goes like this
		centre = (r2 / (r1 + r2) ) * pos1 + (r1 / (r1 + r2) ) * pos2
		angles of tangents are common to both discs
		knowing distance to centre and radius,
		# angle found from arctan(dCentre / radius)
		# marvel at the idiocy
		dCentre is hypotenuse, radius is adjacent
		angle found from arccos(radius / dCentre)
		
		angle only changes with DISPARITY
		both flipped = neither flipped, for angle
		"""

		r1 = self.radius
		r2 = self.next.uRadius

		totalRadii = plug.addLinearPlugs(r1, r2)
		# safety limits
		#totalRadii = plug.setPlugLimits(totalRadii, min=0.0001)
		r2OverTotal = plug.dividePlug(r2, totalRadii)
		r1OverTotal = plug.dividePlug(r1, totalRadii)

		lhs = plug.multPlugs(r2OverTotal, self.pos)
		rhs = plug.multPlugs(r1OverTotal, self.next.pos)

		centre = plug.addCompoundPlugs(lhs, rhs)

		# dot product to check sign of difference in weights
		toCentre = plug.vecFromTo(self.pos, centre)
		dot = plug.dotProduct(vecToNext, toCentre)
		dotBool = plug.boolFromNormalisedPlug(dot)


		# arccos( dCentre / radius )
		distance = plug.plugDistance(centre, self.pos)
		overRadius = plug.dividePlug(self.uRadius, distance)+"X"

		acos = plug.trigPlug(overRadius, mode="arccosine", res=24)
		degrees = plug.plugToDegrees(acos)
		#negDegrees = plug.multPlugs(degrees, -1)
		negDegrees = plug.multPlugs(degrees, plug.multLinearPlugs(dot, -1))
		comp = ECA("composeMatrix", n=self.name + "nextAngleRotMat")

		#comp.con(degrees, "inputRotateZ")
		comp.con(negDegrees, "inputRotateZ")

		# flip angle setup for next point
		nextRot = comp + ".outputMatrix"
		invNextRot = plug.invertMatrixPlug(nextRot)
		self.nextMatSwitch.con(nextRot, "input[0]")
		self.nextMatSwitch.con(invNextRot, "input[1]")

		nextMat = plug.multMatrixPlugs([shearMat, nextRot])
		self.next.prevRotMat.con(nextMat, "inMatrix")


		nextDir = plug.vectorMatrixMultiply( (dot, 0, 0),  nextMat, normalise=False)

		nextDir = plug.normalisePlug(nextDir)
		self.next.prevRotVec.con(nextDir, "input1")
		nextPos = plug.multPlugs(self.uRadius, nextDir)
		self.nextPoint.con(nextPos, "translate")


		prevRot = self.prevRotMat + ".outMatrix"
		invPrevRot = plug.invertMatrixPlug(prevRot)
		self.prevMatSwitch.con(invPrevRot, "input[0]")
		self.prevMatSwitch.con(prevRot, "input[1]")


		# connect to flip on previous wheel

		prevFlip = plug.reversePlug(flipXOR) + "X"
		prevFlipMult = plug.normaliseBoolPlug(prevFlip)


		prevDir = self.prevRotVec + ".output"
		prevDir = plug.normalisePlug(prevDir)
		prevPos = plug.multPlugs(self.uRadius, prevDir)

		self.prevPoint.con(prevPos, "translate")


		# marker for now
		marker = ECA("locator", n=self.name + "hCentreMarker")
		marker.con(centre, "translate")

		self.arc.con(self.prevPoint + ".translate", "point1")
		self.arc.con(self.midpoint + ".translate", "point2")
		self.arc.con(self.nextPoint + ".translate", "point3")

		self.line = curve.curveFromCvPlugs(
			[self.nextPoint + ".worldPosition[0]",
		    self.next.prevPoint + ".worldPosition[0]"],
			useApi=0, name=self.name + "nextLine")

		EdNode.prefixStack.remove(self.name)



	def makeProxy(self):
		""" create a polygon cylinder representing
		wheel orientation and size
		move to dedicated proxy/polygon lib sometime maybe """

		proxy = EdNode(cmds.polyCylinder(
			axis=[0,1,0], ch=0, n=self.name + "_proxy",
		height=0.1, radius=0.4)[0])

		return proxy








"""
wheel dag structure

main group
	vectorOrient # point constrained to main point, oriented to cross product
		offsetOrient # allows rotation around common axis for canted wheels
		proxy # same as above, works as control for now as well
			


"""




