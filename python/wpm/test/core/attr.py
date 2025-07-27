

from wpm.core import plug, node
from wpm import WN, cmds


def testPlugs():

	tf = WN.Transform.create("tfA")

	trPlug = tf.translate_.MPlug

	result = tuple(plug.broadcast(0.1, trPlug))
	print(result)

	# uvpin, array of compounds
	uvPin = WN.UvPin.create() # HATE the casing on the classname
	#TODO: whitelist/override some string atoms for generated names
	# result = plug.broadcast(
	# 	(0, 1), [uvPin.coordinate_[0], uvPin.coordinate_[1], uvPin.coordinate_[2]]
	# )
	# for i in result: print(i[0], i[1].name())


	result = plug.broadcast(
		(0, 1), [uvPin.coordinate_[0], uvPin.coordinate_[1], uvPin.coordinate_[2]]
	)
	for i in result:
		print(i[0], i[1].name())
		cmds.setAttr(i[1].name(), i[0])


"""

issue:

broadcast(
	(0.1, 0.2),
	[ uvPin.coord[0], uvPin.coord[1] ]
)

we want each to get value of (0.1, 0.2)

could say tuples can't be split? only duplicated?



plug[1:4] - array plug

plug[1:4]["coordinate*"] - all direct children matching "coordinate*"

plug("**/test[0]/**") - return all plugs matching given path?



"""
