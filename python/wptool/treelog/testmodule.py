
from tree.dev.log.logfn import treeLog

#import dis
def loopTestFn():

	def _dynamicFn():
		treeLog(
			"eyy")

	for n in range(2):
		treeLog("loop" + str(n))
	# 	innerLoopFn()
		for i in "ab":
			treeLog(



				i




			)
			treeLog("second log")

			_dynamicFn()


#print(dis.code_info(loopTestFn))
dis.dis(loopTestFn)
def innerLoopFn():
	loopLog("inner loop fn")


def mainTestFn():

	treeLog("main fn begin")

	for i in range(3):
		treeLog("loop log")
		treeLog(loopFn())
		for n in range(2):
			treeLog("compound loop log")

	midFn()

	treeLog("main fn end")

	pass

def loopFn():
	treeLog("called in loop")

def midFn():
	treeLog("mid fn begin")

	innerFn()

	treeLog("mid fn halfway")

	innerFn()

	treeLog("mid fn end")

	pass


def innerFn():
	treeLog("inner fn begin")

