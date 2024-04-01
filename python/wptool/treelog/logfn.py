
import inspect, sys, re, functools, itertools, pprint, ast, textwrap
import typing as T

from weakref import WeakKeyDictionary
from collections import namedtuple

import dis # menacing
import opcode # jesus that's menacing

from wptree import Tree



# all ops that can create indents (other than if / else)
indentOps = {k : v for k, v in opcode.opmap.items() if "SETUP_" in k and not "ANNOTATIONS" in k}

# all ops that can set up loop blocks
loopOpNames = ("SETUP_LOOP",
                   "FOR_ITER",
                   )
loopOps = {k : v for k, v in opcode.opmap.items() if k in loopOpNames}
# for loopOpName in ("SETUP_LOOP",
#                    "FOR_ITER",
#                    ):
# 	if loopOpName in opcode.opmap:
# 		loopOps[loopOpName] = opcode.opmap[loopOps]
#loopOps = {k : opcode.opmap[k] for k in ("SETUP_LOOP",)}

logTree = Tree("logRoot")

# match for "*_i(digits)"
pattern = re.compile(r".+_i\d+")


# tuple to hold individual code line reference
# source is just base string, not full code object
CodeLine = namedtuple("CodeLine", ("source", "line"))

FnCall = namedtuple("FnCall", ("fn", "line", "file"))

LoopIteration = namedtuple("LoopIteration", ("loopNode", "valueTuple"))


# map of all distinct frame stacks ever called -
# {string path : (tree branch? , number of calls)}
LoopCallTrace = namedtuple("LoopCallTrace", ("branch", "nCalls"))
allCalls = {} # type: T.Dict[tuple, LoopCallTrace]

# maps of functions to their decompiled ast trees
fnAstMap = {} # type: T.Dict[function : ast.AST]

# maps of individual calls to number of
# loops directly above each, locally
# this does not track usage of loops, only definitions
callLocalLoopMap = {} #type: T.Dict[T.Tuple[str, int], T.Tuple]


callStackCountMap = {} #type: T.Dict[T.Tuple, int]


def increasingChains(baseSeq):
	"""For seq ABCD, returns
	A, AB, ABC, ABCD """
	for i in range(len(baseSeq) ):
		yield baseSeq[ :i + 1]


class RemoveDefinitionTransformer(ast.NodeTransformer):
	"""remove the FunctionDef nodes from a block of ast code"""

	def __init__(self):
		super(RemoveDefinitionTransformer, self).__init__()
		self.defCounter = 0
	def visit_FunctionDef(self, node):
		print("visit function def", node, node.name, self.defCounter)
		if self.defCounter:
			print("remove definition node")
			return None

		self.defCounter += 1
		return self.generic_visit(node)


def treeLog(msg):
	print("")
	stack = inspect.stack()
	callerFrames = stack[1:-1]
	reverse_stack = tuple(reversed(stack))

	# test subchains recursively from top-level call -
	# eg for A ( B ( C () ) ), test
	# A, then AB, then ABC, etc
	# this isn't necessary as we have a semi-reliable hash for loops

	pathway = []
	for i, frameData in enumerate(reversed(callerFrames)
			):

		# look up function object
		# first check calling frame's parent's locals
		# for internally-defined functions
		frame = frameData.frame
		code = frame.f_code
		fn_lookup = frame.f_back.f_locals.get(
			code.co_name,
			# then check globals
			frame.f_globals.get(
				code.co_name)
		)
		call = FnCall(fn_lookup, frameData.lineno, frameData.filename)
		pathway.append(call)


		# get any loops in function

		# look up function object
		# first check calling frame's parent's locals
		# for internally-defined functions
		fn_lookup = frame.f_back.f_locals.get(
				code.co_name,
			# then check globals
				frame.f_globals.get(
					code.co_name
				)
			)

		print("fn lookup", fn_lookup)

		astNode = fnAstMap.get(fn_lookup)
		if astNode is None:
			transformer = RemoveDefinitionTransformer()
			rawNode = ast.parse(textwrap.dedent(
				inspect.getsource(fn_lookup)))
			transformer.visit(rawNode)
			astNode = rawNode
			fnAstMap[fn_lookup] = astNode
		#print("astnode", astNode)

		"""if log function is directly called in stack, get
		loops above it - if another function is called next in stack,
		get loops above that instead"""

		fnLookupName = reverse_stack[ i + 2].frame.f_code.co_name

		loops = loopsInFunctionAst(astNode, frame,
		                           lookupFnName=fnLookupName)
		print("------")
		print("found loops", loops)


		for loopNode in loops:
			loopValues = loopValueMap(
				loopNode, frame.f_locals, frame.f_globals)
			loopValueHash = dictToHashTuple(loopValues)
			pathway.append(LoopIteration(loopNode=loopNode,
			                             valueTuple=loopValueHash))
	print("path")
	print(pathway)

	# process pathway for tree to outer function
	treePath = []
	niceNames = []
	for segment in pathway:
		treePath.append(str(segment))
		niceNames.append(niceNamePathSegment(segment))

	# add to tree
	fnBranch = logTree(niceNames)

	logFrame = stack[1]
	lineName = "ln " + str(logFrame.frame.f_lineno)
	logBranch = fnBranch(lineName)
	logBranch.value = msg

def niceNamePathSegment(segment):
	if isinstance(segment, FnCall):
		return segment.fn.__name__
	elif isinstance(segment, LoopIteration):
		valueDisplay = "loop: " + ";".join(
			i[0] + "=" + i[1] for i in segment.valueTuple)
		return valueDisplay


def loopsInFunctionAst(fnAst, frameObj, lookupFnName=treeLog.__name__):
	"""fnAst is a dynamic Module ast node """
	# get local line number of log call within block
	print("---")
	code = frameObj.f_code
	print("frame", frameObj)
	localLineno = frameObj.f_lineno - code.co_firstlineno

	# remove any internal definitions
	#pprint.pprint(ast.dump(fnAst))

	# add parent attribute to ast nodes
	for i in ast.walk(fnAst):
		for child in ast.iter_child_nodes(i):
			child.parent = i
	fnAst.parent = None

	# filter nodes that are function calls
	#print("lookup fn name", lookupFnName)
	fnNodes = []
	for i in ast.walk(fnAst):
		if isinstance(i, ast.Call) and i.func.id == lookupFnName:
			fnNodes.append(i)

	#print("localLine", localLineno)
	for i in fnNodes:
		maxLine = maxFnCallLine(i) - 1
		#print("fnNode", i, i.func.id, maxLine)
	fnPoints = [(i, maxFnCallLine(i) - 1) for i in fnNodes]
	#print("fnPoints", fnPoints)
	fnNode = next(filter(lambda x: x[1] == localLineno, fnPoints))[0]

	parents = astParents(fnNode)
	loops = [i for i in parents if
	         isinstance(i, (ast.For, ast.While))]
	# backwards - reverse here
	loops = list(reversed(loops))
	return loops


def loopValueMap(loopAstNode, frameLocals, frameGlobals):
	"""iterate over nodes contained in loop target to find
	the loop's local variables - look them up in frameLocals
	as a way of identifying this specific iteration"""
	localNames = set()
	for node in [loopAstNode.target] + list(ast.iter_child_nodes(loopAstNode.target)):
		if isinstance(node, ast.Name):
			localNames.add(node.id)
	localValues = {name : frameLocals[name] for name in localNames}
	#print("local values", localValues)
	return localValues


def dictToHashTuple(baseDict):
	"""super inefficient and explicit for now"""
	return tuple( (k, str(baseDict[k]), str(id((baseDict[k])))) for k in sorted(baseDict.keys()))
	# return ", ".join(tuple(k + "=" + str(baseDict[k]) for k in sorted(baseDict.keys())))


def astParents(childNode, includeChild=True):
	parents = [childNode] if includeChild else []
	node = childNode
	while node.parent:
		parents.append(node.parent)
		node = node.parent
	return parents

def maxFnCallLine(fnCallNode:ast.Call):
	"""return max line of any arguments in function call -
	this line is passed to the frame of the function"""
	lineNumbers = [(i, i.lineno) for i in ast.walk(fnCallNode)
		if getattr(i, "lineno", None)]
	#print(lineNumbers)
	maxLine = max(i[1] for i in lineNumbers)
	#print("maxLine", maxLine)
	return maxLine



"""
for interest, introduction to what bytecode stuff looks like

	frame = callerFrames[0].frame
	code = frame.f_code

	# ALL THIS is to retrieve the right ast node for the call,
	# then check for any loop nodes preceding it

	instructions = dis.get_instructions(code)
	#pprint.pprint([i for i in instructions])
	#pprint.pprint([i for i in dis.findlabels(code.co_code)])

	#pprint.pprint(dis.dis(code))

	rawBytes = code.co_code
	lastByteIndex = frame.f_lasti
	lastByteCmd = rawBytes[lastByteIndex]
	#lastByteTest = dis.disassemble(code, frame.f_lasti)
	lastByteCmdName = opcode.opname[lastByteCmd]
	print("last", lastByteIndex, lastByteCmd, lastByteCmdName)
	print("co tabs", code.co_lnotab)

	# get byte instruction on line
	lastInstruction = [i for i in instructions
	        if i.offset == lastByteIndex][0]
	print("last instruction", lastInstruction)
	lastInstLine = lastInstruction.starts_line
	print(lastInstLine)

"""

