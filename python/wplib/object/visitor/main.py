
from __future__ import annotations

import types, pprint
import typing as T
import inspect
from types import FunctionType
from collections import defaultdict
from dataclasses import dataclass
from collections import deque, namedtuple
from typing import TypedDict

from wplib import log, fnlib as wfunction
from wplib.log import getDefinitionStrLink, getLineLink
#from wplib.sentinel import Sentinel
from wplib.object import Adaptor
from wplib.inheritance import superClassLookup, SuperClassLookupMap, isNamedTupleInstance, isNamedTupleClass
from wplib.object.namespace import TypeNamespace
from wplib.constant import MAP_TYPES, SEQ_TYPES, LITERAL_TYPES

from .adaptor import *

"""
import pretty_errors
pretty_errors.configure(
    separator_character = '*',
    filename_display    = pretty_errors.FILENAME_EXTENDED,
    line_number_first   = True,
    #display_link        = True,
    lines_before        = 0,
    lines_after         = 0,
    line_color          = pretty_errors.RED + '> ' + pretty_errors.default_config.line_color,
    code_color          = '  ' + pretty_errors.default_config.line_color,
	display_trace_locals=True,
    truncate_code       = False,
    display_locals      = True,
	truncate_locals=0,
	infix="\t"

)
"""

ChildData = namedtuple("ChildData",
                       ["key", "obj", "data"],
                       defaults=[None, None, None])



#PARAMS_T = T.Dict[str, T.Any]
PARAMS_T = VisitPassParams



class VisitObjectData(TypedDict):
	"""individual data passed to visit function,
	with richer record of context

	maybe make this optional
	"""
	base : T.Any # root object of the visit
	visitPassParams : VisitPassParams # pass params for current object


def visitFnTemplate(
          obj:T.Any,
          visitor:DeepVisitor,
          visitObjectData:VisitObjectData,
          #visitPassParams:VisitPassParams,
          )->T.Any:
	"""template function to override for custom transform"""
	raise NotImplementedError

# test that function argspec can be compared
# print("argspec", inspect.getfullargspec(visitFnTemplate))
# for k, v in inspect.getfullargspec(visitFnTemplate)._asdict().items():
# 	print(k, v)




class DeepVisitOp:
	"""helper class to define operations on visited objects"""

	def visit(self,
	          obj:T.Any,
              visitor:DeepVisitor,
              visitObjectData:VisitObjectData,
              #visitPassParams:VisitPassParams,
              )->T.Any:
		"""template function to override for custom transform"""
		raise NotImplementedError



class DeepVisitor:
	"""base class for visit and transform operations over all elements
	of a data structure.

	For now a transformation cannot add or remove elements - maybe add later
	using extensions to visitData.

	Run filter function over all elements for now, leave any filtering to
	client code.

	Might also be useful to have a lazy generator / structure that evaluates only when pathed into?
	"""

	VisitObjectData = VisitObjectData
	VisitPassParams = VisitPassParams
	DeepVisitOp = DeepVisitOp

	@classmethod
	def checkVisitFnSignature(cls, fn:visitFnType):
		"""check that the given function has the correct signature"""
		fnId = f"\n{fn} def {getDefinitionStrLink(fn)} \n"
		if not isinstance(fn, (types.FunctionType, types.MethodType)):
			raise TypeError(f"visit function " + fnId + " is not a function")
		# if fn.__code__.co_argcount != 4:
		# 	raise TypeError(f"visit function {fn} does not have 4 arguments")
		argSeq = ("obj", "visitor", "visitObjectData", "visitPassParams")
		#if fn.__code__.co_varnames[-4:] != argSeq:
		# if not (set(argSeq) <= set(fn.__code__.co_varnames)):
		# 	raise TypeError(f"visit function " + fnId + f"does not have correct argument names\n{argSeq} \n{argSeq[-4:]}\n{fn.__code__.co_varnames}")
		return True

	# separate method for every permutation of iteration direction - excessive but I can understand it
	def _iterRecursiveTopDownDepthFirst(self,
	                                    #parentObj:T.Any,
	                                    parentObjData:CHILD_T,
	                                    visitParams:VisitPassParams,
	                                    )->T.Generator[CHILD_T, None, None]:
		"""iterate over all objects top-down"""
		#print("iter rec top down", parentObjData, type(parentObjData), tuple(parentObjData))

		parentKey, parentObj, parentChildType = parentObjData
		adaptorType = VisitAdaptor.adaptorForType(
			type(parentObj))
		assert adaptorType, f"no visit adaptor for type {type(parentObj)}"
		nextObjs : CHILD_LIST_T = adaptorType.childObjects(parentObj, visitParams)
		for childData  in nextObjs:
			if visitParams.passChildDataObjects:
				yield childData
			else:
				yield childData.obj
			yield from self._iterRecursiveTopDownDepthFirst(
				#nextObj,
				childData,
				visitParams)

	def _iterRecursiveTopDownBreadthFirst(self,
	                                      #parentObj:T.Any,
	                                      parentObjData: CHILD_T,
	                                      visitParams:VisitPassParams,
	                                      )->T.Generator[CHILD_T, None, None]:
		"""iterate over all objects top-down"""
		parentKey, parentObj, parentChildType = parentObjData
		adaptorType = VisitAdaptor.adaptorForType(
			type(parentObj))
		assert adaptorType, f"no visit adaptor for type {type(parentObj)}"
		nextObjs : CHILD_LIST_T = adaptorType.childObjects(parentObj)
		for key, nextObj, data  in nextObjs:
			if visitParams.passChildDataObjects:
				yield key, nextObj, data
			else:
				yield nextObj
		for childData  in nextObjs:
			yield from self._iterRecursiveTopDownBreadthFirst(
				#nextObj,
				childData,
				visitParams)

	def _applyRecursiveTopDownDepthFirst(
			self,
			parentObj:T.Any,
			visitParams:VisitPassParams,
			)->T.Generator[tuple, None, None]:
		"""apply a function to all entries, yield input and result"""
		adaptorType : type[VisitAdaptor] = VisitAdaptor.adaptorForType(
			type(parentObj))
		assert adaptorType, f"no visit adaptor for type {type(parentObj)}"
		nextObjs : CHILD_LIST_T = adaptorType.childObjects(
			parentObj,
			params=visitParams
		)
		for childData  in nextObjs:
			visitData = VisitObjectData(
				base=parentObj,
				# visitResult=None,
				# childType=childType,
				# key=key,
				visitPassParams=visitParams,
			)
			if visitParams.passChildDataObjects: # pass in childData if desired
				yield childData, visitParams.visitFn(
					childData, self, visitData)
			else:
				yield childData.obj, visitParams.visitFn(
					childData.obj, self, visitData)
			yield from self._applyRecursiveTopDownDepthFirst(childData.obj, visitParams)


	def _transformRecursiveTopDownDepthFirst(
			self,
			parentObjData:CHILD_T,

			visitParams:VisitPassParams,
			)->T.Any:
		"""transform all objects top-down"""

		key, parentObj, childType = parentObjData

		# transform
		visitData = VisitObjectData(
			base=parentObj,
			# visitResult=None,
			# childType=childType,
			visitPassParams=visitParams,
			#key=key,
		)
		result = visitParams.visitFn(
			parentObj, self, visitData)
		if result is None:
			return result
		#print("result", result)

		# get child objects
		adaptor = VisitAdaptor.adaptorForType(type(result))
		assert adaptor, f"no visit adaptor for type {type(result)}"
		nextObjs : CHILD_LIST_T = VisitAdaptor.adaptorForType(
			type(result)).childObjects(result, visitParams)
		resultObjs = []
		#for nextObj, childType in nextObjs:
		for childData in nextObjs:
			# transform child objects
			resultObjs.append(
				(key, self._transformRecursiveTopDownDepthFirst(
					childData,
				 visitParams),
				 childType)
				 )
		# create new object from transformed child objects
		adaptor = VisitAdaptor.adaptorForType(
			type(result))
		try:
			newObj = adaptor.newObj(result, resultObjs, visitParams)
		except Exception as e:
			print("error making new object:")
			print("base", parentObj, type(parentObj))
			print("result", result, type(result))
			print("resultObjs", resultObjs)
			raise e

		#print("newObj", newObj)
		return newObj

	def _transformRecursiveBottomUpDepthFirst(
			self,
			parentObjData:CHILD_T,
			visitParams:VisitPassParams,
			#childType:ChildType.T=None
			)->T.Any:
		"""transform all objects top-down"""
		key, parentObj, childType = parentObjData
		#log("transform", parentObjData, visitParams)
		# transform
		visitData = VisitObjectData(
			base=parentObj,
			# visitResult=None,
			# childType=childType,
			# key=key,
			visitPassParams=visitParams,
		)
		nextObjs : CHILD_LIST_T = VisitAdaptor.adaptorForType(
			type(parentObj)).childObjects(parentObj, visitParams)
		# get child objects
		resultObjs = []
		for childData in nextObjs:
			resultObjs.append(
				(key, self._transformRecursiveBottomUpDepthFirst(
					childData,
					visitParams),
									 childType)
				 )
			#print("resultObjs", resultObjs)
		#log("resultObjs", resultObjs)
		adaptor = VisitAdaptor.adaptorForType(parentObj)
		try:
			#log("call newobj", parentObj, resultObjs)
			newObj = adaptor.newObj(parentObj, resultObjs, visitParams)
		except Exception as e:
			print("Cannot make new object:")
			print("base", parentObj)
			print("resultObjs", resultObjs)
			raise e

		# if this is the top level we skip, if desired
		if not visitParams.visitRoot:
			if parentObj is visitParams.rootObj:
				return newObj

		# transform object
		result = visitParams.visitFn(
			newObj, self, visitData)
		#log("visit tf result", result, type(result))
		return result





	def dispatchPass(self,
	                 fromObj:T.Any,
	                 passParams:VisitPassParams,
	                 #visitFnTemplate:visitFnType=None,
	                 **kwargs
	                 ):
		"""dispatch a single pass of the visitor
		"""

		if passParams.transformVisitedObjects and passParams.visitFn is None:
			raise ValueError("Cannot transform objects without a visit function")

		passParams.rootObj = passParams.rootObj or fromObj

		# check signature of visit function
		if passParams.visitFn is not None:
			pureFnSuccess, data = wfunction.checkFunctionSpecsMatch(
				visitFnTemplate,
				passParams.visitFn,
				allowExtensions=True
			)
			if not pureFnSuccess:
				raise TypeError(f"visit function {passParams.visitFn}\n{inspect.getsource(passParams.visitFn)}"
				                f"at {getDefinitionStrLink(passParams.visitFn)}\n"
				                f"must match template function signature\n{inspect.getfullargspec(visitFnTemplate).args}\n"
				                "failures", data)



		baseChildData = ChildData("", fromObj, None)
		# if no function, just iterate over stuff
		if passParams.visitFn is None:
			return self._iterRecursiveTopDownDepthFirst(
				#fromObj,
				baseChildData,
				passParams)


		self.checkVisitFnSignature(passParams.visitFn)

		passParams.visitKwargs = passParams.visitKwargs or {}
		passParams.visitKwargs.update(kwargs)

		if not passParams.transformVisitedObjects: # apply function over structure
			return self._applyRecursiveTopDownDepthFirst(
				baseChildData,
				passParams
			)

		# transform and return a new structure

		# switch top-down / bottom-up
		if passParams.topDown:
			return self._transformRecursiveTopDownDepthFirst(
				baseChildData,
				passParams,
			)
		else: # bottom-up
			return self._transformRecursiveBottomUpDepthFirst(
				baseChildData,
				passParams,
				)


if __name__ == '__main__':

	pass



	visitor = DeepVisitor(
	)

	structure = {
		"key1": "value1",
		(2, 4, "fhffhs"): ["value2", [], 3, 4, 5],
		"key3": "value3",
	}
	yieldPass = visitor.dispatchPass(
		structure,
		passParams=VisitPassParams()
	)

	for i in yieldPass:
		#print("yield", i)
		pass

	# apply print function
	def printArgsVisit(obj, visitor, visitObjectData:VisitObjectData):
		#print(obj, visitor, visitData, visitParams)
		return obj

	visitPass = visitor.dispatchPass(
		structure,
		VisitPassParams(visitFn=printArgsVisit)
	)
	for old, new in visitPass:
		#print("visited", old, new)
		pass

	#raise
	def addOneTransform(obj, visitor, visitObjectData):
		#print("addOneTransform", obj)
		if isinstance(obj, int):
			obj += 1
		return obj

	visitor = DeepVisitor()

	structure = [
		1, [2, [3, 4], 2], 1
	]
	print("structure ", structure)
	newStructure = visitor.dispatchPass(structure, VisitPassParams(
		transformVisitedObjects=False,
		visitFn=addOneTransform
	)) # generator, since transform is False
	# print("newStructure", newStructure)
	# for i in newStructure:
	# 	print("new", i)
	# 	pass

	newStructure = visitor.dispatchPass(structure, VisitPassParams(
		transformVisitedObjects=True,
		visitFn=addOneTransform
	))
	print("new struct", newStructure)

	print("structure ", structure)
	newStructure = visitor.dispatchPass(structure, VisitPassParams(
		transformVisitedObjects=True,
		topDown=False,
		visitFn=addOneTransform
	))
	print("new struct", newStructure)



	pass






