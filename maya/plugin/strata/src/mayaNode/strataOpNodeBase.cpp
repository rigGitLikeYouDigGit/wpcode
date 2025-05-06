
#include <stdlib.h>
#include <cstdlib>
#include "../macro.h"
#include "../api.h"
#include "strataOpNodeBase.h"
#include "../lib.h"


using namespace ed;


// use this to check that we've defined the attribute MObjects once -
// otherwise we just add them to the new nodes
int StrataOpNodeBase::strataAttrsDefined = 0;


//DEFINE_STRATA_STATIC_MOBJECTS(StrataOpNodeBase);

//DEFINE_STATIC_NODE_CPP_MEMBERS(STRATABASE_STATIC_MEMBERS, StrataOpNodeBase);


//
//template <typename StrataOpT>
//void StrataOpNodeBase<StrataOpT>::testNoTemplateFn() {}
//
////template<class StrataOpT>
////void StrataOpNodeBase<StrataOpT>::testFn() {}
//
////template<typename NodeT>
////void StrataOpNodeBase::testFn<NodeT>() {}
//
//template<typename StrataOpT, typename NodeT>
//void StrataOpNodeBase<StrataOpT>::testFn() {}
//
//template<typename StrataOpT, typename NodeT>
//void StrataOpNodeBase<StrataOpT>::testFn<NodeT>() {}
//
//template<typename StrataOpT> 
//template<typename NodeT>
//void StrataOpNodeBase<StrataOpT>::testFn<NodeT>() {}
//
//hahahahahahahahahaahaaaaha kill me


//template<typename StrataOpT>
//const std::string StrataOpNodeTemplate<StrataOpT>::getOpNameFromNode(MObject& nodeObj) {

//template <typename NodeT>
//const int StrataOpNodeBase::getOpIndexFromNode(const MObject& nodeObj) {
//	/* return a default name for strata op -
//	if nothing defined in string field, use name of node itself*/
//	MFnDependencyNode depFn(nodeObj);
//	MStatus s;
//	MPlug opNameFieldPlug = depFn.findPlug(NodeT::aStOutput, false, &s);
//	if (s.error()) {
//		DEBUGS("error getting op index field for node " + depFn.name());
//		return -1;
//	}
//	
//	return opNameFieldPlug.asInt(); // NB - this might cause loops - if so, don't put it in driven.
//}


//MStatus StrataOpNodeBase::setOpIndexOnNode(MDataBlock& data, int index) {
//	MStatus s;
//	data.outputValue(aStOutput).setInt(-1);
//	data.outputValue(aStOutput).setInt(index);
//	return s;
//}



MStatus StrataOpNodeBase::setFreshGraph(MObject& nodeObj//, MDataBlock& data
) {
	MS s;
	DEBUGS("base set fresh graph")
	opGraphPtr = std::make_shared<StrataOpGraph>();
	// need to extend in templated class to add a new node to the graph
	return s;
}
MStatus StrataOpNodeBase::setFreshGraph(MObject& nodeObj, MDataBlock& data) {
	MS s;
	DEBUGS("base set fresh graph")
		opGraphPtr = std::make_shared<StrataOpGraph>();
	// need to extend in templated class to add a new node to the graph
	return s;
}



