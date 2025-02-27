#pragma once



#include <cstdint>
#include <memory>

#include "../MInclude.h"
#include "../api.h"
#include "../stratacore/manifold.h"
#include "../stratacore/opgraph.h"

#include "strataMayaLib.h"


class StrataGraphNode : public MPxNode {
	/* master maya node representing single strata
	op graph - output its pointer address in memory as long
	NOPE - turns out a pointer in c++ won't fit in a long, and has to be
	a special uintptr_t integer type

	can't find any resources on splitting it into 2 integers, so to connect later
	nodes, to this one, we'll use the normal bool balancewheel method
	*/


public:
	StrataGraphNode() {}
	virtual ~StrataGraphNode() {}



	static MTypeId kNODE_ID;// = const MTypeId(0x00122C1C);
	static MString kNODE_NAME;// = MString("curveFrame");

	// single opgraph for this object
	// statewise, each individual op node should first check that
	// unique pointer apparently models ownership - hopefully safer than raw pointers everywhere
	std::shared_ptr<ed::StrataOpGraph> opGraph;

	// map of opIndex -> maya node
	std::unordered_map<int, MObjectHandle> indexMObjHandleMap;

	// attribute MObjects
	static MObject aStGraphName;
	static MObject aStGraph;


	static void* creator() {
		StrataGraphNode* newObj = new StrataGraphNode();
		return newObj;
	}

	static MStatus initialize();

	virtual void postConstructor();

	virtual MStatus compute(const MPlug& plug, MDataBlock& data);

	MStatus connectionMade(const MPlug& plug,
		const MPlug& otherPlug,
		bool 	asSrc
	);

	MStatus connectionBroken(const MPlug& plug,
		const MPlug& otherPlug,
		bool 	asSrc
	);

	MStatus legalConnection(const MPlug& plug,
		const MPlug& otherPlug,
		bool 	asSrc,
		bool& isLegal
	)		const;

	template <typename T>
	static MStatus checkLegalConnectionFromStrataGraphNode(
		const MPlug& plug,
		const MPlug& otherPlug,
		bool 	asSrc,
		bool& isLegal
	) {
		/* check if the incoming plug is the strataGraph connection -
		* if so, check that the incoming node is a StrataGraphNode

		asSrc	is this plug a source of the connection
		the docs and argument names around plug connection direction are riddles

		return kSuccess if fine, or if this plug is not the graph plug
		*/
		if (plug.attribute() != T::aStGraph) {
			return MS::kSuccess;
		}// this is the graph input plug
		
		// check that incoming plug is the stGraph bool connection from StrataGraphNode
		if (otherPlug.attribute() != StrataGraphNode::aStGraph) {
			return MS::kFailure;
		}
		return MS::kSuccess;
	}
	
	// convenience method for other nodes to get the connected graph node
	static MStatus getConnectedStrataOpGraph(
		MObject& thisNodeObj, MObject& graphIncomingConnectionAttr,
		std::weak_ptr<ed::StrataOpGraph>& opGraphPtr
	);
};


