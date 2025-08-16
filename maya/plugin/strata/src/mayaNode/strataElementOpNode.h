#pragma once


#include "../MInclude.h"
#include "strataMayaLib.h"
#include "../strataop/elementOp.h"
#include "strataOpNodeBase.h"

#include "../exp/expParse.h"

/*
element node allows either defining AND/OR RETRIEVING strata data for any
kind of elements in sequence

if input connections are given in maya, we define the data;
else we can take from the "out" attributes to display the data in transforms, nurbs etc - 

loop the output of these maya nodes back in, and connect the transform or curve shape message to
"fitTarget" etc, and we can look back through the graph and remove any local deltas, reproject
shape if resolution changes?

although it might still be more correct to do a full pull->edit->push model?

at some point we will always have to scrub local CV edits from a curve in Maya


LOWERING RESOLUTION
then
INCREASING RESOLUTION in moving UI slider
on curve will 
PERMANENTLY LOSE SHAPE DATA
and we can't undo it, because Maya will have lost the deltas?
hmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm

ok got it, kind of -
simply never SHORTEN the saved array of curve points.
changing resolution gives the best-fitting interpolation
of new points against saved
changing shape just sets those more dense points to best fit of lower-res target?

then a separate one-off to trim data to exact curve result





DRIVER != PARENT

element DRIVERs are discrete element connected as topology - a point driving another point acts as a parent constraint,
a point driving an edge defines a point always on that edge, etc

element PARENT provides full SPACE for entire element - smooth changes to parent are propagated through 
child elements.
so using a face as a parent, any elements are defined within the metric space of that face - edges by default are geodesics, points by default follow the nearest point on the surface, etc

for sanity's sake, only allow one parent for now

*/


/// experiment for making declaring / defining attr mobjects less painful - working well

/* for strata elements, different element types need different attribute datas - 
* 
* point - 
*	finalDriverMatrix - final weighted "home" matrix from all this point's drivers
*	finalLocalOffsetMatrix - final weighted local offset from this point's driver matrix
	finalWorldMatrix - final matrix result of point, only thing used in subsequent elements

	also need data for parent coords + individual parent offset

	point parent - no coords, only offset
	
	edge parent - 
		edge point data
			edgeParam // float attribute bounded at 0

			// for simplicity just define all this parametrisation stuff on the curve by default
			overrideCurveMetric 0-1  // should point define a custom parametrisation for the curve?
			normalizeZeroOne 0-1
			useLength 0-1
			reverse 0-1
			fromEnd 0-1

	face parent - 
		faceParam // UVW vector parametre for where the point is on the given face
		// no idea what else this should be , leave for now

edge - 
	EDGES ALWAYS PASS THROUGH DRIVERS EXACTLY.
	there, solved whole lot of issues

	// explicit separate drivers for start and end?

	// driver entries matched by global index? driver index within edge? driver name?

	for each intermediate curve point - 
	offset vector
	normal vector? so that curve up-frame can always match normal of matched surface
	sure, sounds good, go
	ignore normal for now


	for now only point drivers
	for each driver:
		driver id
		driver weight (other than ends, controls how strongly edge passes through matrix)
		driver tangent vector
		driver up vector
		driver twist
		final driver out param // output final parametre on curve of each driver transform
		
	
	// pin array - override final params at certain points, for bunching of driven transforms
		pin param // override final param value
		pin strength // how strongly this pin should affect 
		pin power // linear, quadratic, etc


face -

	only edge drivers - MAYBE a maximum of 1 driver point, to specify where peak of surface should
		pass through, but even that might be too unstable
	


output element indices as array, in case expressions create multiple?

if this will also be used to edit existing elements,
do we need a blend on every single attribute, to say if it should be overridden?
probably, right?
*/

# define STRATAELEMENTOPNODE_STATIC_MEMBERS(prefix, nodeT) \
prefix MObject nodeT aStElement;\
prefix MObject nodeT aStName;\
prefix MObject nodeT aStDriverExp;\
prefix MObject nodeT aStSpaceExp;\
\
prefix MObject nodeT aStDriverWeightIn;\
prefix MObject nodeT aStMatchWorldSpaceIn;\
\
prefix MObject nodeT aStPointWorldMatrixIn;\
prefix MObject nodeT aStPointDriverLocalMatrixIn;\
\
prefix MObject nodeT aStEdgeCurveIn; \
\
prefix MObject nodeT aStElementOut;\
prefix MObject nodeT aStNameOut;\
prefix MObject nodeT aStGlobalIndex;\
prefix MObject nodeT aStElTypeIndex;\
prefix MObject nodeT aStTypeOut;\
prefix MObject nodeT aStPointDriverMatrixOut; \
prefix MObject nodeT aStPointWeightedDriverMatrixOut; \
prefix MObject nodeT aStPointWeightedLocalOffsetMatrixOut; \
prefix MObject nodeT aStPointFinalWorldMatrixOut; \
prefix MObject nodeT aStEdgeCurveOut;\
\



/*
\
prefix MObject nodeT aStFitTransform;\
prefix MObject nodeT aStFitCurve;\


\
prefix MObject nodeT aStEdgeResolution;\
prefix MObject nodeT aStEdgeNormaliseParam;\
prefix MObject nodeT aStEdgeUseLength;\
prefix MObject nodeT aStEdgeReverse;\
prefix MObject nodeT aStEdgeStartIndex;\
prefix MObject nodeT aStEdgeStartName;\
prefix MObject nodeT aStEdgeStartTwist;\
prefix MObject nodeT aStEdgeEndIndex;\
prefix MObject nodeT aStEdgeEndName;\
prefix MObject nodeT aStEdgeEndTwist;\
prefix MObject nodeT aStEdgeMid;\
prefix MObject nodeT aStEdgeMidIndex;\
prefix MObject nodeT aStEdgeMidName;\
prefix MObject nodeT aStEdgeMidTwist;\
\
prefix MObject nodeT aStFaceDriver;\
prefix MObject nodeT aStFaceDriverIndex;\
prefix MObject nodeT aStFaceDriverName;\

*/


/* got some weird crashes when MPxNode wasn't the first base, so
it looks like we need to redeclare all the strata overridden methods
in the final classes - 
that's fine*/

//class StrataElementOpNode : public StrataOpNodeTemplate<strata::StrataElementOp>, public MPxNode {
//class StrataElementOpNode;
class StrataElementOpNode : public MPxNode, public StrataOpNodeTemplate<strata::StrataElementOp> {
public:
	using thisStrataOpT = strata::StrataElementOp;
	using superT = StrataOpNodeTemplate<strata::StrataElementOp>;
	using thisT = StrataElementOpNode;
	StrataElementOpNode() {}
	virtual ~StrataElementOpNode() {}

	static void* creator() {
		StrataElementOpNode* newObj = new StrataElementOpNode();
		return newObj;
	}

	DECLARE_STATIC_NODE_H_MEMBERS(STRATABASE_STATIC_MEMBERS);
	DECLARE_STATIC_NODE_H_MEMBERS(STRATAELEMENTOPNODE_STATIC_MEMBERS);

	//virtual void postConstructor();

	//static MStatus legalConnection(
	//	const MPlug& plug,
	//	const MPlug& otherPlug,
	//	bool 	asSrc,
	//	bool& isLegal
	//);

	static MTypeId kNODE_ID;// = const MTypeId(0x00122C1C);
	static MString kNODE_NAME;// = MString("curveFrame");

	static  MString     drawDbClassification;
	static  MString     drawRegistrantId;


	static MStatus initialize();

	MStatus StrataElementOpNode::edgeDataFromRawCurve(MStatus& ms, MObject& nodeObj, MDataBlock& data, MDataHandle& elDH, strata::SEdgeData& eData);

	virtual MStatus syncStrataParams(MObject& nodeObj, MDataBlock& data, strata::StrataOp* opPtr, strata::StrataOpGraph* graphPtr);

	virtual MStatus compute(const MPlug& plug, MDataBlock& data);

	//virtual MStatus setDependentsDirty(const MPlug& plugBeingDirtied,
	//	MPlugArray& affectedPlugs
	//);


	// override base class static strata objects, so each leaf class still has attributes
	// initialised separately to the base
	//DECLARE_STRATA_STATIC_MEMBERS;

	/*DECLARE_STATIC_NODE_MEMBERS(
		STRATAADDPOINTSOPNODE_STATIC_MEMBERS)*/

	void postConstructor();

	MStatus legalConnection(
		const MPlug& plug,
		const MPlug& otherPlug,
		bool 	asSrc,
		bool& isLegal
	) const;

	virtual MStatus connectionMade(const MPlug& plug,
		const MPlug& otherPlug,
		bool 	asSrc
	);

	virtual MStatus connectionBroken(const MPlug& plug,
		const MPlug& otherPlug,
		bool 	asSrc
	);

};

