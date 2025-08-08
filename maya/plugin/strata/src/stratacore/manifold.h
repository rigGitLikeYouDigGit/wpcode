
#pragma once

#include <memory>
#include <map>
#include <array>
#include <unordered_set>
#include <set>
#include <unordered_map>
#include <string>
#include <vector>
#include <algorithm>
//#include <cstdint>

#include <Eigen/Dense>
#include <unsupported/Eigen/MatrixFunctions>
#include <unsupported/Eigen/Splines>

#include <maya/MVector.h>
#include <maya/MMatrix.h>

#include "../bezier/bezier.h"

#include "wpshared/enum.h"
#include "../macro.h"
#include "../status.h"
#include "../containers.h"
#include "../api.h"
#include "../lib.h"
#include "../libEigen.h"

#include "../logger.h"
/*
smooth topological manifold, made of points, edges and partial edges

elements may have a parent element, which should form a dag graph

operations might come in here too - for example, a later stage of the
graph might override, or extend a previous patch


we have the dag of influence / operation order, operating on and
generating the dg of the data structure -

examining the system at one layer, should return the state of the mesh at that moment

edges to store curvature and tangent ACROSS them at various points? for c2 smoothness


split the mesh components into topology and rich data - one part leads to structural modification,
the other is just changing float values
LATER
since it doesn't matter yet

hopefully we don't need too intense optimisations in the structure, since we're only talking on the order of thousands of elements


FOR NOW, because I can't work out how to do the path stuff, have a separate copy of
the whole structure for each generation of the operation graph

using ints not pointers in topo types for easier copying, serialising if it comes to it etc



for interaction through maya, consider setting up attributes in a map, and entering the attr name you want to affect on the maya node?


*/

namespace ed {

	/* eventually we might use some kind of interning,
	so trying to alias name strings right now
	to make it easier to update later */
	using StrataName = std::string;

	using namespace Eigen;

//int a = seqIndex(-5, 3);

// don't need full int here but MAYBE in the long run, we'll need to make an enum attribute in maya for it
	//BETTER_ENUM(SElType, int, point, edge, face);

	enum SElType : short{ point, edge, face};

	struct StrataManifold;

	int constexpr stMaxParents = 3;

	int constexpr ST_TARGET_MODE_LOCAL = 0;
	int constexpr ST_TARGET_MODE_GLOBAL = 1;
	//int MATRIX_MODE_LOCAL_O

	int constexpr ST_EDGE_DENSE_NPOINTS = 10;

	//struct ExpValue;

	struct SElement {
		// whenever an element is added to manifold, during graph eval, 
		// that element is IMMUTABLE from then on, within that version of the graph
		// so this system contains absolutely no live behaviour, only static connections and positions


		StrataName name;
		SElType elType = SElType::point;
		// ANOTHER DAY, ANOTHER INABILITY TO USE CONST

		int elIndex = -1; // index within this element's type - the 3rd point, the 3rd face etc
		int globalIndex = -1; // unique global index across all elements
		//std::vector<int> drivers; // topological drivers, not parent spaces
		//std::vector<int> parents; // weighted parent influences

		std::vector<StrataName> drivers; // topological drivers, not parent spaces
		std::vector<StrataName> spaces; // weighted parent influences

		std::vector<std::string> opHistory;

		/* i think we need name vectors for these instead
		*/

		std::vector<int> edges; // edges that draw from this element
		std::vector<int> points; // points that draw from this element
		std::vector<int> faces; // faces that use this edge as a rib or boundary, or pass through this point
		

		// todo: bit fields
		// todo: have all flags indicate a reason an element SHOULDN'T appear?
		bool isInactive = false; // should this element (face) be rendered/meshed?
		bool isInvalid = false; // does this element have an error in its history?
		std::string errorMsg;

		// face attributes for winding and orientation - 
		// structural to result manifold, so put it here instead of in data
		std::vector<bool> edgeOrients; // SURELY vector<bool> is cringe
		// true for forwards, false for backwards
		bool flipFace = 0; // if face as a whole should be flipped, after edge winding

		SElement(StrataName elName, const SElType t=SElType::point) : name(elName), elType(t) {
			//name = elName;
			//elType = t;
		};

		inline bool hasDrivers() { return (drivers.size() > 0); }
		inline bool hasParents() { return (spaces.size() > 0); }

		// neighbours do not include child / driven elements
		// topology functions only return raw ints from these structs - 
		// manifold is needed to get rich pointers

		// ...should these return std::set instead? uggggh

		std::vector<int> otherNeighbourPoints(StrataManifold& manifold) {}
		std::vector<int> otherNeighbourEdges(StrataManifold& manifold) {}
		std::vector<int> otherNeighbourFaces(StrataManifold& manifold, bool requireSharedEdge = true) {}
		/// topological sets - after Keenan Crane
		// star is the set of all edges directly touching this element in it
		// for faces this might include intersectors, not sure yet
		std::vector<int> edgeStar(StrataManifold& manifold) {
		}
		// link is the set of all edges exactly one edge away from this element
		std::vector<int> edgeLink(StrataManifold& manifold) {
		}
		// do the same for other element types


		inline std::vector<int> intersectingElements(StrataManifold& manifold) {
			/* this likely will be a very wide function -
			if we want to define elements and expressions in a more topological way,
			we need a general sense of intersection between elements -

			2 edges that cross are intersecting, regardless of which one is driven
			an edge that helps define a face intersects that face -
				and the expression (face n edge) gives the sub-edge touching only the face.

			maintain an order here if possible - list edge intersections in the direction of this edge,
				list face intersections in the face's winding order, etc
			*/
		}

	};

	/*
	for full separation, we might break things up further?


	an edge is driven by SEdgeDriverData -
	outputs of an edge are EdgeSampleData? - 
	*/

	struct SSampleData { //?
		int index = -1;
		float uvn[3] = { 0, 0, 0 };
	};

	struct SElData {
		int index = -1;
		std::string creatorNode;
	};

	struct SPointDriverData {
	/* may not be entirely irrelevant*/ 
		int index = -1;
	};

	// parent datas always relative in parent space - when applied, recover the original shape of element
	struct SPointSpaceData { // parent data FOR a point, driver could be any type
		//int index = -1;
		std::string name; // name of parent space element
		// has to be robust to storing/retrieving between graph iterations
		float weight = 1.0;
		Vector3f uvn = { 0, 0, 0 }; // uvn separate to offset in case point goes outside parent space area - 
		// eg if point goes off edge of space surface
		Affine3f offset = Eigen::Affine3f::Identity(); // translation is UVN, rotation is relative rotation from that position

		std::string strInfo();
	};

	struct SPointData : SElData {
		SPointDriverData driverData;
		std::vector<SPointSpaceData> spaceDatas = {}; // datas for each driver
		//MMatrix finalMatrix = MMatrix::identity; // final evaluated matrix in world space
		Eigen::Affine3f finalMatrix = Eigen::Affine3f::Identity(); // final evaluated matrix in world space
		//std::vector<std::string> nodeHistory; // each node that has affected this point, starting with creator

		std::string strInfo();
	};


	// data for discrete hard connections between elements
	struct SEdgeDriverData {
		/* struct for a single driver OF an edge - get tangent, normal and twist
		vectors for curve frame

		// do we just use NAN values to show arrays being unused?

		TODO: tension, param pinning etc?

		tangents are held in driver, not parent, since they affect 

		*/
		int index = -1; // index of parent element
		Eigen::Vector3f uvn = { 0, 0, 0 }; // uvn coords of parent element to sample for points on this edge
		//float tan[3] = { NAN, 0, 0 }; // tangent of curve at this point - if left NAN is unused

		// tangents normally inline, unless continuity is not 1

		// tangents should be local to final matrix
		// they're NOT, they're GLOBAL for now, it was too complicated for first version

		Eigen::Vector3f baseTan = { 0, 0, 0 }; // vector from prev to next point
		Eigen::Vector3f prevTan = { -1, 0, 0 }; // tangent leading to point
		Eigen::Vector3f postTan = { 1, 0, 0 }; // tangent after point
		float normal[3] = { NAN, 0, 1 }; // normal of curve at this point - if left NAN is unused
		float orientWeight = 0.0; // how strongly matrix should contribute to curve tangent, vs auto behaviour
		float continuity = 1.0; // how sharply to break tangents - maybe just use this to scale tangents in?
		float twist = 0.0; // how much extra twist to add to point, on top of default curve frame
		Eigen::Affine3f finalMatrix = Eigen::Affine3f::Identity();

		inline Eigen::Vector3f pos() { return finalMatrix.translation(); }

		inline Status& syncMatrix() {
			/* update final matrix from local driver tangents, offsets etc
			*/
		}
	};

	// DRIVER datas are in space of the DRIVER
	// PARENT datas are in space of the PARENT
	// convert between them by multiplying out to world and back - 
	// parent datas always overlap drivers at some point

	/* so to generate a full curve,
	for each separate parent, we transpose driver points and vectors
	into that parent's space,
	then do the spline operations, get a dense vector of UVN parametres, and save that dense data as an SEdgeParentData.
	maybe we also cache the final result in world space.

	this is done on the op that first creates the edge and creates its data.

	on later iterations of the graph,
	if when a parent changes, we re-evaluate the UVNs of each SEdgeParentData.

	*/
	struct SEdgeParentData// : StaticClonable<SEdgeParentData> 
	{
		using thisT = SEdgeParentData;
		using T = SEdgeParentData;
		//DECLARE_DEFINE_CLONABLE_METHODS(thisT)

		int index = -1; // feels cringe to copy the index on all of these  
		// TEEECHNICALLLY this should be independent of any driver - 
		Eigen::ArrayXf weights; // per-dense-point weights for this parent
		Eigen::ArrayX3f cvs; // UVN bezier control points - ordered {pt, tanOut, tanIn, pt, tanOut...} etc
		bez::CubicBezierPath parentCurve; // curve in UVN space of parent, used for final interpolation

		Eigen::MatrixX3f finalNormals; // worldspace normals // hopefully smoothstep interpolation is good enough

		inline bez::ClosestPointSolver* closestSolver() {
			return parentCurve.getSolver();
		}

		void initEmpty() {
			/*initialise variables */
			weights = Eigen::ArrayXf();
			cvs = Eigen::ArrayX3f();
			//parentCurve = bez::CubicBezierPath();
			finalNormals = MatrixX3f();
		}
	};


	struct SEdgeData : SElData//, StaticClonable<SEdgeData> 
	{
		/* need dense final result to pick up large changes in
		parent space
		*/
		using thisT = SEdgeData;
		using T = SEdgeData;
		std::vector<SEdgeDriverData> driverDatas; // drivers of this edge
		//std::array<SEdgeParentData, stMaxParents> parentDatas; // curves in space of each driver
		std::vector<SEdgeParentData> parentDatas; // curves in space of each driver
		
		//int denseCount = 10; // number of dense sub-spans in each segment
		/* TODO: adaptive by arc length? adaptive by screen size?
		*/

		bool closed = false;
		/* don't keep live splines, output from parent system etc - 
		all temporary during construction
		posSpline is FINAL spline of all points on this edge
		*/

		Eigen::ArrayX3d uvnOffsets = {}; // final dense offsets should only be in space of final built curve?
		// maybe???? 

		//// IGNORE FOR NOW
		/// brain too smooth
		// surrender to ancestors
		// become caveman

		//Eigen::MatrixX3d finalPositions; // dense worldspace positions
		bez::CubicBezierPath finalCurve; // dense? final curve // DENSE
		Eigen::MatrixX3f finalNormals = {}; // worldspace normals 

		Eigen::MatrixX3f finalPoints = {}; // densely sampled final points in worldspace - use for querying
		
		int _bufferStartIndex = -1;

		//DECLARE_DEFINE_CLONABLE_METHODS(thisT)


		/*inline bez::ClosestPointSolver* closestSolver() {
			return finalCurve.getSolver();
		}*/

		inline bool isClosed() const {
			if (!driverDatas.size()) {
				return false;
			}
			return driverDatas[0].finalMatrix.translation().isApprox(
				driverDatas.back().finalMatrix.translation());
		}

		inline Eigen::Vector3f samplePos(const float t) {
			/* sample curve at 
			*/
		}

		inline int densePointCount() {
			/* point count before resampling - 
			curve has point at each driver, and (segmentPointCount) points
			in each span between them*/
			//return static_cast<int>(driverDatas.size() + denseCount * (driverDatas.size() - 1));
			if (isClosed()) {
				return static_cast<int>(ST_EDGE_DENSE_NPOINTS * (driverDatas.size()));
			}
			return static_cast<int>(ST_EDGE_DENSE_NPOINTS * (driverDatas.size() - 1));
		}

		inline int densePointCount() const {
			/* point count before resampling -
			curve has point at each driver, and (segmentPointCount) points
			in each span between them*/
			//return static_cast<int>(driverDatas.size() + denseCount * (driverDatas.size() - 1));
			if (isClosed()) {
				return static_cast<int>(ST_EDGE_DENSE_NPOINTS * (driverDatas.size()));
			}
			return static_cast<int>(ST_EDGE_DENSE_NPOINTS * (driverDatas.size() - 1));
		}

		inline int nSpans() {
			if (isClosed()) {
				return static_cast<int>(driverDatas.size());
			}
			return static_cast<int>(driverDatas.size()) - 1;
		}

		inline int nCVs() {
			// number of all cvs including tangent points
			return static_cast<int>((driverDatas.size()) * 3);
		}
		inline int nBezierCVs() {
			// number of cvs in use with bezier curves - basically shaving off start and end
			return static_cast<int>((driverDatas.size() - 1) * 3 + 2);
		}

		inline void rawBezierCVs(Eigen::Array3Xf& arr) {
			// ARRAY MUST BE CORRECTLY SIZED FIRST from nBezierCVs()
			
			//arr.resize(nBezierCVs());
			for (int i = 0; i < driverDatas.size(); i++) {
				if (i != 0) {
					arr.row(i * 3 - 1) = driverDatas[i].pos() + driverDatas[i].prevTan;
				}
				arr.row(i * 3) = driverDatas[i].pos();

				if (i != driverDatas.size() - 1) {
					arr.row(i * 3 + 1) = driverDatas[i].pos() + driverDatas[i].postTan;
				}
			}
		}

		inline void driversForSpan(const int spanIndex, SEdgeDriverData& lower, SEdgeDriverData& upper) {
			lower = driverDatas[spanIndex];
			upper = driverDatas[spanIndex + 1];
		}
		Status& buildFinalBuffers(Status& s);

	};





	struct SFaceData : SElData {
		//std::string name; // probably generated, but still needed for semantics?
	};


	
	/// ATTRIBUTES
	// surely there's a different, correct way to do this?
	// damn I wish I was clever

	struct StrataAttr {
		// maybe attributes should track their own names?
	};
	template<typename VT>
	struct DenseStrataAttr : StrataAttr{
		typedef VT VT;
		std::vector<VT> vals;
	};

	template<typename VT>
	struct SparseStrataAttr : StrataAttr{
		typedef VT VT;
		std::unordered_map<int, VT> vals;
	};

	template<typename VT>
	struct StrataArrayAttr : StrataAttr {
		typedef VT VT;
		std::vector<std::vector<VT>> vals;
	};

	template<typename VT>
	struct SparseStrataArrayAttr : StrataAttr{ // is there a good way to prevent the multiplying here?
		typedef VT VT;
		std::unordered_map<int, std::vector< VT > > vals;
	};

	struct StrataGroup { // maybe groups can track their own element types?
		std::vector<int> contents;
		StrataGroup() {};
		StrataGroup(int size) {
			contents.reserve(size);
			// why can you reserve an unsorted set but not a sorted one?
			///// :)))))) thank you c++
		};
		
		// we allocate new memory for set operations - cringe
		// need a decent span-like container, and decent set behaviour for it
		inline std::unordered_set<int> asSet() {
			return std::unordered_set<int>(contents.begin(), contents.end());
		}
	};

	struct StrataManifold {
		/*

		data store just accumulates across graph iterations until something changes it - 
		also lets you set direct atomic deltas against el names, regardless of 
		whether they exist at that moment or not - probably quite good

		memory coherence?
		no. we don't do that here

		could use topo connections to mark elements in manifold dirty, but not sure of the point, for now
		
		*/

		std::vector<SElement> elements;

		std::unordered_map<std::string, int> nameGlobalIndexMap = {}; // everyone point and laugh 
		std::map<int, int> pointIndexGlobalIndexMap = {};
		std::map<int, int> edgeIndexGlobalIndexMap = {};
		std::map<int, int> faceIndexGlobalIndexMap = {};

		std::unordered_map<std::string, SPointData> pDataMap = {};
		std::unordered_map<std::string, SEdgeData> eDataMap = {};
		std::unordered_map<std::string, SFaceData> fDataMap = {};

		// world matrix transform of this manifold
		Affine3f worldMat = Affine3f::Identity();

		// cached indices for drawing buffers
		int _nEdgeVertexBufferEntries = -1;
		int _edgeVertexBufferEntryStart = -1;

		inline SElData* elData(int globalElId, SElType elT) {
			switch (elT) {
			//case SElType::point: return &pointDatas[pointIndexGlobalIndexMap[globalElId]];
			case SElType::point: {
				//return &pointDatas[el.elIndex];
				auto ptr = pDataMap.find(getEl(globalElId)->name);
				if (ptr == pDataMap.end()) { return nullptr; }
				return &(ptr->second);
				break;
			}
			//case SElType::edge: return &edgeDatas[edgeIndexGlobalIndexMap[globalElId]];
			case SElType::edge: {
				auto ptr = eDataMap.find(getEl(globalElId)->name);
				if (ptr == eDataMap.end()) { return nullptr; }
				return &(ptr->second);
				break;
			}
			case SElType::face: {
				auto ptr = fDataMap.find(getEl(globalElId)->name);
				if (ptr == fDataMap.end()) { return nullptr; }
				return &(ptr->second);
				break;
			}
			default: return nullptr;
			}
		}
		inline SElData* elData(int globalElId) {
			return elData(globalElId, elements[globalElId].elType);
		}
		//inline SElData* elData(std::string elName) {
		//	return elData(globalElId, elements[globalElId].elType);
		//}
		inline SElData* elData(SElement& el) {
			switch (el.elType) {
			case SElType::point: {
				//return &pointDatas[el.elIndex];
				auto ptr = pDataMap.find(el.name);
				if (ptr == pDataMap.end()) {return nullptr;	}
				return &(ptr->second);
				break;
			}
			//case SElType::edge: return &edgeDatas[el.elIndex];
			case SElType::edge: {
				auto ptr = eDataMap.find(el.name);
				if (ptr == eDataMap.end()) { return nullptr; }
				return &(ptr->second);
				break;
			}
			case SElType::face: {
				auto ptr = fDataMap.find(el.name);
				if (ptr == fDataMap.end()) { return nullptr; }
				return &(ptr->second);
				break;
			}
			default: return nullptr;
			}
		}

		// ATTRIBUTES 
		// unsure if wrapping this is useful - for now EVERYTHING explicit
		std::unordered_map<std::string, StrataAttr> attrs;
		std::unordered_map<std::string, StrataGroup> groups;


		void clear() {
			// is it better to just make a new object?

			elements.clear();
			nameGlobalIndexMap.clear();
			//pointDatas.clear();
			pDataMap.clear();
			eDataMap.clear();
			fDataMap.clear();

			attrs.clear();
			groups.clear();

			initAttrs();
		}

		void initAttrs() {
			/* set up core attrs we always expect to find and be populated - 
			currently only path.
			do we put name here too?*/
			StrataArrayAttr<int> path;
			attrs["path"] = path;
		}

		StrataManifold() {
			// only run init when no copy source is given
			initAttrs();
		}


		inline SElement* getEl(const int& globalId) {
			if (globalId >= elements.size()) {
				return nullptr;
			}
			return &elements[globalId];
		}

		inline SElement* getEl(const std::string name) {
			if (!nameGlobalIndexMap.count(name)) {
				return nullptr;
			}
			return &elements[nameGlobalIndexMap[name]];
		}


		SmallList<SElement*> getEls(SmallList<int>& globalIds) {
			SmallList<SElement*, globalIds.MAXSIZE> result;
			result.reserve(globalIds.size());
			for (int gId : globalIds) {
				result.push_back(getEl(gId));
			}
			return result;
		}

		std::vector<SElement*> getEls(std::vector<int>& globalIds) {
			std::vector<SElement*> result;
			result.reserve(globalIds.size());
			for (int gId : globalIds) {
				result.push_back(getEl(gId));
			}
			return result;
		}

		
		SElData* setElData(SElement* el, SElData* data) {
			/* absolutely no idea on how best to do these uniform interfaces for #
			setting data of different types*/
			switch (el->elType) {
			case SElType::point: {
				//pointDatas[el->elIndex] = *static_cast<SPointData*>(data);
				pDataMap[el->name] = *static_cast<SPointData*>(data);
				//return &pointDatas[el->elIndex];
				return &pDataMap.at(el->name);
			}
			case SElType::edge: {
				eDataMap[el->name] = *static_cast<SEdgeData*>(data);
				return &eDataMap.at(el->name);
			}
			case SElType::face: {
				fDataMap[el->name] = *static_cast<SFaceData*>(data);
				return &fDataMap[el->name];
			}
			default: return nullptr;
			}
		}

		Status addElement(
			SElement& el,
			SElement*& outPtr,
			bool allowOverride=false
		) {
			LOG("addElement from other");
			Status s;
			if (nameGlobalIndexMap.find(el.name) != nameGlobalIndexMap.end()) {
				if (!allowOverride) {
					l("laready exists, erroring");
					outPtr = nullptr;

					STAT_ERROR(s, "Name " + el.name + " already found in manifold, halting");
					//return nullptr;
				}
				else {
					outPtr = getEl(el.name);
					if (outPtr->elType != el.elType) {
						l("tried to add other element not matching original type - very illegal");
						outPtr = nullptr;

						STAT_ERROR(s, "El name " + el.name + "tried to override original element of different type");

					}
					return s;
				}
			}
			int globalIndex = static_cast<int>(elements.size());
			elements.push_back(el);
			SElement* elP = &(elements[globalIndex]);
			elP->globalIndex = globalIndex;
			nameGlobalIndexMap[el.name] = globalIndex;

			// get element-specific index map, add element data
			switch (el.elType) {
				case SElType::point: {
					//pointDatas.push_back(SPointData());
					pDataMap.insert({ el.name, SPointData() });
					int elementIndex = static_cast<int>(pointIndexGlobalIndexMap.size()); // get current max key of element set
					elP->elIndex = elementIndex;
					pointIndexGlobalIndexMap[elementIndex] = globalIndex;
					break;
				}
				case SElType::edge: {
					//edgeDatas.push_back(SEdgeData());
					eDataMap.insert({ el.name, SEdgeData() });
					int elementIndex = static_cast<int>(edgeIndexGlobalIndexMap.size());
					elP->elIndex = elementIndex;
					edgeIndexGlobalIndexMap[elementIndex] = globalIndex;
					break;
				}
				case SElType::face: { 
					fDataMap.insert({ el.name, SFaceData() });
					int elementIndex = static_cast<int>(faceIndexGlobalIndexMap.size());
					elP->elIndex = elementIndex;
					faceIndexGlobalIndexMap[elementIndex] = globalIndex;
					break;
				}
			}
			outPtr = elP;
			return s;
		}

		Status addElement(
			const std::string name,
			const SElType elT,
			SElement*& outPtr
		) {
			return addElement(SElement(name, elT), outPtr);
		}

		//bool windFaceEdgeOrients(SElement* face) {
		//	/* for each driver edge of a face,
		//	* check direction of each driver edge, add its
		//	* state to the direction vector
		//	*/
		//	face->edgeOrients.clear();

		//	std::vector<SElement*> visitedDrivers;
		//	int prevEnds[2] = { NULL, NULL }; // edges or points
		//	// if one of these is an edge, is it guaranteed to show up in 
		//	// face's own edges?
		//	// not always, consider edges from a 5-pole - 
		//	// seems degenerate to have one child edge come off a parent at u=0,
		//	// that's just a point with extra steps, but maybe it could happen?

		//	for (int driverGId : face->drivers) {
		//		SElement* driver = getEl(driverGId);
		//		// skip anything not an edge
		//		if (driver->elType != SElType::edge) {
		//			continue;
		//		}
		//		// check if this is the first edge visited
		//		if (prevEnds[0] == NULL) {
		//			prevEnds[0] = driver->drivers[0];
		//			prevEnds[1] = driver->drivers[seqIndex(-1, driver->drivers.size())];
		//			// use this edge as reference for forwards
		//			face->edgeOrients.push_back(true);
		//			visitedDrivers.push_back(driver);
		//			continue;
		//		}

		//		// check for the same edge included twice in a face - by definition one
		//		// time would be forwards, one would be backwards.
		//		// this should also catch degenerate cases like loops, 
		//		// where an edge starts and ends at the same point
		//		if (seqContains(visitedDrivers, driver)) {
		//			ptrdiff_t visitedIndex = std::distance(visitedDrivers.begin(), std::find(visitedDrivers.begin(), visitedDrivers.end(), driver));
		//			bool prevState = face->edgeOrients[visitedIndex];
		//			if (prevState) { // previously the edge was the correct orientation
		//				// so now flip it
		//				prevEnds[0] = visitedDrivers[visitedIndex]->drivers[1];
		//				prevEnds[1] = visitedDrivers[visitedIndex]->drivers[0];
		//				face->edgeOrients.push_back(false);
		//			}
		//			else {// previously the edge was backwards
		//				// so now un-flip it
		//				prevEnds[0] = visitedDrivers[visitedIndex]->drivers[0];
		//				prevEnds[1] = visitedDrivers[visitedIndex]->drivers[1];
		//				face->edgeOrients.push_back(true);
		//			}
		//			visitedDrivers.push_back(driver);
		//			continue;
		//		}


		//		// check for easy point-matching case first
		//		if (prevEnds[1] == driver->drivers[0]) { // all good
		//			prevEnds[0] = driver->drivers[0];
		//			prevEnds[1] = driver->drivers[1];
		//			face->edgeOrients.push_back(true);
		//			visitedDrivers.push_back(driver);
		//			continue;
		//		}

		//		if (prevEnds[1] == driver->drivers[1]) { // edge is backwards
		//			prevEnds[0] = driver->drivers[1];
		//			prevEnds[1] = driver->drivers[0];
		//			face->edgeOrients.push_back(false);
		//			visitedDrivers.push_back(driver);
		//			continue;
		//		}

		//		// for edges coming from edges, the driver may literally be one of the previous ends
		//		/* i think this gets really complicated, need to
		//		work out the orientation of the drivers of the driver edge?

		//		"all edges start and end with a point" is a nice rule, let's use it
		//		*/
		//		//if (prevEnds[1] == driver->globalIndex) {
		//		//	prevEnds[0] = driver->globalIndex;
		//		//	prevEnds[1] = driver->drivers[1]; //???????
		//		//	face->edgeOrients.push_back(false);
		//		//	continue;
		//		//}

		//	}
		//}

		bool edgesAreContiguousRing(std::vector<int>& edgeIds) {
			/* might be more c++ to have this more composable -
			maybe have a general check to see if any elements are contiguous between them?
			then some way to filter that it should only check edges, and only move via edges?

			check that each edge contains the next in sequence in its neighbours

			this is quite a loose check actually, if you try and fool it, I guess you can?

			*/
			for (size_t i = 0; i < edgeIds.size(); i++) {
				size_t nextI = seqIndex(i + 1, edgeIds.size());
				int nextEdgeId = edgeIds[nextI];
				SElement* thisEdge = getEl(edgeIds[i]);

				// if next edge is not contained in neighbours, return false
				if (!seqContains(thisEdge->otherNeighbourEdges(*this), nextEdgeId)) {
					return false;
				}
			}
			return true;

		}


		StrataAttr* getAttr(std::string& name) {
			// convenience to get a pointer, handle casting yourself at point of use
			// check pointer is valid
			if (!attrs.count(name)) {
				return nullptr;
			}
			return &(attrs[name]);
		}

		StrataGroup* getGroup(std::string& name, bool create = false, int size = 4) {
			// can't create attrs as easily as groups, so only this gets the default flag
			if (!groups.count(name)) {
				if (!create) {
					return nullptr;
				}
				// create new group if it doesn't exist
				groups[name] = StrataGroup(size);
			}
			return &(groups[name]);
		}


		/*
		merging
		*/

		static constexpr int MERGE_OVERWRITE = 0;
		static constexpr int MERGE_LEAVE = 1;

		Status mergeOther(StrataManifold& other, int mergeMode, Status& s) {
			/*given another manifold, merge it into this one
			* if names are found, update according to merge mode - 
			*	MERGE_OVERWRITE - overwrite this graph's data with matching names in other
			*	MERGE_LEAVE - leave matching names as they are
			* 
			*
			*/
			
			// add any elements not already known by name
			LOG("graph MERGE OTHER");
			l("this graph: " + printInfo());
			l("other graph: " + other.printInfo());
			for (auto& otherEl : other.elements) {
				SElement* newEl;
				// if element not found, add it and copy over its data directly
				if (nameGlobalIndexMap.find(otherEl.name) == nameGlobalIndexMap.end()) {
					addElement(otherEl, newEl);
					setElData(newEl, other.elData(otherEl));
					continue;
				}
				// element found already - do we overwrite?
				newEl = getEl(otherEl.name);
				if (mergeMode == MERGE_OVERWRITE) {
					setElData(newEl, other.elData(otherEl));
				}
			}
			l("after merge, this graph: " + printInfo());
			return s;
		}



		/*
		spatial functions
		static for less coupling to this specific graph's buffers
		*/

		void transform(Affine3f& mat) {
			/* transform entire contents of manifold by given matrix - 
			we still transform final results of elements with drivers,
			this alone mutates a strata manifold
			all our relative offsets are still valid

			it might be worth later on caching all driverless data like this,
			but for now brute force it

			Strata is immutable, EXCEPT for this
			special-case it, and invert transform during backpropagation, wherever it's applied
			*/
			for (auto& el : elements) {
				//if (el.drivers.size()) { // skip elements with drivers
				//	continue;
				//}
				switch (el.elType) {
				case SElType::point: {
					SPointData& data = pDataMap[el.name];
					data.finalMatrix = mat * data.finalMatrix;
					break;
				}
				case SElType::edge: {
					SEdgeData& data = eDataMap[el.name];
					data.finalCurve.transform(mat);
					for (int i = 0; i < static_cast<int>(data.finalNormals.rows()); i++) {
						data.finalNormals.row(i) = mat * Vector3f(data.finalNormals.row(i));
					}
					//data.finalNormals = mat * data.finalNormals;
					break;
				}
				}

			}
		
		}

		static Status& pointPosAt(Status& s, Eigen::Vector3f& out, const SPointData& d, const Eigen::Vector3f& uvn) {
			out = d.finalMatrix * Eigen::Vector3f(uvn);
			return s;
		}
		static Status& edgePosAt(Status& s, Eigen::Vector3f& out, const SEdgeData& d, const Eigen::Vector3f& uvn) {
			/* as above, but just return position -
			may allow faster sampling in future
			
			UVN is (curve param, rotation from normal, distance from curve)
			*/

			// check if we need full matrix
			if (EQ(uvn[1], 0.0f) && EQ(uvn[2], 0.0f)) {
				if (d.parentDatas.size()) {
					out = d.parentDatas[0].parentCurve.eval(uvn[0]);
					return s;
				}
				out = d.finalCurve.eval(uvn[0]);
				return s;
			}
			Eigen::Affine3f curveMat;
			s = edgeDataMatrixAt(s, curveMat, d, uvn);
			out = curveMat.translation();
			return s;
		}

		Status& posAt(Status& s, Eigen::Vector3f& out, int globalIndex, Eigen::Vector3f& uvn)  {
			/* as above, but just return position -
			may allow faster sampling in future*/
			SElement* el = getEl(globalIndex);
			switch (el->elType) {
			case SElType::point: {
				//SPointData& d = pointDatas.at(el->elIndex);
				SPointData& d = pDataMap.at(el->name);
				return pointPosAt(s, out, d, uvn);
				break;
			}
			case SElType::edge: {
				SEdgeData& d = eDataMap.at(el->name);
				return edgePosAt( s, out, d, uvn);
				break;
			}
			}
			return s;
		}
		
		static Status& pointMatrixAt(Status& s, Eigen::Affine3f& out, const SPointData& d, const Eigen::Vector3f& uvn){
			LOG("pointMatrixAt: ");
			out = d.finalMatrix;
			//out.translate(uvn);
			out.translation() = out * uvn;
			return s;
		}
		static inline Status& edgeDataMatrixAt(Status& s, Eigen::Affine3f& out, const SEdgeData& d, const Eigen::Vector3f& uvn
		) {/* if we don't cache a final dense curve for edge data,
			we would have to eval all parents here, then blend between them.

			we could also output the exact curve position? to save a sample?
			*/

			// first make frame
			Vector3f pos = d.finalCurve.eval(uvn(0));
			Vector3f tan = d.finalCurve.tangentAt(uvn(0), pos);
		
			// get normals
			int a; int b; 
			float t = getArrayIndicesTForU(static_cast<int>(d.finalNormals.size()), uvn(0), a, b);
			Vector3f normal = lerp<Vector3f>(d.finalNormals.row(a), d.finalNormals.row(b), smoothstepCubic(t));

			// make base frame
			s = makeFrame(s, out, pos, 
				tan.normalized(), 
				normal.normalized()
			);
			out.translation() = pos;

			// make angle axis describing twist around X
			AngleAxisf baseAA(uvn(1), Vector3f::UnitX());
			out.rotate(baseAA);
			// translate out along twisted normal in Z
			out.translate(Vector3f::UnitZ() * uvn(2));
			
			return s;
		}
		Status& matrixAt(Status& s, Eigen::Affine3f& outMat, SElement* el, Eigen::Vector3f uvn) {
			/* interpolate a spatial element to get a matrix in world space - 
			look up UVN on given element, return that matrix
			*/
			LOG("matrixAt: " + el->name +" " + str(uvn));
			if (el == nullptr) {
				l("el is null");
				outMat = Affine3f::Identity();
				return s;
			}
			//SElement* el = getEl(globalIndex);
			switch (el->elType) {
				case (SElType::point): {
					//SPointData& d = pointDatas[el->elIndex];
					SPointData& d = pDataMap.at(el->name);
					return pointMatrixAt(s, outMat, d, uvn);
				}
				case (SElType::edge): {
					SEdgeData& d = eDataMap[el->name];
					//return edgeMatrixAt(s, out, el->elIndex, uvn);
					return edgeDataMatrixAt(s, outMat, d, uvn);
				}
				default: STAT_ERROR(s, "Cannot eval matrix at UVN for type " + std::to_string(el->elType));
			}
			return s;
		}
		// am i overdoing the copium, or is this way of dispatching by type quite good?

		Status& pointClosestMatrix(Status& s, Eigen::Affine3f& outMat, SPointData& d, const Eigen::Vector3f& worldVec) {
			//outMat = pointDatas[elIndex].finalMatrix;
			outMat = d.finalMatrix;

			return s;
		}
		Status& edgeClosestMatrix(Status& s, Eigen::Affine3f& outMat, SElement* el, const Eigen::Vector3f& worldVec) {
			/* localise matrix by point parent -
			how do we handle local rotations?
			get nearest point to curve
			*/
			//SEdgeData& d = edgeDatas[elIndex];
			SEdgeData& d = eDataMap[ el->name ];

			float u;
			Eigen::Vector3f tan;
			Eigen::Vector3f pos = d.finalCurve.ClosestPointToPath(
				bez::WorldSpace(worldVec.data()),
				d.finalCurve.getSolver(), u,
				tan)
				;
			
			Eigen::Vector3f normal = lerpSampleMatrix<float, 3>(d.finalNormals, u);

			s = makeFrame<float>(s, outMat, pos, tan, normal);
			
			return s;
		}
		Status& edgeClosestMatrix(Status& s, Eigen::Affine3f& outMat, SElement* el, const Eigen::Affine3f& worldMat) {
			return edgeClosestMatrix(s, outMat, el, worldMat.translation());
		}
		Status& closestMatrix(Status& s, Eigen::Affine3f& outMat, SElement* el, const Eigen::Vector3f closePos) {
			// localise a world transform into UVN coordinates in the space of given parent
			// make another function to return a full transform, for point parents
			if (el == nullptr) {
				outMat = Affine3f::Identity();
				return s;
			}
			//SElement* el = getEl(globalIndex);
			switch (el->elType) {
			case (SElType::point): {
				return pointClosestMatrix(s, outMat, pDataMap.at(el->name), closePos);
			}
			case (SElType::edge): {
				return edgeClosestMatrix(s, outMat, el, closePos);
			}
			default: STAT_ERROR(s, "Cannot eval matrix at UVN for type " + std::to_string(el->elType));
			}
			return s;
		}

		Status& pointGetUVN(Status& s, Eigen::Vector3f& outUVN, SPointData& d, const Eigen::Vector3f worldPos) {
			/* return UVN displacement from point matrix
			*/
			//SPointData& d = pointDatas[elIndex];
			//outUVN = (d.finalMatrix.inverse() * worldMat).translation();
			LOG("pointGetUVN");
			outUVN = (d.finalMatrix.inverse() * worldPos);
			return s;
		}

		
		Status& edgeGetUVN(Status& s, Eigen::Vector3f& uvn, SElement* el, const Eigen::Vector3f& worldVec) {
			/* NEED POLAR / CYLINDRICAL conversion for UVN 
			* 
			* for blending, we should probably blend along shortest paths positive or negative, 
			or everything will move in spirals
			but let's save that for when we actually do blending
			*/

			// first get closest matrix on curve
			Eigen::Affine3f curveMat;
			s = edgeClosestMatrix(s, curveMat, el, worldVec);

			//SEdgeData& d = edgeDatas[elIndex];
			SEdgeData& d = eDataMap[el->name];

			float u;
			Eigen::Vector3f tan;
			Eigen::Vector3f pos = d.finalCurve.ClosestPointToPath(
				bez::WorldSpace(worldVec.data()),
				d.finalCurve.getSolver(), u,
				tan)
				;
			uvn(0) = u;
			
			Eigen::Vector3f normal = lerpSampleMatrix<float, 3>(d.finalNormals, u);

			s = makeFrame<float>(s, curveMat, pos, tan, normal);

			uvn(1) = getAngleAroundAxis(
				curveMat * Eigen::Vector3f(0, 0, 1),
				curveMat * Eigen::Vector3f(0, 1, 0),
				(worldVec - curveMat.translation()).normalized()
			);
			uvn(2) = (worldVec - curveMat.translation()).norm();
			return s;
		}


		Status& getUVN(Status& s, Eigen::Vector3f& uvn, SElement* el, const Eigen::Vector3f closePos) {
			// localise a world transform into UVN coordinates in the space of given parent
			// make another function to return a full transform, for point parents
			
			switch (el->elType) {
			case (SElType::point): {
				return pointGetUVN(s, uvn, pDataMap.at(el->name), closePos);
			}
			case (SElType::edge): {
				return edgeGetUVN(s, uvn, el, closePos);
			}
			default: STAT_ERROR(s, "Cannot get UVN for type " + std::to_string(el->elType));
			}
			return s;
		}

		Status& pointSpaceMatrix(Status& s, Affine3f& outMat, SPointData& data);

		Status& computePointData(Status& s, SPointData& data//, bool doProjectToDrivers=false
		);

		Status& pointProjectToDrivers(Status& s, Affine3f& mat, SElement* el);

		Status& edgeParentDataFromDrivers(Status& s, SEdgeData& eData, SEdgeParentData& pData);

		Status& buildEdgeDrivers(Status& s, SEdgeData& eData);

		Status& buildEdgeData(Status& s, SEdgeData& eData);

		Status& buildPointData(Status& s, SPointData& eData) {
			/* construct final dense array for data, assuming all parents and driver indices are set in data

			build base curve matrices in worldspace,
			then get into space of each driver

			but we can only work in worldspace when curve is freshly added, otherwise
			we can only save driver-space versions
			*/

			// Bezier control points for each span

			

			return s;
		}


		Status getIntersection(const SElement& elA, const SElement& elB, MMatrix& matOut, bool& crossFound) {
			/* return a single point where elements intersect.
			- What if there are multiple intersections? some kind of iterator over intersections? no idea
			- should there be some kind of richer Coordinate struct - with element index and UVN to get to point?
			*/

		}


		///////////////////
		/// collate/flatten final state of manifold for drawing - deformation as dense data
		static int constexpr CURVE_SHAPE_RES = 20; // how many points to sample each curve?
		// surely we'll have to break this up for longer edges

		inline Float3Array getPointPositionArray(Status& s) {
			/* flattened float array of positions for all points
			I suppose this would be where struct of arrays wins over struct of arrays
			TODO: struct of arrays
			*/
			Float3Array result(static_cast<int>(pDataMap.size()));
			int i = 0;
			for (auto& p : pointIndexGlobalIndexMap) {
				result[i] = pDataMap.at(getEl(p.second)->name).finalMatrix.translation().data();
				i += 1;
			}
			return result;
		}

		template<typename arrT>
		inline arrT getPointPositionArray(Status& s) {
			/* flattened float array of positions for all points
			I suppose this would be where struct of arrays wins over struct of arrays
			TODO: struct of arrays
			*/
			arrT result(static_cast<int>(pDataMap.size()));
			int i = 0;
			for (auto& p : pointIndexGlobalIndexMap) {
				result[i] = pDataMap.at(getEl(p.second)->name).finalMatrix.translation().data();
				i += 1;
			}
			return result;
		}


		//inline Float3Array getWireframeVertexPositionArray(Status& s);
		//	/* return flat float3 array for vector positions on points and curves
		//	* 
		//	* order as [point positions, dense curve positions]
		//	* 
		//	* each point has 4 coords - point itself, and then 0.1 units in x, y, z of that point
		//	*/

		int getWireframePointGnomonVertexPositionLength() {
			return static_cast<int>(pDataMap.size()) * 4;
		}
		Float3Array getWireframePointGnomonVertexPositionArray(Status& s);
		Status& getWireframePointGnomonVertexPositionArray(Status& s, Float3* outArr, int startIndex);
		IndexList getWireframePointIndexArray(Status& s);

		Status& getWireframeSingleEdgeGnomonVertexPositionArray(Status& s, Float3Array& outArr, SElement* el, int arrStartIndex);

		//Status& setWireframeSingleEdgeVertexPositionArray(Status& s, Float3Array& outArr, SElement* el, int arrStartIndex);

		int getWireframeEdgeVertexPositionLength() {
			int result = 0;
			for (auto& p : eDataMap) {
				result += p.second.densePointCount();
			}
			return result;

		}
		Float3Array getWireframeEdgeVertexPositionArray(Status& s);
		Status& getWireframeEdgeVertexPositionArray(Status& s, Float3* outArr, int startIndex);


		IndexList getWireframeEdgeVertexIndexList(Status& s);

		void setGnomonIndexList(unsigned int* result, unsigned int i);

		Float3Array getWireframeEdgeGnomonVertexPositionArray(Status& s);

		IndexList getWireframeEdgeGnomonVertexIndexList(Status& s);


		std::string printInfo() {
			std::string result = "<manifold - nPts: " + str(pDataMap.size()) + ", nEdges: " + str(eDataMap.size()) + " >";
			return result;
		}
		/*TODO: proper dump for everything - every driver of every element.
		might as well just serialise things at that point.
		*/

		/////////////

		static inline bool _isNotAlnum(char c)
		{
			return !std::isalnum(c);
		}
		static Status validateElName(const std::string& elName) {
			Status s;
			if (std::find_if(elName.begin(), elName.end(), _isNotAlnum) == elName.end()) {
				return s;
			}
			STAT_ERROR(s, "element name: " + elName + " contains invalid characters, must only be alphanumeric");
			
		}

		
	};

}


