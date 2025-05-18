
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

#include "wpshared/enum.h"
#include "../macro.h"
#include "../status.h"
#include "../containers.h"
#include "../api.h"
#include "../lib.h"
#include "../libEigen.h"
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


using vectors for everything, I don't care for now

// do we actually need separate types for elements? not for most things

NO WE DON'T

no SPoints, SEdges - only SElements, with int enum telling their type


*/

namespace ed {

	/* eventually we might use some kind of interning,
	so trying to alias name strings right now
	to make it easier to update later */
	using StrataName = std::string;

//int a = seqIndex(-5, 3);

// don't need full int here but MAYBE in the long run, we'll need to make an enum attribute in maya for it
	//BETTER_ENUM(StrataElType, int, point, edge, face);

	enum StrataElType : short{ point, edge, face};

	struct StrataManifold;

	int constexpr stMaxParents = 3;

	struct SElement {
		// whenever an element is added to manifold, during graph eval, 
		// that element is IMMUTABLE from then on, within that version of the graph
		// so this system contains absolutely no live behaviour, only static connections and positions


		StrataName name;
		StrataElType elType = StrataElType::point;
		// ANOTHER DAY, ANOTHER INABILITY TO USE CONST

		int elIndex = -1; // index within this element's type - the 3rd point, the 3rd face etc
		int globalIndex = -1; // unique global index across all elements
		//std::vector<int> drivers; // topological drivers, not parent spaces
		//std::vector<int> parents; // weighted parent influences

		std::vector<StrataName> drivers; // topological drivers, not parent spaces
		std::vector<StrataName> parents; // weighted parent influences

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

		SElement(StrataName elName, const StrataElType t=StrataElType::point) : name(elName), elType(t) {
			//name = elName;
			//elType = t;
		};


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


	an edge is driven by EdgeDriverData -
	outputs of an edge are EdgeSampleData? - 
	*/

	struct SSampleData { //?
		int index = -1;
		float uvn[3] = { 0, 0, 0 };
	};

	struct SElData {
		int index = -1;
	};


	// parent datas always relative in parent space - when applied, recover the original shape of element
	struct SPointParentData { // parent data FOR a point, driver could be any type
		int index = -1;
		float weight = 1.0;
		float uvn[3] = { 0, 0, 0 }; // if parent is a point, this is just the literal vector in that point's space
		// also this need only be an orient matrix
		float orientMatrix[9] = {
			1, 0, 0,
			0, 1, 0,
			0, 0, 1
		};
	};

	struct SPointData : SElData {
		std::array<SPointParentData, stMaxParents> parentDatas; // datas for each driver
		//MMatrix finalMatrix = MMatrix::identity; // final evaluated matrix in world space
		Eigen::Affine3d finalMatrix = Eigen::Affine3d::Identity(); // final evaluated matrix in world space
	};


	// data for discrete hard connections between elements
	struct EdgeDriverData {
		/* struct for a single driver OF an edge - get tangent, normal and twist
		vectors for curve frame

		// do we just use NAN values to show arrays being unused?

		TODO: tension, param pinning etc?

		tangents are held in driver, not parent, since they affect 

		*/
		int index = -1; // index of parent element
		double uvn[3] = { 0, 0, 0 }; // uvn coords of parent element to sample for points on this edge
		//float tan[3] = { NAN, 0, 0 }; // tangent of curve at this point - if left NAN is unused

		// tangents normally inline, unless continuity is not 1

		// tangents should be local to final matrix
		// they're NOT, they're GLOBAL for now, it was too complicated for first version

		Eigen::Vector3d baseTan = { 0, 0, 0 }; // vector from prev to next point
		Eigen::Vector3d prevTan = { -1, 0, 0 }; // tangent leading to point
		Eigen::Vector3d postTan = { 1, 0, 0 }; // tangent after point
		float normal[3] = { NAN, 0, 1 }; // normal of curve at this point - if left NAN is unused
		float orientWeight = 0.0; // how strongly matrix should contribute to curve tangent, vs auto behaviour
		double continuity = 1.0; // how sharply to break tangents - maybe just use this to scale tangents in?
		float twist = 0.0; // how much extra twist to add to point, on top of default curve frame
		Eigen::Affine3d finalMatrix = Eigen::Affine3d::Identity();

		inline Eigen::Vector3d pos() { return finalMatrix.translation(); }
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
	struct SEdgeParentData {
		int index = -1; // feels cringe to copy the index on all of these  
		// TEEECHNICALLLY this should be independent of any driver - 
		// re-sampling UVNs on any random parent should transform this curve into that element's space
		std::vector<float> weights;
		std::vector<std::array<float, 3>> uvns;
		//std::vector<float> twists;
	};


	struct SEdgeData : SElData {
		/* need dense final result to pick up large changes in
		parent space
		*/
		std::vector<EdgeDriverData> driverDatas; // drivers of this edge
		std::array<SEdgeParentData, stMaxParents> parentDatas; // curves in space of each driver
		int denseCount = 10; // number of dense sub-points in each segment
		Eigen::Spline3d posSpline;// worldspace dense positions of curve
		Eigen::Spline3d normalSpline; // worldspace dense positions of curve
		//Eigen::MatrixX3d finalPositions; 
		//Eigen::MatrixX3d finalNormals; // worldspace normals

		inline bool isClosed() {
			return driverDatas[0].finalMatrix.translation().isApprox(
				driverDatas.back().finalMatrix.translation());
		}

		inline int densePointCount() {
			/* point count before resampling - 
			curve has point at each driver, and (segmentPointCount) points
			in each span between them*/
			return static_cast<int>(driverDatas.size() + denseCount * (driverDatas.size() - 1));
		}

		inline int nSpans() {
			return static_cast<int>(driverDatas.size()) - 1;
		}

		inline void driversForSpan(const int spanIndex, EdgeDriverData& lower, EdgeDriverData& upper) {
			lower = driverDatas[spanIndex];
			upper = driverDatas[spanIndex + 1];
		}
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
		manifold will have to include more live graph behaviour - 
		whole point is to set up persistent relationships between elements,
		so changing an input by default updates the whole mesh.

		it's not insurmountable.
		consider storing 
		[ element index starting group] ,
		[ node index to eval up to ]

		
		*/

		std::vector<SElement> elements;

		std::unordered_map<std::string, int> nameGlobalIndexMap; // everyone point and laugh 
		std::map<int, int> pointIndexGlobalIndexMap;
		std::map<int, int> edgeIndexGlobalIndexMap;
		std::map<int, int> faceIndexGlobalIndexMap;


		std::vector<SPointData> pointDatas;
		std::vector<SEdgeData> edgeDatas;
		std::vector<SFaceData> faceDatas;


		inline SElData* elData(int globalElId, StrataElType elT) {
			switch (elT) {
			case StrataElType::point: return &pointDatas[pointIndexGlobalIndexMap[globalElId]];
			case StrataElType::edge: return &edgeDatas[edgeIndexGlobalIndexMap[globalElId]];
			case StrataElType::face: return &faceDatas[faceIndexGlobalIndexMap[globalElId]];
			default: return nullptr;
			}
		}
		inline SElData* elData(int globalElId) {
			return elData(globalElId, elements[globalElId].elType);
		}
		inline SElData* elData(SElement& el) {
			switch (el.elType) {
			case StrataElType::point: return &pointDatas[el.elIndex];
			case StrataElType::edge: return &edgeDatas[el.elIndex];
			case StrataElType::face: return &faceDatas[el.elIndex];
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
			pointDatas.clear();
			edgeDatas.clear();
			faceDatas.clear();

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
			case StrataElType::point: {
				pointDatas[el->elIndex] = *static_cast<SPointData*>(data);
				return &pointDatas[el->elIndex];
			}
			case StrataElType::edge: {
				edgeDatas[el->elIndex] = *static_cast<SEdgeData*>(data);
				return &edgeDatas[el->elIndex];
			}
			case StrataElType::face: {
				faceDatas[el->elIndex] = *static_cast<SFaceData*>(data);
				return &faceDatas[el->elIndex];
			}
			default: return nullptr;
			}
		}

		Status addElement(
			SElement& el,
			SElement*& outPtr
		) {
			Status s;
			if (nameGlobalIndexMap.count(el.name)) {
				STAT_ERROR(s, "Name " + el.name + " already found in manifold, halting");
				//return nullptr;
			}
			int globalIndex = static_cast<int>(elements.size());
			elements.push_back(el);
			SElement* elP = &(elements[globalIndex]);
			elP->globalIndex = globalIndex;
			nameGlobalIndexMap[el.name] = globalIndex;

			// get element-specific index map, add element data
			switch (el.elType) {
				case StrataElType::point: {
					pointDatas.push_back(SPointData());
					int elementIndex = static_cast<int>(pointIndexGlobalIndexMap.size()); // get current max key of element set
					elP->elIndex = elementIndex;
					pointIndexGlobalIndexMap[elementIndex] = globalIndex;

				}
				case StrataElType::edge: {
					edgeDatas.push_back(SEdgeData());
					int elementIndex = static_cast<int>(edgeIndexGlobalIndexMap.size());
					elP->elIndex = elementIndex;
					edgeIndexGlobalIndexMap[elementIndex] = globalIndex;
				}
				case StrataElType::face: { 
					faceDatas.push_back(SFaceData());
					int elementIndex = static_cast<int>(faceIndexGlobalIndexMap.size());
					elP->elIndex = elementIndex;
					faceIndexGlobalIndexMap[elementIndex] = globalIndex;
				}
			}
			outPtr = elP;
			return s;
		}

		Status addElement(
			const std::string name,
			const StrataElType elT,
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
		//		if (driver->elType != StrataElType::edge) {
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
			*/
			
			// add any elements not already known by name
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
			return s;
		}



		/*
		spatial functions
		*/
		Status& pointPosAt(Status& s, Eigen::Vector3d& out, int elIndex, double uvn[3]) {
			SPointData& p = pointDatas[elIndex];
			//out = p.finalMatrix * MVector(uvn);			
			out = p.finalMatrix * Eigen::Vector3d(uvn);
			return s;
		}
		Status& edgePosAt(Status& s, Eigen::Vector3d& out, int elIndex, double uvn[3]) {
			/* as above, but just return position -
			may allow faster sampling in future
			
			UVN is (curve param, rotation from normal, distance from curve)
			*/
			
			SEdgeData& e = edgeDatas[elIndex];
			Eigen::Affine3d amat;
			s = splineUVN(s, amat, e.posSpline, e.normalSpline, uvn);
			//out = MVector(amat.translation().data());
			out = Eigen::Vector3d(amat.translation().data());
			return s;
		}
		Status& posAt(Status& s, Eigen::Vector3d& out, int globalIndex, double uvn[3]) {
			/* as above, but just return position -
			may allow faster sampling in future*/
			SElement* el = getEl(globalIndex);
			switch (el->elType) {
			case StrataElType::point: {
				return pointPosAt(s, out, el->elIndex, uvn);
				break;
			}
			case StrataElType::edge: {
				return edgePosAt( s, out, el->elIndex, uvn);
				break;
			}
			}
			return s;
		}
		
		Status& pointMatrixAt(Status& s, Eigen::Affine3d& out, int elIndex, double uvn[3]){
			SPointData& d = pointDatas[elIndex];
			out = d.finalMatrix;
			out.translate( Eigen::Vector3d(uvn));
			return s;
		}
		Status& edgeMatrixAt(Status& s, Eigen::Affine3d& out, int elIndex, double uvn[3]) {
			SEdgeData& d = edgeDatas[elIndex];
			s = splineUVN(s, out, d.posSpline, d.normalSpline, uvn);
			return s;
		}
		Status& matrixAt(Status& s, Eigen::Affine3d& out, int globalIndex, double uvn[3]) {
			/* interpolate a spatial element to get a matrix in world space*/
			SElement* el = getEl(globalIndex);
			switch (el->elType) {
				case (StrataElType::point): {
					return pointMatrixAt(s, out, el->elIndex, uvn);
				}
				case (StrataElType::edge): {
					return edgeMatrixAt(s, out, el->elIndex, uvn);
				}
				default: STAT_ERROR(s, "Cannot eval matrix at UVN for type " + std::to_string(el->elType));
			}
			return s;
		}
		// am i overdoing the copium, or is this way of dispatching by type quite good?

		Status& pointInSpaceOf(Status& s, double outUVN[3], int elIndex, Eigen::Affine3d worldMat) {
			/* localise matrix by point parent -
			how do we handle local rotations?
			*/
			SPointData& d = pointDatas[elIndex];
			auto uvn = (d.finalMatrix.inverse() * worldMat).translation().data();
			outUVN[0] = uvn[0];
			outUVN[1] = uvn[1];
			outUVN[2] = uvn[2];
			return s;
		}

		Status& edgeInSpaceOf(Status& s, double outUVN[3], int elIndex, Eigen::Affine3d worldMat) {
			/* get nearest point on spline to worldMat
			*/
			SEdgeData& d = edgeDatas[elIndex];
			double closeU = closestParamOnSpline(d.posSpline, worldMat.translation());
			Eigen::Affine3d curveMat;
			matrixAtU(s, curveMat, d.posSpline, d.normalSpline, closeU);
			auto uvn = (curveMat.inverse() * worldMat).translation().data();
			outUVN[0] = uvn[0];
			outUVN[1] = uvn[1];
			outUVN[2] = uvn[2];
			return s;
		}

		Status& inSpaceOf(Status& s, double outUVN[3], const int globalIndex, const Eigen::Affine3d worldMat) {
			// localise a world transform into UVN coordinates in the space of given parent
			// make another function to return a full transform, for point parents
			SElement* el = getEl(globalIndex);
			switch (el->elType) {
			case (StrataElType::point): {
				return pointInSpaceOf(s, outUVN, el->elIndex, worldMat);
			}
			case (StrataElType::edge): {
				return edgeInSpaceOf(s, outUVN, el->elIndex, worldMat);
			}
			default: STAT_ERROR(s, "Cannot eval matrix at UVN for type " + std::to_string(el->elType));
			}
			return s;
		}


		Status& buildEdgeParentData(Status& s, SEdgeData& eData) 
		{
			/* set up parent influences from matrix drivers
			* arrive at final spans in each parent space
			*/

			
		}

		// if IMMUTABLE
		//  HOW DO WE DO DRIVERS
		// check for existing data when element op is run?

		/*
		within single element op
		add face F
		add edge E, parent F
		
		on next iteration, need to 


		data "injection" chances at different stages? ie whenever element added, also put in hook for any incoming data?
		check for overrides
		add element / generate element in whichever way 
		THEN check for overrides?
		index by element -> parent -> string attribute ?
		

		data is data, only one
		reduce complexity by propagating back to node parametres?
		otherwise could be lost if element vanishes?

		*/

		MStatus weightEdgeDriverMatrix(std::vector<EdgeDriverData>& driverDatas, int targetIndex, MMatrix& out) {

		}

		Status& buildEdgeData(Status& s, SEdgeData& eData) {
			/* construct final dense array for data, assuming all parents and drivers
			are built
			
			build base curve matrices in worldspace,
			then get into space of each driver 

			but we can only work in worldspace when curve is freshly added, otherwise 
			we can only save driver-space versions
			*/

			// Bezier control points for each span
			std::vector<Eigen::Vector3d> spanPoints;
			spanPoints.resize(eData.nSpans() * 2);

			// set base matrices on all points
			for (int i = 0; i < static_cast<int>(eData.driverDatas.size()); i++) {
				// vector from prev ctl pt to next
				//eData.driverDatas[i].finalMatrix = matrixAt()
				matrixAt(s,
					eData.driverDatas[i].finalMatrix,
					eData.driverDatas[i].index,
					eData.driverDatas[i].uvn
				);
			}

			// set tangent vectors for each one (scaling done later)
			// also includes start and end, as if it were closed
			int nextI, prevI;
			for (int i = 0; i < static_cast<int>(eData.driverDatas.size()); i++) {
				nextI = (i + 1) % static_cast<int>(eData.driverDatas.size());
				prevI = (i - 1) % static_cast<int>(eData.driverDatas.size());

				EdgeDriverData& thisDriver = eData.driverDatas[i];
				EdgeDriverData& prevDriver = eData.driverDatas[prevI];
				EdgeDriverData& nextDriver = eData.driverDatas[nextI];

				// vector from prev ctl pt to next
				auto tanVec = nextDriver.pos() - prevDriver.pos();
				thisDriver.baseTan = tanVec;

				auto toThisVec = thisDriver.pos() - prevDriver.pos();
				// vector from this ctl pt to next
				Eigen::Vector3d nextVec = nextDriver.pos() - thisDriver.pos();

				// forwards tan scale factor
				double nextTanScale = (tanVec.dot(nextVec) - tanVec.dot(toThisVec)) / 3.0;
				nextTanScale = -sminQ(-nextTanScale, 0.0, 0.2);

				// back tan scale factor
				double prevTanScale = tanVec.dot(toThisVec) - (tanVec.dot(toThisVec)) / 3.0;
				prevTanScale = -sminQ(-prevTanScale, 0.0, 0.2);

				thisDriver.postTan = tanVec.normalized() * nextTanScale;
				thisDriver.prevTan = -tanVec.normalized() * prevTanScale;
			}

			// set ends if not continuous
			// if not closed, ends are not continuous
			if (!eData.isClosed()) {
				eData.driverDatas[0].continuity = 0.0;
				eData.driverDatas.back().continuity = 0.0;
			}

			// check continuity
			for (int i = 0; i < static_cast<int>(eData.driverDatas.size()); i++) {
				nextI = (i + 1) % static_cast<int>(eData.driverDatas.size());
				prevI = (i - 1) % static_cast<int>(eData.driverDatas.size());

				EdgeDriverData& thisDriver = eData.driverDatas[i];
				EdgeDriverData& prevDriver = eData.driverDatas[prevI];
				EdgeDriverData& nextDriver = eData.driverDatas[nextI];

				// blend between next tangent point and next point, based on continuity
				double postTanLen = thisDriver.postTan.norm();
				thisDriver.postTan = lerp(
					thisDriver.postTan,
					(lerp(
						nextDriver.pos(),
						Eigen::Vector3d(nextDriver.pos() + nextDriver.prevTan),
						//Eigen::Vector3d(nextDriver.pos() + nextDriver.preTan).matrix(),
						nextDriver.continuity
					) - thisDriver.pos()).eval(),
					thisDriver.continuity
				);
				thisDriver.postTan = thisDriver.postTan.normalized() * postTanLen;

				// prev tan
				double prevTanLen = thisDriver.prevTan.norm();
				thisDriver.prevTan = lerp(
					thisDriver.postTan,
					(lerp(
						prevDriver.pos(),
						Eigen::Vector3d(prevDriver.pos() + prevDriver.postTan),
						prevDriver.continuity
					) - thisDriver.pos()).eval(),
					thisDriver.continuity
				);
				thisDriver.prevTan = thisDriver.prevTan.normalized() * prevTanLen;

			}


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
			Float3Array result(static_cast<int>(pointDatas.size()));
			for (int i = 0; i < static_cast<int>(pointDatas.size()); i++) {
				result[i] = pointDatas[i].finalMatrix.translation().data();
			}
			return result;
		}

		template<typename arrT>
		inline arrT getPointPositionArray(Status& s) {
			/* flattened float array of positions for all points
			I suppose this would be where struct of arrays wins over struct of arrays
			TODO: struct of arrays
			*/
			arrT result(static_cast<int>(pointDatas.size()));
			for (int i = 0; i < static_cast<int>(pointDatas.size()); i++) {
				result[i] = pointDatas[i].finalMatrix.translation().data();
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


		inline Float3Array getWireframePointGnomonVertexPositionArray(Status& s) {
			/* return flat float3 array for gnomon positions only on points
			* each point has 4 coords - point itself, and then 0.1 units in x, y, z of that point
			*/

			Float3Array result(pointDatas.size() * 4);

			for (size_t i = 0; i < pointDatas.size(); i++) {
				result[i * 4] = pointDatas[i].finalMatrix.translation().data();
				result[i * 4 + 1] = pointDatas[i].finalMatrix * Eigen::Vector3d{ 1, 0, 0 };
				result[i * 4 + 2] = pointDatas[i].finalMatrix * Eigen::Vector3d{ 0, 1, 0 };
				result[i * 4 + 3] = pointDatas[i].finalMatrix * Eigen::Vector3d{ 0, 0, 1 };
			}
			return result;
		}
		inline IndexList getWireframePointIndexArray(Status& s) {
			/* return index array for point gnomons
			* intended to emit as separate lines, so half is duplication
			*/
			IndexList result(pointDatas.size() * 3 * 2);
			for (int i = 0; i < static_cast<int>(pointDatas.size()); i++) {
				result[i * 4] = i * 4;
				result[i * 4 + 1] = i * 4 + 1;
				result[i * 4 + 2] = i * 4;
				result[i * 4 + 3] = i * 4 + 2;
				result[i * 4 + 4] = i * 4;
				result[i * 4 + 5] = i * 4 + 3;
			}
			return result;
		}

		Status& getWireframeSingleEdgeGnomonVertexPositionArray(Status& s, Float3Array& outArr, int elIndex, int arrStartIndex) {
			SEdgeData& d = edgeDatas[elIndex];
			int n;
			double u;
			Eigen::Affine3d aff;
			double uvn[3];
			for (int i = 0; i < d.densePointCount(); i++) {
				n = arrStartIndex + (i * 4);
				u = 1.0 / double(d.densePointCount() - 1) * double(i);
				uvn[0] = u;
				uvn[1] = 0; uvn[2] = 0;
				s = edgeMatrixAt(s, aff, elIndex, uvn);

				outArr[n] = d.posSpline(u);
				outArr[n + 1] = aff * Eigen::Vector3d{ 1, 0, 0 };
				outArr[n + 2] = aff * Eigen::Vector3d{ 0, 1, 0 };
				outArr[n + 3] = aff * Eigen::Vector3d{ 0, 0, 1 };
			}
			return s;
		}

		void setGnomonIndexList(unsigned int* result, unsigned int i) {
			result[i * 4] = i * 4;
			result[i * 4 + 1] = i * 4 + 1;
			result[i * 4 + 2] = i * 4;
			result[i * 4 + 3] = i * 4 + 2;
			result[i * 4 + 4] = i * 4;
			result[i * 4 + 5] = i * 4 + 3;
		}

		inline Float3Array getWireframeEdgeGnomonVertexPositionArray(Status& s) {
			// return all edge 
			int totalPositionEntries = 0;
			int totalIndexEntries = 0;
			for (auto& edata : edgeDatas) {
				totalPositionEntries += edata.densePointCount();
				totalIndexEntries += edata.densePointCount() * 4;
			}
			Float3Array posResult(totalPositionEntries);
			IndexList indexResult(totalIndexEntries);
			int posStartIndex = 0;
			for (auto& edata : edgeDatas) {
				SElement& el = elements[edata.index];
				getWireframeSingleEdgeGnomonVertexPositionArray(
					s,
					posResult,
					el.elIndex,
					posStartIndex
				);
				posStartIndex += edata.densePointCount() * 4;
				//setGnomonIndexList()
			}
			return posResult;
		}

		inline IndexList getWireframeEdgeGnomonVertexIndexList(Status& s) {
			int totalIndexEntries = 0;
			for (auto& edata : edgeDatas) {
				totalIndexEntries += edata.densePointCount() * 4;
			}
			IndexList indexResult(totalIndexEntries);
			int posStartIndex = 0;
			for (auto& edata : edgeDatas) {
				SElement& el = elements[edata.index];
				for (unsigned int n = 0; n < static_cast<unsigned int>(edata.densePointCount()); n++) {
					setGnomonIndexList(indexResult.data(), posStartIndex);
					posStartIndex += 6;
				}
				///posStartIndex += edata.densePointCount() * 6;
			}
			return indexResult;
		}

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


