
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
#include <queue>
#include <deque>
//#include <cstdint>

#include <Eigen/Dense>
#include <unsupported/Eigen/MatrixFunctions>
#include <unsupported/Eigen/Splines>

#include <maya/MVector.h>
#include <maya/MMatrix.h>

#include "../bezier/bezier.h"

#include <enum.h>
#include "../macro.h"
#include "../status.h"
#include "../../containers.h"
#include "../api.h"
#include "../lib.h"
#include "../libEigen.h"

#include "../name.h"
#include "element.h"
#include "pointData.h"
#include "edgeData.h"
//#include "faceData.h"
#include "vertexData.h"
#include "intersection.h"
#include "group.h"

#include "../logger.h"
/*
smooth topological manifold, made of points, edges and partial edges

elements may have a domain element, which should form a dag graph

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

namespace strata {


	struct SEdgeSubspaceData {
		/* test subspace as actual element type?
		* should we put in inheritance to base class
		* do we store new dense points and final curves here?
		* 
		* is there a way to integrate this with all the existing anchor
		* setup without adding in NEW cases for subspaces? these should each behave like edges
		*/
		int source = -1;

		// full UVNs are obviously irrelevant here?
		/*std::array<Vector3f, 2> range = { Vector3f{0, 0, 0}, Vector3f{1, 1, 1} };*/
		std::array<float, 2> range = { 0.0, 1.0 };
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

	static inline int orientedEdgeIndex(int unorientedIndex, bool flip){
		/* please pay attention as this is very complicated*/
		return unorientedIndex * 2 + static_cast<int>(flip);
	}
	static inline std::pair<int, bool> unOrientedEdgeIndex(int orientedIndex, bool wantFlip) {
		/* */
		return std::make_pair(orientedIndex / 2, orientedIndex % 2);
	}
	static inline int unOrientedEdgeIndex(int orientedIndex) {
		/* */
		return orientedIndex / 2;
	}
	struct StrataManifold {
		/*

		data store just accumulates across graph iterations until something changes it - 
		also lets you set direct atomic deltas against el names, regardless of 
		whether they exist at that moment or not - probably quite good

		*/

		std::vector<SPoint> points;
		std::vector<SEdge> edges;
		std::vector<std::pair<SElType, int>> globalIdMap = {}; // global index to element type and element index

		std::unordered_map<StrataName, int> nameGlobalIdMap = {};

		//std::unordered_map<StrataName, SPoint> pDataMap = {};
		//std::unordered_map<StrataName, SEdge> eDataMap = {};
		//std::unordered_map<StrataName, SFaceData> fDataMap = {};

		std::vector<Vertex> vertices;
		std::vector<HEdge> hedges;
		std::unordered_map<int, // one edge 
			////std::tuple<int, int, bool, bool, float, float>,  /* inEdge, outEdge, inEdgeFlip, outEdgeFlip */
			//std::tuple<int, int>,  /* inHedge, outHedge */
			//int
			std::unordered_map<int, // other edge
				std::vector<int>> // all vertices formed by their intersections
		> vertexMap; // is this insane

		/*
		{
			edgeA : {
				edgeB : {
					[ vertices ] ??? YES just do this. just do this and move on.
					}
				}
		
		edgeA : {
			uvnA : {
				edgeB : {
					[ vertices ] ???
					}
				}
			}


		*/


		Vertex* getVertex(int vtxId) {
			return &vertices[vtxId];
		}
		
		Vertex* getVertex(
			int eA, float uA, bool dirA,
			int eB, float uB, bool dirB,
			int iPoint=-1
		) {
			/* for now, we only add the 'directed' vertex from
			eA to eB when created here - 
			no attempt to also add the same vertex from eB to eA
			*/
			auto foundA = vertexMap.find(eA);
			if (foundA != vertexMap.end()) {
				auto foundB = foundA->second.find(eB);
				if (foundB != foundA->second.end()) {
					for (int vtxId : foundB->second) {
						Vertex& vtx = vertices[vtxId];
						if (EQ(vtx.edgeUs[0], uA) && EQ(vtx.edgeUs[1], uB)
							&& EQ(vtx.edgeDirs[0], dirA) && EQ(vtx.edgeDirs[1], dirB)
							) 
						{
							return &vertices[vtxId];
						}
					}
				}
			}
			/* not found, add a new vertex*/
			int newIdx = static_cast<int>(vertices.size());
			Vertex newVtx{
				newIdx,
				{eA, eB},
				{uA, uB},
				{dirA, dirB},
				iPoint
			};
			vertices.push_back(newVtx);
			vertexMap[eA][eB].push_back(newIdx);
			SEdge* eDataA = getEdge(eA);
			SEdge* eDataB = getEdge(eB);
			eDataA->vertices.push_back(newIdx);
			eDataB->vertices.push_back(newIdx);

			return &vertices[newIdx];
		}

		HEdge* hedge(int hId) {
			return &hedges[hId];
		}

		std::unordered_map<StrataName, SGroup> groupMap = {};

		// map of intersecting elements
		IntersectionRecord iMap;
		/*TODO: integrate with raw elements?
		*/

		// world matrix transform of this manifold
		Affine3f worldMat = Affine3f::Identity();

		// cached indices for drawing buffers
		int _nEdgeVertexBufferEntries = -1;
		int _edgeVertexBufferEntryStart = -1;

		//inline SElement* elData(int globalElId) {
		//	switch (elT) {
		//	//case SElType::point: return &pointDatas[pointIndexGlobalIndexMap[globalElId]];
		//	case SElType::point: {
		//		//return &pointDatas[el.elIndex];
		//		auto ptr = pDataMap.find(getEl(globalElId)->name);
		//		if (ptr == pDataMap.end()) { return nullptr; }
		//		return &(ptr->second);
		//		break;
		//	}
		//	//case SElType::edge: return &edgeDatas[edgeIndexGlobalIndexMap[globalElId]];
		//	case SElType::edge: {
		//		auto ptr = eDataMap.find(getEl(globalElId)->name);
		//		if (ptr == eDataMap.end()) { return nullptr; }
		//		return &(ptr->second);
		//		break;
		//	}
		//	case SElType::face: {
		//		auto ptr = fDataMap.find(getEl(globalElId)->name);
		//		if (ptr == fDataMap.end()) { return nullptr; }
		//		return &(ptr->second);
		//		break;
		//	}
		//	default: return nullptr;
		//	}
		//}
		//inline SElShapeData* elData(int globalElId) {
		//	return elData(globalElId, elements[globalElId].elType);
		//}
		////inline SElData* elData(StrataName elName) {
		////	return elData(globalElId, elements[globalElId].elType);
		////}
		//inline SElShapeData* elData(SElement& el) {
		//	switch (el.elType) {
		//	case SElType::point: {
		//		//return &pointDatas[el.elIndex];
		//		auto ptr = pDataMap.find(el.name);
		//		if (ptr == pDataMap.end()) {return nullptr;	}
		//		return &(ptr->second);
		//		break;
		//	}
		//	//case SElType::edge: return &edgeDatas[el.elIndex];
		//	case SElType::edge: {
		//		auto ptr = eDataMap.find(el.name);
		//		if (ptr == eDataMap.end()) { return nullptr; }
		//		return &(ptr->second);
		//		break;
		//	}
		//	case SElType::face: {
		//		auto ptr = fDataMap.find(el.name);
		//		if (ptr == fDataMap.end()) { return nullptr; }
		//		return &(ptr->second);
		//		break;
		//	}
		//	default: return nullptr;
		//	}
		//}

		// ATTRIBUTES 
		// unsure if wrapping this is useful - for now EVERYTHING explicit
		std::unordered_map<StrataName, StrataAttr> attrs;
		std::unordered_map<StrataName, StrataGroup> groups;


		void clear() {
			// is it better to just make a new object?

			//elements.clear();
			//nameGlobalIdMap.clear();
			////pointDatas.clear();
			//pDataMap.clear();
			//eDataMap.clear();
			//fDataMap.clear();

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

		//inline const SElement& getEl(const int& globalId) const {
		//	return elements.at(globalId);
		//}

		//inline const SElement& getEl(const int& globalId) const {
		//	return elements.at(globalId);
		//}

		inline const SElement* getEl(const SElement* el) const {
			return el;
		}

		inline const SElement* getEl(const int& globalId) const {
			if (globalId >= globalIdMap.size()) {
				return nullptr;
			}
			auto [elT, elIdx] = globalIdMap[globalId];
			switch (elT) {
				case SElType::point: return &points[elIdx];
				case SElType::edge: return &edges[elIdx];
				//case SElType::face: return &faces[elIdx];
			}
		}
		inline SElement* getEl(SElement* el) {
			return el;
		}
		inline SElement* getEl(const int& globalId) {
			if (globalId >= globalIdMap.size()) {return nullptr;}
			auto [elT, elIdx] = globalIdMap[globalId];
			switch (elT) {
			case SElType::point: return &points[elIdx];
			case SElType::edge: return &edges[elIdx];
				//case SElType::face: return &faces[elIdx];
			}
		}
		inline SEdge* getEdge(const int& globalId) {
			auto [elT, elIdx] = globalIdMap[globalId];
			return &edges[elIdx];
		}

		inline SElement* getEl(const StrataName& name) {
			if (!nameGlobalIdMap.count(name)) {
				return nullptr;
			}
			int globalId = nameGlobalIdMap[name];
			auto [elT, elIdx] = globalIdMap[globalId];
			switch (elT) {
			case SElType::point: return &points[elIdx];
			case SElType::edge: return &edges[elIdx];
				//case SElType::face: return &faces[elIdx];
			}
		}

		//inline const SElement* getElC(const StrataName name) const {
		//	if (!nameGlobalIdMap.count(name)) {
		//		return nullptr;
		//	}
		//	//return &elements[nameGlobalIndexMap[name]];
		//	return &elements.at(nameGlobalIdMap.at(name));
		//	//return &elements[nameGlobalIndexMap[name]];
		//}


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

		template<class iterator_type>
		std::vector<SElement*> getEls(iterator_type it, iterator_type end) const {
			std::vector<SElement*> result;
			//result.reserve(globalIds.size());
			//for (StrataName& gId : globalIds) {
			while(it != end){
				result.push_back(getElC(*it));
				it++;
			}
			return result;
		}

		template<typename T>
		int getElIndex(T el) {
			return getEl(el)->globalIndex;
		}

		//template<class in_iterator_type, class out_iterator_type>
		//void getElIndices(in_iterator_type it, in_iterator_type end,
		//	) const {
		//	std::vector<SElement*> result;
		//	//result.reserve(globalIds.size());
		//	//for (StrataName& gId : globalIds) {
		//	while (it != end) {
		//		result.push_back(getElC(*it));
		//		it++;
		//	}
		//	return result;
		//}

		
		//SElShapeData* setElData(SElement* el, SElShapeData* data) {
		//	/* absolutely no idea on how best to do these uniform interfaces for #
		//	setting data of different types*/
		//	switch (el->elType) {
		//	case SElType::point: {
		//		//pointDatas[el->elIndex] = *static_cast<SPointData*>(data);
		//		pDataMap[el->name] = *static_cast<SPoint*>(data);
		//		//return &pointDatas[el->elIndex];
		//		return &pDataMap.at(el->name);
		//	}
		//	case SElType::edge: {
		//		eDataMap[el->name] = *static_cast<SEdge*>(data);
		//		return &eDataMap.at(el->name);
		//	}
		//	case SElType::face: {
		//		fDataMap[el->name] = *static_cast<SFaceData*>(data);
		//		return &fDataMap[el->name];
		//	}
		//	default: return nullptr;
		//	}
		//}

		Status addElement(
			SElement& el,
			SElement*& outPtr,
			bool allowOverride=false
		) {
			LOG("addElement from other");
			Status s;
			if (nameGlobalIdMap.find(el.name) != nameGlobalIdMap.end()) {
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
			int globalIndex = static_cast<int>(globalIdMap.size());
			int elementIndex;
			el.globalIndex = globalIndex;
			switch (el.elType) {
			case SElType::point: {
				elementIndex = static_cast<int>(points.size());
				el.elIndex = elementIndex;
				points.push_back(static_cast<SPoint&>(el)); 
				break;
			}
			case SElType::edge: {
				elementIndex = static_cast<int>(edges.size());
				el.elIndex = elementIndex;
				edges.push_back(static_cast<SEdge&>(el)); 
				break;
			}
			/*case SElType::face: {
				int elementIndex = static_cast<int>(faces.size());
				el.elIndex = elementIndex;
				faces.push_back(static_cast<SFace&>(el)); 
				break;
			}*/
			}
			globalIdMap.push_back({ el.elType, elementIndex });
			SElement* elP = getEl(globalIndex);
			nameGlobalIdMap[el.name] = globalIndex;

			outPtr = elP;
			return s;
		}

		Status addElement(
			const StrataName name,
			const SElType elT,
			SElement*& outPtr
		) {
			return addElement(SElement(name, elT), outPtr);
		}

		void renameElement(
			SElement* el, StrataName newName
		) {
			/* USE WITH EXTREME CAUTION - ideally only for assigning a name to a constructed element
			* won't try and rename name references in dependents - use this only on a new element
			*/
			nameGlobalIdMap.erase(el->name);
			nameGlobalIdMap[newName] = el->globalIndex;
			el->name = newName;
		}

		Status& intersectingElements(
			Status& s,
			SElement* el,
			std::vector<SElement*>& outEls
		);

		bool elementsCouldIntersect(
			SElement* elA,
			SElement* elB
		);

		//bool windFaceEdgeOrients(SElement* face) {
		//	/* for each anchor edge of a face,
		//	* check direction of each anchor edge, add its
		//	* state to the direction vector
		//	*/
		//	face->edgeOrients.clear();

		//	std::vector<SElement*> visitedAnchors;
		//	int prevEnds[2] = { NULL, NULL }; // edges or points
		//	// if one of these is an edge, is it guaranteed to show up in 
		//	// face's own edges?
		//	// not always, consider edges from a 5-pole - 
		//	// seems degenerate to have one child edge come off a domain at u=0,
		//	// that's just a point with extra steps, but maybe it could happen?

		//	for (int anchorGId : face->anchors) {
		//		SElement* anchor = getEl(anchorGId);
		//		// skip anything not an edge
		//		if (anchor->elType != SElType::edge) {
		//			continue;
		//		}
		//		// check if this is the first edge visited
		//		if (prevEnds[0] == NULL) {
		//			prevEnds[0] = anchor->anchors[0];
		//			prevEnds[1] = anchor->anchors[seqIndex(-1, anchor->anchors.size())];
		//			// use this edge as reference for forwards
		//			face->edgeOrients.push_back(true);
		//			visitedAnchors.push_back(anchor);
		//			continue;
		//		}

		//		// check for the same edge included twice in a face - by definition one
		//		// time would be forwards, one would be backwards.
		//		// this should also catch degenerate cases like loops, 
		//		// where an edge starts and ends at the same point
		//		if (seqContains(visitedAnchors, anchor)) {
		//			ptrdiff_t visitedIndex = std::distance(visitedAnchors.begin(), std::find(visitedAnchors.begin(), visitedAnchors.end(), anchor));
		//			bool prevState = face->edgeOrients[visitedIndex];
		//			if (prevState) { // previously the edge was the correct orientation
		//				// so now flip it
		//				prevEnds[0] = visitedAnchors[visitedIndex]->anchors[1];
		//				prevEnds[1] = visitedAnchors[visitedIndex]->anchors[0];
		//				face->edgeOrients.push_back(false);
		//			}
		//			else {// previously the edge was backwards
		//				// so now un-flip it
		//				prevEnds[0] = visitedAnchors[visitedIndex]->anchors[0];
		//				prevEnds[1] = visitedAnchors[visitedIndex]->anchors[1];
		//				face->edgeOrients.push_back(true);
		//			}
		//			visitedAnchors.push_back(anchor);
		//			continue;
		//		}


		//		// check for easy point-matching case first
		//		if (prevEnds[1] == anchor->anchors[0]) { // all good
		//			prevEnds[0] = anchor->anchors[0];
		//			prevEnds[1] = anchor->anchors[1];
		//			face->edgeOrients.push_back(true);
		//			visitedAnchors.push_back(anchor);
		//			continue;
		//		}

		//		if (prevEnds[1] == anchor->anchors[1]) { // edge is backwards
		//			prevEnds[0] = anchor->anchors[1];
		//			prevEnds[1] = anchor->anchors[0];
		//			face->edgeOrients.push_back(false);
		//			visitedAnchors.push_back(anchor);
		//			continue;
		//		}

		//		// for edges coming from edges, the anchor may literally be one of the previous ends
		//		/* i think this gets really complicated, need to
		//		work out the orientation of the anchors of the anchor edge?

		//		"all edges start and end with a point" is a nice rule, let's use it
		//		*/
		//		//if (prevEnds[1] == anchor->globalIndex) {
		//		//	prevEnds[0] = anchor->globalIndex;
		//		//	prevEnds[1] = anchor->anchors[1]; //???????
		//		//	face->edgeOrients.push_back(false);
		//		//	continue;
		//		//}

		//	}
		//}

		//bool edgesAreContiguousRing(std::vector<int>& edgeIds) {
		//	/* might be more c++ to have this more composable -
		//	maybe have a general check to see if any elements are contiguous between them?
		//	then some way to filter that it should only check edges, and only move via edges?

		//	check that each edge contains the next in sequence in its neighbours

		//	this is quite a loose check actually, if you try and fool it, I guess you can?

		//	*/
		//	for (size_t i = 0; i < edgeIds.size(); i++) {
		//		size_t nextI = seqIndex(i + 1, edgeIds.size());
		//		int nextEdgeId = edgeIds[nextI];
		//		SElement* thisEdge = getEl(edgeIds[i]);

		//		// if next edge is not contained in neighbours, return false
		//		if (!seqContains(thisEdge->otherNeighbourEdges(*this), nextEdgeId)) {
		//			return false;
		//		}
		//	}
		//	return true;

		//}


		StrataAttr* getAttr(StrataName& name) {
			// convenience to get a pointer, handle casting yourself at point of use
			// check pointer is valid
			if (!attrs.count(name)) {
				return nullptr;
			}
			return &(attrs[name]);
		}

		StrataGroup* getGroup(StrataName& name, bool create = false, int size = 4) {
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

		/* is there a benefit ever to using enums over simple integers like this?*/
		static constexpr int MERGE_OVERWRITE = 0;
		static constexpr int MERGE_LEAVE = 1;
		static constexpr int MERGE_UNION = 2;

		Status& mergeOther(StrataManifold& other, int mergeMode, Status& s);

		/* groups */
		SGroup* addGroup(StrataName& groupName, SElType elType);
		SGroup* getGroup(StrataName& groupName);
		void addToGroup(SGroup* grp, SElement* el);
		void removeFromGroup(SGroup* grp, SElement* el);
		Status& deleteGroup(Status& s, StrataName& groupName);

		Status& renameGroup(Status& s, StrataName& startName, StrataName& newName, int mergeMode);

		/*
		spatial functions
		static for less coupling to this specific graph's buffers
		*/

		void transform(Affine3f& mat) {
			/* transform entire contents of manifold by given matrix - 
			we still transform final results of elements with anchors,
			this alone mutates a strata manifold
			all our relative offsets are still valid

			it might be worth later on caching all anchorless data like this,
			but for now brute force it

			Strata is immutable, EXCEPT for this
			special-case it, and invert transform during backpropagation, wherever it's applied
			*/
			for (auto& el : points) {
				el.finalMatrix = mat * el.finalMatrix;
			}
			for (auto& el : edges) {
				for (int i = 0; i < static_cast<int>(el.finalPositions.rows()); i++) {
					// Convert row to Vector3f, transform, assign back
					Eigen::Vector3f pos = el.finalPositions.row(i);
					el.finalPositions.row(i) = mat * pos;
				}
			}
		}

		//static Status& matrixAt(Status& s, Eigen::Affine3f& outMat, int globalIndex, const Vector3f& uvn, const Vector3f& mode = {0, 0, 0});
		static Status& matrixAt(Status& s, Eigen::Affine3f& outMat, SElement* el , const Vector3f& uvn, const Vector3f& mode = { 0, 0, 0 });
		//static Status& posAt(Status& s, Eigen::Affine3f& outMat, int globalIndex, const SCoord& uvn, );
		static Status& posAt(Status& s, Eigen::Affine3f& outMat, SElement* el, const Vector3f& uvn, const Vector3f& mode = { 0, 0, 0 });



		Status& closestMatrix(Status& s, Eigen::Affine3f& outMat, SElement* el, const Eigen::Vector3f closePos);

		Status& getUVN(Status& s, Eigen::Vector3f& uvn, SElement* el, const Eigen::Vector3f& closePos);

		Status& pointSpaceMatrix(Status& s, Affine3f& outMat, SPoint& data);

		Status& computePointData(Status& s, SPoint& data//, bool doProjectToAnchors=false
		);

		Status& pointProjectToAnchors(Status& s, Affine3f& mat, SElement* el);

		Status& edgeDomainDataFromAnchors(Status& s, SEdge& eData, SEdgeSpaceData& pData);

		Status& buildEdgeAnchors(Status& s, SEdge& eData);

		Status& buildEdgeData(Status& s, SEdge& eData);

		Status& buildFaceAnchors(Status& s, SFaceData& fData);

		Status& buildFaceData(Status& s, SFaceData& fData);


		Status& buildPointData(Status& s, SPoint& eData) {
			/* construct final dense array for data, assuming all domains and anchor indices are set in data

			build base curve matrices in worldspace,
			then get into space of each anchor

			but we can only work in worldspace when curve is freshly added, otherwise
			we can only save anchor-space versions
			*/

			// Bezier control points for each span

			

			return s;
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
			Float3Array result(static_cast<int>(points.size()));
			int i = 0;
			for (auto& p : points) {
				result[i] = p.finalMatrix.translation().data();
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
			return static_cast<int>(points.size()) * 4;
		}
		Float3Array getWireframePointGnomonVertexPositionArray(Status& s);
		Status& getWireframePointGnomonVertexPositionArray(Status& s, Float3* outArr, int startIndex);
		IndexList getWireframePointIndexArray(Status& s);

		Status& getWireframeSingleEdgeGnomonVertexPositionArray(Status& s, Float3Array& outArr, SElement* el, int arrStartIndex);

		//Status& setWireframeSingleEdgeVertexPositionArray(Status& s, Float3Array& outArr, SElement* el, int arrStartIndex);

		int getWireframeEdgeVertexPositionLength() {
			int result = 0;
			for (auto& e : edges) {
				result += e.densePointCount();
			}
			return result;

		}
		Float3Array getWireframeEdgeVertexPositionArray(Status& s);
		Status& getWireframeEdgeVertexPositionArray(Status& s, Float3* outArr, int startIndex);


		IndexList getWireframeEdgeVertexIndexList(Status& s);
		IndexList getWireframeEdgeVertexIndexList(Status& s, SEdge& eData);
		IndexList getWireframeEdgeVertexIndexListPATCH(Status& s);

		void setGnomonIndexList(unsigned int* result, unsigned int i);

		Float3Array getWireframeEdgeGnomonVertexPositionArray(Status& s);

		IndexList getWireframeEdgeGnomonVertexIndexList(Status& s);


		StrataName printInfo() {
			StrataName result = "<manifold - nPts: " + str(points.size()) + ", nEdges: " + str(edges.size()) + " >";
			return result;
		}
		/*TODO: proper dump for everything - every anchor of every element.
		might as well just serialise things at that point.
		*/

		/////////////

		static inline bool _isNotAlnum(char c)
		{
			return !std::isalnum(c);
		}
		//static Status validateElName(const StrataName& elName) {
		//	Status s;
		//	if (std::find_if(elName.begin(), elName.end(), _isNotAlnum) == elName.end()) {
		//		return s;
		//	}
		//	STAT_ERROR(s, "element name: " + elName + " contains invalid characters, must only be alphanumeric");
		//	
		//}

		
	};

}


