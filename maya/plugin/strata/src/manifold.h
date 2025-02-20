
#pragma once

#include <memory>
#include <map>
#include <unordered_set>
#include <set>
#include <unordered_map>
#include <string>
#include <vector>
#include <algorithm>
//#include <cstdint>

#include <maya/MVector.h>
#include <maya/MMatrix.h>

#include "wpshared/enum.h"
#include "containers.h"

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

*/

using namespace ed;

// don't need full int here but MAYBE in the long run, we'll need to make an enum attribute in maya for it
BETTER_ENUM(SElType, uShort, point, edge, face);

struct StrataElement {
	std::string name;
	uShort elIndex;
	uShort globalIndex;
	std::vector<uShort> drivers; // topological drivers, not parent spaces
	std::vector<uShort> edges; // edges that draw from this element
	std::vector<uShort> points; // points that draw from this element
	std::vector<uShort> faces; // faces that use this edge as a rib or boundary
	SElType::_enumerated elType = SElType::point;

	bool isActive = true; // should this element (face) be rendered/meshed?
	bool isValid = true; // does this element have an error in its history?
	std::string errorMsg;

	StrataElement(std::string elName) {
		name = elName;
	};

	// neighbours do not include child / driven elements
	// topology functions only return raw ints from these structs - 
	// manifold is needed to get rich pointers

	// ...should these return std::set instead? uggggh

	inline SmallList<uShort, 32> otherNeighbourPoints(StrataManifold& manifold) {
	}
	inline SmallList<uShort, 32> otherNeighbourEdges(StrataManifold& manifold) {
	}
	inline SmallList<uShort, 32> otherNeighbourFaces(StrataManifold& manifold, bool requireSharedEdge=true) {
	}

	/// topological sets - after Keenan Crane
	// star is the set of all edges directly touching this element in it
	// for faces this might include intersectors, not sure yet
	inline SmallList<uShort, 32> edgeStar(StrataManifold& manifold) {
	}
	// link is the set of all edges exactly one edge away from this element
	inline SmallList<uShort, 32> edgeLink(StrataManifold& manifold) {
	}
	// do the same for other element types
	

	inline std::vector<uShort> intersectingElements(StrataManifold& manifold) {
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

struct SPoint : StrataElement {
	uShort driver;
	SElType elType = SElType::point;

	//SPoint() = default;
	SPoint(std::string elName) : StrataElement(elName) {
	}

	inline SmallList<uShort, 32> otherNeighbourPoints(StrataManifold& manifold) {
		// return list of pointers to neighbours of the same element type
		// return all points connected to edges here
		SmallList<uShort, 32> result;
		for (uShort edgeId : edges) {
			StrataElement* edgePtr = manifold.getEl(edgeId);
			for (uShort ptId : edgePtr->points) {
				if (ptId == globalIndex) { continue; }
				result.push_back(ptId);
			}
		}
		return result;
	}
	inline SmallList<uShort, 32> otherNeighbourEdges(StrataManifold& manifold) {
		// return list of pointers to neighbours of the same element type
		// return all points connected to edges here
		SmallList<uShort, 32> result;
		for (uShort edgeId : edges) {
			result.push_back(edgeId);
		}
		return result;
	}

	inline SmallList<uShort, 32> edgeStar(StrataManifold& manifold) {
		return otherNeighbourEdges(manifold);
	}

	inline SmallList<uShort, 32> otherNeighbourFaces(StrataManifold& manifold) {
		// return list of pointers to neighbours of the same element type
		// return all points connected to edges here
		SmallList<uShort, 32> result;
		std::set<uShort> foundFaces;
		for (uShort edgeId : edges) {
			StrataElement* edge = manifold.getEl(edgeId);
			foundFaces.insert(edge->faces.begin(), edge->faces.end());
		}
		for (uShort faceId : foundFaces) {
			result.push_back(faceId);
		}
		return result;
	}



};

struct SEdge : StrataElement{
	SElType elType = SElType::edge;
	SEdge(std::string elName) : StrataElement(elName) {
	}

	inline SmallList<uShort, 32> otherNeighbourPoints(StrataManifold& manifold) {
		// return list of pointers to neighbours of the same element type
		// return end points of this edge only, for now
		SmallList<uShort, 32> result(2);

		if (manifold.getEl(drivers[0])->elType == SElType::point) {
			result.push_back(drivers[0]);
		}
		if (manifold.getEl(drivers[1])->elType == SElType::point) {
			result.push_back(drivers[1]);
		}
		return result;
	}

	
};

struct SFace : StrataElement{
	SElType elType = SElType::face;
	SFace(std::string elName) : StrataElement(elName) {
	}
};

struct SPointData {
	std::string name;
	MMatrix matrix;
};

struct SEdgeData {
	std::string name; // probably generated, but still needed for semantics?
};

struct SFaceData {
	std::string name; // probably generated, but still needed for semantics?
};

struct STid {
	SElType elType;
	uShort elIndex;
};

struct StrataManifold {
	// using vectors here for now

	std::vector<SPoint> points;
	std::vector<SEdge> edges;
	std::vector<SFace> faces;

	uShort globalIndex = 0; // ticks up any time an element is added
	std::unordered_map<uShort, STid> globalIndexElTypeMap; // this just feels ridiculous
	std::unordered_map<std::string, uShort> nameGlobalIndexMap; // everyone point and laugh 

	inline StrataElement* getEl(uShort globalId){
		STid elId = globalIndexElTypeMap[globalId];
		switch (elId.elType) {
			case SElType::point:
				return &points[elId.elIndex];
			case SElType::edge:
				return &edges[elId.elIndex];
			case SElType::face:
				return &faces[elId.elIndex];
;		}
	}

	std::vector<SPointData> pointDatas;
	std::vector<SEdgeData> edgeDatas;
	std::vector<SFaceData> faceDatas;

	std::map<std::string, uShort> pointNameIdMap;
	std::map<std::string, uShort> edgeNameIdMap;
	std::map<std::string, uShort> faceNameIdMap;

	SmallList<StrataElement*> getEls(SmallList<uShort>& globalIds) {
		SmallList<StrataElement*, globalIds.MAXSIZE> result;
		result.reserve(globalIds.size());
		for (uShort gId : globalIds) {
			result.push_back(getEl(gId));
		}
		return result;
	}

	std::vector<StrataElement*> getEls(std::vector<uShort>& globalIds) {
		std::vector<StrataElement*> result;
		result.reserve(globalIds.size());
		for (uShort gId : globalIds) {
			result.push_back(getEl(gId));
		}
		return result;
	}


	SPoint* addPoint(SPoint& el, SPointData elData) {
		// god what a mess
		// still not sure how unique-pointers and ownership work here, but
		// it's a hassle to make sure what we return is still a valid object in the main vector
		uShort elementIndex = static_cast<int>(points.size());
		points.push_back(el);
		SPoint* elP = &points[elementIndex];
		elP->globalIndex = globalIndex;
		elP->elIndex = elementIndex;
		STid elId{ SElType::point, elementIndex };
		globalIndexElTypeMap[globalIndex] = elId;
		nameGlobalIndexMap[el.name] = globalIndex;
		globalIndex += 1;
		return elP;
	}
	SEdge* addEdge(SEdge& el, SEdgeData elData) {
		uShort elementIndex = static_cast<int>(edges.size());
		edges.push_back(el);
		SEdge* elP = &edges[elementIndex];
		elP->globalIndex = globalIndex;
		elP->elIndex = elementIndex;
		STid elId{ SElType::edge, elementIndex };
		globalIndexElTypeMap[globalIndex] = elId;
		nameGlobalIndexMap[el.name] = globalIndex;
		globalIndex += 1;
		return elP;
	}
	SFace* addFace(SFace& el, SFaceData elData) {
		uShort elementIndex = static_cast<int>(faces.size());
		faces.push_back(el);
		SFace* elP = &faces[elementIndex];
		elP->globalIndex = globalIndex;
		elP->elIndex = elementIndex;
		STid elId{ SElType::face, elementIndex };
		globalIndexElTypeMap[globalIndex] = elId;
		nameGlobalIndexMap[el.name] = globalIndex;
		globalIndex += 1;
		return elP;
	}


	/*inline std::vector<StrataElement>* vecForType(SElType t) { // this doesn't work either to mask the type
		if (SElType::point == static_cast<int>(t)) {
			return &points;
		}
	}*/
	/*switch (t) {
		SElType::point:
			return points;
		SElType::edge:
			return edges;
	}*/
	//}



	/*
	auto elByGlobalIndex(const int globalIndex) {
		StrataElement* el = globalIndexElMap[globalIndex];
		switch (el->elType) {
		case SElType::point:
			return static_cast<SPoint*>(el);
		case SElType::edge:
			return static_cast<SEdge*>(el);
		case SElType::face:
			return static_cast<SFace*>(el);
		}
	}
	//// I guess you have to deal with the dynamic type at the point of use,
	//// every time
	//// no point doing a cast here and then some kind of type check later
	*/
};




