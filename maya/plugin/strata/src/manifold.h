
#pragma once

#include <memory>
#include <map>
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
*/

using namespace ed;

// don't need full int here but MAYBE in the long run, we'll need to make an enum attribute in maya for it
BETTER_ENUM(SElType, int, point, edge, face);

struct StrataElement {
	std::string name;
	uShort elIndex;
	uShort globalIndex;
	SElType elType = SElType::point;

	StrataElement(std::string elName) {
		name = elName;
	};
};

struct SPoint : StrataElement {
	SmallList<uShort, 32> edges;
	SElType elType = SElType::point;

	//SPoint() = default;
	SPoint(std::string elName) : StrataElement(elName) {
	}
};

struct SEdge : StrataElement{
	SmallList<uShort, 128> drivers; // can hold points and/or edges
	SmallList<uShort, 32> faces; // faces that use this edge as a rib or boundary
	SElType elType = SElType::edge;

	SEdge(std::string elName) : StrataElement(elName) {
	}
};

struct SFace : StrataElement{
	std::string name;
	SmallList<uShort, 32> edges;
	SElType elType = SElType::face;

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
	std::map<uShort, STid> globalIndexElTypeMap; // this just feels ridiculous
	std::map<std::string, uShort> nameGlobalIndexMap; // everyone point and laugh 

	inline StrataElement* elFromGlobalIndex(uShort globalId){
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




