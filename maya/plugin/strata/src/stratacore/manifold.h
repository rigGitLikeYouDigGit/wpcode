
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
#include "../macro.h"
#include "../status.h"
#include "../containers.h"

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




//int a = seqIndex(-5, 3);

// don't need full int here but MAYBE in the long run, we'll need to make an enum attribute in maya for it
	//BETTER_ENUM(SElType, int, point, edge, face);

	enum SElType { NONE, point, edge, face};

	struct StrataManifold;

	struct SElement {
		std::string name;
		int elIndex = -1; // index within this element's type - the 3rd point, the 3rd face etc
		int globalIndex = -1; // unique global index across all elements
		std::vector<int> drivers; // topological drivers, not parent spaces
		std::vector<int> edges; // edges that draw from this element
		std::vector<int> points; // points that draw from this element
		std::vector<int> faces; // faces that use this edge as a rib or boundary
		//SElType::_enumerated elType = SElType::point;
		SElType elType = SElType::NONE; // not initialised by default

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

		SElement(std::string elName, const SElType t) : name(elName), elType(t) {
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

		/*
		eventually move to iterators for topo operations, if speed isn't an issue

		template<long FROM, long TO>
class Range {
public:
	class iterator {
		long num = FROM;
	public:
		iterator(long _num = 0) : num(_num) {}
		iterator& operator++() {num = TO >= FROM ? num + 1: num - 1; return *this;}
		iterator operator++(int) {iterator retval = *this; ++(*this); return retval;}
		bool operator==(iterator other) const {return num == other.num;}
		bool operator!=(iterator other) const {return !(*this == other);}
		long operator*() {return num;}
		// iterator traits
		using difference_type = long;
		using value_type = long;
		using pointer = const long*;
		using reference = const long&;
		using iterator_category = std::forward_iterator_tag;
	};
	iterator begin() {return FROM;}
	iterator end() {return TO >= FROM? TO+1 : TO-1;}
};

		*/



	};

	//struct SPoint : SElement {
	//	int driver = -1;
	//	SElType elType = SElType::point;

	//	//SPoint() = default;
	//	SPoint(std::string elName) : SElement(elName) {
	//	}

	//	inline std::vector<int> otherNeighbourPoints(StrataManifold& manifold);
	//	inline std::vector<int> otherNeighbourEdges(StrataManifold& manifold);

	//	inline std::vector<int> edgeStar(StrataManifold& manifold) {
	//		return otherNeighbourEdges(manifold);
	//	}

	//	inline std::vector<int> otherNeighbourFaces(StrataManifold& manifold);
	//};

	//struct SEdge : SElement {
	//	SElType elType = SElType::edge;
	//	SEdge(std::string elName) : SElement(elName) {
	//	}

	//	inline std::vector<int> otherNeighbourPoints(StrataManifold& manifold);

	//	inline std::vector<int> otherNeighbourEdges(StrataManifold& manifold);

	//};

	//struct SFace : SElement {
	//	SElType elType = SElType::face;

	//	std::vector<bool> edgeOrients; // SURELY vector<bool> is cringe
	//	// true for forwards, false for backwards
	//	bool flipFace = 0; // if face as a whole should be flipped, after edge winding

	//	SFace(std::string elName) : SElement(elName) {
	//	}
	//};

	struct SElData {
		std::string name;
		int index;
	};

	struct SPointData : SElData{
		
		MMatrix matrix;
	};

	struct SEdgeData : SElData {
		
	};

	struct SFaceData : SElData {
		//std::string name; // probably generated, but still needed for semantics?
	};

	//struct STid {
	//	//SElType::_enumerated elType;
	//	SElType elType;
	//	int elIndex;

	//	//STid(SElType et, int index) : elType(et), elIndex(index) {}

	//};
	
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

		//std::vector<SPoint> points;
		//std::vector<SEdge> edges;
		//std::vector<SFace> faces;

		int globalIndex = 0; // ticks up any time an element is added
		//std::unordered_map<int,  STid> globalIndexElTypeMap; // this just feels ridiculous
		std::unordered_map<std::string, int> nameGlobalIndexMap; // everyone point and laugh 
		std::map<int, int> pointIndexGlobalIndexMap;
		std::map<int, int> edgeIndexGlobalIndexMap;
		std::map<int, int> faceIndexGlobalIndexMap;


		std::vector<SPointData> pointDatas;
		std::vector<SEdgeData> edgeDatas;
		std::vector<SFaceData> faceDatas;

		// ATTRIBUTES 
		// unsure if wrapping this is useful - for now EVERYTHING explicit
		std::unordered_map<std::string, StrataAttr> attrs;
		std::unordered_map<std::string, StrataGroup> groups;

		StrataAttr* getAttr(std::string& name) {
			// convenience to get a pointer, handle casting yourself at point of use
			// check pointer is valid
			if (!attrs.count(name)) {
				return nullptr;
			}
			return &(attrs[name]);
		}

		StrataGroup* getGroup(std::string& name, bool create=false, int size=4) {
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

		void clear() {
			// is it better to just make a new object?

			/*points.clear();
			edges.clear();
			faces.clear();*/
			elements.clear();
			globalIndex = 0;
			//globalIndexElTypeMap.clear();
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
			return &elements[globalId];
			/*STid elId = globalIndexElTypeMap[globalId];
			switch (elId.elType) {
			case SElType::point:
				return &points[elId.elIndex];
			case SElType::edge:
				return &edges[elId.elIndex];
			case SElType::face:
				return &faces[elId.elIndex];
				;
			}*/
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


		SElement* addElement(SElement& el, SPointData elData) {
			// god what a mess
			// still not sure how unique-pointers and ownership work here, but
			// it's a hassle to make sure what we return is still a valid object in the main vector
			int globalIndex = static_cast<int>(elements.size());

			std::map<int, int>* localGlobalIdMapPtr;
			//std::map<int, int>& localGlobalIdMapPtr = pointIndexGlobalIndexMap;
			switch (el.elType) {
				case SElType::point: localGlobalIdMapPtr = &pointIndexGlobalIndexMap;
				case SElType::edge: localGlobalIdMapPtr = &edgeIndexGlobalIndexMap;
				case SElType::face: localGlobalIdMapPtr = &faceIndexGlobalIndexMap;
				//case SElType::point: localGlobalIdMapPtr = pointIndexGlobalIndexMap;
				//case SElType::edge: localGlobalIdMapPtr = edgeIndexGlobalIndexMap;
				//case SElType::face: localGlobalIdMapPtr = faceIndexGlobalIndexMap;
			}
			int elementIndex = static_cast<int>(
				((localGlobalIdMapPtr->rbegin()))->first
				); // get current max key of element set
			elements.push_back(el);
			SElement* elP = &(elements[elementIndex]);
			elP->globalIndex = globalIndex;
			elP->elIndex = elementIndex + 1;
			//localGlobalIdMapPtr->insert(elP->elIndex, elP->globalIndex);
			nameGlobalIndexMap[el.name] = globalIndex;
			return elP;
		}
		//SEdge* addEdge(SEdge& el, SEdgeData elData) {
		//	int elementIndex = static_cast<int>(edges.size());
		//	edges.push_back(el);
		//	SEdge* elP = &edges[elementIndex];
		//	elP->globalIndex = globalIndex;
		//	elP->elIndex = elementIndex;
		//	STid elId{ SElType::edge, elementIndex };
		//	//globalIndexElTypeMap[globalIndex] = elId;
		//	globalIndexElTypeMap.insert(std::make_pair(globalIndex, elId));
		//	nameGlobalIndexMap[el.name] = globalIndex;
		//	globalIndex += 1;
		//	return elP;
		//}
		//SFace* addFace(SFace& el, SFaceData elData) {
		//	int elementIndex = static_cast<int>(faces.size());
		//	faces.push_back(el);
		//	SFace* elP = &faces[elementIndex];
		//	elP->globalIndex = globalIndex;
		//	elP->elIndex = elementIndex;
		//	STid elId{ SElType::face, elementIndex };
		//	globalIndexElTypeMap.insert(std::make_pair(globalIndex, elId));
		//	nameGlobalIndexMap[el.name] = globalIndex;
		//	globalIndex += 1;
		//	return elP;
		//}

		bool windFaceEdgeOrients(SElement* face) {
			/* for each driver edge of a face,
			* check direction of each driver edge, add its
			* state to the direction vector
			*/
			face->edgeOrients.clear();

			std::vector<SElement*> visitedDrivers;
			int prevEnds[2] = { NULL, NULL }; // edges or points
			// if one of these is an edge, is it guaranteed to show up in 
			// face's own edges?
			// not always, consider edges from a 5-pole - 
			// seems degenerate to have one child edge come off a parent at u=0,
			// that's just a point with extra steps, but maybe it could happen?

			for (int driverGId : face->drivers) {
				SElement* driver = getEl(driverGId);
				// skip anything not an edge
				if (driver->elType != SElType::edge) {
					continue;
				}
				// check if this is the first edge visited
				if (prevEnds[0] == NULL) {
					prevEnds[0] = driver->drivers[0];
					prevEnds[1] = driver->drivers[seqIndex(-1, driver->drivers.size())];
					// use this edge as reference for forwards
					face->edgeOrients.push_back(true);
					visitedDrivers.push_back(driver);
					continue;
				}

				// check for the same edge included twice in a face - by definition one
				// time would be forwards, one would be backwards.
				// this should also catch degenerate cases like loops, 
				// where an edge starts and ends at the same point
				if (seqContains(visitedDrivers, driver)) {
					ptrdiff_t visitedIndex = std::distance(visitedDrivers.begin(), std::find(visitedDrivers.begin(), visitedDrivers.end(), driver));
					bool prevState = face->edgeOrients[visitedIndex];
					if (prevState) { // previously the edge was the correct orientation
						// so now flip it
						prevEnds[0] = visitedDrivers[visitedIndex]->drivers[1];
						prevEnds[1] = visitedDrivers[visitedIndex]->drivers[0];
						face->edgeOrients.push_back(false);
					}
					else {// previously the edge was backwards
						// so now un-flip it
						prevEnds[0] = visitedDrivers[visitedIndex]->drivers[0];
						prevEnds[1] = visitedDrivers[visitedIndex]->drivers[1];
						face->edgeOrients.push_back(true);
					}
					visitedDrivers.push_back(driver);
					continue;
				}


				// check for easy point-matching case first
				if (prevEnds[1] == driver->drivers[0]) { // all good
					prevEnds[0] = driver->drivers[0];
					prevEnds[1] = driver->drivers[1];
					face->edgeOrients.push_back(true);
					visitedDrivers.push_back(driver);
					continue;
				}

				if (prevEnds[1] == driver->drivers[1]) { // edge is backwards
					prevEnds[0] = driver->drivers[1];
					prevEnds[1] = driver->drivers[0];
					face->edgeOrients.push_back(false);
					visitedDrivers.push_back(driver);
					continue;
				}

				// for edges coming from edges, the driver may literally be one of the previous ends
				/* i think this gets really complicated, need to
				work out the orientation of the drivers of the driver edge?

				"all edges start and end with a point" is a nice rule, let's use it
				*/
				//if (prevEnds[1] == driver->globalIndex) {
				//	prevEnds[0] = driver->globalIndex;
				//	prevEnds[1] = driver->drivers[1]; //???????
				//	face->edgeOrients.push_back(false);
				//	continue;
				//}

			}
		}

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


		/*inline std::vector<SElement>* vecForType(SElType t) { // this doesn't work either to mask the type
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
			SElement* el = globalIndexElMap[globalIndex];
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

}


