

#include <string>
#include <vector>
#include "../status.h"
#include "../macro.h"

#include "manifold.h"
#include "libManifold.h"

#include "libFace.h"

using namespace strata;

struct EdgeSpan {
	std::string name;
	std::array<float, 2> params;
};

/* need sorted i points along each edge*/

struct Vertex {
	/* are we actually doing houdini things in here?
	unique corner on a face - 
	I think this can also link to a unique subpatch

	order of edges matches winding order of face
	*/
	int index = -1;
	std::array<int, 2> edgeIds;
	std::array<float, 2> edgeUs;
	std::array<bool, 2> edgeFlips; /* does the direction of each edge match the face winding order*/

};

struct SingleFaceBuildData {
	std::vector<EdgeSpan> edges; /* ordered edges to use to create this face -
	not guaranteed to connect?*/

	/* 2 crossing edges could connect to 4 separate faces - */
};


struct EdgeCircuitExtraData {
	/* passed in to graph iteration predicates for edge circuit paths
	
	- if a next target already appeared in path, that is a closed path
	- multiple closed paths with same start/endpoint might appear
	- index by start edge
	
	*/
	StrataManifold& manifold;
	std::vector<Vertex> vertices;
	std::unordered_set<int> visitedEdges; /* probably not needed*/
	std::unordered_set<int> visitedVertices; /* used during iteration to cull duplicate paths*/
	std::unordered_set<int> validEdges; /* used during iteration to cull duplicate paths*/
	std::map<int, std::vector<Vertex>> closedPaths; /* separate paths, indexed by start VERTEX 
	one VERTEX maps to one CLOSED PATH -> one CLOSED FACE
	
	*/

	std::unordered_map<
		std::tuple<int, int, bool, bool>,  /* inEdge, outEdge, inEdgeFlip, outEdgeFlip */
		Vertex*
	> vertexMap; // is this insane

	EdgeCircuitExtraData(StrataManifold& manifold_) : manifold(manifold_) {};

	Vertex* getVertex(int eA, int eB, bool flipA, bool flipB) {
		auto found = vertexMap.find({ eA, eB, flipA, flipB });
		if (found == vertexMap.end()) {
			return nullptr;
		}
		return found->second;
	}

	Vertex* createVertex(
		int eA, int eB, float uA, float uB, bool flipA, bool flipB
	) {
		/* make new vertex, add entry in map
should this be a double-layer map again?
*/
		int newIndex = vertices.size();
		vertices.emplace_back();
		vertices.back().index = newIndex;
		vertices.back().edgeIds = { eA, eB };
		vertices.back().edgeFlips = { flipA, flipB };
		vertices.back().edgeUs = { uA, uB };
		vertexMap.insert({ { eA, eB, flipA, flipB }, &vertices.back() });
		return &vertices.back();
	}
};


void _processFoundClosedPath(
	std::vector<int>& idPath,
	EdgeCircuitExtraData& eData,
	StrataManifold& manifold
) {

}

struct EdgePathNextIdsPred : NextIdsPred {

	/* optionally pass in whole node path up to this one - last in vector*/
	template< typename ExtraT=EdgeCircuitExtraData* >
	std::vector<int> operator()(
		std::vector<int>& idPath, // VERTEX index
		GraphVisitor::VisitHistory& history,
		ExtraT extraData = nullptr
		) {
		/*
		idPath: vector of nodes from source, including this one

		return vector of new DIRECT destinations from this node -
		externally these will be added on to paths

		look up all connected edges, remove all that have already been visited in this path?
		*/
		std::vector<int> result;
		EdgeCircuitExtraData& exData = *extraData;
		StrataManifold& manifold = exData.manifold;
		IntersectionRecord& rec = manifold.iMap;
		Vertex& vertex = exData.vertices[idPath.back()];
		
		/* get current edge we're travelling along*/
		int outEdge = vertex.edgeIds[1]; 
		/* check if we're travelling backwards*/
		bool outEdgeFlip = vertex.edgeFlips[1];
		/* u coord on current edge of origin vertex*/
		float origU = vertex.edgeUs[1];

		/* get possible intersection points on this edge */
		for (auto& p : rec.elUVNPointMap.at(outEdge)) {
			/* disregard points lower (if straight) or higher (if flip) */
			bool shouldSkip = outEdgeFlip ? (p.first.x() < origU) : (p.first.x() > origU);
			if (shouldSkip) {
				continue;
			}
			/* here we find / consider every valid intersection point on this edge for vertices - 
			trust to the history to discount them, or else to pull separate ring paths out of them

			SHOULD WE put intersection points at the tips of all edges, to they're more visible to
			processes like this?
			*/
			
			IntersectionPoint& pt = p.second;
			// I think this should be a map of some kind on intersection point
			
			int ptThisEdgeIndex = vectorIndex(pt.elements, outEdge);
			for (int i = 0; i < static_cast<int>(pt.elements.size()); i++) {
				SElement* otherEl = manifold.getEl(pt.elements[i]);
				/* only care about other edges*/
				if (otherEl->elType != SElType::edge) {
					continue;
				}
				/* only care about other VALID edges */
				if (exData.validEdges.find(otherEl->globalIndex) == exData.validEdges.end()) {
					continue;
				}

				Vertex* lookupVertex;

				for (bool nextFlip : {true, false}) {
					lookupVertex = exData.getVertex(
						outEdge, otherEl->globalIndex, outEdgeFlip, nextFlip
					);
					if (lookupVertex == nullptr) {
						lookupVertex = exData.createVertex(
							outEdge, otherEl->globalIndex,
							pt.uvns[ptThisEdgeIndex].x(), pt.uvns[i].x(),
							outEdgeFlip, nextFlip);
						/* first time vertex created, auto-add it to valid next destinations*/
						result.push_back(lookupVertex->index);
					}
					else {
						/* if this vertex is already marked visited, skip?
						this is mainly to prevent infinite crawling, not sure on logic of
						when to add to this set*/
						if (exData.visitedVertices.find(lookupVertex->index)) {
							continue;
						}
					}
				}
				

			}
		
		}

		eData.visitedEdges.insert(idPath);

		/*NEXT DESTINATIONS - get intersecting edges, filter against already visited*/
		for (auto& p : rec.elMap[idPath.back()]) {
			/* first check for a CLOSED PATH
			has to be closed/match in direction as well.
			kill me.
			*/

			/* if next intersected edge is CLOSED, yield 2 VERTICES - 
			one going off in each direction*/

			auto foundInPath = std::find(idPath.begin(), idPath.end(), p.first);
			if (foundInPath != idPath.end()) { /* check if there is already a path starting and ending at previous element*/
				int startEdge = *foundInPath;
				auto foundPath = eData.closedPaths.fi
				if()
			}
		}
		return result;
	}
};

void getEdgeCircuitPaths(
	Status& s,
	StrataManifold& manifold,
	std::vector<int> edgeIsland,
	std::vector<std::vector<Vertex>>& vertexPaths
) {
	/* for each circuit contained in edges, 
	return a list of vertices to use to build faces

	closed edges make this quite annoying
	*/
	IntersectionPoint startPt;
	bool found = false;
	std::unordered_set<int> islandSet(edgeIsland.begin(), edgeIsland.end());

	std::vector<Vertex> vertexPath;

	for (auto& p : manifold.iMap.elMap[edgeIsland[0]]) {
		if (islandSet.find(p.first) == islandSet.end()) { /* only consider intersecting elements in island*/
			continue; 
		}
		/* p is an edge in the island connected to the first*/
		for (auto& ptrPair : p.second) { /* iterate over all possible intersections between these 2 edges*/
			/* if intersection is not a point (somehow) skip */
			if (ptrPair.second != Intersection::POINT) {
				continue;
			}
			startPt = manifold.iMap.points[ptrPair.first];
			vertexPath.emplace_back();
			Vertex& v = vertexPath.back();
			v.edgeIds[0] = edgeIsland[0]
			found = true;
			break;
		}
		break;
	}
	if (!found) {
		/* RETURN ERROR, can't continue*/
	}
}


struct ItIntersectingElements {
	/* iterator to run over all connected and intersecting elements -
	might be excessive, just trying new things. 
	need to keep record of layers iterated, so we can skip a branch?
	one day I'll learn how to template this properly, the logic is the
	same as the graph iteration we have in the StrataManifold already 
	*/
	IntersectionRecord& rec;
	
	int elId;
	std::unordered_set<int> found;
	std::unordered_set<int>* whitelist = nullptr;

	ItIntersectingElements(
		IntersectionRecord& rec_
	) : rec(rec_) {
		elId = rec.elMap.begin()->first;
	}

	void _next() {
		
	}

	ItIntersectingElements& operator++(int n) {

		return *this;
	}
};


int getNextEl(
	IntersectionRecord& record,
	std::unordered_set<int>& found,
	int start
) {
	/* this gets exponentially slower 
	as we loop through more elements already found -
	honestly never felt like more of a fraud*/
	for (auto& p : record.elMap[start]) {
		/* if already found connected element, skip it*/
		if (found.find(p.first) != found.end()) {
			continue;
		}
		/* new element found - flag as visited, then return it*/
		found.insert(p.first);
		return p.first;
	}
	return -1; 
}

void connectedEls(
	IntersectionRecord& record,
	std::unordered_set<int>& checked,
	std::unordered_set<int>* letList,
	std::deque<int>& toCheck,
	std::vector<int>& allConnected
) {
	/* index islands by their min el index, since each element can only appear in one
	*/
	int start = toCheck.front();
	toCheck.pop_front();
	checked.insert(start);
	for (auto& p : record.elMap[start]) {
		/* has el already been visited*/
		if (checked.find(p.first) != checked.end()) { 
			continue;
		}
		checked.insert(p.first);

		/* is el part of the let list*/
		if (letList != nullptr) {
			if (letList->find(p.first) == letList->end()) {
				continue;
			}
		}
		allConnected.push_back(p.first);
		toCheck.push_front(p.first);
	}
}

std::vector<int> connectedElIsland(
	IntersectionRecord& record,
	std::unordered_set<int>& checked,
	std::unordered_set<int>* letList,
	int startIndex
) {
	std::deque<int> toCheck = { startIndex };
	std::vector<int> island = { startIndex };
	checked.insert(startIndex);
	while (toCheck.size()) {
		connectedEls(
			record,
			checked,
			letList,
			toCheck,
			island);
	}
	std::sort(island.begin(), island.end());
	return island;
}

void connectedElIsland(
	IntersectionRecord& record,
	std::unordered_set<int>& checked,
	std::unordered_set<int>* letList,
	int startIndex,
	std::vector<int>& island
) {
	std::deque<int> toCheck = { startIndex };
	island.push_back(startIndex);
	checked.insert(startIndex);
	while (toCheck.size()) {
		connectedEls(
			record,
			checked,
			letList,
			toCheck,
			island);
	}
	std::sort(island.begin(), island.end());
	//return island;
}

//Status& findClosedEdgePaths(
//	IntersectionRecord& record,
//	std::vector<int> edgeIsland,
//	
//)

Status& strata::makeFaceGroup(
	Status& s,
	StrataManifold& manifold,
	SGroup* grp,
	std::vector<std::string>& elNames
) {
	/*
	* - filter elements to see which are edges, which are points to match
	* - filter edge islands to check for disconnected patches - those will create tubes when connected
	* - for each island, work out separate face
	* 
	* CANNOT MIX CLOSED AND OPEN BOUNDARIES ON ISLANDS.
	*  - if all are open boundaries, do a simple rail
	*  - if all are closed, tube
	* 
	*/
	auto filtered = filterElementsByTypeSet(manifold, elNames.begin(), elNames.end());
	auto& edgeSet = std::get<1>(filtered);

	/* get allowed edge indices */
	std::unordered_set<int> edgeIndexSet;
	for (auto& i : edgeSet) {
		edgeIndexSet.insert(manifold.getElIndex(i));
	}
	std::unordered_set<int> checked;
	std::vector<std::vector<int>> edgeIslands;
	
	for (auto& index : edgeIndexSet) {
		if (checked.find(index) != checked.end()) {
			continue;
		}
		edgeIslands.emplace_back();
		connectedElIsland(
			manifold.iMap,
			checked,
			&edgeIndexSet,
			index,
			edgeIslands.back()
		);
	}


	/* only consider edges */
	std::vector<std::set<SElement*>> edgeIslands;

	/* map edge-edge corners to single faces this way*/
	std::map<SElement*, std::map<SElement*, SingleFaceBuildData>> edgeEdgeToFaceMap;

	return s;
}