

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
	std::unordered_set<int> visitedEdges; /* used during iteration to cull duplicate paths*/
	std::map<int, std::vector<Vertex>> closedPaths; /* separate paths, indexed by start edge*/

	EdgeCircuitExtraData(StrataManifold& manifold_) : manifold(manifold_) {};
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
		std::vector<int>& idPath,
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
		EdgeCircuitExtraData& eData = *extraData;
		StrataManifold& manifold = eData.manifold;
		IntersectionRecord& rec = manifold.iMap;

		eData.visitedEdges.insert(idPath);

		/*NEXT DESTINATIONS - get intersecting edges, filter against already visited*/
		for (auto& p : rec.elMap[idPath.back()]) {
			/* first check for a CLOSED PATH
			has to be closed/match in direction as well.
			kill me.
			*/
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