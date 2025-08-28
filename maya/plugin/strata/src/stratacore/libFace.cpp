

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

struct SingleFaceBuildData {
	std::vector<EdgeSpan> edges; /* ordered edges to use to create this face -
	not guaranteed to connect?*/

	/* 2 crossing edges could connect to 4 separate faces - */
};

Status& _makeFaceGroupFromEdgeIslands(
	std::vector<std::set<SElement*>>& edgeIslands
) {

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

	ItIntersectingElements(
		IntersectionRecord& rec_
	) : rec(rec_) {
		elId = rec.elMap.begin()->first;
	}

	ItIntersectingElements& operator++(int n) {

		return *this;
	}




};

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

	/* only consider edges */
	std::vector<std::set<SElement*>> edgeIslands;

	/* map edge-edge corners to single faces this way*/
	std::map<SElement*, std::map<SElement*, SingleFaceBuildData>> edgeEdgeToFaceMap;


	/* find all edges intersecting each edge,
	check if they're included in face group setup

	group to connected islands, then process the islands
	somehow


	elA - elB - elC

	if iteration finds elA and elC before elB, it won't realise they are connected - 
	

	*/
	auto& edgeSet = std::get<1>(filtered);
	for (SElement* edgeEl : edgeSet) {

		float uVals[2] = { 0.0, 1.0 };
		auto& connectedIndexMap = manifold.iMap.elMap[edgeEl->globalIndex];
		for (auto& p : connectedIndexMap) {
			/* get connected element*/
			SElement* connectEl = manifold.getEl(p.first);
			/* check if part of the face group expression*/
			if (edgeSet.find(connectEl) == edgeSet.end()) {
				// if not, skip
				continue;
			}
			// we've found an included edge that intersects this edge, find intersection's uValue on this edge
			/* edge KEY in face is its LOCAL INTERSECTION with that face?
			so it's enough to say  (eA > eB) < eC to find the span?

			*/
			
			
		}

	}


	return s;
}