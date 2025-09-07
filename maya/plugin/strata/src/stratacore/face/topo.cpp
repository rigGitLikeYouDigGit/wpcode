

#include <string>
#include <vector>
#include "../../status.h"
#include "../../macro.h"
#include "../../visitor.h"

#include "../manifold.h"
#include "../libManifold.h"

#include "topo.h"

#include "shape.h"

using namespace strata;

struct EdgeSpan {
	std::string name;
	std::array<float, 2> params;
};

/* need sorted i points along each edge*/


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
	//std::vector<Vertex> vertices;
	std::unordered_set<int> visitedEdges; /* probably not needed*/
	std::unordered_set<int> visitedVertices; /* used during iteration to cull duplicate paths*/
	std::unordered_set<int> validEdges; /* used during iteration to cull duplicate paths*/
	std::unordered_map<int, std::vector<int>>& closedPaths; /* separate VERTEX INDEX paths, indexed by LOWEST VERTEX 
	one VERTEX maps to one CLOSED PATH -> one CLOSED FACE
	
	*/

	EdgeCircuitExtraData(
		StrataManifold& manifold_,
		std::unordered_map<int, std::vector<int>>& closedPaths_
		) : manifold(manifold_), closedPaths(closedPaths_) {};

};


/* TEMP TEMP TEMP*/
int getAnyVertex(
	StrataManifold& manifold,
	int edgeId,
	EdgeCircuitExtraData& exData,
	std::unordered_set<int>& validEdges
) {
	/* return any vertex lying on the edge
	as a start point for graph iteration

	vertices should be built at the same time we update intersections, I know this now
	*/
	SEdgeData& eData = manifold.eDataMap[manifold.getEl(edgeId)->name];
	if (!eData.vertices.size()) {
		return -1;
	}
	if (!eData._verticesSorted) {
		eData.sortVertices(manifold);
	}
	return eData.vertices[0];
}

struct EdgePathNextIdsPred : NextIdsPred {

	/* optionally pass in whole node path up to this one - last in vector*/
	template< typename ExtraT=EdgeCircuitExtraData* >
	std::vector<int> operator()(
		std::vector<int>& idPath, // VERTEX index
		GraphVisitor& visitor,
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
		Vertex& vertex = manifold.vertices[idPath.back()];
		
		/* get current edge we're travelling along*/
		int outEdge = vertex.edgeIds[1]; 
		/* check if we're travelling backwards*/
		bool outEdgeDir = vertex.edgeDirs[1];
		/* u coord on current edge of origin vertex*/
		float origU = vertex.edgeUs[1];
		SEdgeData& eData = manifold.eDataMap[manifold.getEl(outEdge)->name];
		if (!eData._verticesSorted) {
			eData.sortVertices(manifold);
		}
		/* start going forwards to the next vertex on this edge*/
		auto vIterStart = std::find(eData.vertices.begin(), eData.vertices.end(), idPath.back());
		vIterStart++;
		while (vIterStart != eData.vertices.end()) {
			
			Vertex* testV = manifold.getVertex(*vIterStart);
			/* check that source edge of new vertex is out edge of old one */
			if (testV->edgeIds[0] != outEdge) {
				vIterStart++;
				continue;
			}

			/* check that testV doesn't have the same U value as our original -
pretty sure that's always invalid
TODO: check this for direction along edge and flip sign
*/
			if (outEdgeDir) { // forwards along this edge, only want higher u values
				if (!(testV->edgeUs[0] > origU + 0.0001)) {
					vIterStart++;
					continue;
				}
			}
			else {
				if (!(testV->edgeUs[0] < origU - 0.0001)) {
					vIterStart++;
					continue;
				}
			}

			/* if this vertex is already marked visited, skip?
this set is only updated when a closed path is found*/
			if (exData.visitedVertices.find(testV->index) != exData.visitedVertices.end()) {
				vIterStart++;
				continue;
			}


			/* check backwards in the current path for that vertex - 
			* - same edge in, out, same direction on each.
			if found, it's a CLOSED PATH,
			but we still need to check if it's the shortest path
			*/
						
			// check if we find lookupVertex index in this path
			auto foundInIdPath = std::find(idPath.begin(), idPath.end(), testV->index);
			if (foundInIdPath == idPath.end()) { // nothing found, yield it for iteration
				result.push_back(testV->index);
				vIterStart++;
				continue;
			}

			/* FOUND in prev path -
			I think the breadth-first approach guarantees it's the shortest path already
			*/

			int minIndex = *std::min(foundInIdPath, idPath.end());
			/* add new closed path*/
			exData.closedPaths.insert({ minIndex,
				{foundInIdPath, idPath.end()} });
			exData.visitedVertices.insert(foundInIdPath, idPath.end());
			/* don't yield this index
			SHOULD WE just return nothing for this iteration at all? as in we've found a single
			closed path, this graph crawler is complete?
			*/

			return std::vector<int>();

			/* check that testV connects a valid edge*/
			vIterStart++;
		}

		return result;
	}
};

Status& getVertexCircuitPaths(
	Status& s,
	StrataManifold& manifold,
	std::vector<int>& edgeIsland,
	std::unordered_map<int, std::vector<int>>& vertexPaths
) {
	/* for each circuit contained in edges, 
	return a list of vertices to use to build faces

	closed edges make this quite annoying

	find a single vertex and start graph iteration - 
	vertices should be part of manifold
	*/
	IntersectionPoint startPt;
	std::unordered_set<int> islandSet(edgeIsland.begin(), edgeIsland.end());

	EdgeCircuitExtraData exData(manifold, vertexPaths);
	int firstVertex = getAnyVertex(
		manifold, edgeIsland[0],
		exData,
		islandSet
	);
	if (firstVertex < 0) { // no crossovers found
		return s;
	}

	GraphVisitor visitor;
	EdgePathNextIdsPred nextIdsPred;
	//VisitPred visitPred;
	std::vector<std::vector<int>> nodePaths;
	nodePaths.push_back({ firstVertex });
	std::vector<std::unordered_set<int>> generations;

	visitor.visit(
		nodePaths,
		generations,
		//visitPred,
		nextIdsPred,
		&exData,
		GraphVisitor::kBreadthFirst
	);
	return s;
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
			island
		);
	}
	std::sort(island.begin(), island.end());
	//return island;
}

std::string _getNameForFace(
	StrataManifold& man,
	std::vector<int>& vertexPath
) {
	/* TODO:
	super basic and bad for now - consider how we would communicate
	face "enclosed by" edges
	*/
	std::string baseName = "f";
	for (auto vId : vertexPath) {
		Vertex* v = man.getVertex(vId);
		SElement* vEdgeEl = man.getEl(v->edgeIds[0]);
		baseName += vEdgeEl->name;
	}
	return baseName;
}

Status& strata::makeNewFaceTopo(
	Status& s,
	StrataManifold& man,
	std::vector<int>& vertexPath,
	SFaceCreationParams& faceCreateParams, /* should this be packed in some other way?*/
	SElement*& el
) {
	/* from a given guaranteed closed vertex path,
	create a new strata face
	populate the passed element handle
	*/
	s = man.addElement(
		_getNameForFace(
			man, vertexPath
		),
		SElType::face,
		el
	);
	SFaceData& fData = man.fDataMap.at(el->name);
	/* set vertex ids */
	fData.vertices = vertexPath;
	/* add edges to face drivers */
	for (auto vId : vertexPath) {
		SElement* edgeEl = man.getEl(man.getVertex(vId)->edgeIds[1]);
		el->drivers.push_back(edgeEl->globalIndex);
	}
	return s;
}


Status& strata::makeFaceGroup(
	Status& s,
	StrataManifold& manifold,
	SGroup* grp,
	SFaceCreationParams& faceCreateParams,
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

	/* 
	 connected islands found - process each one to find closed vertex paths
	* TODO: add mode for "broad" vs "narrow" - do we try and fill in extra missing border curves
	* or only take exactly what we have
	*/

	for (auto& edgeIsland : edgeIslands) {
		std::unordered_map<int, std::vector<int>> vertexPaths;
		s = getVertexCircuitPaths(
			s,
			manifold,
			edgeIsland,
			vertexPaths
		);

		CWRSTAT(s, "error getting vertex path");

		/* for each closed vertex path, make a new face for that path */
		
		for (auto& vP : vertexPaths) {

			SElement* el = nullptr;
			s = makeNewFaceTopo(
				s,
				manifold,
				vP.second,
				faceCreateParams,
				el
			);
		}

	}

	return s;
}