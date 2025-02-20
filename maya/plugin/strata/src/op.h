#pragma once

#include <memory>
#include <algorithm>

#include "manifold.h"

typedef StrataOpGraph;

/*TODO

for adding points, try and generate descriptive names (that might also double as
history paths) based on the topology of each one - 

so for points named A, B, C, D,
prefix each p:
pA, pB, pC, pD

each edge prefix e:
e(pA, pB), e(pB, pC) - to uniquely identify an edge, we only consider its endpoints (??????)

each face prefix f:

f( 
	e(pA, pB), 
	e(pB, pC), 
	e(pC, pD), 
	e(pD, pA) 
)
order of edges matters, direction does not - 
each face works out its own orientation,
and we will conform all orientations in a single pass later


LATER later, try and create faces only based on intersections between edges:
f( 
	e(
		p( e(pA, pB) ∩ e(pF, pG) )1, 
		pC
	),
	...
)
	eg an edge, FROM the INTERSECTION POINT of 2 other edges, TO another normal point
	add the 1? since at the limit, 2 rings on a complex surface might intersect at any number of points?  


but there's no key to type ∩ ...

...do we literally just write n, u etc?

*/

struct StrataOp {
	// maybe later we try and split topological and smooth operations
	// in maya nodes we have separate flags for "data dirty" and "topo dirty"
	std::string name;
	StrataOpGraph* graph;
	uShort index;
	
	// parent node, if scope ever happens;
	uShort parent = USNULL;

	// input connections from nodes
	uShort drivers[8] = { USNULL };

	// computed result of function
	// maybe we can say the result is always an element or array of elements?
	SmallList<uShort, 8> result;

	// does this node need all preceding spatial data to be up to date, before
	// operating on topology?
	// EG are we selecting by proximity, normals, interior points, winding number etc
	inline bool topoDependsOnData() {
		return false;
	}

	// dirty flags from previous nodes
	bool topoDirty = true; // topo needs recompute (more expensive, modifies strata structure)
	bool dataDirty = true; // data needs recompute

	// for dirty propagation, does each node need to know its descendents?
	// or can this node just check if it should be dirty when needed?
	// nodes need to know about the graph to make the raw index connections work
	// painful

	inline StrataOp* getOp(uShort opIndex) {
		return &(graph->ops[opIndex]);
	}

	inline void syncDirty() {
		for (uShort id : drivers) {
			if (getOp(id)->dataDirty) { dataDirty = true; }
			if (getOp(id)->topoDirty) { topoDirty = true; }
		}
	}
	
	virtual StrataManifold* evalTopo(StrataManifold& manifold) {}
	virtual StrataManifold* evalData(StrataManifold& manifold) {}

	inline void reset() {
		result.clear();
	}

};

struct AddPointsOp : StrataOp {
	// add one or more points to the graph
	std::vector<MMatrix> matrices;
	std::vector<std::string> names;

	SmallList<uShort, 256> result; // add a maximum of 256 points per op?
	
	virtual StrataManifold* evalTopo(StrataManifold& manifold) {
		manifold.points.reserve(matrices.size());
		for (size_t i = 0; i < matrices.size(); i++) {
			SPoint el(names[i]);
			SPointData elData;
			elData.matrix = matrices[i];
			SPoint* resultpt = manifold.addPoint(el, elData);
			result.push_back(resultpt->globalIndex);
		}
		return &manifold;
	}

	virtual StrataManifold* evalData(StrataManifold& manifold) {
		// update the matrix of each point
		for (size_t i = 0; i < matrices.size(); i++) {
			manifold.pointDatas[result[i]].matrix = matrices[i];
		}
		return &manifold;
	}

};

struct AddEdgesOp : StrataOp {
	// add hardcoded edges to the graph
	// no support for complex inputs
	// do we assume that each single input will be an array of points? 
	
	std::vector<std::string> names;
	std::vector<std::vector<uShort>> driverGlobalIds;

	virtual StrataManifold* evalTopo(StrataManifold& manifold) {
		manifold.edges.reserve(names.size());
		for (size_t i = 0; i < names.size(); i++) {
			SEdge el(names[i]);
			SEdgeData elData;
			//elData.matrix = matrices[i];
			SEdge* resultEl = manifold.addEdge(el, elData);
			result.push_back(resultEl->globalIndex);

			

			// update driver indices for this edge
			std::copy(driverGlobalIds[i].begin(), driverGlobalIds[i].end(), resultEl->drivers.begin());
			// add this edge to output edges of each driver
			for (uShort driverGlobalId : driverGlobalIds[i]) {
				StrataElement* driverPtr = manifold.elFromGlobalIndex(driverGlobalId);

				// check all drivers are points (for now)
				if (uShort(driverPtr->elType) != uShort(SElType::point)) {
					resultEl->isValid = false;
					resultEl->errorMsg = "driver " + driverPtr->name + " is not a point, not allowed in addEdgesOp";
				}
				// check if any of the drivers are already marked invalid
				if (!driverPtr->isValid) { // an invalid source invalidates all elements after it, like a NaN
					resultEl->isValid = false;
					resultEl->errorMsg = "driver " + driverPtr->name + " is already invalid";
				}
				driverPtr->edges.push_back(resultEl->globalIndex);
			}
		}
		return &manifold;
	}
};

struct AddFacesOp : StrataOp {
	// explicitly add faces, explicitly supply border edges for each
	std::vector<std::string> names;
	std::vector<std::vector<uShort>> driverGlobalIds;

	virtual StrataManifold* evalTopo(StrataManifold& manifold) {
		manifold.faces.reserve(names.size());
		for (size_t i = 0; i < names.size(); i++) {

			// check that all these elements are edges (for now)
			bool areEdges = true;

			// how do we check that all these edges are contiguous?

			SFace el(names[i]);
			SFaceData elData;
			//elData.matrix = matrices[i];
			SFace* resultEl = manifold.addFace(el, elData);
			result.push_back(resultEl->globalIndex);

			// update driver indices for this edge
			std::copy(driverGlobalIds[i].begin(), driverGlobalIds[i].end(), resultEl->drivers.begin());

			// add this edge to output edges of each driver
			for (uShort driverGlobalId : driverGlobalIds[i]) {
				StrataElement* driverPtr = manifold.elFromGlobalIndex(driverGlobalId);

				// check all drivers are edges
				if (uShort(driverPtr->elType) != uShort(SElType::edge)) {
					el.isValid = false;
					el.errorMsg = "driver " + driverPtr->name + " is not an edge, not allowed in addFacesOp";
				}
				// check if any of the drivers are already marked invalid
				if (!driverPtr->isValid) { // an invalid source invalidates all elements after it, like a NaN
					el.isValid = false;
					el.errorMsg = "driver " + driverPtr->name + " is already invalid";
				}

				driverPtr->faces.push_back(resultEl->globalIndex);
			}

		}
		return &manifold;
	}

};




