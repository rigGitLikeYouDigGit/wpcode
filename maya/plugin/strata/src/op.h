#pragma once

#include <memory>
#include <algorithm>

#include "manifold.h"

typedef StrataOpGraph;

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
	std::vector<std::vector<uShort>> indices;

	virtual StrataManifold* evalTopo(StrataManifold& manifold) {
		manifold.points.reserve(names.size());
		for (size_t i = 0; i < names.size(); i++) {
			SEdge el(names[i]);
			SEdgeData elData;
			//elData.matrix = matrices[i];
			SEdge* resultEl = manifold.addEdge(el, elData);
			result.push_back(resultEl->globalIndex);

			// update driver indices for this edge
			std::copy(indices[i].begin(), indices[i].end(), resultEl->drivers.begin());
		}
		return &manifold;
	}
};

