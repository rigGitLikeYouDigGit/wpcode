#pragma once

#include <memory>
#include <unordered_set>
#include <algorithm>

#include "../macro.h"
#include "manifold.h"


namespace ed {

	struct StrataOpGraph;

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

	struct ExpNode {
		/* individual node for parametre expressions in nodes -
		* eval-ing whole expression tree should end up as a list of ints?
		* paired lists of opId, element gId?
		*
		* fine to recurse in evaluation here
		*/
	};

	struct StrataOp {
		// maybe later we try and split topological and smooth operations
		// in maya nodes we have separate flags for "data dirty" and "topo dirty"
		std::string name;
		StrataOpGraph* graph;
		int index;

		// parent node, if scope ever happens;
		int parent = -1;

		// temp values used in traversal
		int temp_inDegree = 0;
		int temp_generationId = 0;
		//int temp_visited = 0;

		// data travelling in the graph is always (?) a manifold, even if it only contains a group of points
		// first input (i0) is default, and always corresponds to the main incoming manifold data to write to
		std::vector<int> inputs;
		std::unordered_map<std::string, int> inputAliases;

		StrataManifold result; // test caching on all nodes, not sure how expensive that might get

		// does this node need all preceding spatial data to be up to date, before
		// operating on topology?
		// EG are we selecting by proximity, normals, interior points, winding number etc
		inline bool topoDependsOnData() {
			return false;
		}

		// dirty flags from previous nodes
		bool topoDirty = true; // topo needs recompute (more expensive, modifies strata structure)
		bool dataDirty = true; // data needs recompute

		bool paramsDirty = true; // param expressions have changed, need recompiling (later)
		ExpNode* rootExpNode = nullptr;

		// for dirty propagation, does each node need to know its descendents?
		// or can this node just check if it should be dirty when needed?
		// nodes need to know about the graph to make the raw index connections work
		// painful

		//inline StrataOp* getOp(int opIndex) {
		//	return &(graph->ops[opIndex]);
		//}
		//inline StrataOp* getOp(std::string opName) {
		//	return &(graph->ops[graph->opNameIndexMap[opName]]);
		//}
		StrataOp* getOp(int opIndex);
		StrataOp* getOp(std::string opName);

		//void signalIOChanged() {
		//	/*reach back out to graph, set its structure dirty flag.
		//	* yes I KNOW this is crazy dependency, maybe the graph should be the only thing that
		//	* tracks IO on nodes?
		//	* maybe we should do a proper signalling system?
		//	*/
		//	graph->graphChanged = true;

		//}

		void signalIOChanged();

		inline void syncDirty() {
			for (int id : inputs) {
				if (getOp(id)->dataDirty) { dataDirty = true; }
				if (getOp(id)->topoDirty) { topoDirty = true; }
			}
		}

		virtual ExpNode* evalParams() {
			/* run over all node parametre expressions , compile individual ASTs
			*/
			return &(ExpNode());
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

		SmallList<int, 256> result; // add a maximum of 256 points per op?

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
				manifold.pointDatas[result[static_cast<int>(i)]].matrix = matrices[static_cast<int>(i)];
			}
			return &manifold;
		}

	};

	struct AddEdgesOp : StrataOp {
		// add hardcoded edges to the graph
		// no support for complex inputs
		// do we assume that each single input will be an array of points? 

		std::vector<std::string> names;
		std::vector<std::vector<int>> driverGlobalIds;

		virtual StrataManifold* evalTopo(StrataManifold& manifold) {
			manifold.edges.reserve(names.size());
			for (size_t i = 0; i < names.size(); i++) {
				SEdge el(names[i]);
				SEdgeData elData;
				//elData.matrix = matrices[i];
				SEdge* resultEl = manifold.addEdge(el, elData);
				//result.push_back(resultEl->globalIndex);



				// update driver indices for this edge
				std::copy(driverGlobalIds[i].begin(), driverGlobalIds[i].end(), resultEl->drivers.begin());
				// add this edge to output edges of each driver
				for (int driverGlobalId : driverGlobalIds[i]) {
					StrataElement* driverPtr = manifold.getEl(driverGlobalId);

					// check all drivers are points (for now)
					if (int(driverPtr->elType) != int(SElType::point)) {
						resultEl->isInvalid = true;;
						resultEl->errorMsg = "driver " + driverPtr->name + " is not a point, not allowed in addEdgesOp";
					}
					// check if any of the drivers are already marked invalid
					if (!driverPtr->isInvalid) { // an invalid source invalidates all elements after it, like a NaN
						resultEl->isInvalid = true;
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
		std::vector<std::vector<int>> driverGlobalIds;

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
				//result.push_back(resultEl->globalIndex);


				// update driver indices for this edge
				std::copy(driverGlobalIds[i].begin(), driverGlobalIds[i].end(), resultEl->drivers.begin());

				if (!manifold.edgesAreContiguousRing(resultEl->drivers)) {
					el.isInvalid = true;
					el.errorMsg = "driver edges for " + resultEl->name + " do not form contiguous ring, \
					invalid border for face";
				}

				// add this edge to output edges of each driver
				for (int driverGlobalId : driverGlobalIds[i]) {
					StrataElement* driverPtr = manifold.getEl(driverGlobalId);

					// check all drivers are edges
					if (int(driverPtr->elType) != int(SElType::edge)) {
						el.isInvalid = true;
						el.errorMsg = "driver " + driverPtr->name + " is not an edge, not allowed in addFacesOp";
					}
					// check if any of the drivers are already marked invalid
					if (!driverPtr->isInvalid) { // an invalid source invalidates all elements after it, like a NaN
						el.isInvalid = true;
						el.errorMsg = "driver " + driverPtr->name + " is already invalid";
					}

					driverPtr->faces.push_back(resultEl->globalIndex);
				}

			}
			return &manifold;
		}

	};


	struct LoftFaceOp : StrataOp {
		/* 2 separate operations - first create 2 new
		edges at extremities of input edges,
		then use those as borders

		if you loft 2 rings, add 2 equivalent edges in different directions?
		or just a single edge, included in the face twice
		has to be this, since any shape modifications have to affect both "sides" of
		the face exactly
		*/
	};


	struct RailFaceOp : StrataOp {
		/* add a face covering more than one edge in
		u and v - face MUST be rectangular (however we
		work that out)
		*/
	};

}

template<>
struct std::hash<ed::StrataOp>
{
	std::size_t operator()(const ed::StrataOp& s) const noexcept
	{
		return std::hash<int>{}(s.index);

		//std::size_t h1 = std::hash<std::string>{}(s.first_name);
		//std::size_t h2 = std::hash<std::string>{}(s.last_name);
		//return h1 ^ (h2 << 1); // or use boost::hash_combine
	}
};
