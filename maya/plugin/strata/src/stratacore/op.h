#pragma once

#include <memory>
#include <unordered_set>
#include <algorithm>

#include "../macro.h"
#include "manifold.h"
#include "../exp/expParse.h"

#include "../dirtyGraph.h"

#include "opgraph.h"

namespace ed {



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


	/* BACK PROPAGATION
	
	*/

	constexpr int SDELTAMODE_LOCAL = 0; // local on top of original final matrix - by default, adding empty matrix does nothing
	constexpr int SDELTAMODE_WORLD = 1; // direct snap to given target in worldspace
	

	/* worldspace snap can only act once, other wise it's an eternal pin in the graph -
	so all of these work out to saving to space data,
	but only affects how we gather matrices from Maya?
	maybe the mode makes no difference here
	*/

	/* UVN should be allowed
	with separate flags?*/

	struct SPointDataDelta {
		SPointData data;
		int matrixMode = SDELTAMODE_LOCAL;
		int uvnMode = SDELTAMODE_LOCAL;

	};

	struct SAtomMatchTarget {
		//int index;
		int matrixMode = SDELTAMODE_LOCAL;
		int uvnMode = SDELTAMODE_LOCAL;

		Affine3f matrix;
		Vector3f uvn;
		/*std::vector<Affine3f> matrices;
		std::vector<Vector3f> uvns;*/
	};


	// always use only relative matrices and deltas to match with previous components 
	struct SAtomBackDeltaGroup {
		/* overall thing representing wavefront of elements to backpropagate
		*/

		/* one element might have multiple outputs
		and targets to match, from multiple following nodes
		*/
		std::map<StrataName, std::vector<SAtomMatchTarget>> targetMap;


		void mergeOther(SAtomBackDeltaGroup& other) {
			for (auto& p : other.targetMap) {
				targetMap.at(p.first).insert(
					targetMap.at(p.first).end(),
					p.second.begin(),
					p.second.end()
				);
			}
		}
	};



	struct StrataAuxData : EvalAuxData {

	};

	struct StrataOp : EvalNode<StrataManifold> {

		using EvalNode::EvalNode;

		using graphT = StrataOpGraph;
				
		// dense map of { param string name : param object }
		//std::map<std::string, expns::Expression > paramNameExpMap;

		// couldn't find any general way to define parametres here, just do it node by node for now


		// parent node, if scope ever happens;
		int parent = -1;

		// does this node need all preceding spatial data to be up to date, before
		// operating on topology?
		// EG are we selecting by proximity, normals, interior points, winding number etc
		inline bool topoDependsOnData() {
			return false;
		}

		// dirty flags from previous nodes
		bool connectionsDirty = true; // graph structure needs rebuild (most expensive, redo all graph tables)
		bool topoDirty = true; // topo needs recompute (more expensive, modifies strata structure)
		bool dataDirty = true; // data needs recompute

		bool paramsDirty = true; // param expressions have changed, need recompiling (later)

		//std::unordered_set<int> elements; // elements created by this node 
		std::vector<int> elements; // elements created by this node 

		/* test saving created element data on the ops that create them????
		*/
		std::map<StrataName, SPointData> opPointDataMap;
		std::map<StrataName, SEdgeData> opEdgeDataMap;


		void signalIOChanged();


		virtual void preReset() {
			// before node value is reset in graph
			// reset node data
			static_cast<EvalGraph<StrataManifold>*>(graphPtr)->nodeDatas[index] = NodeData();
		}
		/*virtual Status evalTopo(StrataManifold& manifold, Status& s) { return s; }
		virtual Status evalData(StrataManifold& manifold, Status& s) { return s; }*/

		virtual Status makeParams() { return Status(); }

		virtual graphT* getGraphPtr() { return reinterpret_cast<graphT*>(graphPtr); }


		bool isOutputNode() {
			StrataOpGraph* ptr = getGraphPtr();
			return (ptr->outputIndex == index);
		}

		/* BACK-PROPAGATION */
		SAtomBackDeltaGroup backDeltasToMatch;

		/* 2 pass system - 
		BACKWARDS fitting node parametres to given result, outputting required incoming geometry as
		further geo in turn
		
		then
		FORWARDS once all nodes have been fit as best as possible - add on local offsets on top of the incoming geo input
		*/

		/* initial gather will only run from head output node (which will probably be a
		merge node, from a shape node in maya*/
		virtual Status& gatherBackDeltas(Status& s, StrataManifold& finalManifold, SAtomBackDeltaGroup& result) {
			/* get initial deltas from end node
			*/
			return s;
		}

			/* assume this will run parallel between nodes - each object should only know immediate components to affect,
			and result will only be components taken in by this node*/
		virtual SAtomBackDeltaGroup bestFitBackDeltas(Status* s, StrataManifold& finalManifold, SAtomBackDeltaGroup& front) {
			/* pass in deltas to match -
			
			we work out what INPUTS (if any) would best match the given targets- 
			return new AtomDeltaGroup representing that.

			SAVE initial target deltas on this op to match later

			those deltas may not be able to be matched exactly - 
			subsequent method builds final offsets on this node, from previous node's best effort
			*/

			// save deltas on this node
			backDeltasToMatch = front;
			SAtomBackDeltaGroup result(front); // copy so we pass through any other elements
			// erase any elements created by this node from result
			for (int i : elements) { 
				auto name = finalManifold.getEl(i)->name;
				auto found = result.targetMap.find(name);
				if (found != result.targetMap.end()) {
					result.targetMap.erase(found->first);
				}
			}
			return result;
			}

		virtual Status& setBackOffsetsAfterDeltas(
			Status& s, StrataManifold& manifold) {
			/* finalise offsets on top of fitted inputs to match 
			saved deltas
			this node will have been evaluated - make sure to 
			edit the strata manifold element datas too?
			or we just point back into these nodes to retrieve data

			*/
			return s;
		}

		Status& runBackPropagation(
			Status& s, 
			StrataOp* fromNode, 
			StrataManifold& finalManifold, 
			SAtomBackDeltaGroup deltaGrp,
			StrataAuxData& auxData
		) {
			/* overall top-level back prop function
			assume deltas have already been gathered outside of this

			from that work out what nodes created the affected elements.
			check nodes breadth-first from output node backwards,
			calling prop methods on those nodes if they created elements in delta group
			*/
			//std::set<StrataOp*> toVisit({ fromNode });
			//std::set<StrataOp*> nextToVisit; // don't know how to do breadth-first in leet code

			// get generations in history of nodes
			std::vector<std::vector<int>> generations = getGraphPtr()->nodesInHistory(fromNode->index, true);

			/* backwards pass, getting target deltas for elements/ nodes to match
			*/
			for(int i = 0; i < static_cast<int>(generations.size()); i++) {

				std::vector<int>& toVisit = generations[i];
				std::vector<SAtomBackDeltaGroup> resultDeltas(toVisit.size()); // results of this iteration of back-prop
				for (int n = 0; n < static_cast<int>(toVisit.size()); n++) { // parallel this
					StrataOp* op = getGraphPtr()->getNode<StrataOp>(toVisit[n]);
					resultDeltas[n] = op->bestFitBackDeltas(&s, finalManifold, deltaGrp);
					CRMSG(s, "error running back propagation on node " + op->name);
				}

				// collate delta fronts
				for (int n = 1; n < static_cast<int>(resultDeltas.size()); n++) {
					resultDeltas[0].mergeOther(resultDeltas[n]);
				}
				deltaGrp = resultDeltas[0]; // need to do a copy here because we create the intermediate vals in the loop scope
			}

			/* now forwards pass, eval'ing nodes and setting final offsets
			*/
			for (int i = 0; i < static_cast<int>(generations.size()); i++) {
				
				std::vector<int>& toVisit = *(generations.rbegin() + i);
				for (int n = 0; n < static_cast<int>(toVisit.size()); n++) { // parallel this
					StrataOp* op = getGraphPtr()->getNode<StrataOp>(toVisit[n]);
					s = getGraphPtr()->evalGraph(s, toVisit[n], &auxData); // eval node with new best-fit params
					//StrataManifold& m = op->value();
					s = op->setBackOffsetsAfterDeltas(s, op->value());
					op->setDirty(true);
					getGraphPtr()->nodePropagateDirty(op->index);
				}
			}

			return s;
		}

	};




	//struct LoftFaceOp : StrataOp {
	//	/* 2 separate operations - first create 2 new
	//	edges at extremities of input edges,
	//	then use those as borders

	//	if you loft 2 rings, add 2 equivalent edges in different directions?
	//	or just a single edge, included in the face twice
	//	has to be this, since any shape modifications have to affect both "sides" of
	//	the face exactly
	//	*/
	//};


	//struct RailFaceOp : StrataOp {
	//	/* add a face covering more than one edge in
	//	u and v - face MUST be rectangular (however we
	//	work that out)
	//	*/
	//};

}
//
//template<>
//struct std::hash<ed::StrataOp>
//{
//	std::size_t operator()(const ed::StrataOp& s) const noexcept
//	{
//		return std::hash<int>{}(s.index);
//
//		//std::size_t h1 = std::hash<std::string>{}(s.first_name);
//		//std::size_t h2 = std::hash<std::string>{}(s.last_name);
//		//return h1 ^ (h2 << 1); // or use boost::hash_combine
//	}
//};
