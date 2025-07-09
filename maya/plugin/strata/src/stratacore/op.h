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

		Affine3f matrix = Affine3f::Identity();
		Vector3f uvn = { 0, 0, 0 };
		/*std::vector<Affine3f> matrices;
		std::vector<Vector3f> uvns;*/
		int spaceIndex = -1; // if not -1, this target applies locally only to the relative data in this space

		inline std::string strInfo() {
			std::stringstream matStr;
			matStr << matrix.matrix() ;

			std::stringstream uvnStr;
			uvnStr << uvn;

			return "<tgt: " + matStr.str() + ", " + uvnStr.str() + ">";
		}
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
		inline std::string strInfo() {
			std::string result = "<dgrp: ";
			for (auto& p : targetMap) {
				result += "{" + p.first + ":";
				for (auto& v : p.second) {
					result += v.strInfo() + ",";
				}
				result += "}";
			}	
			result += ">";
			return result;
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
		std::vector<int> elements = {}; // elements created by this node 

		/* test saving created element data on the ops that create them????
		*/
		std::map<StrataName, SPointData> opPointDataMap; /* use as check if element created on previous cycle*/
		std::map<StrataName, SEdgeData> opEdgeDataMap;

		/* BACK-PROPAGATION */
		SAtomBackDeltaGroup backDeltasToMatch;


		//virtual StrataOp* clone_impl() const { return new StrataOp(*this); };
		template <typename T>
		T* clone_impl() const { return new T(*reinterpret_cast<const T*>(this)); }
		virtual StrataOp* clone_impl() const {
			LOG("OP BASE clone_impl() - WRONG");
			return clone_impl<StrataOp>(); };

		virtual std::unique_ptr<DirtyNode> clone() const { 
			LOG("OP BASE clone()");
			return std::unique_ptr<StrataOp>(this->clone_impl()); }

		virtual Status eval(StrataManifold& value,
			EvalAuxData* auxData, Status& s);

		void signalIOChanged();


		virtual void preReset() {
			// before node value is reset in graph
			// reset node data
			//static_cast<EvalGraph<StrataManifold>*>(graphPtr)->nodeDatas[index] = NodeData();
		}
		/*virtual Status evalTopo(StrataManifold& manifold, Status& s) { return s; }
		virtual Status evalData(StrataManifold& manifold, Status& s) { return s; }*/

		virtual Status makeParams() { return Status(); }

		virtual graphT* getGraphPtr() { return reinterpret_cast<graphT*>(graphPtr); }


		bool isOutputNode() {
			StrataOpGraph* ptr = getGraphPtr();
			return (ptr->_outputIndex == index);
		}


		/* 2 pass system - 
		BACKWARDS fitting node parametres to given result, outputting required incoming geometry as
		further geo in turn
		
		then
		FORWARDS once all nodes have been fit as best as possible - add on local offsets on top of the incoming geo input
		*/

		/* initial gather will only run from head output node (which will probably be a
		merge node, from a shape node in maya*/
		virtual Status& gatherBackDeltas(Status& s, StrataManifold& finalManifold, SAtomBackDeltaGroup& result);

			/* assume this will run parallel between nodes - each object should only know immediate components to affect,
			and result will only be components taken in by this node*/
		virtual SAtomBackDeltaGroup bestFitBackDeltas(Status* s, StrataManifold& finalManifold, SAtomBackDeltaGroup& front);

		/* we now treat offset targets within normal evaluation
		*/
		//virtual Status& setBackOffsetsAfterDeltas(
		//	Status& s, StrataManifold& manifold);

		

		virtual Status& runBackPropagation(
			Status& s,
			StrataOp* fromNode,
			StrataManifold& finalManifold,
			SAtomBackDeltaGroup deltaGrp,
			StrataAuxData& auxData
		);

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
