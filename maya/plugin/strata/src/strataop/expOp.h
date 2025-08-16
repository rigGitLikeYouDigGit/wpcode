#pragma once

#include "../stratacore/op.h"
#include "../stratacore/opGraph.h"
#include "../stratacore/manifold.h"

/* 
* 
* 

*/

//namespace strata {
//
//	const int EL_OP_GLOBAL = 0;
//	const int EL_OP_LOCAL = 1;
//
//	struct ElOpParam {
//		StrataName name;
//		expns::Expression driverExp;
//		expns::Expression spaceExp;
//		int spaceMode = EL_OP_GLOBAL;
//		SPointData pData;
//		Affine3f pOffset; // I DON'T KNOW WHERE THIS GOES
//		SEdgeData eData;
//	};
//
//	struct StrataElementOp : StrataOp {
//		/* add one or more points to the graph
//		could alo use a snippet to do the same thing
//		*/
//		using StrataOp::StrataOp;
//
//		//// populate these if literal worldspace inputs are given
//		//std::map<StrataName, SPointData> namePointDataMap; 
//		//std::map<StrataName, SEdgeData> nameEdgeDataMap; 
//
//		std::map<StrataName, ElOpParam> paramMap;
//		// is this the best way to iterate over dictionary in creation order?
//		std::vector<StrataName> paramNames = {};
//
//		virtual Status makeParams();
//
//		//std::vector<StrataName> elementsAdded = {}; // temp, used to pass information back out of op compute to maya
//		std::vector<StrataName> elementsAdded = {}; // temp, used to pass information back out of op compute to maya
//
//		///template <typename AuxT>
//		virtual Status eval(StrataManifold& value, EvalAuxData* auxData, Status& s);
//
//		virtual SAtomBackDeltaGroup bestFitBackDeltas(Status* s, StrataManifold& finalManifold, SAtomBackDeltaGroup& front);
//		Status& pointProcessTargets(Status& s, StrataManifold& finalManifold, SAtomBackDeltaGroup& deltaGrp, SElement* el);
//
//		//virtual Status& setBackOffsetsAfterDeltas(Status& s, StrataManifold& manifold);
//
//		virtual StrataElementOp* clone_impl() const;
//
//	};
//}