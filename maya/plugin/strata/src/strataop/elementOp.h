#pragma once


#pragma once

#include "../stratacore/op.h"
#include "../stratacore/opGraph.h"
#include "../stratacore/manifold.h"

/* op to directly add or edit 
element data in strata -

node takes array of element structs as input, for each


if name is already found / expression matches (?) we don't edit topology directly
( that's against ethos of strata ), only update data in flowing geo with data provided

otherwise we treat as new element, setting up parents, relations, el type etc


(in maya analogue, control dirty attributes in case any data is pulled out for edge curves, point tfs etc)

each node in st graph should be able to ping "eval to here now" and have that be the cached result manifold 



internally elementOp needs to track which elements depend on which, 
then invert those sequences for back-propagation

*/

namespace ed {

	struct ElOpParam {
		StrataName name;
		expns::Expression driverExp;
		expns::Expression spaceExp;
		SPointData pData;
		Affine3f pOffset; // I DON'T KNOW WHERE THIS GOES
		SEdgeData eData;
	};

	struct StrataElementOp : StrataOp {
		/* add one or more points to the graph
		could alo use a snippet to do the same thing
		*/
		using StrataOp::StrataOp;

		//// populate these if literal worldspace inputs are given
		//std::map<StrataName, SPointData> namePointDataMap; 
		//std::map<StrataName, SEdgeData> nameEdgeDataMap; 

		std::map<StrataName, ElOpParam> paramMap;

		virtual Status makeParams();

		std::vector<StrataName> elementsAdded; // temp, used to pass information back out of op compute to maya

		///template <typename AuxT>
		virtual Status eval(StrataManifold& value, EvalAuxData* auxData, Status& s);

		virtual SAtomBackDeltaGroup bestFitBackDeltas(Status* s, StrataManifold& finalManifold, SAtomBackDeltaGroup& front);
		Status& pointProcessTargets(Status& s, StrataManifold& finalManifold, SAtomBackDeltaGroup& deltaGrp, SElement* el);
		
		virtual Status& setBackOffsetsAfterDeltas(Status& s, StrataManifold& manifold);

	};


}


