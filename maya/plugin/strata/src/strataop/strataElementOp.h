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

*/

namespace ed {

	struct StrataElementParam {

		std::string name;
		std::string exp; // if supplied, element matching will be assigned given name
		SElType elType = SElType::NONE;

		// uninitialised spatial data should just give default behaviour if not supplied
	};


	struct StrataElementOp : StrataOp {
		// add one or more points to the graph
		std::vector<StrataElementParam> params;
		

		Status evalTopo(StrataManifold& manifold, Status& s) {
			//manifold.pointDatas.reserve(names.size());
			//manifold.points.reserve(names.size());

			//std::string outGroupName = name + ":out";
			//StrataGroup* outGroup = manifold.getGroup(outGroupName, true, static_cast<int>(names.size()));

			//for (size_t i = 0; i < names.size(); i++) {
			//	SPoint el(names[i]);
			//	SPoint* resultpt = manifold.addPoint(el, pointDatas[i]);
			//	outGroup->contents.push_back(resultpt->globalIndex);
			//}
			return s;
		}

		Status evalData(StrataManifold& manifold, Status& s) {
			// update the matrix of each point
			//std::string outGroupName = name + ":out";
			//StrataGroup* outGroup = manifold.getGroup(outGroupName, false);
			//for (size_t i = 0; i < pointDatas.size(); i++) {
			//	manifold.pointDatas[outGroup->contents[static_cast<int>(i)]] = pointDatas[static_cast<int>(i)];
			//}
			return s;
		}

	};


}


