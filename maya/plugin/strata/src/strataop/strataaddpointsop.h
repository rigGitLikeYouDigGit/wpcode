#pragma once

#include "../stratacore/op.h"
#include "../stratacore/opGraph.h"
#include "../stratacore/manifold.h"

/* op to add multiple new points to strata manifold.
create new elements, new point datas, assign indices, group as node result - that's it*/

namespace ed {




	struct StrataAddPointsOp : StrataOp {
		// add one or more points to the graph
		std::vector<std::string> names;
		std::vector<SPointData> pointDatas; // build separately

		virtual void evalTopo(StrataManifold& manifold) {
			manifold.pointDatas.reserve(names.size());
			manifold.points.reserve(names.size());
			
			std::string outGroupName = name + ":out";
			StrataGroup* outGroup = manifold.getGroup(outGroupName, true, names.size());

			for (size_t i = 0; i < names.size(); i++) {
				SPoint el(names[i]);
				SPoint* resultpt = manifold.addPoint(el, pointDatas[i]);
				outGroup->contents.push_back(resultpt->globalIndex);
			}
		}

		virtual void evalData(StrataManifold& manifold) {
			// update the matrix of each point
			std::string outGroupName = name + ":out";
			StrataGroup* outGroup = manifold.getGroup(outGroupName, false);
			for (size_t i = 0; i < pointDatas.size(); i++) {
				manifold.pointDatas[outGroup->contents[static_cast<int>(i)]] = pointDatas[static_cast<int>(i)];
			}
			//return &manifold;
		}

	};


	struct StrataAddPointsOp : public StrataOp {

		// only parametre to node is names, and then pointDatas
		std::vector<std::string> pointNamesToAdd;

		void evalTopo(StrataManifold& manifold) {
			/* add new point elements, group them as this node's output
			*/
			std::string outGroupName = name + ":out"; 
			StrataGroup* outGroup = manifold.getGroup(outGroupName);
			for (size_t i = 0; i < pointNamesToAdd.size(); i++) {
				SPoint* newPoint = manifold.addPoint(SPoint(pointNamesToAdd[i]), SPointData());
				// add global index to out group
				outGroup->contents.insert(newPoint->globalIndex);
			}
		}
		void evalData(StrataManifold& manifold) {}

	};

}
