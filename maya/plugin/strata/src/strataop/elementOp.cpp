
#include "elementOp.h"
#include "../stringLib.h"

using namespace ed;
using namespace ed::expns;

Status StrataElementOp::makeParams() {
	Status s;
	paramNameExpMap["pNew"] = Expression("");
	return s;
}


Status StrataElementOp::eval(StrataManifold& value, 
	EvalAuxData* auxData, Status& s) 
{
	/*
	IF EXISTING DATA FOR ELEMENT FOUND IN MANIFOLD:
		use that
	ELSE:
		use data from op's parametres
	
	*/
	DEBUGSL("EL OP EVAL");
	//StrataElementOp* opPtr = static_cast<StrataElementOp*>(node);
	elementsAdded.clear();
	elementsAdded.reserve(paramNameExpMap.size());
	StrataElementOp* opPtr = this;
	DEBUGS("n params to eval: " + std::to_string(paramNameExpMap.size()));
	for (auto& p : paramNameExpMap) {
		if (p.first[0] == '!') {// parent expression
			continue;
		}
		SElement* outPtr = nullptr;
		ExpAuxData expAuxData;
		expAuxData.manifold = &value;
		std::vector<ExpValue>* resultVals = nullptr;

		/* first evaluate driver expression*/
		DEBUGS("eval param: " + p.first + " : " + p.second.srcStr);
		switch (p.first[0]) { // get the type of element to add by the first letter of the name

			case 'p': { // a point has no drivers (yet, later allow declaring points as the direct output of topo operations
				DEBUGS("adding new point");
				s = value.addElement(SElement(p.first, StrataElType::point), outPtr);
				CWRSTAT(s, "Error adding new element");
				break;
			}

			case 'e': {
				/* eval expression to get list of driving elements */

				s = p.second.result(resultVals, &expAuxData);
				CWRSTAT(s, "error getting exp result, halting");
				if (resultVals == nullptr) {
					STAT_ERROR(s, "edge driver result gave a nullptr");
				}
				std::vector<int> drivers = expAuxData.expValuesToElements(*resultVals, s);
				CWRSTAT(s, "error converting final exp result to strata drivers, halting");

				// add new edge element 
				s = value.addElement(SElement(p.first, StrataElType::edge), outPtr);
				CWRSTAT(s, "Error adding new edge element: " + p.first + " halting");

				SEdgeData& edgeData = value.edgeDatas.at(outPtr->elIndex);
				// add drivers to new edge

				/* TODO: drivers still need coords - 
				how do we specify them in exp, 
				or how do we pass them in from maya node
				smooth data should not be in expressions
				*/

				for (auto& i : drivers) {
					SElement* elPtr = value.getEl(i);
					outPtr->drivers.push_back(i);
					edgeData.driverDatas.push_back(EdgeDriverData());
					EdgeDriverData& driverData = edgeData.driverDatas.at(edgeData.driverDatas.size()-1);
					driverData.index = i;
					MMatrix result;
					float defaultCoords[3] = {0.5, 0.5, 0.5};
					if (elPtr->elType == StrataElType::point) { // no offsets on points
						defaultCoords[0] = 0;
						defaultCoords[1] = 0;
						defaultCoords[2] = 0;
					}
					s = value.matrixAt(
						elPtr->globalIndex,
						defaultCoords,
						result, s);
					CWRSTAT(s, "error getting default driver matrix");
					driverData.driverMatrix = result;
					elPtr->edges.push_back(outPtr->globalIndex);
				}
				break;
			}
			case 'f': {
				/* eval expression to get list of driving elements */
				ExpAuxData expAuxData;
				std::vector<ExpValue>* resultVals = nullptr;
				 
				s = p.second.result(resultVals, &expAuxData);
				CWRSTAT(s, "error getting exp result, halting");
				if (resultVals == nullptr) {
					STAT_ERROR(s, " getting result returned a nullptr");
				}
				std::vector<int> drivers = expAuxData.expValuesToElements(*resultVals, s);
				CWRSTAT(s, "error converting final exp result to strata drivers, halting");

				// add new edge element 
				s = value.addElement(SElement(p.first, StrataElType::face), outPtr);
				CWRSTAT(s, "Error adding new face element: " + p.first + " halting");

				// add drivers to new edge
				for (auto& i : drivers) {
					SElement* elPtr = value.getEl(i);
					outPtr->drivers.push_back(i);
					elPtr->faces.push_back(outPtr->globalIndex);
				}
				break;
			}
			default: {
				STAT_ERROR(s, "INVALID ELEMENT PREFIX: " + p.first + "; must begin with one of p, e, f");
			}
		}
		if (outPtr == nullptr) {
			STAT_ERROR(s, "outPtr not set correctly when adding element: " + p.first + ", halting");
		}
		elementsAdded.push_back(outPtr->globalIndex);
		/* eval parent expression*/

		//if (paramNameExpMap.count(p.first + "!") == 0) {
		if (paramNameExpMap.count("!" + p.first) == 0) {
			DEBUGS("no parent expression found for element " + p.first + " - continuing");
			continue;
		}
		//Expression& parentExp = paramNameExpMap[p.first + "!"];
		Expression& parentExp = paramNameExpMap["!" + p.first];
		trimEnds(parentExp.srcStr);
		//return s;
		// check if expression defines a parent point or driver
		resultVals = nullptr;
		switch (p.first[0]) { // get the type of element added again
			case 'p': {
				if (!parentExp.srcStr.empty()) {
					DEBUGS("parent expression str not empty, eval ing exp");
					//return s;  // works up to here
					s = p.second.result(resultVals, &expAuxData);
					CWRSTAT(s, "Error evaling point exp result, halting");
					if (resultVals == nullptr) {
						STAT_ERROR(s, "point parent result gave nullptr");
					}
					//return s;


					std::vector<int> drivers = expAuxData.expValuesToElements(*resultVals, s); // error in resultValuesToElements
					/* issue seems to be in derefing vector of ExpValues*/
					CWRSTAT(s, "error converting final exp result to strata drivers, halting");
					// add drivers to new point (should only be 1 I think?)
					for (auto& i : drivers) {
						SElement* elPtr = value.getEl(i);
						outPtr->drivers.push_back(i);
						elPtr->points.push_back(outPtr->globalIndex);
					}
				}

				DEBUGS("ended empty exp check");
				//return s;
				// check if we have data for this point - if so, set its matrix
				if (opPtr->namePointDataMap.count(p.first)) {
					DEBUGS("found point data for " + p.first);
					DEBUGS("manifold pointDatas length:" + std::to_string(value.pointDatas.size()));
					DEBUGS("newEl: " + std::to_string(outPtr->elIndex) + ", " + outPtr->name);
					//return s;
					value.pointDatas[outPtr->elIndex] = opPtr->namePointDataMap[p.first];
				}
				elementsAdded.push_back(outPtr->globalIndex);
				DEBUGS("finish adding point");
				break;
			}
		}
		
	}
	DEBUGSL("EL EVAL COMPLETE");

	return s;
}

//std::vector<StrataElementParam> params;
//virtual StrataElementOp* clone_impl() const override {
//	return new StrataElementOp(*this); 
//};


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