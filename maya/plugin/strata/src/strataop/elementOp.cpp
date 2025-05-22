
#include "elementOp.h"
#include "../stringLib.h"

using namespace ed;
using namespace ed::expns;

Status StrataElementOp::makeParams() {
	Status s;
	paramNameExpMap["pNew"] = Expression("");
	return s;
}

Status& pointEvalDriverExpression(
	Status& s, SElement*& outPtr, StrataManifold& value, ExpAuxData& expAuxData,
	std::vector<ExpValue>* resultVals,
	const std::string& paramName, ed::expns::Expression& exp) {
	/* a separate function here may be slightly overkill
	
	DO WE allow input data for pure drivers, and PROJECT that data on to driver geo?
	
	for points, it's the nearest point on driver geo to the target point?
	for edges, flatten target curve into driver surface?
	*/
	s = value.addElement(SElement(paramName, StrataElType::point), outPtr);
	CWRSTAT(s, "Error adding new element");
	return s;
}

Status& edgeEvalDriverExpression(
	Status& s, SElement*& outPtr, StrataManifold& value, ExpAuxData& expAuxData,
	std::vector<ExpValue>* resultVals,
	const std::string& paramName, ed::expns::Expression& exp) {

	s = exp.result(resultVals, &expAuxData);
	CWRSTAT(s, "error getting exp result, halting");
	if (resultVals == nullptr) {
		STAT_ERROR(s, "edge driver result gave a nullptr");
	}
	std::vector<int> drivers = expAuxData.expValuesToElements(*resultVals, s);
	CWRSTAT(s, "error converting final exp result to strata drivers, halting");

	// add new edge element 
	s = value.addElement(SElement(paramName, StrataElType::edge), outPtr);
	CWRSTAT(s, "Error adding new edge element: " + paramName + " halting");

	SEdgeData& edgeData = value.edgeDatas.at(outPtr->elIndex);
	// add drivers to new edge

	/* TODO: drivers still need coords -
	how do we specify them in exp,
	or how do we pass them in from maya node
	smooth data should not be in expressions

	MAYBE we have a constraint that curves can only be driven directly by
	points? each direct exp must resolve to a point - leaving freedom as to how we
	arrive at that point (via closest-point, param lookup on surface beforehand, etc)

	ONLY POINTS FOR NOW
	CUT ALL THE SCOPE
	GET THE DEMO OUT

	*/

	for (auto& i : drivers) {
		SElement* driverPtr = value.getEl(i);
		//outPtr->drivers.push_back(i);
		outPtr->drivers.push_back(driverPtr->name);
		edgeData.driverDatas.push_back(EdgeDriverData());
		EdgeDriverData& driverData = edgeData.driverDatas.at(edgeData.driverDatas.size() - 1);
		driverData.index = i;

		double defaultCoords[3] = { 0.0, 0.0, 0.0 };
		// TODO: HOW DO WE DEFINE DRIVER PARAMETRES
		if (driverPtr->elType == StrataElType::edge) {
			defaultCoords[0] = 0.5;
		}

		s = value.matrixAt(s, driverData.finalMatrix,
			driverPtr->globalIndex, defaultCoords
		);
		CWRSTAT(s, "ERROR getting MatrixAt for driver el: " + driverPtr->name + " for edge: " + outPtr->name);

		driverPtr->edges.push_back(outPtr->globalIndex);

	}

	s = value.buildEdgeData(s, edgeData);

	return s;
}


Status& faceEvalDriverExpression(
	Status& s, SElement*& outPtr, StrataManifold& value, ExpAuxData& expAuxData,
	std::vector<ExpValue>* resultVals,
	const std::string& paramName, ed::expns::Expression& exp) 
{
	//ExpAuxData expAuxData;
	//std::vector<ExpValue>* resultVals = nullptr;

	s = exp.result(resultVals, &expAuxData);
	CWRSTAT(s, "error getting exp result, halting");
	if (resultVals == nullptr) {
		STAT_ERROR(s, " getting result returned a nullptr");
	}
	std::vector<int> drivers = expAuxData.expValuesToElements(*resultVals, s);
	CWRSTAT(s, "error converting final exp result to strata drivers, halting");

	// add new edge element 
	s = value.addElement(SElement(paramName, StrataElType::face), outPtr);
	CWRSTAT(s, "Error adding new face element: " + paramName + " halting");

	// add drivers to new edge
	for (auto& i : drivers) {
		SElement* elPtr = value.getEl(i);
		//outPtr->drivers.push_back(i);
		outPtr->drivers.push_back(elPtr->name);
		elPtr->faces.push_back(outPtr->globalIndex);
	}
	return s;
}


Status& evalDriverExpression(
	Status& s, SElement*& outPtr, StrataManifold& value, ExpAuxData& expAuxData,
	std::vector<ExpValue>* resultVals,
	const std::string& paramName, ed::expns::Expression& exp) {

	//auto p = std::make_pair(paramName, exp);

	/* first evaluate driver expression*/
	//DEBUGS("eval param: " + p.first + " : " + p.second.srcStr);
	DEBUGS("eval param: " + paramName + " : " + exp.srcStr);
	switch (paramName[0]) { // get the type of element to add by the first letter of the name
		case 'p': { // a point has no drivers (yet, later allow declaring points as the direct output of topo operations
			DEBUGS("adding new point");
			s = pointEvalDriverExpression(s, outPtr, value, expAuxData,
				resultVals,
				paramName, exp
			);
			break;
		}

		case 'e': {
			/* eval expression to get list of driving elements */
			s = edgeEvalDriverExpression(s, outPtr, value, expAuxData,
				resultVals,
				paramName, exp);
			break;
		}
		case 'f': {
			/* eval expression to get list of driving elements */
			s = faceEvalDriverExpression(s, outPtr, value, expAuxData,
				resultVals,
				paramName, exp);
			break;
		}
		default: {
			STAT_ERROR(s, "INVALID ELEMENT PREFIX: " + paramName + "; must begin with one of p, e, f");
		}
	}
	return s;
}


Status& pointEvalParentExpression(
	Status& s, SElement*& outPtr, StrataManifold& value, ExpAuxData& expAuxData,
	std::vector<ExpValue>* resultVals,
	const std::string& paramName, ed::expns::Expression& parentExp) {

	//auto p = std::make_pair(paramName, exp);

	/* first evaluate driver expression*/
	//DEBUGS("eval param: " + p.first + " : " + p.second.srcStr);
	return s;
}



Status& evalParentExpression(
	Status& s, SElement*& outPtr, StrataManifold& value, ExpAuxData& expAuxData,
	std::vector<ExpValue>* resultVals,
	const std::string& paramName, ed::expns::Expression& parentExp) {
	/* this switches based on type of child element - 
	final functions probably switch based on parent type
	*/

	//auto p = std::make_pair(paramName, exp);

	/* first evaluate driver expression*/
	//DEBUGS("eval param: " + p.first + " : " + p.second.srcStr);
	DEBUGS("eval param: " + paramName + " : " + parentExp.srcStr);
	if (!parentExp.srcStr.empty()) {

	}
	s = parentExp.result(resultVals, &expAuxData);
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
		outPtr->drivers.push_back(elPtr->name);
		elPtr->points.push_back(outPtr->globalIndex);
	}

	switch (paramName[0]) { // get the type of element to add by the first letter of the name
		case 'p': { // a point has no drivers (yet, later allow declaring points as the direct output of topo operations
			DEBUGS("adding new point");
			s = pointEvalParentExpression(s, outPtr, value, expAuxData,
				resultVals,
				paramName, parentExp
			);
			break;
		}

		//case 'e': {
		//	/* eval expression to get list of driving elements */
		//	s = edgeEvalDriverExpression(s, outPtr, value, expAuxData,
		//		resultVals,
		//		paramName, exp);
		//	break;
		//}
		
		//case 'f': {
		//	/* eval expression to get list of driving elements */
		//	s = faceEvalDriverExpression(s, outPtr, value, expAuxData,
		//		resultVals,
		//		paramName, exp);
		//	break;
		//}
		default: {
			STAT_ERROR(s, "INVALID ELEMENT PREFIX: " + paramName + "; must begin with one of p, e, f");
		}
	}
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
	
	driver, then parent for each element - since some may depend on others,
		each must be fully constructed before continuing

	TODO: subdivision for edges
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

		s = evalDriverExpression(
			s, outPtr, value, expAuxData, resultVals,
			p.first, p.second
		);

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
		
		///////////// PARENT //////////

		Expression& parentExp = paramNameExpMap["!" + p.first];
		trimEnds(parentExp.srcStr);
		
		/* IF parent data for certain parent is not found:
		*	IF other parent datas ARE found? IGNORE for now, only one parent
		*	compute parent data in parent's space
		*/

		resultVals = nullptr;
		switch (p.first[0]) { // get the type of element added again
			case 'p': {
				if (!parentExp.srcStr.empty()) {
					DEBUGS("parent expression str not empty, eval ing exp");
					//return s;  // works up to here

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