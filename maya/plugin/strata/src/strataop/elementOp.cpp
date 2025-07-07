
#include "elementOp.h"
#include "../stringLib.h"
#include "../logger.h"

using namespace ed;
using namespace ed::expns;

Status StrataElementOp::makeParams() {
	Status s;
	//paramNameExpMap["pNew"] = Expression("");
	return s;
}

Status& pointEvalParam( 
	Status& s, StrataElementOp& op, SElement*& outPtr, StrataManifold& value, ExpAuxData& expAuxData,
	ElOpParam& param) {
	/* a separate function here may be slightly overkill
	
	DO WE allow input data for pure drivers, and PROJECT that data on to driver geo?
	
	for points, it's the nearest point on driver geo to the target point?
	for edges, flatten target curve into driver surface?

	allow point drivers for snapping positions,
	then save relative position in space as override,
	reapply that in later iterations

	IF DATA IS ALREADY FOUND from previous graph iteration:
		add element
		compute data from spaces
		(maybe then snap to drivers)
		done
	*/
	LOG("eval param: " + param.name);
	std::vector<ExpValue> resultVals;

	s = value.addElement(SElement(param.name, StrataElType::point), outPtr);

	if (s) {
		l("error adding element: " + s.msg);
		return s;
	}

	/* we don't care about outputs etc - 
	when this op is created afresh, its saved data will be empty,
	so we just need to look if node has data saved on it*/
	bool ALREADY_DATA = false;
	auto prevData = op.opPointDataMap.find(param.name);
	ALREADY_DATA = (prevData != op.opPointDataMap.end());
	
	
	if (ALREADY_DATA) {
		l("already data found");
		s = value.computePointData(s, prevData->second);
		value.pDataMap[param.name] = prevData->second;

		// save built data to this node's data map
		op.opPointDataMap[param.name] = value.pDataMap[param.name];
		return s;
	}

	
	// no data found, make new
	value.pDataMap[param.name] = param.pData;
	SPointData& pData = value.pDataMap[param.name];
	pData.creatorNode = op.name; // this op created this data
	pData.index = outPtr->globalIndex;

	// check if point has a driver - if it's a point, snap to it
	s = param.driverExp.result(&resultVals, &expAuxData);
	CWRSTAT(s, "error reading driver exp");
	std::vector<int> drivers = expAuxData.expValuesToElements(resultVals, s);
	CWRSTAT(s, "error converting driver exp to elements");

	// check for spaces
	s = param.spaceExp.result(&resultVals, &expAuxData);
	CWRSTAT(s, "error reading space exp");
	std::vector<int> spaces = expAuxData.expValuesToElements(resultVals, s);
	CWRSTAT(s, "error converting space exp to elements");

	/* check if matrix is specified locally, or UVNs given*/
	if (param.spaceMode == EL_OP_GLOBAL) {
		l("creating global data");

		if (drivers.size()) {
			auto driverEl = value.getEl(drivers[0]);
			pData.driverData.index = drivers[0];
			if (driverEl->elType == StrataElType::point) {
				pData.finalMatrix = value.pDataMap[driverEl->name].finalMatrix;
			}
		}


		if (spaces.size()) {
			for (auto i : spaces) {
				auto spaceEl = value.getEl(i);
				// get final matrix in this space

				SPointSpaceData sd;
				s = value.getUVN(s, sd.uvn, spaceEl, pData.finalMatrix.translation());
				sd.name = spaceEl->name;
				Affine3f sampleMat;
				s = value.matrixAt(s, sampleMat, spaceEl, sd.uvn);
				sd.offset = sampleMat.inverse() * pData.finalMatrix;

				pData.spaceDatas.push_back(sd);
			}
		}
		//value.pDataMap[param.name].finalMatrix = param.pData.finalMatrix;
		pData.finalMatrix = param.pData.finalMatrix;

		// save built data to this node's data map
		//op.opPointDataMap[param.name] = value.pDataMap[param.name];
		op.opPointDataMap[param.name] = pData;
		return s;
	}

	// locally specified, more complex
	l("creating local data");
	s = param.spaceExp.result(&resultVals, &expAuxData);
	spaces = expAuxData.expValuesToElements(resultVals, s);

	/* TODO: allow defining locally-defined UVNs for drivers */

	if (!spaces.size()) { // if no spaces, nothing to do, take local as global
		l("no spaces found, using local target as global");
		//value.pDataMap[param.name].finalMatrix = param.pData.finalMatrix;
		pData.finalMatrix = param.pData.finalMatrix;

		// save built data to this node's data map
		//op.opPointDataMap[param.name] = value.pDataMap[param.name];
		op.opPointDataMap[param.name] = pData;
		return s;
	}
	

	/* how do we create an element with 2 parent spaces, but we only specify its
	local transform in one of them?
	
	brother do you want to get this project working or not

	this also only makes sense if the parent is a point for matrices
	*/


	// ADD TO OP POINT DATA MAP HERE
	if (spaces.size() == 1) {
		SElement* spaceEl = value.getEl(spaces[0]);
		if (spaceEl->elType == StrataElType::point) { // if single parent space is a point
			SPointData spaceData = value.pDataMap[spaceEl->name];
			//value.pDataMap[param.name].finalMatrix = spaceData.finalMatrix * param.pData.finalMatrix;
			pData.finalMatrix = spaceData.finalMatrix * param.pData.finalMatrix;
			op.opPointDataMap[param.name] = pData;
			return s;
		}
	}
	std::vector<Affine3f> spaceBlendMats;
	for (int i = 0; i < static_cast<int>(spaces.size()); i++) {
		SElement* spaceEl = value.getEl(spaces[i]);
		if (spaceEl->elType == StrataElType::point) {
			SPointData spaceData = value.pDataMap[spaceEl->name];
			spaceBlendMats.push_back(spaceData.finalMatrix * param.pData.finalMatrix);
		}
	}
	VectorXf weights(spaceBlendMats.size());
	weights.fill(1.0);
	pData.finalMatrix = blendTransforms(spaceBlendMats, weights);
	op.opPointDataMap[param.name] = pData;
	return s;

}

//Status& edgeEvalDriverExpression(
//	Status& s, StrataElementOp& op, SElement*& outPtr, StrataManifold& value, ExpAuxData& expAuxData,
//	std::vector<ExpValue>* resultVals,
//	const std::string& paramName, ed::expns::Expression& exp) {
//
//	s = exp.result(resultVals, &expAuxData);
//	CWRSTAT(s, "error getting exp result, halting");
//	if (resultVals == nullptr) { // empty driver, could just be passed in
//		if (op.nameEdgeDataMap.find(paramName) == op.nameEdgeDataMap.end()) {
//			STAT_ERROR(s, "edge driver result gave a nullptr and no literal curve given");
//		}
//	}
//	// add new edge element 
//	s = value.addElement(SElement(paramName, StrataElType::edge), outPtr);
//	CWRSTAT(s, "Error adding new edge element: " + paramName + " halting");
//	SEdgeData& edgeData = value.edgeDatas.at(outPtr->elIndex);
//
//	// if literal curve passed in, just set its shape and return
//	if (op.nameEdgeDataMap.find(paramName) != op.nameEdgeDataMap.end()) {
//		edgeData.finalCurve = op.nameEdgeDataMap.find(paramName)->second.finalCurve;
//		edgeData.finalNormals = op.nameEdgeDataMap.find(paramName)->second.finalNormals;
//	}
//
//	std::vector<int> drivers = expAuxData.expValuesToElements(*resultVals, s);
//	CWRSTAT(s, "error converting final exp result to strata drivers, halting");
//
//	if (drivers.empty()) {
//		DEBUGS("empty drivers for curve driver exp " + exp.srcStr);
//		return s;
//	}
//	
//	// add drivers to new edge
//
//	/* TODO: drivers still need coords -
//	how do we specify them in exp,
//	or how do we pass them in from maya node
//	smooth data should not be in expressions
//
//	MAYBE we have a constraint that curves can only be driven directly by
//	points? each direct exp must resolve to a point - leaving freedom as to how we
//	arrive at that point (via closest-point, param lookup on surface beforehand, etc)
//
//	ONLY POINTS FOR NOW
//	CUT ALL THE SCOPE
//	GET THE DEMO OUT
//
//	*/
//
//	for (auto& i : drivers) {
//		SElement* driverPtr = value.getEl(i);
//		//outPtr->drivers.push_back(i); 
//		outPtr->drivers.push_back(driverPtr->name);
//		edgeData.driverDatas.push_back(EdgeDriverData());
//		EdgeDriverData& driverData = edgeData.driverDatas.at(edgeData.driverDatas.size() - 1);
//		driverData.index = i;
//
//		Eigen::Vector3f defaultCoords{ 0.0, 0.0, 0.0 };
//		// TODO: HOW DO WE DEFINE DRIVER PARAMETRES
//
//		/*if (driverPtr->elType == StrataElType::point) {
//			driverData.finalMatrix = value.pointDatas[driverPtr->el]
//		}*/
//
//		if (driverPtr->elType == StrataElType::edge) {
//			defaultCoords(0) = 0.5f;
//		}
//
//		s = value.matrixAt(s, driverData.finalMatrix,
//			driverPtr->globalIndex, defaultCoords
//		);
//		CWRSTAT(s, "ERROR getting MatrixAt for driver el: " + driverPtr->name + " for edge: " + outPtr->name);
//
//		driverPtr->edges.push_back(outPtr->globalIndex);
//
//	}
//
//	//s = value.buildEdgeData(s, edgeData);
//
//	return s;
//}
//
//
//Status& faceEvalDriverExpression(
//	Status& s, SElement*& outPtr, StrataManifold& value, ExpAuxData& expAuxData,
//	std::vector<ExpValue>* resultVals,
//	const std::string& paramName, ed::expns::Expression& exp) 
//{
//	//ExpAuxData expAuxData;
//	//std::vector<ExpValue>* resultVals = nullptr;
//
//	s = exp.result(resultVals, &expAuxData);
//	CWRSTAT(s, "error getting exp result, halting");
//	if (resultVals == nullptr) {
//		STAT_ERROR(s, " getting result returned a nullptr");
//	}
//	std::vector<int> drivers = expAuxData.expValuesToElements(*resultVals, s);
//	CWRSTAT(s, "error converting final exp result to strata drivers, halting");
//
//	// add new edge element 
//	s = value.addElement(SElement(paramName, StrataElType::face), outPtr);
//	CWRSTAT(s, "Error adding new face element: " + paramName + " halting");
//
//	// add drivers to new edge
//	for (auto& i : drivers) {
//		SElement* elPtr = value.getEl(i);
//		//outPtr->drivers.push_back(i);
//		outPtr->drivers.push_back(elPtr->name);
//		elPtr->faces.push_back(outPtr->globalIndex);
//	}
//	return s;
//}




Status StrataElementOp::eval(StrataManifold& value, 
	EvalAuxData* auxData, Status& s) 
{
	/*
	
	set up drivers
	set up temp final matrix if given
	put drivers into spaces, convert to uvn

	get per-space offsets

	build full data object

	then CHECK OVERRIDE MAP for any entries to override

	then check if element needs snapping / projecting to driver of higher dimension


	*/
	LOG("EL OP EVAL");
	//StrataElementOp* opPtr = static_cast<StrataElementOp*>(node);
	elementsAdded.clear();
	elementsAdded.reserve(paramMap.size());
	StrataElementOp* opPtr = this;
	l("n params to eval: " + std::to_string(paramMap.size()));

	// use one data struct throughout to allow reusing variables between params
	ExpAuxData expAuxData;

	//for (auto& p : paramMap) {
	for( auto paramName : paramNames){
		//auto p = std::make_pair(paramName, paramMap[paramName]);
		ElOpParam& param = paramMap[paramName];
		SElement* outPtr = nullptr;
		expAuxData.manifold = &value;
		std::vector<ExpValue>* resultVals = nullptr;
		auto& op = *this;

		switch (paramName[0]) { // get the type of element to add by the first letter of the name
			case 'p': { // a point has no drivers (yet, later allow declaring points as the direct output of topo operations
				l("adding new point");
				s = pointEvalParam(s, op, outPtr, value, expAuxData,
					param
				);
				break;
			}

			default: {
				STAT_ERROR(s, "INVALID ELEMENT PREFIX: " + paramName + "; must begin with one of p, e, f");
			}
		}

		if (outPtr == nullptr) {
			STAT_ERROR(s, "outPtr not set correctly when adding element: " + paramName + ", halting");
		}
		elementsAdded.push_back(outPtr->name);

		/* eval parent expression*/		
	}
	l("EL EVAL COMPLETE");
	return s;
}

/* should we RE-RUN driver stuff after parents built?
if points need to be snapped back to driver curves, 
curves to surfaces etc.

we don't use delta matrices here yet, might be worth changing that
*/

Status& StrataElementOp::pointProcessTargets(Status& s, StrataManifold& finalManifold, SAtomBackDeltaGroup& deltaGrp, SElement* el) {
	/* blend any target matrices given*/
	LOG("EL OP point process targets: " + el->name);
	int nTargets = static_cast<int>(deltaGrp.targetMap[el->name].size());
	if (!nTargets) {
		return s;
	}
	VectorXf weights(nTargets);
	std::vector<Affine3f> mats(nTargets);

	/* for now just average weights
	later on maybe weight it more towards larger deltas
	*/
	float w = 1.0f / float(nTargets);
	for (int i = 0; i < nTargets; i++) {
		weights(i) = w;
		mats[i] = deltaGrp.targetMap[el->name][i].matrix;
	}
	// blend matrices together
	Affine3f targetMat = blendTransforms(mats, weights);
	
	// save target matrix, to be matched absolutely with final offsets
	deltaGrp.targetMap[el->name][0].matrix = targetMat;
	
	// if this point has no drivers or spaces, just move it
	if ((!el->drivers.size()) && (!el->spaces.size())) {
		paramMap[el->name].pData.finalMatrix = targetMat;
		return s;
	}

	// if spaces, invert the current uvn + offset
	if (el->spaces.size()) {
		for (int i = 0; i < static_cast<int>(el->spaces.size()); i++) {
			SAtomMatchTarget target;
			SElement* spaceEl = finalManifold.getEl(el->spaces[i]);
			SPointSpaceData& spaceData = finalManifold.pDataMap[el->name].spaceDatas[i];
			Vector3f uvn = spaceData.uvn;

			target.matrix = targetMat * spaceData.offset.inverse();
			deltaGrp.targetMap[el->spaces[i]].push_back(target);
		}
	}


	if (el->drivers.size()) {
		for (int i = 0; i < static_cast<int>(el->drivers.size()); i++) {
			SAtomMatchTarget target;
			target.matrix = targetMat;
			// get nearest UVN for driver?
			SElement* driverEl = finalManifold.getEl(el->drivers[i]);
			
			// don't change any set UVN coords here, need to maintain the offset here
			//finalManifold.getUVN(s, target.uvn, driverEl, targetMat.translation()); /* ?????*/
			///* this is IMMEDIATELY susceptible to nearest-point jumping within a driver element - do we care? */

			deltaGrp.targetMap[el->drivers[i]].push_back(target);
		}
	}

	return s;
}

SAtomBackDeltaGroup StrataElementOp::bestFitBackDeltas(Status* s, StrataManifold& finalManifold, SAtomBackDeltaGroup& front) {

	/* work backwards through elements added by this op, check if each name appears
	* in delta front
	* if it does, process it, and add any drivers (even within this node) 
	* to the front as well
	*/
	LOG("EL OP best fit back deltas, " + str(front.targetMap.size()));
	Status& stat = *s;
	for (int i = 0; i < static_cast<int>(elementsAdded.size()); i++) {
		std::string& name = elementsAdded.rbegin()[i];
		auto found = front.targetMap.find(name);
		if (found == front.targetMap.end()) {
			l("no targets found for created element: " + name + ", skipping");
			continue;
		}
		
		SElement* el = finalManifold.getEl(name);
		ElOpParam& param = paramMap[name];
		switch (el->elType) {
			case StrataElType::point: {
				stat = pointProcessTargets(stat, finalManifold, front, el);
			}
		}

	}
	backDeltasToMatch = front; // save deltas to match for later

	return front;
}

Status& StrataElementOp::setBackOffsetsAfterDeltas(Status& s, StrataManifold& manifold) {
	/* iterate over elements created in FORWARDS order this time - compare offsets,
	add offsets to matrices, then snap to drivers where needed
	
	BUT we need to set the offset first, before computing elements that rely on it
	I will now consume my internal organs
	*/
	LOG("EL OP setBackOffsets");
	for (int i = 0; i < static_cast<int>(elementsAdded.size()); i++) {
		std::string& name = elementsAdded[i];

		auto foundToMatch = backDeltasToMatch.targetMap.find(name);
		if (foundToMatch == backDeltasToMatch.targetMap.end()) {
			/* no effects found for this element, skip*/
			l("no target found for el " + name + ", skipping");
			continue;
		}
		/* above is rechecked each iteration, so even internal moving around of later elements should
		propagate properly*/

		/* check that a found element does have a target - this should always be the case
		*/
		if (!foundToMatch->second.size()) {
			l("found no targets for found el " + name + ", skipping");
			continue;
		}

		SElement* el = manifold.getEl(name);

		ElOpParam& param = paramMap[name];

		switch (el->elType) {
			case StrataElType::point: {
				l("point offsets:" + el->name);
				SAtomMatchTarget& target = foundToMatch->second[0];
				SPointData& pData = manifold.pDataMap[name];

				if (pData.finalMatrix.isApprox(target.matrix)) {
					l("point already matched, no offset needed");
					continue;
				}
			
				// get final offset
				Affine3f offset = pData.finalMatrix.inverse() * target.matrix;
				param.pOffset = offset;

				

				// ideally we just match the target here
				pData.finalMatrix = target.matrix;
			
				// project to drivers if any
				if (el->drivers.size()) {
					l("projecting point to drivers");
					s = manifold.pointProjectToDrivers(s, pData.finalMatrix, el);
				}
				break;
			}
		}

	}
	return s;
}



StrataElementOp* StrataElementOp::clone_impl() const {
	LOG("EL OP CLONE");
	return StrataOp::clone_impl<StrataElementOp>();
};

