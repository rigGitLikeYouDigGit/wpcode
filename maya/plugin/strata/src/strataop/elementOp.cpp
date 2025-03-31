
#include "elementOp.h"

using namespace ed;
using namespace ed::expns;

Status StrataElementOp::makeParams() {
	paramNameExpMap["pNew"] = Expression("");
}


Status StrataElementOp::eval(StrataOp* node, StrataManifold& value, 
	EvalAuxData* auxData, Status& s) 
{
	for (auto& p : node->paramNameExpMap) {
		SElement* outPtr;
		switch (p.first[0]) { // get the type of element to add by the first letter of the name
			case 'p': {
				s = value.addElement(SElement(p.first, StrataElType::point), outPtr ); 
				CWRSTAT(s, "Error adding new element");
			}
			case 'e': {
				/* eval expression to get list of driving elements */
				ExpAuxData expAuxData;
				std::vector<ExpValue>* resultVals;
				
				s = p.second.result(resultVals, &expAuxData);
				CWRSTAT(s, "error getting exp result, halting");
				std::vector<SElement*> drivers = expAuxData.expValuesToElements(*resultVals, s);
				CWRSTAT(s, "error converting final exp result to strata drivers, halting");

				// add new edge element 
				s = value.addElement(SElement(p.first, StrataElType::edge), outPtr);
				CWRSTAT(s, "Error adding new edge element: " + p.first + " halting");

				// add drivers to new edge
				for (auto& i : drivers) {
					outPtr->drivers.push_back(i->globalIndex);
					i->edges.push_back(outPtr->globalIndex);
				}
				
			}
			case 'f': {
				/* eval expression to get list of driving elements */
				ExpAuxData expAuxData;
				std::vector<ExpValue>* resultVals;

				s = p.second.result(resultVals, &expAuxData);
				CWRSTAT(s, "error getting exp result, halting");
				std::vector<SElement*> drivers = expAuxData.expValuesToElements(*resultVals, s);
				CWRSTAT(s, "error converting final exp result to strata drivers, halting");

				// add new edge element 
				s = value.addElement(SElement(p.first, StrataElType::face), outPtr);
				CWRSTAT(s, "Error adding new face element: " + p.first + " halting");

				// add drivers to new edge
				for (auto& i : drivers) {
					outPtr->drivers.push_back(i->globalIndex);
					i->faces.push_back(outPtr->globalIndex);
				}

			}
		}
		
	}


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