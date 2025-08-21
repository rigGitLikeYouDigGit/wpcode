#pragma once
#include "manifold.h"

#include "../exp/expParse.h"

namespace strata {

	Status& getIntersections(
		Status& s, 
		StrataManifold& manifold,
		IntersectionRecord& result, 
		int elA, int elB
	);

	static std::vector<SElement*> elsFromExpValue(
		StrataManifold& manifold,
		expns::ExpValue& expV
	) {
		/* return vector with all elements found from exp.
		* I don't know if there's a more efficient way other than allocating a new vector for it
		*/
		std::vector<SElement*> result;
		result.reserve(expV.stringVals.size() + expV.numberVals.size());
		for (auto& s : expV.stringVals) {
			SElement* el = manifold.getEl(s);
			if (el == nullptr) {
				continue;
			}
			result.push_back(el);
		}
		for (auto& f : expV.numberVals) {
			int index = round(f);
			SElement* el = manifold.getEl(index);
			if (el == nullptr) {
				continue;
			}
			result.push_back(el);
		}
		return result;
	}
	static Status& elementGreaterThan(
		Status& s,
		StrataManifold& manifold,
		expns::ExpValue& expA,
		expns::ExpValue& expB,
		expns::ExpValue& expOut
		/* do we guarantee this will always output a single element?
		or should it also be an expValue? since could be
		multiple sub-elements that satisfy greater-than?
		*/
	);
}
