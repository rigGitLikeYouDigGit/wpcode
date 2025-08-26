#pragma once
#include "manifold.h"

#include "../exp/expParse.h"

namespace strata {

	Status& updateElementIntersections(
		Status& s,
		StrataManifold& manifold,
		SElement* el,
		IntersectionRecord& record
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
			int index = static_cast<int>(round(f));
			SElement* el = manifold.getEl(index);
			if (el == nullptr) {
				continue;
			}
			result.push_back(el);
		}
		return result;
	}
	//static Status& elementGreaterThan(
	//	Status& s,
	//	StrataManifold& manifold,
	//	expns::ExpValue& expA,
	//	expns::ExpValue& expB,
	//	expns::ExpValue& expOut
	//	/* do we guarantee this will always output a single element?
	//	or should it also be an expValue? since could be
	//	multiple sub-elements that satisfy greater-than?
	//	*/
	//);


	/* for composite operations we're gonna be real stupid with it,
	every step creates an intermediate element*/
	Status& elementGreaterThan(
		Status& s,
		StrataManifold& manifold,
		int idA,
		//int idB,
		std::vector<int> idsB,
		std::vector<int>& elsOut
		/*
		* return a new element in the subspace of elA, starting at intersection with elB

		invalid if elA is a point, you can't have a point subspace
		*/
	);


	Status& elementGreaterThan(
		Status& s,
		StrataManifold& manifold,
		std::vector<int>& elsA,
		std::vector<int>& elsB,
		std::vector<int>& elsOut
		/* do we guarantee this will always output a single element?
		or should it also be an expValue? since could be
		multiple sub-elements that satisfy greater-than?
		*/
	);


	Status& elementLessThan(
		Status& s,
		StrataManifold& manifold,
		int idA,
		//int idB,
		std::vector<int> idsB,
		std::vector<int>& elsOut
	);


	Status& elementLessThan(
		Status& s,
		StrataManifold& manifold,
		std::vector<int>& elsA,
		std::vector<int>& elsB,
		std::vector<int>& elsOut
		/* do we guarantee this will always output a single element?
		or should it also be an expValue? since could be
		multiple sub-elements that satisfy greater-than?
		*/
	);
}
