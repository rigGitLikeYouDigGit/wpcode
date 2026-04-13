#pragma once

#include <vector>
#include <tuple>
#include "manifold.h"

#include "../exp/expParse.h"

namespace strata {

	std::tuple< std::vector<SElement*>, std::vector<SElement*>, std::vector<SElement*> >
		filterElementsByType(StrataManifold& manifold, std::vector<int> inIds);

	std::tuple< std::vector<SElement*>, std::vector<SElement*>, std::vector<SElement*> >
		filterElementsByType(const StrataManifold& manifold, std::vector<SElement*> inIds);

	template<class iterator_type>
	std::tuple< std::vector<SElement*>, std::vector<SElement*>, std::vector<SElement*> >
		filterElementsByType(const StrataManifold& manifold, iterator_type it, iterator_type end)
	{
		std::tuple< std::vector<SElement*>, std::vector<SElement*>, std::vector<SElement*> > result;
		while (it != end) {
			SElement* ptr = manifold.getElC(*it)
			it++;
			switch (ptr->elType) {
			case SElType::point: std::get<0>(result).push_back(ptr); continue;
			case SElType::edge: std::get<1>(result).push_back(ptr); continue;
			case SElType::face: std::get<2>(result).push_back(ptr); continue;
			}
			
		}
		//while (it != end) {
		//	result.push_back(getElC(*it));
		//	it++;
		//}
		return result;
	}

	template<class iterator_type>
	std::tuple< std::set<SElement*>, std::set<SElement*>, std::set<SElement*> >
		filterElementsByTypeSet(StrataManifold& manifold, iterator_type it, iterator_type end)
	{
		std::tuple< std::set<SElement*>, std::set<SElement*>, std::set<SElement*> > result;
		while (it != end) {
			SElement* ptr = manifold.getEl(*it);
			it++;
			switch (ptr->elType) {
			case SElType::point: std::get<0>(result).insert(ptr); continue;
			case SElType::edge: std::get<1>(result).insert(ptr); continue;
			case SElType::face: std::get<2>(result).insert(ptr); continue;
			}
		}
		return result;
	}

	/* could we pass in the desired return type for elements as a template param?
	template<elT>
	vector<elT> processEls(it, it){
		vector<elT> result;
		result.push_back(getEl<elT>)
		return result
		}
		?
	
	
	*/

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
