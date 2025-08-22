
/* splitting out more complex strata functions into this library - 
we shouldn't have to recompile the core data structure every time
we add a new way to manipulate it*/
#include "manifold.h"

#include "../exp/expParse.h"

using namespace strata;
using namespace strata::expns;

/* consider, would intersections be easier if we flattened all
our geometry to dense discrete point sets? Probably

as in a curve is just the set of points along the bezier eq, 
so any connection to it must also use one of those points?

intersections should be indexed by driver and UVN, at least for points -
a common intersectionPoint should hold uvns of all contributing drivers
*/

Status& examineIntersections(Status& s,
	StrataManifold& manifold,
	IntersectionRecord& result,
	int idxA, int idxB
)
{	/* once we know 2 elements could intersect,
	this function updates all the actual intersection events
	*/
	return s;
}

/*
WANT:

elA intersection list:

[ uvnA : driver 1, uvnB : driver 2, curveIntersection3 : neighbour 1 ]


*/

Status& elDriverIntersections(
	Status& s,
	StrataManifold& manifold,
	SElement* el,
	IntersectionRecord& record
) {
	//if (record.pointMap.find(el->globalIndex) == record.pointMap.end()) {
	//	record.pointMap[el->globalIndex];
	//}
	//auto& driverIntersectMap = record.iMap.at(el->globalIndex);


	switch (el->elType) {
	case SElType::point: {
		IntersectionPoint* ptr = nullptr; // point can only ever intersect AT a point
		SPointData& pData = manifold.pDataMap[el->name];
		SPointDriverData& driverData = pData.driverData; // only single driver for points

		auto lookup = record.posPointMap.find(
			toKey(pData.finalMatrix.translation())
		);
		if (lookup == record.posPointMap.end()) { // make new point for this element
			record.points.emplace_back();
			ptr = &record.points.back();
			ptr->pos = pData.finalMatrix.translation();
			ptr->elements.push_back(el->globalIndex);
			ptr->uvns.push_back(Vector3f(0, 0, 0));
			record.pointMap[el->globalIndex][Vector3i(0, 0, 0)] = ptr; //uvns on a point are zero
			record.posPointMap[toKey(pData.finalMatrix.translation())] = ptr;

		}
		else {
			ptr = lookup->second;
			record.pointMap[el->globalIndex][Vector3i(0, 0, 0)] = ptr;
		}

		for (auto driveIdx : el->drivers) { // loop not necessary for points, keeping for consistency
			SElement* driverEl = manifold.getEl(driveIdx);

			switch (driverEl->elType) {
			case SElType::point: { // point-point intersection, just a single point
				/* look up existing intersection by position, as the canonical way to share them?
				seems SUPER dodgy but makes logic easier for now
				*/
				SPointData& dPData = manifold.pDataMap[driverEl->name];

				ptr->elements.push_back(driverEl->globalIndex);
				ptr->uvns.push_back(Vector3f(0, 0, 0));
				record.pointMap[driverEl->globalIndex][Vector3i(0, 0, 0)] = ptr;

				record.elMap[el->globalIndex][driverEl->globalIndex].push_back({ ptr, nullptr });
				record.elMap[driverEl->globalIndex][el->globalIndex].push_back({ ptr, nullptr });

				/* recurse??? */
				s = elDriverIntersections(s, manifold, driverEl,
					record);
				break;
			}
			case SElType::edge: {
				/* point-curve intersection - DO NOT RECURSE here?
				or do - this curve might be driven by a point at this exact same position, etc
				*/
				SEdgeData& dEData = manifold.eDataMap[driverEl->name];
				ptr->elements.push_back(driverEl->globalIndex);
				ptr->uvns.push_back(driverData.uvn);
				record.pointMap[driverEl->globalIndex][toKey(driverData.uvn)] = ptr;

				record.elMap[el->globalIndex][driverEl->globalIndex].push_back({ ptr, nullptr });
				record.elMap[driverEl->globalIndex][el->globalIndex].push_back({ ptr, nullptr });

				s = elDriverIntersections(
					s,
					manifold,
					driverEl,
					record);
				break; // break driver switch
			}
			}
			break; // break loop
		}
		break; // break eltype point switch
	}
	case SElType::edge: {
		
		SEdgeData& eData = manifold.eDataMap[el->name];
		for (int i = 0; i < el->drivers.size(); i++) { // loop not necessary for points, keeping for consistency
			int driveIdx = el->drivers[i];
			SElement* driverEl = manifold.getEl(driveIdx);
			SEdgeDriverData& driverData = eData.driverDatas[i];
			switch (driverEl->elType) {
			case SElType::point: {
				//SPointData& dPData = manifold.pDataMap[driverEl->name];
				IntersectionPoint* ptr = nullptr;
				auto found = record.posPointMap.find(toKey(
					driverData.pos()));
				if (found == record.posPointMap.end()) { /* add point driver to record */
					record.points.emplace_back();
					ptr = &record.points.back();
					record.posPointMap[toKey(driverData.pos())] = ptr;
				}
				else {
					ptr = found->second;
				}
				// add driver
				ptr->elements.push_back(driverEl->globalIndex);
				ptr->pos = driverData.pos();
				ptr->uvns.push_back(driverData.uvn);

				/* add edge point at UVN*/
				ptr->elements.push_back(el->globalIndex);
				ptr->uvns.push_back(Vector3f(driverData.uOnEdge, 0, 0)	);

				// bidirectional lookups between elements
				record.elMap[el->globalIndex][driverEl->globalIndex].push_back({ ptr, nullptr });
				record.elMap[driverEl->globalIndex][el->globalIndex].push_back({ ptr, nullptr });
				
				// recurse
				s = elDriverIntersections(
					s,
					manifold,
					driverEl,
					record);
				continue;
			}
			case SElType::edge: {
				SEdgeData& dEData = manifold.eDataMap[driverEl->name];
				IntersectionPoint* ptr = nullptr;
				auto found = record.posPointMap.find(toKey(
					driverData.pos()));
				if (found == record.posPointMap.end()) { /* add point driver to record */
					record.points.emplace_back();
					ptr = &record.points.back();
					record.posPointMap[toKey(driverData.pos())] = ptr;
				}
				else {
					ptr = found->second;
				}
				// add driver
				ptr->elements.push_back(driverEl->globalIndex);
				ptr->pos = driverData.pos();
				ptr->uvns.push_back(driverData.uvn);

				// add edge

				/* add edge point at UVN*/
				ptr->elements.push_back(el->globalIndex);
				ptr->uvns.push_back(Vector3f(driverData.uOnEdge, 0, 0));

				// bidirectional lookups between elements
				record.elMap[el->globalIndex][driverEl->globalIndex].push_back({ ptr, nullptr });
				record.elMap[driverEl->globalIndex][el->globalIndex].push_back({ ptr, nullptr });

				// recurse
				s = elDriverIntersections(
					s,
					manifold,
					driverEl,
					record);
				continue;
			}
			}
	}
	}

	return s;

}

Status& getIntersectionMap(
	Status& s,
	StrataManifold& manifold,
	SElement* el,
	IntersectionRecord& record
) {
	/* populate all records for the given element?
	*/
	std::unordered_set<int> visited;
	std::deque<int> toCheck(el->drivers.begin(), el->drivers.end());

	while (toCheck.size()) {
		int checkIdx = toCheck.front();
		toCheck.pop_front();
		// check if we've already checked this element
		if (visited.find(checkIdx) != visited.end()) {
			continue;
		}
		// mark as visited
		visited.insert(checkIdx);

		SElement* checkEl = manifold.getEl(checkIdx);

		switch (el->elType) {
		case SElType::point: {
			/* 
			need prev el, prev el type, prev intersection point?
			an intersection can only ever DECAY down dimensions, 
			a point won't ever expand to a curve if you track it further
			*/
		}
		}

	}

	return s;
}


Status& getIntersections(Status& s, 
	StrataManifold& manifold,
	IntersectionRecord& result,
	int idxA, int idxB) 
{
	/* write out logic for most complicated 
	* case you can think of, then do it again for another one
	* 
	* could probably have separate function to get history intersection, but
	* the validity of that depends on what kinds of elements it routes through
	* 
	* whichever has the higher index MUST be the later element?
	* trace back from later element ->
	*	hit a point?
	*		fan forwards, check for exact matches
	*	hit a face?
	*		fan forwards in one step, check for elements directly driven
	*/

	// curve/curve, including potential history and surface driver interaction
	SElement* elA = manifold.getEl(idxA);
	SElement* elB = manifold.getEl(idxB);

	std::unordered_set<int> foundSet;

	int later = ((idxA > idxB) ? idxA : idxB);
	int target = ((idxA > idxB) ? idxB : idxA);

	while (true) {

		break;
	}

	/* for each, if EITHER is a point, look backwards til we find an element that's not?
	*/
	SElement* thisElA = elA;
	SElement* thisElB = elB;
	int thisIdA = idxA;
	int thisIdB = idxB;

	/*std::unordered_set<int> toCheckA = { idxA };
	std::unordered_set<int> toCheckB = { idxB };*/
	std::queue<int> toCheckA;
	toCheckA.push(idxA);
	std::queue<int> toCheckB;
	toCheckB.push(idxB);

	while (toCheckA.size() && toCheckB.size()) {
		thisIdA = toCheckA.back();
		toCheckA.pop();
		thisIdB = 
	}

	for (auto& idx : elA->drivers) {
		if(foundSet.find())
	}


	
	return s;
}


Status& elementGreaterThan(
	Status& s,
	StrataManifold& manifold,
	ExpValue& expA,
	ExpValue& expB,
	ExpValue& expOut
	/* do we guarantee this will always output a single element?
	or should it also be an expValue? since could be 
	multiple sub-elements that satisfy greater-than?
	*/
) {
	/* QUICK DIRTY sketch for now -
	this should probably
	only return some kind of UVN coordinates,
	that we can then pass into a separate function to generate
	sub-elements

	should we just overload this function for every permutation you
	can get out of the expression system?
	*/

	

	return s;
}

