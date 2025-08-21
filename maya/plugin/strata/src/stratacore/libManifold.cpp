
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
	if (record.iMap.find(el->globalIndex) == record.iMap.end()) {
		record.iMap[el->globalIndex];
	}
	auto& driverIntersectMap = record.iMap.at(el->globalIndex);

	for (auto driveIdx : el->drivers) {
		SElement* driverEl = manifold.getEl(driveIdx);

		
		switch (el->elType) {
		case SElType::point:{
			switch (driverEl->elType) {
			case SElType::point: { // point-point intersection, just a single point
				record.points.emplace_back();
				IntersectionPoint& pt = record.points.back();
				pt.type = pt.POINT;
				pt.elements = { el->globalIndex, driverEl->globalIndex };
				pt.uvns = { Vector3f(0, 0, 0), Vector3f(0, 0, 0) };
				record.iMap[el->globalIndex][driverEl->globalIndex].push_back(
					std::make_pair(&pt, nullptr)
				);
			}
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

