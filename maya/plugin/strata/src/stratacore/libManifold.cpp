
/* splitting out more complex strata functions into this library - 
we shouldn't have to recompile the core data structure every time
we add a new way to manipulate it*/
#include "manifold.h"

#include "libManifold.h"
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


std::tuple< std::vector<SElement*>, std::vector<SElement*>, std::vector<SElement*> >
strata::filterElementsByType(const StrataManifold& manifold, std::vector<SElement*> els)
{
	/* from input indices, return vectors for points, edges and faces
	TODO: how should we work with groups?
	how should we store and track groups in the manifold?
	*/
	std::tuple< std::vector<SElement*>, std::vector<SElement*>, std::vector<SElement*> > result;
	for (auto ptr : els) {
		switch (ptr->elType) {
		case SElType::point: std::get<0>(result).push_back(ptr); continue;
		case SElType::edge: std::get<1>(result).push_back(ptr); continue;
		case SElType::face: std::get<2>(result).push_back(ptr); continue;
		}
	}
	return result;
}

std::tuple< std::vector<SElement*>, std::vector<SElement*>, std::vector<SElement*> >
strata::filterElementsByType(StrataManifold& manifold, std::vector<int> inIds)
{
	/* from input indices, return vectors for points, edges and faces
	TODO: how should we work with groups?
	how should we store and track groups in the manifold?
	*/
	return filterElementsByType(manifold, manifold.getEls(inIds));
}


/*
WANT:

elA intersection list:

[ uvnA : driver 1, uvnB : driver 2, curveIntersection3 : neighbour 1 ]


*/


/* to UPDATE intersections without remapping everything every time

on element created, query by POSITIONS covered by new element - 
only update intersections that match them

from an element's DIRECT DRIVERS, we can get spatial data to query?
plus AABB checks for surfaces, curves + surfaces
*/

Status& _updatePointDriverIntersections(
	Status& s,
	StrataManifold& manifold,
	SElement* el,
	IntersectionRecord& record//,
	//IntersectionPoint* ip
) {
	/* called on new point (el)
	* check if any elements cross through this position 
	*/
	/* only called when we know for sure point has a driver */
	SPointData& pData = manifold.pDataMap[el->name];
	IntersectionPoint* ptr = record.getPointByVectorPosition(pData.finalMatrix.translation()); 
	// point can only ever intersect AT a point
	
	//SPointDriverData& driverData = pData.driverData; // only single driver for points

	if(ptr == nullptr){
		ptr = record.newPoint(pData.finalMatrix.translation());
		//ptr->pos = pData.finalMatrix.translation();
		//record.posPointMap[toKey(pData.finalMatrix.translation())] = ptr->index;

		//ptr->elUVNMap[el->globalIndex] = Vector3f(0, 0, 0);
		//ptr->elements.push_back(el->globalIndex);
		//ptr->uvns.push_back(Vector3f(0, 0, 0));
		ptr->elUVNMap[el->globalIndex].push_back(Vector3f(0, 0, 0)); /* this point's coord for this intersection is 0,0,0*/

	}
	record.elUVNPointMap[el->globalIndex][Vector3i(0, 0, 0)] = ptr->index; //uvns on a point are zero

	if (!el->drivers.size()) {
		return s;
	}
	SElement* driverEl = manifold.getEl(el->drivers[0]);
	SPointDriverData& driverData = pData.driverData; // only single driver for points

	switch (driverEl->elType) {
	case SElType::point: { // point-point intersection, just a single point

		ptr->elUVNMap[driverEl->globalIndex].push_back(Vector3f(0, 0, 0));
		record.elUVNPointMap[driverEl->globalIndex][Vector3i(0, 0, 0)] = ptr->index;

		record.elMap[el->globalIndex][driverEl->globalIndex].push_back({ ptr->index, Intersection::POINT });
		record.elMap[driverEl->globalIndex][el->globalIndex].push_back({ ptr->index, Intersection::POINT });

		return s;
	}
	case SElType::edge: {
		/* edge driving point
		*/
		SEdgeData& dEData = manifold.eDataMap[driverEl->name];

		auto found = ptr->elUVNMap.find(driverEl->globalIndex);
		if (found == ptr->elUVNMap.end()) { /* edge not already included in point*/
			ptr->elUVNMap.insert({ driverEl->globalIndex, {driverData.uvn} });
		}
		else {
			/* check if edge already includes this point -
			maybe edge loops back on itself multiple times
			*/
			
			auto newFound = std::find(found->second.begin(), found->second.end(), driverData.uvn);
			if (newFound == found->second.end()) {
				found->second.push_back(driverData.uvn);
			}
		}

		record.elUVNPointMap[driverEl->globalIndex][toKey(driverData.uvn)] = ptr->index;

		record.elMap[el->globalIndex][driverEl->globalIndex].push_back({ ptr->index, Intersection::POINT });
		record.elMap[driverEl->globalIndex][el->globalIndex].push_back({ ptr->index, Intersection::POINT });

		return s;
	}
	}
	return s;
	}


Status& _updateEdgeVertices(
	Status& s,
	StrataManifold& manifold,
	SElement* el,
	IntersectionRecord& record
) {
	/* add any vertices between this edge and any others 
	* run after updating intersections to be safe
	*/

	for (auto& p : record.elMap[el->globalIndex]) {
		/* all elements connected to this edge 
		*/
		SElement* otherEl = manifold.getEl(p.first);
		/* skip if not an edge */
		if (otherEl->elType != SElType::edge) {
			continue;
		}
		for (auto& otherP : p.second) {
			/* all intersections between this edge and that element
			*/
			/* check intersection type even though it should always be a point between curves*/
			if (otherP.second != Intersection::POINT) {
				continue;
			}

			IntersectionPoint& p = record.points[otherP.first];
			/* TECHNICALLY if both edges loop back on themselves at this exact point, could have multiple
			UVNs for each one.
			At what point do we just say no and slap the user
			*/

			for (Vector3f& srcUVN : p.elUVNMap[el->globalIndex]) {
				for (Vector3f& dstUVN : p.elUVNMap[otherEl->globalIndex]) {

					/* update both entries in vertex map - might waste data but it's easier to check
					* for directions, only important that each ordered pair of edge indices gets opposite sets of vertex orientations
					* 
					* (A, B) -> (0, 0), (1, 1)
					* (B, A) -> (0, 1), (1, 0)
					* 
					* logic works - now filter if edge ends at this point, corner cases
					*/
					manifold.getVertex(
						el->globalIndex, srcUVN.x(), false,
						otherEl->globalIndex, dstUVN.x(), false,
						p.index
					);
					manifold.getVertex(
						el->globalIndex, srcUVN.x(), true,
						otherEl->globalIndex, dstUVN.x(), true,
						p.index
					);
					manifold.getVertex(
						otherEl->globalIndex, dstUVN.x(), false,
						el->globalIndex, srcUVN.x(), true,
						p.index
					);
					manifold.getVertex(
						otherEl->globalIndex, dstUVN.x(), true,
						el->globalIndex, srcUVN.x(), false,
						p.index
					);
				}
			}


		}
	}

	return s;
}

Status& _updateEdgeDriverIntersections(
	Status& s,
	StrataManifold& manifold,
	SElement* el, // edge element
	IntersectionRecord& record
	) {
	SEdgeData& eData = manifold.eDataMap[el->name];

	for (int i = 0; i < el->drivers.size(); i++) { // 
		int driveIdx = el->drivers[i];
		SElement* driverEl = manifold.getEl(driveIdx);
		SEdgeDriverData& driverData = eData.driverDatas[i];
		switch (driverEl->elType) { 
		case SElType::point: {// point driver of curve - intersection is point
			/* I think this should be guaranteed to be found already 
			as an IntersectionPoint, no?
			*/
			IntersectionPoint* ptr = record.getPointByVectorPosition(driverData.pos());
			if(ptr == nullptr){
				ptr = record.newPoint(driverData.pos());
			}

			// add driver
			ptr->elUVNMap[driverEl->globalIndex].push_back(driverData.uvn);
			ptr->elUVNMap[el->globalIndex].push_back(Vector3f(driverData.uOnEdge, 0, 0));

			// bidirectional lookups between elements
			record.elMap[el->globalIndex][driverEl->globalIndex].push_back({ ptr->index, Intersection::POINT });
			record.elMap[driverEl->globalIndex][el->globalIndex].push_back({ ptr->index, Intersection::POINT });
			continue;
		}
		case SElType::edge: {
			SEdgeData& dEData = manifold.eDataMap[driverEl->name];
			IntersectionPoint* ptr = record.getPointByVectorPosition(driverData.pos());
			if(ptr == nullptr){
				ptr = record.newPoint(driverData.pos());

			}
			// add driver
			ptr->elUVNMap[driverEl->globalIndex].push_back(driverData.uvn);
			ptr->elUVNMap[el->globalIndex].push_back({ driverData.uOnEdge, 0, 0 });

			// bidirectional lookups between elements
			record.elMap[el->globalIndex][driverEl->globalIndex].push_back({ ptr->index, Intersection::POINT });
			record.elMap[driverEl->globalIndex][el->globalIndex].push_back({ ptr->index, Intersection::POINT });
			continue;
		}
		}
	}
	s = _updateEdgeVertices(s,
		manifold,
		el,
		record
	);
	return s;
}

	Status& strata::updateElementIntersections(
		Status & s,
		StrataManifold & manifold,
		SElement * el,
		IntersectionRecord & record
	) {
		/* call this on each element as it's added - 
		WAY easier than trying to do it on demand
		
		TODO: faces and AABB queries	
		*/
		switch (el->elType) {
			case SElType::point: {
				_updatePointDriverIntersections(
					s,
					manifold,
					el,
					record
					//ptr
				);
				break; // break eltype point switch
			}
		case SElType::edge: {
			_updateEdgeDriverIntersections(
				s,
				manifold,
				el,
				record
			);
			break;
			}
		}
	
	return s;

}


//
//Status& getIntersections(Status& s, 
//	StrataManifold& manifold,
//	IntersectionRecord& record,
//	int idxA, int idxB) 
//{
//	/* write out logic for most complicated 
//	* case you can think of, then do it again for another one
//	* 
//	* could probably have separate function to get history intersection, but
//	* the validity of that depends on what kinds of elements it routes through
//	* 
//	* whichever has the higher index MUST be the later element?
//	* trace back from later element ->
//	*	hit a point?
//	*		fan forwards, check for exact matches
//	*	hit a face?
//	*		fan forwards in one step, check for elements directly driven
//	*/
//
//	return record.getIntersectionsBetweenEls(idxA, idxB);
//}




/* for composite operations we're gonna be real stupid with it,
every step creates an intermediate element*/
Status& strata::elementGreaterThan(
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
) {
	//if(idA == idB){ // what are you doing
	//	elsOut.push_back(idA);
	//	return s;
	//}
	SElement* elA = manifold.getEl(idA);
	switch (elA->elType) {
	case SElType::point: {
		//STAT_ERROR(s, "Cannot pass point as first argument of comparison - can't have a point subspace");

		/* we try and take the APEX approach of not freaking out about data we don't understand, 
		so here just return this id*/
		elsOut.push_back(idA);
		return s;
	}
	case SElType::edge: {
		auto vP = manifold.iMap.getIntersectionsBetweenEls(
			idA, idsB
		);
		if (!vP.size()) { // no intersections at all, just return the original element unchanged
			elsOut.push_back(idA);
			return s;
		}
		/* we have at least 1 intersection, find the highest crossing U coord*/
		float maxU = 0.0;

		for (int i = 0; i < static_cast<int>(vP.size()); i++) {
			/* */
			IntersectionPoint* ptr = vP[i].first;
			if (ptr == nullptr) { /* intersection is not a point*/
				continue;
			}

			/* loop over elements connected to this point -
			cumbersome to do it this way, might be better somehow with maps,
			or a map to vectors of coordinates
			*/
			for (int n = 0; n < ptr->elements.size(); n++) {
				if(std::find(idsB.begin(), idsB.end(), ptr->elements[n]) != idsB.end()) {
					maxU = std::max(maxU, ptr->uvns[n].x());
				}
			}
		}
		if (EQ(maxU, 1.0)) { /* no span of edge higher than 1.0 -
			what do we do?
			return -1 as sign that operation has failed? no valid result?
			*/
			elsOut.push_back(-1);
			return s;
		}

		if (EQ(maxU, 0.0)) { /* entire edge is valid, just return it
			*/
			elsOut.push_back(idA);
			return s;
		}

		SEdgeData& baseData = manifold.eDataMap[elA->name];
		/* create new edge starting at maxU 
		
		TODO: don't include names on new edge that don't touch it - 
		so if a wide-reaching expression includes it and some neighbours, 
		don't name new edge with half the mesh.
		much more reusable that way
		*/
		SElement* newEl;
		std::string newName = elA->name + ">(";
		for (auto idB : idsB) {
			//newName += std::to_string(idB) + "_";
			SElement* driverEl = manifold.getEl(idB);
			if (driverEl == nullptr) {
				continue;
			}
			newName += driverEl->name + "_";
		}

		/* if we have an existing element with that name, great, we're already done*/
		newEl = manifold.getEl(newName);
		if (newEl != nullptr) {
			elsOut.push_back(newEl->globalIndex);
			return s;
		}

		manifold.addElement(newName, SElType::edge, newEl);
		SEdgeData& newEdgeData = manifold.eDataMap[newName];
		s = baseData.getSubData(s, newEdgeData, maxU, 1.0);
		/* update intersections of newly created element in map*/
		s = updateElementIntersections(
			s,
			manifold,
			newEl,
			manifold.iMap
		);
		elsOut.push_back(newEl->globalIndex);
	}
	}
	
	return s;
}

Status& strata::elementGreaterThan(
	Status& s,
	StrataManifold& manifold,
	std::vector<int>& elsA,
	std::vector<int>& elsB,
	std::vector<int>& elsOut
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

	for (auto& index : elsA) {
		s = elementGreaterThan(s, manifold,
			index, elsB, elsOut);
	}
	return s;
}


/* same as above but less
factor this later when everything works*/
Status& strata::elementLessThan(
	Status& s,
	StrataManifold& manifold,
	int idA,
	//int idB,
	std::vector<int> idsB,
	std::vector<int>& elsOut
	/*
	*/
) {

	SElement* elA = manifold.getEl(idA);
	switch (elA->elType) {
	case SElType::point: {
		elsOut.push_back(idA);
		return s;
	}
	case SElType::edge: {
		auto vP = manifold.iMap.getIntersectionsBetweenEls(
			idA, idsB
		);
		if (!vP.size()) { // no intersections at all, just return the original element unchanged
			elsOut.push_back(idA);
			return s;
		}
		/* we have at least 1 intersection, find the lowest crossing U coord*/
		float minU = 0.0;

		for (int i = 0; i < static_cast<int>(vP.size()); i++) {
			/* */
			IntersectionPoint* ptr = vP[i].first;
			if (ptr == nullptr) { /* intersection is not a point*/
				continue;
			}

			/* loop over elements connected to this point -
			cumbersome to do it this way, might be better somehow with maps,
			or a map to vectors of coordinates
			*/
			for (int n = 0; n < ptr->elements.size(); n++) {
				if (std::find(idsB.begin(), idsB.end(), ptr->elements[n]) != idsB.end()) {
					minU = std::min(minU, ptr->uvns[n].x());
				}
			}
		}
		if (EQ(minU, 0.0)) { /* no span of edge higher than 1.0 -
			what do we do?
			return -1 as sign that operation has failed? no valid result?
			*/
			elsOut.push_back(-1);
			return s;
		}

		if (EQ(minU, 1.0)) { /* entire edge is valid, just return it
			*/
			elsOut.push_back(idA);
			return s;
		}

		SEdgeData& baseData = manifold.eDataMap[elA->name];
		/* create new edge starting at maxU */
		SElement* newEl;
		std::string newName = elA->name + "<(";
		for (auto idB : idsB) {
			//newName += std::to_string(idB) + "_";
			SElement* driverEl = manifold.getEl(idB);
			if (driverEl == nullptr) {
				continue;
			}
			newName += driverEl->name + "_";
		}

		/* if we have an existing element with that name, great, we're already done*/
		newEl = manifold.getEl(newName);
		if (newEl != nullptr) {
			elsOut.push_back(newEl->globalIndex);
			return s;
		}

		manifold.addElement(newName, SElType::edge, newEl);
		SEdgeData& newEdgeData = manifold.eDataMap[newName];
		s = baseData.getSubData(s, newEdgeData, 0.0, minU);
		/* update intersections of newly created element in map*/
		s = updateElementIntersections(
			s,
			manifold,
			newEl,
			manifold.iMap
		);
		elsOut.push_back(newEl->globalIndex);
	}
	}

	return s;
}

Status& strata::elementLessThan(
	Status& s,
	StrataManifold& manifold,
	std::vector<int>& elsA,
	std::vector<int>& elsB,
	std::vector<int>& elsOut

) {

	for (auto& index : elsA) {
		s = elementLessThan(s, manifold,
			index, elsB, elsOut);
	}
	return s;
}



