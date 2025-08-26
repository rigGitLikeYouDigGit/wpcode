

#include <string>
#include <vector>
#include "../status.h"
#include "../macro.h"

#include "manifold.h"
#include "libManifold.h"


using namespace strata;


Status& makeFaceGroup(
	Status& s,
	StrataManifold& manifold,
	SFaceData&,
	SElement* el,
	std::vector<std::string>& edgeNames
) {
	/*
	* - filter elements to see which are edges, which are points to match
	* - filter edge islands to check for disconnected patches - those will create tubes when connected
	* - for each island, work out separate face
	* 
	* CANNOT MIX CLOSED AND OPEN BOUNDARIES ON ISLANDS.
	*  - if all are open boundaries, do a simple rail
	*  - if all are closed, tube
	*/

	return s;
}