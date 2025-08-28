#pragma once

/* lib functions for face creation and topology,
not specifically tessellation

this will get complicated but for now keep as simple as possible
*/

/* can we have a 'headerInclude.h' so we
get all the STL stuff like this without unneeded header dependencies?
*/
#include <string>
#include <vector>
#include "../status.h"
#include "../macro.h"

//#include "manifold.h"
//#include "libManifold.h"


namespace strata {

	struct StrataManifold;
	struct SFaceData;
	struct SElement;

	Status& makeFaceGroup(
		Status& s,
		StrataManifold& manifold,
		SGroup* grp,
		std::vector<std::string>& edgeNames
	);

}



