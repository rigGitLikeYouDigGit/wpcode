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
#include "../../status.h"
#include "../../macro.h"
#include "../types.h"


namespace strata {



	Status& makeNewFaceTopo( /* */
		Status& s,
		StrataManifold& man,
		std::vector<int>& vertexPath,
		SFaceCreationParams& faceCreateParams, /* should this be packed in some other way?*/
		SElement*& el
	);

	Status& makeFaceGroup(
		Status& s,
		StrataManifold& manifold,
		SGroup* grp,
		SFaceCreationParams& faceCreateParams,
		std::vector<std::string>& edgeNames
	);

	//float evalBezPatch

}



