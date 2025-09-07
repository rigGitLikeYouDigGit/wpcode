#pragma once

#include "../types.h"

namespace strata {

	Status& makeNewFaceData( /* */
		Status& s,
		StrataManifold& man,
		std::vector<int>& vertexPath,
		SFaceCreationParams& faceCreateParams, /* should this be packed in some other way?*/
		SElement*& el
	);


}