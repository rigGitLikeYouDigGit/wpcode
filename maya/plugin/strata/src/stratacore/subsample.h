#pragma once

#include "element.h"
#include "pointData.h"
#include "edgeData.h"
#include "faceData.h"


namespace strata {

	struct SubElement {
		/* subsample a single element
		to define new boundaries or a smaller part of the same space?

		how do we subsample a face? boundary edges?
		further subsamples for edges?

		*/
		int parent = -1;
		SElType parentType = SElType::edge;

		float edgeUs[2]; // still unsure if we'll ever need super-fancy edge samples
		//Vector3f edgeUVNs[2];

 	};


}



