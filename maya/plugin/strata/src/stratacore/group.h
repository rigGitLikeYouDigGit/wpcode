#pragma once

#include "element.h"
#include "pointData.h"
#include "edgeData.h"
#include "faceData.h"

/* if an element expression creates multiple individual 
elements, those will be connected in a group of that 
same name - 

Groups should expand automatically for intersections, operators etc

*/

namespace strata {
	

	struct SGroup {
		/* wow what a complex and worthwhile object
		*/
		//int globalIndex = -1; /* groups don't share the same indexing as real elements */
		SElType elT = SElType::point;
		StrataName name;
		std::vector<int> globalIndices;
	};
}


