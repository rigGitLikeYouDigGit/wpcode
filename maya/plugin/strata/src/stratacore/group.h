#pragma once
#include <set>
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
		std::string name;
		//std::vector<int> globalIndices;
		//std::vector<std::string> elNames;
		std::set<std::string> elNames;

		/* type-specific data? */
		std::vector<std::string> uLines; /* edge sequence to use for U param in face groups */
		std::vector<std::string> vLines; /* edge sequence to use for V param in face groups */
	};
}


