#pragma once

#include "../name.h"
#include "element.h"


namespace strata {

	// domain datas always relative in domain space - when applied, recover the original shape of element
	struct SPointSpaceData { // domain data FOR a point, anchor could be any type
		/* stores affine offset from single SCoord - in domain, blend these together to 
		get final matrix*/
		float weight = 1.0;
		AffineCompact3f offset = AffineCompact3f::Identity(); // translation is UVN, rotation is relative rotation from that position

		std::string strInfo();
	};

	struct SPoint : SElement {

		SmallList<SPointSpaceData, 1> spaceDatas = {}; // datas for each anchor
		AffineCompact3f finalMatrix = AffineCompact3f::Identity(); // final evaluated matrix in world space

		std::string strInfo();
	};

}
