#pragma once

#include <vector>
#include "MInclude.h"
#include "api.h"
#include "macro.h"

namespace ed {

	std::vector<MMatrix> curveMatricesFromDriverDatas(
		std::vector<MMatrix> controlMats, int segmentPointCount);


}
