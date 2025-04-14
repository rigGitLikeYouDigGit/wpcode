#pragma once

#include "../MInclude.h"
#include "../macro.h"

#include "../api.h"
#include "../stratacore/manifold.h"


namespace ed {

	MPointArray mPointsFromManifoldPoints(const StrataManifold& manifold) {
		MPointArray result(manifold.pointDatas.size());
		for (int i = 0; i < result.length(); i++) {
			result[i] = manifold.pointDatas[i].matrix[3];
		}
		return result;
	}
	MMatrixArray mMatrixArrayFromManifoldPoints(const StrataManifold& manifold) {
		MMatrixArray result(manifold.pointDatas.size());
		for (int i = 0; i < result.length(); i++) {
			result[i] = manifold.pointDatas[i].matrix;
		}
		return result;
	}

}

