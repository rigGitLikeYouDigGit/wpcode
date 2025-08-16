#pragma once

#include <vector>
#include <memory>
#include <iterator>

#include "containers.h"
#include "iterator.h"
#include "macro.h"

namespace strata {

	template <typename VT>
	struct NDArray {
		std::vector<VT> data;
		SmallList<int, 4> dims;
		SmallList<int, 4> strides; // how many flat values a single index increment in each dimension adds
		/* eg for a 4x4 matrix:
		dims = {4, 4}
		strides = {4, 1} // last is always 1, maybe trim it

		for a 3x13x4x4 array of matrices:
		dims = {3, 13, 4, 4}
		strides = {198, 16, 4, 1}
		*/

		VT defaultVal;

		enum class BroadcastMode {
			Strict, // require all dimensions to line up exactly
			Trim, // curtail to the shortest length
			ExtendLast, // extend by the last value to max array length
			ExtendDefault // extend by the default
		};

		inline void _buildDimStrides() {
			strides.clear();
			strides.reserve(dims.size());
			int rowLength = 1;
			for (int i = 0; i < dims.size(); i++) {
				strides.push(rowLength);
				rowLength *= dims[i];
			}
		}

		NDArray(int iDims) {
			dims.push_back(iDims);
		}


		inline int flatIndex(const int* indexPath, const int& indexPathLen=1) {
			int result = 0;
			for (int i = 0; i < indexPathLen; i++) {
				result += strides[i] * indexPath[i];
			}
			return result;
		}
		inline int flatIndex(const SmallList<int> indexList) {
			return flatIndex(indexList.data(), indexList.size());
		}

		inline SmallList<int, 4> pathIndex(int flatIndex) {
			/* given flattened index, return the path to get to it*/
			SmallList<int, 4> pathIndex;
			for (int i = 0; i < dims.size(); i++) {
				pathIndex.push_back(flatIndex / strides[i]);
				flatIndex = flatIndex % strides[i];
			}
			return pathIndex;
		}

		inline VT accessSingle(int flatIndex) { return data[flatIndex]; }
		inline VT* pointSingle(int flatIndex) { return *data[flatIndex]; }

		//inline 

		template<typename otherVT=int>
		Status broadcast(
			const NDArray<otherVT>& otherArr,
			BroadcastMode mode,
			const int* desiredDimsA = { 1, }, const int nDimsA = 1,
			const int* desiredDimsB = {1, }, const int nDimsB = 1
			) {
			/* return pairs of PAIRS OF start, end indices? 
			*/ 

			
		}
	};


}
