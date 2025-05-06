

#pragma once
#ifndef ED_LIB_EIGEN



#include <Eigen/Dense>
#include <unsupported/Eigen/MatrixFunctions>

#include "libEigen.h"



using namespace ed;

std::vector<MMatrix> ed::curveMatricesFromDriverDatas(
	std::vector<MMatrix> controlMats, int segmentPointCount) {
	/* interpolate rational-root matrices between drivers, and then add drivers and interpolated mats to result*/
	std::vector<MMatrix> result;
	result.reserve(controlMats.size() + segmentPointCount * (controlMats.size() - 1));

	/* TODO: parallelise segments here if matrix roots are costly*/
	for (size_t i = 0; i < (controlMats.size() - 1); i++) {
		result.push_back(controlMats[i]);

		// get relative matrix from this point to the next
		auto relMat = toEigen(controlMats[i + 1] * controlMats[i].inverse());

		// get square root of matrix, for single midpoint; cubic for 2, etc
		//Eigen::MatrixPower<Eigen::Matrix4cd> relMatPower(relMat);
		Eigen::MatrixPower<Eigen::Matrix4d> relMatPower(relMat);
		auto step = relMatPower(1.0 / float(segmentPointCount + 1));

		// raise that root matrix to the same power as its segment point index
		for (size_t n = 0; n < segmentPointCount; n++) {
			//result.push_back( controlMats[i] * toMMatrix<Eigen::Matrix4cd>(
			result.push_back(controlMats[i] * toMMatrix<Eigen::Matrix4d>(
				step.pow(static_cast<double>(n + 1))));
		}

	}
	result.push_back(controlMats.back());
	return result;
}

/* the templating is beyond me but it would be nice to have a general function
for 'setArrayLength', 'addToArray' etc
that could accept any of these
*/

////template <typename arrT, typename T>
//void setArrayLength(MMatrixArray& arr, int n) {
//	arr.setLength(n);
//}
//
//void setArrayLength(std::vector<>& arr, int n) {
//	arr.setLength(n);
//}



//
//static Eigen::MatrixX4d ed::curveMatricesFromDriverDatas(
//	MMatrixArray controlMats, int segmentPointCount) {
//	/* interpolate rational-root matrices between drivers, and then add drivers and interpolated mats to result*/
//	MMatrixArray result;
//	result.setLength(controlMats.length() + segmentPointCount * (controlMats.length() - 1));
//
//	/* TODO: parallelise segments here if matrix roots are costly*/
//	for (size_t i = 0; i < (controlMats.length() - 1); i++) {
//		result[i] = controlMats[i];
//
//		// get relative matrix from this point to the next
//		auto relMat = toEigen(controlMats[i + 1] * controlMats[i].inverse());
//
//		// get square root of matrix, for single midpoint; cubic for 2, etc
//		//Eigen::MatrixPower<Eigen::Matrix4cd> relMatPower(relMat);
//		Eigen::MatrixPower<Eigen::Matrix4d> relMatPower(relMat);
//		auto step = relMatPower(1.0 / float(segmentPointCount + 1));
//
//		// raise that root matrix to the same power as its segment point index
//		for (size_t n = 0; n < segmentPointCount; n++) {
//			//result.push_back( controlMats[i] * toMMatrix<Eigen::Matrix4cd>(
//			result.push_back(controlMats[i] * toMMatrix<Eigen::Matrix4d>(
//				step.pow(static_cast<double>(n + 1))));
//		}
//
//	}
//	result.push_back(controlMats.back());
//	return result;
//}
#endif // !ED_LIB_EIGEN