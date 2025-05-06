
#pragma once

#include "api.h"
#include "macro.h"

#include "lib.h"

namespace ed {
	/* FUNCTIONS TAKE QUATERNIONS OF FORM X-Y-Z-W

	*/

	template<typename T>
	T* quatTo4x4Mat(T* q, T* m) {
		// from the id software paper SIMD-From-Quaternion-to-Matrix-and-Back

		// we expect m to be 4x4, no interaction with 4th row or column
		// 
		///// example uses a 3x4 joint matrix, and a quaternion of { float[4] quat, float[3] translation}
		//m[0 * 4 + 3] = q[4];
		//m[1 * 4 + 3] = q[5];
		//m[2 * 4 + 3] = q[6]; 
		// 
		//float x2 = q[0] + q[0];
		//float y2 = q[1] + q[1];
		//float z2 = q[2] + q[2];
		//{
		//	float xx2 = q[0] * x2;
		//	float yy2 = q[1] * y2;
		//	float zz2 = q[2] * z2;
		//	m[0 * 4 + 0] = 1.0f - yy2 - zz2;
		//	m[1 * 4 + 1] = 1.0f - xx2 - zz2;
		//	m[2 * 4 + 2] = 1.0f - xx2 - yy2;
		//}
		//{
		//	float yz2 = q[1] * z2;
		//	float wx2 = q[3] * x2;
		//	m[2 * 4 + 1] = yz2 - wx2;
		//	m[1 * 4 + 2] = yz2 + wx2;
		//}
		//{
		//	float xy2 = q[0] * y2;
		//	float wz2 = q[3] * z2;
		//	m[1 * 4 + 0] = xy2 - wz2;
		//	m[0 * 4 + 1] = xy2 + wz2;
		//}
		//{
		//	float xz2 = q[0] * z2;
		//	float wy2 = q[3] * y2;
		//	m[0 * 4 + 2] = xz2 - wy2;
		//	m[2 * 4 + 0] = xz2 + wy2;
		//}

		////// weird SSE version below

		T x2 = q[0] + q[0];
		T y2 = q[1] + q[1];
		T z2 = q[2] + q[2];
		T w2 = q[3] + q[3];
		T yy2 = q[1] * y2;
		T xy2 = q[0] * y2;
		T xz2 = q[0] * z2;
		T yz2 = q[1] * z2;
		T zz2 = q[2] * z2;
		T wz2 = q[3] * z2;
		T wy2 = q[3] * y2;
		T wx2 = q[3] * x2;
		T xx2 = q[0] * x2;
		m[0 * 4 + 0] = -yy2 - zz2 + 1.0f;
		m[0 * 4 + 1] = xy2 + wz2;
		m[0 * 4 + 2] = xz2 - wy2;
		//m[0 * 4 + 3] = q[4];
		m[1 * 4 + 0] = xy2 - wz2;
		m[1 * 4 + 1] = -xx2 - zz2 + 1.0f;
		m[1 * 4 + 2] = yz2 + wx2;
		///m[1 * 4 + 3] = q[5];
		m[2 * 4 + 0] = xz2 + wy2;
		m[2 * 4 + 1] = yz2 - wx2;
		m[2 * 4 + 2] = -xx2 - yy2 + 1.0f;
		//m[2 * 4 + 3] = q[6];
		return m;
	}


	template<typename T>
	T* quatTo3x3Mat(T* quat, T* m) {
		// from the id software paper SIMD-From-Quaternion-to-Matrix-and-Back
		// quat is X-Y-Z-W

		// we expect m to be 3x3
		// 
		///// example uses a 3x4 joint matrix, and a quaternion of { float[4] quat, float[3] translation}

		T x2 = q[0] + q[0];
		T y2 = q[1] + q[1];
		T z2 = q[2] + q[2];
		T w2 = q[3] + q[3];
		T yy2 = q[1] * y2;
		T xy2 = q[0] * y2;
		T xz2 = q[0] * z2;
		T yz2 = q[1] * z2;
		T zz2 = q[2] * z2;
		T wz2 = q[3] * z2;
		T wy2 = q[3] * y2;
		T wx2 = q[3] * x2;
		T xx2 = q[0] * x2;
		m[0 * 3 + 0] = -yy2 - zz2 + 1.0f;
		m[0 * 3 + 1] = xy2 + wz2;
		m[0 * 3 + 2] = xz2 - wy2;
		//m[0 * 4 + 3] = q[4];
		m[1 * 3 + 0] = xy2 - wz2;
		m[1 * 3 + 1] = -xx2 - zz2 + 1.0f;
		m[1 * 3 + 2] = yz2 + wx2;
		///m[1 * 4 + 3] = q[5];
		m[2 * 3 + 0] = xz2 + wy2;
		m[2 * 3 + 1] = yz2 - wx2;
		m[2 * 3 + 2] = -xx2 - yy2 + 1.0f;
		//m[2 * 4 + 3] = q[6];
		return m;
	}


	template<typename T>
	static T* x4MatToQuat(T* m, T* q) {

		//float* q = &jointQuats[i].q;
		//T* q[4];

		// diagonal sign check
		if (m[0 * 4 + 0] + m[1 * 4 + 1] + m[2 * 4 + 2] > 0.0f) {
			T t = +m[0 * 4 + 0] + m[1 * 4 + 1] + m[2 * 4 + 2] + 1.0f;
			T s = ReciprocalSqrt<T>(t) * 0.5f;
			q[3] = s * t;
			q[2] = (m[0 * 4 + 1] - m[1 * 4 + 0]) * s;
			q[1] = (m[2 * 4 + 0] - m[0 * 4 + 2]) * s;
			q[0] = (m[1 * 4 + 2] - m[2 * 4 + 1]) * s;
		}
		else if (m[0 * 4 + 0] > m[1 * 4 + 1] && m[0 * 4 + 0] > m[2 * 4 + 2]) {
			T t = +m[0 * 4 + 0] - m[1 * 4 + 1] - m[2 * 4 + 2] + 1.0f;
			T s = ReciprocalSqrt<T>(t) * 0.5f;
			q[0] = s * t;
			q[1] = (m[0 * 4 + 1] + m[1 * 4 + 0]) * s;
			q[2] = (m[2 * 4 + 0] + m[0 * 4 + 2]) * s;
			q[3] = (m[1 * 4 + 2] - m[2 * 4 + 1]) * s;
		}
		else if (m[1 * 4 + 1] > m[2 * 4 + 2]) {
			T t = -m[0 * 4 + 0] + m[1 * 4 + 1] - m[2 * 4 + 2] + 1.0f;
			T s = ReciprocalSqrt<T>(t) * 0.5f;
			q[1] = s * t;
			q[0] = (m[0 * 4 + 1] + m[1 * 4 + 0]) * s;
			q[3] = (m[2 * 4 + 0] - m[0 * 4 + 2]) * s;
			q[2] = (m[1 * 4 + 2] + m[2 * 4 + 1]) * s;
		}
		else {
			T t = -m[0 * 4 + 0] - m[1 * 4 + 1] + m[2 * 4 + 2] + 1.0f;
			T s = ReciprocalSqrt<T>(t) * 0.5f;
			q[2] = s * t;
			q[3] = (m[0 * 4 + 1] - m[1 * 4 + 0]) * s;
			q[0] = (m[2 * 4 + 0] + m[0 * 4 + 2]) * s;
			q[1] = (m[1 * 4 + 2] + m[2 * 4 + 1]) * s;
		}

		return q;
	}

	//static MDoubleArray ed::uniformKnotsForCVs(int nCVs, int degree) {
	//	// for degree = 1, k=2
	//	// degree = 2, k=3 etc
	//	// degree = 3, k=4
	//	// for a cubic curve with 6 CVs, knots should be thus
	//	// [ 0, 0, 0, 0, 1, 2, 3, 3, 3, 3 ]
	//	// for cubic curve with 4 CVs:
	//	// [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
	//	// 

	//	MDoubleArray arr(nCVs + degree - 1, float(nCVs) - float(degree));
	//	float knotVal = 0.0;
	//	for (int i = 0; i < (nCVs); i++) {
	//		if (i < (degree - 1)) {
	//			arr[i] = knotVal;
	//			continue;
	//		}
	//		arr[i] = knotVal;
	//		knotVal += 1.0;
	//	}
	//	return arr;
	//}

	template<typename T>
	inline T* slerp(T* qa, T* qb, T* qm, T t) {
		/* adapted from euclideanspace.com https://www.euclideanspace.com/maths/algebra/realNormedAlgebra/quaternions/slerp/

		quats should be X-Y-Z-W

		matches maya, but not eigen :(

		qa is quat1
		qb is quat2
		qm is result quat
		t is scalar value [0.0, 1.0]

		quat pointers should all be 4-long
		*/

		// Calculate angle between them.
		//T cosHalfTheta = qa.w * qb.w + qa.x * qb.x + qa.y * qb.y + qa.z * qb.z;
		//T cosHalfTheta = qa[0] * qb[0] + qa[1] * qb[1] + qa[2] * qb[2] + qa[3] * qb[3];
		T cosHalfTheta = qa[3] * qb[3] + qa[0] * qb[0] + qa[1] * qb[1] + qa[2] * qb[2];

		/* /////OPTIONAL BLOCK but seems to give better behaviour */
		if (cosHalfTheta < 0) {
			//qb.w = -qb.w; qb.x = -qb.x; qb.y = -qb.y; qb.z = qb.z;
			//qb[0] = -qb[0]; qb[1] = -qb[1]; qb[2] = -qb[2]; qb[3] = qb[3];
			qb[3] = -qb[3]; qb[0] = -qb[0]; qb[1] = -qb[1]; qb[2] = qb[2];
			cosHalfTheta = -cosHalfTheta;
		}
		/////

		// if qa=qb or qa=-qb then theta = 0 and we can return qa
		if (abs(cosHalfTheta) >= 1.0) {
			//qm.w = qa.w; qm.x = qa.x; qm.y = qa.y; qm.z = qa.z;
			//qm[0] = qa[0]; qm[1] = qa[1]; qm[2] = qa[2]; qm[3] = qa[3];
			qm[3] = qa[3]; qm[0] = qa[0]; qm[1] = qa[1]; qm[2] = qa[2];
			return qm;
		}
		// Calculate temporary values.
		T halfTheta = acos(cosHalfTheta);
		T sinHalfTheta = sqrt(1.0 - cosHalfTheta * cosHalfTheta);
		// if theta = 180 degrees then result is not fully defined
		// we could rotate around any axis normal to qa or qb
		if (fabs(sinHalfTheta) < 0.001) { // fabs is floating point absolute
			//qm.w = (qa.w * 0.5 + qb.w * 0.5);
			//qm.x = (qa.x * 0.5 + qb.x * 0.5);
			//qm.y = (qa.y * 0.5 + qb.y * 0.5);
			//qm.z = (qa.z * 0.5 + qb.z * 0.5);
			qm[0] = (qa[0] * 0.5 + qb[0] * 0.5);
			qm[1] = (qa[1] * 0.5 + qb[1] * 0.5);
			qm[2] = (qa[2] * 0.5 + qb[2] * 0.5);
			qm[3] = (qa[3] * 0.5 + qb[3] * 0.5);
			return qm;
		}
		T ratioA = sin((1 - t) * halfTheta) / sinHalfTheta;
		T ratioB = sin(t * halfTheta) / sinHalfTheta;
		//calculate Quaternion.
		/*qm.w = (qa.w * ratioA + qb.w * ratioB);
		qm.x = (qa.x * ratioA + qb.x * ratioB);
		qm.y = (qa.y * ratioA + qb.y * ratioB);
		qm.z = (qa.z * ratioA + qb.z * ratioB);*/
		qm[0] = (qa[0] * ratioA + qb[0] * ratioB);
		qm[1] = (qa[1] * ratioA + qb[1] * ratioB);
		qm[2] = (qa[2] * ratioA + qb[2] * ratioB);
		qm[3] = (qa[3] * ratioA + qb[3] * ratioB);
		return qm;
	}

	// TODO: simd
	template<typename T, int N>
	static inline T* lerpN(T* a, T* b, T* out, T t) {
		for (int i = 0; i < N; i++) {
			out[i] = a[i] + t * (b[i] - a[i]);
		}
		return out;
	}

	//MDoubleArray ed::uniformKnotsForCVs(int nCVs, int degree)
	//{
	//		// for degree = 1, k=2
	//// degree = 2, k=3 etc
	//// degree = 3, k=4
	//// for a cubic curve with 6 CVs, knots should be thus
	//// [ 0, 0, 0, 0, 1, 2, 3, 3, 3, 3 ]
	//// for cubic curve with 4 CVs:
	//// [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
	//// 

	//	MDoubleArray arr(nCVs + degree - 1, float(nCVs) - float(degree));
	//	float knotVal = 0.0;
	//	for (int i = 0; i < (nCVs); i++) {
	//		if (i < (degree - 1)) {
	//			arr[i] = knotVal;
	//			continue;
	//		}
	//		arr[i] = knotVal;
	//		knotVal += 1.0;
	//	}
	//	return arr;
	//}


	MDoubleArray uniformKnotsForCVs(int nCVs, int degree)
	{
		return MDoubleArray();
	}

	MMatrix interpolateMMatrixArray(std::vector<MMatrix>& mmatrixArr, MMatrix& out, float t) {
		/* assuming steadily spaced keypoints in arr, interpolate at param t
		slerp rotation component
		*/
		t = fmin(fmax(t, 0.0f), 1.0f);

		int start = static_cast<int>(mmatrixArr.size() * t);
		int end = static_cast<int>(mmatrixArr.size() * t) + 1;
		float fraction = mmatrixArr.size() * t - start;

		double quatA[4];
		double quatB[4];
		x4MatToQuat<double>(MMatrixFlatData(mmatrixArr[start]), quatA);
		x4MatToQuat<double>(MMatrixFlatData(mmatrixArr[end]), quatB);
		double quatOut[4];
		slerp<double>(quatA, quatB, quatOut, fraction);
		quatTo4x4Mat(quatOut, MMatrixFlatData(out));

		lerpN<double, 3>(
			MMatrixFlatData(mmatrixArr[end]) + 12,
			MMatrixFlatData(mmatrixArr[end]) + 12,
			MMatrixFlatData(out) + 12,
			fraction
		);
		return out;
	}

}


