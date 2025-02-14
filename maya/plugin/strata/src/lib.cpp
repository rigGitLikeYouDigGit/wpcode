

#include <maya/MMatrix.h>
#include <maya/MQuaternion.h>

#include "api.h"
#include "macro.h"

namespace ed {

	typedef unsigned int uint;

	BETTER_ENUM(STDriverType, int, point, line, face);

	template<typename T>
	T* quatTo4x4Mat(T* quat, T* m) {
		// from the id software paper SIMD-From-Quaternion-to-Matrix-and-Back

		// we expect m to be 4x4
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
	inline T* quatTo4x4Mat(T* quat) {
		T* m[16];
		return quatTo4x4Mat(m);
	}

	template<typename T>
	T* quatTo3x3Mat(T* quat, T* m) {
		// from the id software paper SIMD-From-Quaternion-to-Matrix-and-Back

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
	inline T* quatTo3x3Mat(T* quat) {
		T* m[9];
		return quatTo3x3Mat(quat, m);
	}

	template<typename T>
	T ReciprocalSqrt(T x) {
		long i;
		T y, r;
		y = x * 0.5f;
		i = *(long*)(&x);
		i = 0x5f3759df - (i >> 1); // mr carmack please calm down
		r = *(T*)(&i);
		r = r * (1.5f - r * r * y);
		return r;
	}

	template<typename T>
	T* x4MatToQuat(T* m) {

		//float* q = &jointQuats[i].q;
		T* q[4];
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





	////TODO: get a proper shared library for this
}
