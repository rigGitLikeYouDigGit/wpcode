#pragma once

#include <vector>
#include "api.h"
#include "macro.h"

namespace ed {

	typedef unsigned int uint;

	BETTER_ENUM(STDriverType, int, point, line, face); // used in maya enum attr

	/* handy way to work more easily with vertex buffer memory - cast
	it to vector of float types like this */
	struct Float2
	{
		Float2() {}
		Float2(float x, float y)
			: x(static_cast<float>(x)), y(static_cast<float>(y)) {}
		Float2(MVector v)
			: x(static_cast<float>(v[0])), y(static_cast<float>(v[1])) {}
		Float2(MPoint v)
			: x(static_cast<float>(v[0])), y(static_cast<float>(v[1])) {}
		float x;
		float y;
	};
	struct Float3
	{
		Float3() {}
		Float3(float x, float y, float z)
			: x(static_cast<float>(x)), y(static_cast<float>(y)), z(static_cast<float>(z)) {}
		Float3(MVector v)
			: x(static_cast<float>(v[0])), y(static_cast<float>(v[1])), z(static_cast<float>(v[2])) {}
		Float3(MPoint v)
			: x(static_cast<float>(v[0])), y(static_cast<float>(v[1])), z(static_cast<float>(v[2])) {}
		float x;
		float y;
		float z;
	};
	typedef std::vector<Float3>       Float3Array;
	typedef std::vector<Float2>       Float2Array;
	typedef std::vector<unsigned int> IndexList;


	/* FUNCTIONS TAKE QUATERNIONS OF FORM X-Y-Z-W

	*/

	template<typename T>
	T* quatTo4x4Mat(T* q, T* m);
		// from the id software paper SIMD-From-Quaternion-to-Matrix-and-Back

		// we expect m to be 4x4, no interaction with 4th row or column
		// 
		///// example uses a 3x4 joint matrix, and a quaternion of { float[4] quat, float[3] translation}
		

	template<typename T>
	inline T* quatTo4x4Mat(T* quat) {
		T* m[16];
		return quatTo4x4Mat(quat, m);
	}

	template<typename T>
	T* quatTo3x3Mat(T* quat, T* m);
		// from the id software paper SIMD-From-Quaternion-to-Matrix-and-Back
		// quat is X-Y-Z-W

		// we expect m to be 3x3
		// 
		///// example uses a 3x4 joint matrix, and a quaternion of { float[4] quat, float[3] translation}

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
	static T* x4MatToQuat(T* m, T* q);


	template<typename T>
	inline T* slerp(T* qa, T* qb, T* qm, T t);


	template<typename T>
	static inline void WXYZ_to_XYZW(T* q) {
		T temp = q[0];
		q[0] = q[1];
		q[1] = q[2];
		q[2] = q[3];
		q[3] = temp;
	}

	template<typename T>
	static inline void XYZW_to_WXYZ(T* q) {
		T temp = q[3];
		q[3] = q[2];
		q[2] = q[1];
		q[1] = q[0];
		q[0] = temp;
	}


	// TODO: simd
	// lerp N values of a with N values of b by weight t and put result in out
	template<typename T, int N>
	static inline T* lerpN(T* a, T* b, T* out, T t);

	MMatrix interpolateMMatrixArray(std::vector<MMatrix>& mmatrixArr, MMatrix& out, float t);

}