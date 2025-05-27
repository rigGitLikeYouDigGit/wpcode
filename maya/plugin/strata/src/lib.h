#pragma once

#include <vector>
#include <math.h>
#include "api.h"
#include "macro.h"
#include "bezier/bezier.h"

namespace ed {

	typedef unsigned int uint;
	using std::min;
	using std::max;

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
		Float3() : x(0), y(0), z(0) {}
		Float3(float x, float y, float z)
			: x(static_cast<float>(x)), y(static_cast<float>(y)), z(static_cast<float>(z)) {}
		Float3(MVector v)
			: x(static_cast<float>(v[0])), y(static_cast<float>(v[1])), z(static_cast<float>(v[2])) {}
		Float3(MPoint v)
			: x(static_cast<float>(v[0])), y(static_cast<float>(v[1])), z(static_cast<float>(v[2])) {}
		Float3(float* v)
			: x(static_cast<float>(v[0])), y(static_cast<float>(v[1])), z(static_cast<float>(v[2])) {}
		Float3(const float* v)
			: x(static_cast<float>(v[0])), y(static_cast<float>(v[1])), z(static_cast<float>(v[2])) {
		}
		Float3(double* v)
			: x(static_cast<float>(v[0])), y(static_cast<float>(v[1])), z(static_cast<float>(v[2])) {}
		Float3(Eigen::Vector3f& v)
			: x(static_cast<float>(v.data()[0])), y(static_cast<float>(v.data()[1])), z(static_cast<float>(v.data()[2])) {}
		Float3(Eigen::Vector3f&& v)
			: x(static_cast<float>(v[0])), y(static_cast<float>(v.data()[1])), z(static_cast<float>(v.data()[2])) {
		}
		Float3(Eigen::Matrix<float, 3, 1, 0, 3, 1>&& v)
			: x(static_cast<float>(v[0])), y(static_cast<float>(v.data()[1])), z(static_cast<float>(v.data()[2])) {
		}
		Float3(Eigen::Matrix<double, 3, 1, 0, 3, 1> v)
			: x(static_cast<float>(v.data()[0])), y(static_cast<float>(v.data()[1])), z(static_cast<float>(v.data()[2])) {}
		Float3(Eigen::Array<double, 3, 1, 0, 3, 1> v)
			: x(static_cast<float>(v.data()[0])), y(static_cast<float>(v.data()[1])), z(static_cast<float>(v.data()[2])) {}
		float x;
		float y;
		float z;
		Float3& operator=(const float*& v) {
			x = v[0];
			y = v[1];
			z = v[2];
			return *this;
		}
		Float3& operator=(const Eigen::Vector3f& v) {
			return operator=(v.data());
		}
		Float3& operator=(const Eigen::MatrixBase<float>& v) {
			x = *v.begin();
			y = *(v.begin()+1);
			z = *(v.begin()+2);
			return *this;
		}
	};
	typedef std::vector<Float3>       Float3Array;
	typedef std::vector<Float2>       Float2Array;
	typedef std::vector<unsigned int> IndexList;


	template<typename T>
	inline T uss(T in) {
		/* unsigned-to-signed conversion - 
		[0,1] -> [-1,1] */
		return (2.0 * in) - 1.0;
	}

	template<typename T>
	inline T sus(T in) {
		/* signed-to-unsigned conversion -
		[-1,1] -> [0,1] */
		return (in + 1.0) / 2.0;
	}

	template<typename T>
	inline T lerp( T a, T b, T t ) {
		return b * t + (1.0 - t) * b;
	}

	template<typename T, typename N>
	inline T lerp( T a, T b, N t) {
		return b * t + (1.0 - t) * b;
	}

	template<typename T>
	inline T clamp(T in, T inMin, T inMax) {
		return std::min(inMax, std::max(inMin, in));
	}

	template<typename T>
	inline T mapTo01(T in, T inMin, T inMax) {
		return (in - inMin) / (inMax - inMin);
	}

	template<typename T>
	inline T mapTo01(T in, T inMin, T inMax, bool clamp=true) {
		return clamp((in - inMin) / (inMax - inMin), inMin, inMax);
	}

	template<typename T>
	inline T remap(T in, T inMin, T inMax, T outMin, T outMax) {
		return lerp(mapTo01(in, inMax, inMin), outMin, outMax);
	}

	template<typename T>
	inline T remap(T in, T inMin, T inMax, T outMin, T outMax, bool clamp=true) {
		return clamp( lerp( mapTo01(in, inMax, inMin), outMin, outMax ) );
	}

	/* what's the best way to compose things like this? if I wanted a remap function with a quadratic 
	smooth clamp?*/

	/* Inigo Quilez smooth min functions - normally only need exp and quadratic */
	// exponential
	template<typename T>
	T sminExp(T a, T b, T k)
	{
		k *= 1.0;
		T r = exp2(-a / k) + exp2(-b / k);
		return -k * log2(r);
	}
	// root
	template<typename T>
	T sminRoot(T a, T b, T k)
	{
		k *= 2.0;
		T x = b - a;
		return 0.5 * (a + b - sqrt(x * x + k * k));
	}
	// sigmoid
	template<typename T>
	T sminSig(T a, T b, T k)
	{
		k *= log(2.0);
		T x = b - a;
		return a + x / (1.0 - exp2(x / k));
	}
	// quadratic polynomial
	template<typename T>
	T sminQ(T a, T b, T k)
	{
		k *= 4.0;
		T h = max(k - abs(a - b), 0.0) / k;
		return min(a, b) - h * h * k * (1.0 / 4.0);
	}
	// cubic polynomial
	template<typename T>
	T sminC(T a, T b, T k)
	{
		k *= 6.0;
		T h = max(k - abs(a - b), 0.0) / k;
		return min(a, b) - h * h * h * k * (1.0 / 6.0);
	}
	// quartic polynomial
	template<typename T>
	T sminQuart(T a, T b, T k)
	{
		k *= 16.0 / 3.0;
		T h = max(k - abs(a - b), 0.0) / k;
		return min(a, b) - h * h * h * (4.0 - h) * k * (1.0 / 16.0);
	}
	// circular
	template<typename T>
	T sminCirc(T a, T b, T k)
	{
		k *= 1.0 / (1.0 - sqrt(0.5));
		T h = max(k - abs(a - b), 0.0) / k;
		return min(a, b) - k * 0.5 * (1.0 + h - sqrt(1.0 - h * (h - 2.0)));
	}
	//// circular geometrical
	//float smin(float a, float b, float k)
	//{
	//	k *= 1.0 / (1.0 - sqrt(0.5));
	//	return max(k, min(a, b)) -
	//		length(max(k - vec2(a, b), 0.0));
	//}

	template<typename T>
	inline T smoothstepCubic(T x) {
		return x * x * (3.0 - 2.0 * x);
	}

	template<typename T>
	inline T smoothstepCubicRational(T x) {
		return x * x * x / (3.0 * x * x - 3.0 * x + 1.0);
	}

	template<typename T>
	inline T invSmoothstepCubicRational(T x) {
		T a = pow(x, 1.0 / 3.0);
		return a / (a + pow(1.0 - x, 1.0 / 3.0));

	}

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


	/*
	curves - do we cache arrays of densely sampled values?
	if yes, we have to either save many values, 
	OR resample those values in a cubic spline built on the fly
	so we still pay cubic cost anyway
	*/

	template<typename T>
	Eigen::MatrixX3f makeRMFNormals(
		const bez::CubicBezierPath& crv,
		const Eigen::MatrixX3f& targetNormals,
		const Eigen::VectorXf& normalUVals,
		const int nSamples
	) {/* 
		normals are target vectors, normalUVals are targets to match
		TODO: maybe get twist in here

		at each targetNormal uValue, normals of RMF are pointed as close to that direction as
		we can

		having defined normals means we can parallelise between each one? 

		for now just do single pass reflection RMF from first normal,
		TODO

		using double reflection system from the microsoft paper

		 0 to n − 1 do
		Begin
			1) v1 := xi+1 − xi ;						/*compute reflection vector of R1. 
			2) c1 := v1 · v1;
			3) rLi := ri − (2/c1) ∗ (v1 · ri) ∗ v1;		/*compute rL		i = R1ri . 
			4) tLi := ti − (2/c1) ∗ (v1 · ti) ∗ v1;		/*compute tL	i = R1ti . 
			5) v2 := ti+1 − tLi ;						/*compute reflection vector of R2. 
			6) c2 := v2 · v2;
			7) ri+1 := rLi − (2/c2) ∗ (v2 · rLi ) ∗ v2; /*compute ri+1 = R2rLi . 
			8) si+1 := ti+1 × ri+1;						/*compute vector si+1 of Ui+1. 
			9) Ui+1 := (ri+1,si+1, ti+1);

		*/

		Eigen::MatrixX3f resultNs(nSamples);

		//Eigen::Vector3f ri = targetNormals.row(0);
		resultNs.row(0) = targetNormals.row(0);

		for (int i = 0; i < nSamples - 1; i++) {
			float param = (0.9999 / float(nSamples-1) * float(i));
			float nextParam = (1.0 / float(nSamples - 1) * float(i + 1));
			auto xi = crv.eval(param);
			auto ti = (crv.eval(param + 0.0001) - xi).normalized();

			auto xiPlus1 = crv.eval(nextParam);
			auto v1 = xiPlus1 - xi;
			auto c1 = v1.dot(v1);
			auto rLi = resultNs.row(i) - (2.0 / c1) * (v1.dot(resultNs.row(i))) * v1;
			auto tLi = ti - (2.0 / c1) * (v1.dot(ti)) * v1;

			auto tiPlus1 = (crv.eval(nextParam + 0.0001) - xiPlus1).normalized(); // next point's tangent
			auto v2 = tiPlus1 - tLi;
			auto c2 = v2.dot(v2);
			auto riPlus1 = rLi - (2.0 / c2) * (v2.dot(rLi)) * v2; // final reflected normal
			resultNs.row(i + 1) = riPlus1.normalized();
		}
		return resultNs;
	}
}