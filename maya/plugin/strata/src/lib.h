#pragma once

#include <vector>
#include <math.h>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <map>
#include "api.h"
#include "macro.h"
//#include "bezier/bezier.h"


/* if C warns me about losing precision adding 2.0 to a float,
* I don't care about it
*/

#pragma warning(push)
#pragma warning( disable : 4244)
namespace ed {

	template<typename T>
	inline std::string str(T any) {
		return std::to_string(any);
	}

	//inline std::string str(enum any) {
	//	return static_cast<int>(any);
	//}

	inline std::string str(std::string any) {
		return any;
	}
	inline std::string str(const char* any) {
		return std::string(any);
	}

	template<typename T>
	inline std::string str(std::vector<T> any) {
		std::string result = "{";
		for (T& s : any) {
			result += str(s);
		}
		result += "len:" + str(any.size()) + "}";
		return result;
	}
	//inline std::string str(const char** any) {
	//	return std::string(any);
	//}

	typedef unsigned int uint;
	using std::min;
	using std::max;

	BETTER_ENUM(STDriverType, int, point, line, face) // used in maya enum attr
	
	template<typename mapT>
	std::set<typename mapT::key_type> mapKeysToSet(mapT& m) {
		std::set<typename mapT::key_type>result;
		for (auto& p : m) {
			result.insert(p.first);
		}
		return result;
	}

	template<typename mapT>
	std::unordered_set<typename mapT::key_type> mapKeysToUSet(mapT& m) {
		std::unordered_set<typename mapT::key_type>result;
		for (auto& p : m) {
			result.insert(p.first);
		}
		return result;
	}

	template<typename T>
	bool anyIntersectionVectorUSet(std::vector<T> a, std::unordered_set<T> b) {
		for (auto i : a) {
			if (b.find(a) != b.end()) {
				return true;
			}
		}
		return false;
	}

	template<typename Key, typename Value>
	using Map = std::map<Key, Value>;

	template<typename Key, typename Value>
	using MapIterator = typename Map<Key, Value>::iterator;

	template<typename Key, typename Value>
	class MapKeyIterator : public MapIterator<Key, Value> {

	public:

		MapKeyIterator() : MapIterator<Key, Value>() {};
		MapKeyIterator(MapIterator<Key, Value> it_) : MapIterator<Key, Value>(it_) {};

		Key* operator -> () { return (Key* const)&(MapIterator<Key, Value>::operator -> ()->first); }
		Key operator * () { return MapIterator<Key, Value>::operator * ().first; }
	};

	template<typename Key, typename Value>
	class MapValueIterator : public MapIterator<Key, Value> {

	public:

		MapValueIterator() : MapIterator<Key, Value>() {};
		MapValueIterator(MapIterator<Key, Value> it_) : MapIterator<Key, Value>(it_) {};

		Value* operator -> () { return (Value* const)&(MapIterator<Key, Value>::operator -> ()->second); }
		Value operator * () { return MapIterator<Key, Value>::operator * ().second; }
	};

	/* handy way to work more easily with vertex buffer memory - cast
	it to vector of float types like this */
	struct Float2
	{
		Float2() { x = 0.0f; y = 0.0f; }
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
		//Float3(Eigen::Matrix<float, 3, 1, 0, 3, 1>&& v)
		//	: x(static_cast<float>(v[0])), y(static_cast<float>(v.data()[1])), z(static_cast<float>(v.data()[2])) {
		//}
		Float3(Eigen::Matrix<double, 3, 1, 0, 3, 1> v)
			: x(static_cast<float>(v.data()[0])), y(static_cast<float>(v.data()[1])), z(static_cast<float>(v.data()[2])) {}
		Float3(Eigen::Array<double, 3, 1, 0, 3, 1> v)
			: x(static_cast<float>(v.data()[0])), y(static_cast<float>(v.data()[1])), z(static_cast<float>(v.data()[2])) {}
		float x = 0.0f;
		float y = 0.0f;
		float z = 0.0f;
		Float3& operator=(const float*& v) {
			x = v[0];
			y = v[1];
			z = v[2];
			return *this;
		}

		Float3& operator=(const Eigen::Vector3<float>& v) {
			x = v[0];
			y = v[1];
			z = v[2];
			return *this;
		}
	};

	struct Float4
	{
		Float4() : x(0), y(0), z(0), w(0) {}
		Float4(float x, float y, float z)
			: x(static_cast<float>(x)), y(static_cast<float>(y)), z(static_cast<float>(z)), w(0.0f) {
		}
		Float4(float x, float y, float z, float w)
			: x(static_cast<float>(x)), y(static_cast<float>(y)), z(static_cast<float>(z)), w(static_cast<float>(w)) {
		}
		Float4(MVector v)
			: x(static_cast<float>(v[0])), y(static_cast<float>(v[1])), z(static_cast<float>(v[2])) {
		}
		Float4(MPoint v)
			: x(static_cast<float>(v[0])), y(static_cast<float>(v[1])), z(static_cast<float>(v[2])) {
		}
		Float4(float* v)
			: x(static_cast<float>(v[0])), y(static_cast<float>(v[1])), z(static_cast<float>(v[2])) {
		}
		Float4(const float* v)
			: x(static_cast<float>(v[0])), y(static_cast<float>(v[1])), z(static_cast<float>(v[2])) {
		}
		Float4(double* v)
			: x(static_cast<float>(v[0])), y(static_cast<float>(v[1])), z(static_cast<float>(v[2])) {
		}
		Float4(Eigen::Vector3f& v)
			: x(static_cast<float>(v.data()[0])), y(static_cast<float>(v.data()[1])), z(static_cast<float>(v.data()[2])) {
		}
		Float4(Eigen::Vector3f&& v)
			: x(static_cast<float>(v[0])), y(static_cast<float>(v.data()[1])), z(static_cast<float>(v.data()[2])) {
		}
		//Float4(Eigen::Matrix<float, 3, 1, 0, 3, 1>&& v)
		//	: x(static_cast<float>(v[0])), y(static_cast<float>(v.data()[1])), z(static_cast<float>(v.data()[2])) {
		//}
		Float4(Eigen::Matrix<double, 3, 1, 0, 3, 1> v)
			: x(static_cast<float>(v.data()[0])), y(static_cast<float>(v.data()[1])), z(static_cast<float>(v.data()[2])) {
		}
		Float4(Eigen::Array<double, 3, 1, 0, 3, 1> v)
			: x(static_cast<float>(v.data()[0])), y(static_cast<float>(v.data()[1])), z(static_cast<float>(v.data()[2])) {
		}
		float x = 0.0f;
		float y = 0.0f;
		float z = 0.0f;
		float w = 0.0f;
		Float4& operator=(const float*& v) {
			x = v[0];
			y = v[1];
			z = v[2];
			return *this;
		}

		Float4& operator=(const Eigen::Vector3<float>& v) {
			x = v[0];
			y = v[1];
			z = v[2];
			return *this;
		}
	};

	typedef std::vector<Float4>       Float4Array;
	typedef std::vector<Float3>       Float3Array;
	typedef std::vector<Float2>       Float2Array;
	typedef std::vector<unsigned int> IndexList;


	template<typename T>
	inline T uss(T in) {
		/* unsigned-to-signed conversion - 
		[0,1] -> [-1,1] */
		float a = 2.0;
		return (2.0 * in) - 1.0;
	}

	template<typename T>
	inline T sus(T in) {
		/* signed-to-unsigned conversion -
		[-1,1] -> [0,1] */
		return (in + T(1.0)) / T(2.0);
	}

	template<typename T>
	inline T lerp( T a, T b, T t ) {
		return b * t + (T(1.0) - t) * b;
	}

	template<typename T, typename N>
	inline T lerp( T a, T b, N t) {
		return b * t + (N(1.0f) - t) * b;
	}

	template Eigen::Vector3f lerp<Eigen::Vector3f, float>(Eigen::Vector3f a, Eigen::Vector3f b, float t);

	inline float lerp(float a, float b, float t) {
		return b * t + (float(1.0f) - t) * b;
	}

	inline Eigen::Vector3f lerp(Eigen::Vector3f a, Eigen::Vector3f b, float t) {
		return b * t + (float(1.0f) - t) * b;
	}

	/* test a way to pass in different modes of interpolation
	to more complex functions -  
	*/
	template<typename T, typename N=float>
	struct Lerp {
		static T fn(T a, T b, N t) { return lerp<T, T, N>(a, b, t); }
	};


	template <typename T>
	static inline T clamp(T low, T high, T x) {
		return std::min(high, std::max(low, x));
	}
	template <typename T>
	static inline T clamp01(T x) {
		return std::min(static_cast<T>(0.0), std::max(static_cast<T>(1.0), x));
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
	inline T sminExp(T a, T b, T k)
	{
		k *= 1.0;
		T r = exp2(-a / k) + exp2(-b / k);
		return -k * log2(r);
	}
	// root
	template<typename T>
	inline T sminRoot(T a, T b, T k)
	{
		k *= 2.0;
		T x = b - a;
		return 0.5 * (a + b - sqrt(x * x + k * k));
	}
	// sigmoid
	template<typename T>
	inline T sminSig(T a, T b, T k)
	{
		k *= log(2.0);
		T x = b - a;
		return a + x / (1.0 - exp2(x / k));
	}
	// quadratic polynomial
	template<typename T>
	inline T sminQ(T a, T b, T k)
	{
		k *= 4.0;
		T h = max(k - abs(a - b), T(0.0)) / k;
		return min(a, b) - h * h * k * (T(1.0) / T(4.0));
	}
	// cubic polynomial
	template<typename T>
	inline T sminC(T a, T b, T k)
	{
		k *= 6.0;
		T h = max(k - abs(a - b), T(0.0)) / k;
		return min(a, b) - h * h * h * k * T(1.0 / 6.0);
	}
	// quartic polynomial
	template<typename T>
	inline T sminQuart(T a, T b, T k)
	{
		k *= T(16.0 / 3.0);
		T h = max(k - abs(a - b), T(0.0)) / k;
		return min(a, b) - h * h * h * (T(4.0) - h) * k * T(1.0 / 16.0);
	}
	// circular
	template<typename T>
	inline T sminCirc(T a, T b, T k)
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


	/* smoothstep functions
	any appearance of competence I exhibit is a thin bin bag over
	the actual work of iq
	*/
	inline float smoothstepCubic(float x) // generally fine, cheap, c1, linear c2
	{
		return x * x * (3.0 - 2.0 * x);
	}
	inline float inv_smoothstepCubic(float x)
	{
		return 0.5 - sin(asin(1.0 - 2.0 * x) / 3.0);
	}

	

	inline float smoothstepQuarticPolynomial(float x)
	{
		return x * x * (2.0 - x * x);
	}
	inline float inv_smoothstepQuarticPolynomial(float x)
	{
		return sqrt(1.0 - sqrt(1.0 - x));
	}

	inline float smoothstepQuinticPolynomial(float x)
	{
		return x * x * x * (x * (x * 6.0 - 15.0) + 10.0);
	}

	inline float smoothstepQuadraticRational(float x)
	{ // slightly rounder than cubic, c1, sigmoid c2 
		return x * x / (2.0 * x * x - 2.0 * x + 1.0);
	}
	inline float inv_smoothstepQuadraticRational(float x)
	{
		return (x - sqrt(x * (1.0 - x))) / (2.0 * x - 1.0);
	}

	inline float smoothstepCubicRational(float x)
	{	/* smooth fillet, distinct straight regions in sigmoid,
		c2 continuous*/
		return x * x * x / (3.0 * x * x - 3.0 * x + 1.0);
	}
	inline float inv_smoothstepCubicRational(float x)
	{		float a = pow(x, 1.0 / 3.0);
		float b = pow(1.0 - x, 1.0 / 3.0);
		return a / (a + b);
	}


	inline float smoothstepRational(float x, float n)
	{		return pow(x, n) / (pow(x, n) + pow(1.0 - x, n));
	}
	inline float inv_smoothstepRational(float x, float n)
	{		return smoothstepRational(x, 1.0 / n);
	}

	inline float smoothstepPiecewiseQuadratic(float x)
	{
		return (x < 0.5) ?
			2.0 * x * x :
			2.0 * x * (2.0 - x) - 1.0;
	}
	inline float inv_smoothstepPiecewiseQuadratic(float x)
	{
		return (x < 0.5) ?
			sqrt(0.5 * x) :
			1.0 - sqrt(0.5 - 0.5 * x);
	}

	inline float smoothstepPiecewisePolynomial(float x, float n)
	{
		return (x < 0.5) ?
			0.5 * pow(2.0 * x, n) :
			1.0 - 0.5 * pow(2.0 * (1.0 - x), n);
	}
	inline float inv_smoothstepPiecewisePolynomial(float x, float n)
	{
		return (x < 0.5) ?
			0.5 * pow(2.0 * x, 1.0 / n) :
			1.0 - 0.5 * pow(2.0 * (1.0 - x), 1.0 / n);
	}

	inline float smoothstepTrigonometric(float x)
	{
		return 0.5 - 0.5 * cos(PI * x);
	}
	inline float inv_smoothstepTrigonometric(float x)
	{
		return acos(1.0 - 2.0 * x) / PI;
	}



	/* FUNCTIONS TAKE QUATERNIONS OF FORM X-Y-Z-W

	*/

	template<typename T>
	inline T* quatTo4x4Mat(T* q, T* m);
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
	inline T* quatTo3x3Mat(T* quat, T* m);
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
	inline T ReciprocalSqrt(T x) {
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

	void interpolateMMatrixArray(std::vector<MMatrix>& mmatrixArr, MMatrix& out, float t);

	template <typename T=float>
	inline T lerpArray(const T* arr, const int nVals, float t) {
		
		t = clamp(0.0, 0.99999, t) * float(nVals);
		int lower = floor(t);
		return lerp(
			arr[lower],
			arr[lower + 1],
			t - float(nVals)
		);
	}

	//template <typename T = float>
	//T lerpArray(const T* arr, const int nVals, float t) {

	//	t = clamp(0.0, 0.99999, t) * float(nVals);
	//	int lower = floor(t);
	//	return lerp(
	//		arr[lower],
	//		arr[lower + 1],
	//		t - float(nVals)
	//	);
	//}


	/*
	curves - do we cache arrays of densely sampled values?
	if yes, we have to either save many values, 
	OR resample those values in a cubic spline built on the fly
	so we still pay cubic cost anyway
	*/

}

#pragma warning( pop )