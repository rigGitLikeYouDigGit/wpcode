#pragma once

#include <vector>
#include <math.h>
#include <set>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <map>
#include "macro.h"
#include "api.h"


//#include "bezier/bezier.h"


/* if C warns me about losing precision adding 2.0 to a float,
* I don't care about it
*/

#pragma warning(push)
#pragma warning( disable : 4244)
namespace strata {




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
	bool anyIntersectionVectorUSet(std::vector<T>& a, std::unordered_set<T>& b) {
		for (auto i : a) {
			if (b.find(a) != b.end()) {
				return true;
			}
		}
		return false;
	}

	template<typename VT>
	int vectorIndex(std::vector<VT>& vec, VT& v) {
		auto found = std::find(vec.begin(), vec.end(), v);
		if (found == vec.end()) {
			return -1;
		}
		return static_cast<int>(std::distance(vec.begin(), found));
	}

	inline int iNext(int orig, int n) {
		return (orig + n + 1) % n;
	}
	
	inline int iPrev(int orig, int n) {
		return (orig + n - 1) % n;
	}

	/* predicate object to pass to std::sort, to bundle a
	sortable value in a tuple of attached values
	*/
	template <int kIndex = 0>
	struct TupleSorter {

		template<typename tupleT>
		bool operator()(const tupleT& left, const tupleT& right) {
			return std::get<kIndex>(left) < std::get<kIndex>(right);
			//return left.second < right.second;
		}
	};

	template <typename containerT, int kIndex = 0>
	static void sortTupleContainer(containerT& cont) {
		std::sort(cont.begin(), cont.end(), TupleSorter<kIndex>());
	}

	template <typename kT, typename containerT, int kIndex = 0>
	static int indexForTupleValue(containerT& container, kT& key) {
		/* for a container with tuple values, check for the given key
		value - return the first index that matches, or -1

		for small sequences this is fine
		*/
		int i = 0;
		for (auto& item : container) {
			if (std::get<kIndex>(item) == key) {
				return i;
			}
			i++;
		}
		return -1;
	}

	/**
	 * Argsort(currently support ascending sort)
	 * @tparam T array element type
	 * @param array input array
	 * @return indices w.r.t sorted array
	 */
	template<typename T>
	std::vector<size_t> argsort(const std::vector<T>& array) {
		std::vector<size_t> indices(array.size());
		std::iota(indices.begin(), indices.end(), 0);
		std::sort(indices.begin(), indices.end(),
			[&array](int left, int right) -> bool {
			// sort indices according to corresponding array element
			return array[left] < array[right];
		});

		return indices;
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

	template<typename VT>
	struct Span {
		VT* ptr = nullptr;
		int l = 0;
	};

	template< typename VT >
	struct OffsetBuffer {
		/* keep array of index offsets into flat buffer, for storing short
		* but irregular lists
		* TODO: maybe try and keep multiple kinds of value buffer, accept another buffer for
		* indices etc
		*/
		std::vector<int> offsets;
		std::vector<VT> values;

		inline int entryLength(int index) {
			if (index == 0) {
				return offsets[1];
			}
			return offsets[index] - offsets[index - 1];
		}

		/* NO SUPPORT FOR CHANGING LENGTHS AFTER SETTING, can't ripple-update the entire buffer*/
		inline void set(int index, VT* values_, int length){
			for (int i = 0; i < length; i++) {
				values[i] = values_[i];
			}
			if (index) {
				offsets[index] = length + offsets[index - 1];
			}
			else {
				offsets[index] = length;
			}
		}

		inline void reserve(int nEntries, int valueSize = 4) {
			offsets.reserve(nEntries);
			values.reserve(nEntries * valueSize);
		}

		inline void append(int index, VT* values_, int length) {
			for (int i = 0; i < length; i++) {
				values.push_back(values_[i]);
			}
			
			if (index) {
				offsets.push_back(length + offsets[index - 1]);
			}
			else {
				offsets.push_back(length);
			}
		}
		inline Span<VT> entry(int index) {
			int offset = 0;
			if (index) {
				offset = offsets[index - 1];
			}
			return Span<VT>{
				values.data() + (offset),
					offset - offsets[index]
			};
		}

	};
	/* TODO: investigate if it's better to store separate vectors or spans within a vector for differently
	sized entries? 
	so group all len-3s together, then 4s etc.
	but then your offsets still need to store start-end integers
	pointless ignore me
	*/

	//template<typename T>
	inline int divNearestInt(int val, int divisor) {
		int c = (int)val / divisor;
		int d = val % divisor;
		if (d >= (divisor / 2)) {
			return c + 1;
		}
		return c;
	}
	inline int divNearestInt(float val, float divisor) {
		int c = (int)val / (int)divisor;
		float d = val - c * divisor;
		if (d > (0.5f * divisor)) {
			return c + 1;
		}
		return c;
	}

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
	static inline T clamp(T x, T low, T high) {
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
	inline T mapTo01(T in, T inMin, T inMax, bool doClamp) {
		return clamp((in - inMin) / (inMax - inMin), inMin, inMax);
	}

	template<typename T>
	inline T remap(T in, T inMin, T inMax, T outMin, T outMax) {
		return lerp(mapTo01(in, inMax, inMin), outMin, outMax);
	}

	template<typename T>
	inline T remap(T in, T inMin, T inMax, T outMin, T outMax, bool doClamp) {
		return clamp(
			lerp( 
				mapTo01(in, inMax, inMin, true), 
				outMin, 
				outMax ), 
			outMin, 
			outMax);
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

	int factorial(const int n)
	{
		int f = 1;
		for (int i = 1; i <= n; ++i)
			f *= i;
		return f;
	}

	float binomialCoefficient(int n, int v) {
		/* 
		in notation usually
		(	n	)
		(		)
		(	k	)
		*/
		return float(factorial(n)) / 
			float(factorial(v) * factorial(n - v));
	}

	float bernsteinBasis(int n, int v, float x) {
		/*
		see definition at https://en.wikipedia.org/wiki/Bernstein_polynomial

		*/
		return binomialCoefficient(n, v) * pow(x, v) * pow((1.0f - x), n - v);
	}

}

#pragma warning( pop )