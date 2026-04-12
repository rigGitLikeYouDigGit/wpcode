#pragma once

#include <vector>
#include <utility>
#include <tuple>
#include <map>
#include <unordered_map>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include "MInclude.h"

#include <unsupported/Eigen/Splines>

#include "api.h"
#include "macro.h"
#include "status.h"
#include "lib.h"
#include "AABB.h"

namespace bez {
	class CubicBezierSpline;
	struct CubicBezierPath;
}

namespace strata {
	/* copying various from Free Electron,
	adapting to Eigen types

	nomenclature:
	U - GLOBAL coordinate
	t - LOCAL coordinate, usually only between 2 discrete values for a lerp
	*/

	using namespace Eigen;

	/* 
jank utils for working with float values as map keys,
for looking up intersections by UVN coords*/
	constexpr float E = 0.00001f;
	// returns massive int keys - maybe fine
	inline int toKey(float k) {
		//return trunc(k / E);
		return static_cast<int>(trunc(k * 100000.0f));
	}
	inline int toKey(double k) {
		return static_cast<int>(trunc(k * 100000.0f));
	}

	inline Vector3i toKey(Vector3f k) {
		return Vector3i(toKey(k.x()), toKey(k.y()), toKey(k.z()));
	}

	struct Vector3iCompare
	{ /* to use int vectors as janky map keys */
		bool operator()(const Vector3i& lhs, const Vector3i& rhs) const {
			return std::lexicographical_compare(
				lhs.data(), lhs.data() + 3,
				rhs.data(), rhs.data() + 3
			);
		}
		bool operator() (const Vector3f& lhs, const Vector3i& rhs) const
		{
			Vector3i iLhs = toKey(lhs);
			return std::lexicographical_compare(
				iLhs.data(), iLhs.data() + 3,
				rhs.data(), rhs.data() + 3
			);
		}
		bool operator() (const Vector3i& lhs, const Vector3f& rhs) const
		{
			Vector3i iRhs = toKey(rhs);
			return std::lexicographical_compare(
				lhs.data(), lhs.data() + 3,
				iRhs.data(), iRhs.data() + 3
			);
		}
		bool operator() (const Vector3f& lhs, const Vector3f& rhs) const
		{
			Vector3i iLhs = toKey(lhs);
			Vector3i iRhs = toKey(rhs);
			return std::lexicographical_compare(
				iLhs.data(), iLhs.data() + 3,
				iRhs.data(), iRhs.data() + 3
			);
		}
	};
	
	template <typename V>
	using Vector3iMap = std::map<Vector3i, V, Vector3iCompare>;
	template <typename V>
	using Vector3iUMap = std::unordered_map<Vector3i, V, Vector3iCompare>;

	/* eigen to string functions - a little spaghett*/
	inline std::string str(const Vector3f& any) {
		std::stringstream ss;
		ss << any;
		return ss.str();
	}
	inline std::string str(const Affine3f& any) {
		std::stringstream ss;
		ss << any.matrix();
		return ss.str();
	}

/* sketch for a way of testing tradeoffs between caching vs computing - 
* Not sure if it's viable for more complex things.
need to be careful nothing modifies _dir if cached
*/
#define ST_RAY_CACHE_DIR 1
	struct Ray {
		Vector3f o;
		Vector3f span;
#if ST_RAY_CACHE_DIR
		Vector3f _dir;
		//inline Vector3f dir() {
		//	return _dir;
		//}
		inline Vector3f& dir() {
			return _dir;
		}
#else
		inline Vector3f dir() {
			return span.normalized();
		}
#endif

	};

	struct SampleIndices {
    Eigen::Index lower;  // Changed from int
    Eigen::Index upper;  // Changed from int
    float t;
    
    // Convenience: check if this is a valid sample
    bool isValid() const { return lower >= 0 && upper >= 0; }
	};

	/**
	 * @brief Get array indices and interpolation factor for global parameter U
	 *
	 * @param nEntries Total number of entries in array
	 * @param u Global parameter [0,1] (0 = start, 1 = end)
	 * @return SampleIndices containing {lower, upper, t}
	 *
	 * Examples:
	 *   u=0.0   -> {0, 0, 0.0}      (clamped at start)
	 *   u=0.25  -> {1, 2, 0.0}      (for nEntries=10)
	 *   u=1.0   -> {9, 9, 1.0}      (clamped at end)
	 */
	inline SampleIndices getIndicesTForU(int nEntries, float u) {
		// Clamp to [0, 1]
		if (u <= 0.0f) {
			return { 0, 0, 0.0f };
		}
		if (u >= 1.0f) {
			return { nEntries - 1, nEntries - 1, 1.0f };
		}

		// Map u to continuous index space [0, nEntries)
		float indexF = u * static_cast<float>(nEntries - 1);
		int lower = static_cast<int>(std::floor(indexF));
		int upper = lower + 1;

		// Clamp upper (defensive, shouldn't trigger if u < 1.0)
		if (upper >= nEntries) {
			upper = nEntries - 1;
		}

		// Local t between lower and upper
		float t = indexF - static_cast<float>(lower);

		return { lower, upper, t };
	}

	/**
	 * @brief Overload accepting output parameters (legacy compatibility)
	 */
	inline float getIndicesTForU(int nEntries, float u, int& lower, int& upper) {
		SampleIndices idx = getIndicesTForU(nEntries, u);
		lower = int(idx.lower);
		upper = int(idx.upper);
		return idx.t;
	}
	template<typename Derived>
	inline SampleIndices getIndicesTForU(
		const Eigen::DenseBase<Derived>& entryCoords,
		float u
	) {
		auto nEntries = entryCoords.size();

		// Clamp to array bounds
		if (u <= entryCoords.coeff(0)) {
			return { 0, 0, 0.0f };
		}
		if (u >= entryCoords.coeff(nEntries - 1)) {
			return { nEntries - 1, nEntries - 1, 1.0f };
		}

		// Binary search for bracketing indices
		Eigen::Index lower = 0;
		Eigen::Index upper = nEntries - 1;

		while (upper - lower > 1) {
			Eigen::Index mid = (lower + upper) / 2;
			if (entryCoords.coeff(mid) <= u) {
				lower = mid;
			}
			else {
				upper = mid;
			}
		}

		// Compute local interpolation factor
		float coordLower = entryCoords.coeff(lower);
		float coordUpper = entryCoords.coeff(upper);
		float t = (u - coordLower) / (coordUpper - coordLower);

		return { lower, upper, t };
	}

	template<typename Derived>
	inline typename Derived::Scalar sampleArray(
		const SampleIndices& idx,
		const Eigen::DenseBase<Derived>& arr
	) {
		static_assert(Derived::ColsAtCompileTime == 1, "Use multi-column overload");

		using Scalar = typename Derived::Scalar;
		return arr.coeff(idx.lower) * (Scalar(1) - idx.t) +
			arr.coeff(idx.upper) * idx.t;
	}

	/**
	 * @brief Sample multi-column array using pre-computed indices
	 *
	 * @param idx Pre-computed sample indices and interpolation factor
	 * @param arr Input matrix (rows = samples, cols = dimensions)
	 * @return Interpolated row vector
	 */
	template<typename Derived>
	inline auto sampleArray(
		const SampleIndices& idx,
		const Eigen::DenseBase<Derived>& arr
	) -> typename std::enable_if<
		Derived::ColsAtCompileTime != 1,
		typename Derived::RowXpr
	>::type {
		return arr.row(idx.lower) * (1.0f - idx.t) +
			arr.row(idx.upper) * idx.t;
	}

	/**
	 * @brief Smoothly sample 1D array using pre-computed indices
	 *
	 * @param idx Pre-computed sample indices and interpolation factor
	 * @param arr Input array
	 * @return Smoothly interpolated scalar value (cubic Hermite)
	 */
	template<typename Derived>
	inline typename Derived::Scalar sampleArraySmooth(
		const SampleIndices& idx,
		const Eigen::DenseBase<Derived>& arr
	) {
		static_assert(Derived::ColsAtCompileTime == 1, "Use multi-column overload");

		using Scalar = typename Derived::Scalar;
		float tSmooth = idx.t * idx.t * (3.0f - 2.0f * idx.t);

		return arr.coeff(idx.lower) * (Scalar(1) - tSmooth) +
			arr.coeff(idx.upper) * tSmooth;
	}

	/**
	 * @brief Smoothly sample multi-column array using pre-computed indices
	 *
	 * @param idx Pre-computed sample indices and interpolation factor
	 * @param arr Input matrix
	 * @return Smoothly interpolated row vector (cubic Hermite)
	 */
	template<typename Derived>
	inline auto sampleArraySmooth(
		const SampleIndices& idx,
		const Eigen::DenseBase<Derived>& arr
	) -> typename std::enable_if<
		Derived::ColsAtCompileTime != 1,
		typename Derived::RowXpr
	>::type {
		float tSmooth = idx.t * idx.t * (3.0f - 2.0f * idx.t);

		return arr.row(idx.lower) * (1.0f - tSmooth) +
			arr.row(idx.upper) * tSmooth;
	}


	// ============================================================================
// Convenience overloads: compute indices inline
// ============================================================================

/**
 * @brief Sample array at U parameter (linear interpolation)
 */
	template<typename Derived>
	inline auto sampleArrayAtU(const Eigen::DenseBase<Derived>& arr, float u) {
		return sampleArray(getIndicesTForU(arr.rows(), u), arr);
	}

	/**
	 * @brief Sample array at U parameter (smooth interpolation)
	 */
	template<typename Derived>
	inline auto sampleArrayAtUSmooth(const Eigen::DenseBase<Derived>& arr, float u) {
		return sampleArraySmooth(getIndicesTForU(arr.rows(), u), arr);
	}

	/**
	 * @brief Sample array with non-uniform coordinates (linear)
	 */
	template<typename DerivedCoords, typename DerivedData>
	inline auto sampleArrayAtU(
		const Eigen::DenseBase<DerivedCoords>& entryCoords,
		const Eigen::DenseBase<DerivedData>& arr,
		float u
	) {
		return sampleArray(getIndicesTForU(entryCoords, u), arr);
	}

	/**
	 * @brief Sample array with non-uniform coordinates (smooth)
	 */
	template<typename DerivedCoords, typename DerivedData>
	inline auto sampleArrayAtUSmooth(
		const Eigen::DenseBase<DerivedCoords>& entryCoords,
		const Eigen::DenseBase<DerivedData>& arr,
		float u
	) {
		return sampleArraySmooth(getIndicesTForU(entryCoords, u), arr);
	}

	// ============================================================================
	// Broadcasting: array of SampleIndices
	// ============================================================================

	/**
	 * @brief Sample at multiple pre-computed indices (linear)
	 *
	 * @param indices Array of pre-computed sample indices
	 * @param arr Data array to sample
	 * @return Array of interpolated values
	 *
	 * Example:
	 *   std::vector<SampleIndices> indices;
	 *   for (float u : uParams) indices.push_back(getIndicesTForU(100, u));
	 *   ArrayXf results = sampleArray(indices, data);
	 */
	template<typename Derived>
	inline auto sampleArray(
		const std::vector<SampleIndices>& indices,
		const Eigen::DenseBase<Derived>& arr
	) {
		using Scalar = typename Derived::Scalar;
		int nSamples = indices.size();

		if constexpr (Derived::ColsAtCompileTime == 1) {
			Eigen::Array<Scalar, Eigen::Dynamic, 1> result(nSamples);
			for (int i = 0; i < nSamples; ++i) {
				result[i] = sampleArray(indices[i], arr);
			}
			return result;
		}
		else {
			constexpr int Cols = Derived::ColsAtCompileTime;
			int nCols = arr.cols();
			Eigen::Array<Scalar, Eigen::Dynamic, Cols> result(nSamples, nCols);
			for (int i = 0; i < nSamples; ++i) {
				result.row(i) = sampleArray(indices[i], arr);
			}
			return result;
		}
	}

	/**
	 * @brief Sample at multiple pre-computed indices (smooth)
	 */
	template<typename Derived>
	inline auto sampleArraySmooth(
		const std::vector<SampleIndices>& indices,
		const Eigen::DenseBase<Derived>& arr
	) {
		using Scalar = typename Derived::Scalar;
		int nSamples = indices.size();

		if constexpr (Derived::ColsAtCompileTime == 1) {
			Eigen::Array<Scalar, Eigen::Dynamic, 1> result(nSamples);
			for (int i = 0; i < nSamples; ++i) {
				result[i] = sampleArraySmooth(indices[i], arr);
			}
			return result;
		}
		else {
			constexpr int Cols = Derived::ColsAtCompileTime;
			int nCols = arr.cols();
			Eigen::Array<Scalar, Eigen::Dynamic, Cols> result(nSamples, nCols);
			for (int i = 0; i < nSamples; ++i) {
				result.row(i) = sampleArraySmooth(indices[i], arr);
			}
			return result;
		}
	}

	// ============================================================================
	// Broadcasting: array of U parameters (delegates to indices version)
	// ============================================================================

	/**
	 * @brief Sample at multiple U parameters (linear, uniform spacing)
	 */
	template<typename Derived, typename UDerived>
	inline auto sampleArrayAtU(
		const Eigen::DenseBase<Derived>& arr,
		const Eigen::DenseBase<UDerived>& uParams
	) {
		std::vector<SampleIndices> indices;
		indices.reserve(uParams.size());
		for (int i = 0; i < uParams.size(); ++i) {
			indices.push_back(getIndicesTForU(arr.rows(), uParams.coeff(i)));
		}
		return sampleArray(indices, arr);
	}

	/**
	 * @brief Sample at multiple U parameters (smooth, uniform spacing)
	 */
	template<typename Derived, typename UDerived>
	inline auto sampleArrayAtUSmooth(
		const Eigen::DenseBase<Derived>& arr,
		const Eigen::DenseBase<UDerived>& uParams
	) {
		std::vector<SampleIndices> indices;
		indices.reserve(uParams.size());
		for (int i = 0; i < uParams.size(); ++i) {
			indices.push_back(getIndicesTForU(arr.rows(), uParams.coeff(i)));
		}
		return sampleArraySmooth(indices, arr);
	}

	/**
	 * @brief Sample at multiple U parameters (linear, non-uniform spacing)
	 */
	template<typename DerivedCoords, typename DerivedData, typename DerivedU>
	inline auto sampleArrayAtU(
		const Eigen::DenseBase<DerivedCoords>& entryCoords,
		const Eigen::DenseBase<DerivedData>& arr,
		const Eigen::DenseBase<DerivedU>& uParams
	) {
		std::vector<SampleIndices> indices;
		indices.reserve(uParams.size());
		for (int i = 0; i < uParams.size(); ++i) {
			indices.push_back(getIndicesTForU(entryCoords, uParams.coeff(i)));
		}
		return sampleArray(indices, arr);
	}

	/**
	 * @brief Sample at multiple U parameters (smooth, non-uniform spacing)
	 */
	template<typename DerivedCoords, typename DerivedData, typename DerivedU>
	inline auto sampleArrayAtUSmooth(
		const Eigen::DenseBase<DerivedCoords>& entryCoords,
		const Eigen::DenseBase<DerivedData>& arr,
		const Eigen::DenseBase<DerivedU>& uParams
	) {
		std::vector<SampleIndices> indices;
		indices.reserve(uParams.size());
		for (int i = 0; i < uParams.size(); ++i) {
			indices.push_back(getIndicesTForU(entryCoords, uParams.coeff(i)));
		}
		return sampleArraySmooth(indices, arr);
	}
	// ============================================================================
	// Single-value sampling (scalar U)
	// ============================================================================

	/**
	 * @brief Sample 1D array (VectorXf, ArrayXf) at parameter U
	 *
	 * @tparam Derived Eigen expression type (deduced)
	 * @param arr Input array (VectorXf, ArrayXf, etc.)
	 * @param u Global parameter [0,1]
	 * @return Interpolated scalar value
	 *
	 * Example:
	 *   ArrayXf weights(100);
	 *   float w = sampleArrayAtU(weights, 0.5f);  // Sample at midpoint
	 */
	//template<typename Derived>
	//inline auto sampleArrayAtU(const Eigen::DenseBase<Derived>& arr, float u) {
	//	SampleIndices idx = getIndicesTForU(arr.rows(), u);

	//	if constexpr (Derived::ColsAtCompileTime == 1) {
	//		// 1D case
	//		return arr.coeff(idx.lower) * (1.0f - idx.t) +
	//			arr.coeff(idx.upper) * idx.t;
	//	}
	//	else {
	//		// Multi-column case
	//		return arr.row(idx.lower) * (1.0f - idx.t) +
	//			arr.row(idx.upper) * idx.t;
	//	}
	//}

	/**
	 * @brief Sample multi-column array (MatrixXf, ArrayXXf) at parameter U
	 *
	 * @tparam Derived Eigen expression type (deduced)
	 * @param arr Input matrix (rows = samples, cols = dimensions)
	 * @param u Global parameter [0,1]
	 * @return Interpolated row vector (fixed-size or dynamic)
	 *
	 * Example:
	 *   ArrayX3f positions(100, 3);
	 *   Vector3f pos = sampleArrayAtU(positions, 0.75f);  // Sample at 75%
	 */
	template<typename Derived>
	inline auto sampleArrayAtU(
		const Eigen::DenseBase<Derived>& arr,
		float u
	) -> typename std::enable_if<
		Derived::ColsAtCompileTime != 1,  // Multi-column
		typename Derived::RowXpr
	>::type {
		SampleIndices idx = getIndicesTForU(arr.rows(), u);

		// Lerp between lower and upper rows
		auto lower_row = arr.row(idx.lower);
		auto upper_row = arr.row(idx.upper);

		return lower_row * (1.0f - idx.t) + upper_row * idx.t;
	}

	// ============================================================================
	// Broadcasting: array of U parameters
	// ============================================================================

	/**
	 * @brief Sample 1D array at multiple U parameters (broadcast)
	 *
	 * @tparam Derived Input array type
	 * @tparam UDerived U parameter array type
	 * @param arr Input data array (VectorXf, ArrayXf)
	 * @param uParams Array of U parameters [0,1]
	 * @return Array of interpolated values (same size as uParams)
	 *
	 * Example:
	 *   ArrayXf data(100);
	 *   ArrayXf uSamples = ArrayXf::LinSpaced(50, 0.0f, 1.0f);
	 *   ArrayXf results = sampleArrayAtU(data, uSamples);  // 50 samples
	 */
	template<typename Derived, typename UDerived>
	inline Eigen::Array<typename Derived::Scalar, Eigen::Dynamic, 1>
		sampleArrayAtU(
			const Eigen::DenseBase<Derived>& arr,
			const Eigen::DenseBase<UDerived>& uParams
		) {
		using Scalar = typename Derived::Scalar;
		int nSamples = uParams.size();

		Eigen::Array<Scalar, Eigen::Dynamic, 1> result(nSamples);

		for (int i = 0; i < nSamples; ++i) {
			result[i] = sampleArrayAtU(arr, uParams.coeff(i));
		}

		return result;
	}

	/**
	 * @brief Sample multi-column array at multiple U parameters (broadcast)
	 *
	 * @tparam Derived Input matrix type
	 * @tparam UDerived U parameter array type
	 * @param arr Input data matrix (MatrixXf, ArrayXXf)
	 * @param uParams Array of U parameters [0,1]
	 * @return Matrix of interpolated rows (nSamples × nCols)
	 *
	 * Example:
	 *   ArrayX3f positions(100, 3);
	 *   ArrayXf uSamples = ArrayXf::LinSpaced(20, 0.0f, 1.0f);
	 *   ArrayX3f sampled = sampleArrayAtU(positions, uSamples);  // (20, 3)
	 */
	template<typename Derived, typename UDerived>
	inline auto sampleArrayAtU(
		const Eigen::DenseBase<Derived>& arr,
		const Eigen::DenseBase<UDerived>& uParams
	) -> typename std::enable_if<
		Derived::ColsAtCompileTime != 1,  // Multi-column
		Eigen::Array<typename Derived::Scalar, Eigen::Dynamic, Derived::ColsAtCompileTime>
	>::type {
		using Scalar = typename Derived::Scalar;
		constexpr int Cols = Derived::ColsAtCompileTime;

		int nSamples = uParams.size();
		int nCols = arr.cols();

		Eigen::Array<Scalar, Eigen::Dynamic, Cols> result(nSamples, nCols);

		for (int i = 0; i < nSamples; ++i) {
			result.row(i) = sampleArrayAtU(arr, uParams.coeff(i));
		}

		return result;
	}

	// ============================================================================
	// Smooth sampling (cubic Hermite interpolation)
	// ============================================================================

	/**
	 * @brief Smoothly sample array using cubic Hermite spline (C1 continuous)
	 *
	 * @param arr Input array
	 * @param u Global parameter [0,1]
	 * @return Smoothly interpolated value
	 *
	 * Uses smoothstep (3t² - 2t³) for natural cubic interpolation
	 */
	template<typename Derived>
	inline auto sampleArrayAtUSmooth(
		const Eigen::DenseBase<Derived>& arr,
		float u
	) -> decltype(sampleArrayAtU(arr, u)) {
		SampleIndices idx = getIndicesTForU(arr.rows(), u);

		// Cubic Hermite blend
		float tSmooth = idx.t * idx.t * (3.0f - 2.0f * idx.t);

		auto lower = (Derived::ColsAtCompileTime == 1)
			? arr.coeff(idx.lower)
			: arr.row(idx.lower);
		auto upper = (Derived::ColsAtCompileTime == 1)
			? arr.coeff(idx.upper)
			: arr.row(idx.upper);

		return lower * (1.0f - tSmooth) + upper * tSmooth;
	}

	/**
	 * @brief Smooth sampling with broadcast support
	 */
	template<typename Derived, typename UDerived>
	inline auto sampleArrayAtUSmooth(
		const Eigen::DenseBase<Derived>& arr,
		const Eigen::DenseBase<UDerived>& uParams
	) -> decltype(sampleArrayAtU(arr, uParams)) {
		int nSamples = uParams.size();
		auto result = sampleArrayAtU(arr, uParams);  // Get correct result type

		for (int i = 0; i < nSamples; ++i) {
			result.row(i) = sampleArrayAtUSmooth(arr, uParams.coeff(i));
		}

		return result;
	}

	/* example
	// ============================================================================
// Example 1: Single scalar sample
// ============================================================================
ArrayXf weights(100);
weights.setLinSpaced(0.0f, 10.0f);

float w = sampleArrayAtU(weights, 0.5f);  // Sample at midpoint
// Result: ~5.0

// ============================================================================
// Example 2: Single vector sample
// ============================================================================
ArrayX3f positions(100, 3);
// ... populate ...

Vector3f pos = sampleArrayAtU(positions, 0.25f);  // Sample at 25%
// Returns: interpolated (x, y, z)

// ============================================================================
// Example 3: Broadcasting (numpy-style)
// ============================================================================
ArrayX3f edgePositions(100, 3);  // Dense edge samples
ArrayXf uParams = ArrayXf::LinSpaced(20, 0.0f, 1.0f);  // 20 query points

ArrayX3f sampled = sampleArrayAtU(edgePositions, uParams);
// Result: (20, 3) array of interpolated positions

// ============================================================================
// Example 4: Smooth sampling (for normals, avoiding faceting)
// ============================================================================
ArrayX3f normals(100, 3);
Vector3f smoothNormal = sampleArrayAtUSmooth(normals, 0.5f);
// Uses cubic Hermite instead of linear lerp

// ============================================================================
// Example 5: Your original resampleVectorArray, simplified
// ============================================================================
template<typename T>
inline Eigen::ArrayX3<T> resampleVectorArray(
    const Eigen::MatrixX3<T>& inVs,
    float start, float end,
    int nSamples,
    bool normalise = true
) {
    // Map [start, end] to [0, 1] U-space
    ArrayXf uParams = ArrayXf::LinSpaced(nSamples, start, end);
    
    // Broadcast sample
    ArrayX3f result = sampleArrayAtU(inVs, uParams);
    
    // Normalize if requested
    if (normalise) {
        result = result.rowwise().normalized();
    }
    
    return result;
}
	
	*/




	template<class T>
	//inline void Quaternion<T>::computeAngleAxis(T& radians, Vector<3, T>& axis) const
	inline void computeAngleAxis(Eigen::Quaternion<T>& q, T& radians, Eigen::Vector3<T>& axis)
	{
		//T len = (*this)[0] * (*this)[0] + (*this)[1] * (*this)[1] + (*this)[2] * (*this)[2];
		T len = q[0] * q[0] + q[1] * q[1] + q[2] * q[2];
		if (len == T(0))
		{
			//set(axis, T(0), T(0), T(1));
			axis[0] = T(0);
			axis[1] = T(0);
			axis[2] = T(1);
			radians = T(0);
			return;
		}

		T inv = T(1) / sqrt(len);
		if (q[3] < T(0))
			inv = -inv;

		//set(axis, (*this)[0] * inv, (*this)[1] * inv, (*this)[2] * inv);
		axis[0] = q[0] * inv;
		axis[1] = q[1] * inv;
		axis[2] = q[2] * inv;

		radians = 2.0f * acos(fabs(q[3]));
		if (WP_INVALID_SCALAR(radians))
		{
			radians = T(0);
		}
	}

	template<typename T, typename U>
	inline bool isZero(T a_value, U a_tolerance)
	{
		return (fabs(a_value) > a_tolerance);
		{
			return false;
		}
		return true;
	}

	template <typename T>
	inline bool isZeroV3(const Eigen::Vector3<T>& a_vec, T a_tolerance)
	{
		if (!isZero(a_vec[0], a_tolerance) || !isZero(a_vec[1], a_tolerance) ||
			!isZero(a_vec[2], a_tolerance))
		{
			return false;
		}
		return true;
	}
	
	template<typename T>
	inline bool isZero(T a_value)
	{
		return isZero(a_value, EPS);
	}

	//inline Eigen::AngleAxisd toAngleAxis(const Eigen::Quaterniond& q) {
	//	double theta = acos(q.w()) * 2;
	//	Eigen::AngleAxisd result;
	//	result

	//}

	template<class T>
	//inline Quaternion<T> angularlyScaled(const Quaternion<T>& lhs, const U& rhs)
	inline Eigen::Quaternion<T> angularlyScaled(const Eigen::Quaternion<T>& lhs, const T& rhs)
	{
		T angle;
		Vector<3, T> axis;
		computeAngleAxis(lhs, angle, axis);

		//return Quaternion<T>(angle * rhs, axis);
		
		return Eigen::Quaternion<T>(angle * rhs, axis[0], axis[1], axis[2]);
	}

	MMatrixArray curveMatricesFromAnchorDatas(
		MMatrixArray controlMats, int segmentPointCount,
		int rootIterations
	);

	MPointArray curvePointsFromEditPoints(
		MMatrixArray controlMats, int segmentPointCount
		//int rootIterations
	);

	MPointArray curvePointsFromEditPointsAndTangents(
		MMatrixArray controlMats, int segmentPointCount
		//int rootIterations
	);


	template <typename D=float>
	inline auto pointLineDistanceSquared(
		const Eigen::Vector3<D>& pt1,
		const Eigen::Vector3<D>& pt2,
		const Eigen::Vector3<D>& sample) {
		return ((pt2 - pt1).cross(pt2 - sample)).squaredNorm() / (pt2 - pt1).squaredNorm();
	}

	template <typename D = float>
	inline auto closestTAlongRay(
		const Eigen::Vector3<D>& unitDir,
		const Eigen::Vector3<D>& origin,
		const Eigen::Vector3<D>& sample) {
		return (sample - origin).dot(unitDir);
	}






	//inline Eigen::ArrayXd uniformKnotsForCVs(Status& s, int nCvs, int degree) {
	//	if (degree == 1) {
	//		//Eigen::ArrayXd result()
	//		MDoubleArray result(nCvs, 0.0);
	//		for (int i = 0; i < nCvs; i++) {
	//			result[i] = float(i);
	//		}
	//		return result;
	//	}
	//	MDoubleArray result(degree + nCvs - 1);
	//	int i = 0;
	//	float v = 0;
	//	for (int n = 0; n < degree; n++) {
	//		result[i] = v;
	//		i += 1;
	//	}
	//	for (int n = 0; n < nCvs - degree; n++) {
	//		v += 1.0;
	//		result[i] = v;
	//		i += 1;
	//	}
	//	for (int n = 0; n < degree - 1; n++) {
	//		result[i] = v;
	//		i += 1;
	//	}
	//	return result;
	//}

	template<typename MATRIX, typename T, int N=3>
	inline void setMatrixRow(MATRIX& mat, const int& rowIndex, const T* data) {
		for (int i = 0; i < N; i++) {
			mat(rowIndex, i) = data[i];
		}
	}

	template<typename T>
	inline Status& makeFrame(
		Status& s,
		//Eigen::Matrix4<T>& frameMat,
		Eigen::Transform<T, 3, Eigen::Affine> frameMat,
		const Eigen::Vector3<T>& pos,
		const Eigen::Vector3<T>& tan, 
		const Eigen::Vector3<T>& normal
	){
		/* x is tangent,
		y is up,
		z is normal
		*/
		Eigen::Vector3<T> up = tan.cross(normal);
		Eigen::Vector3<T> normalZ = tan.cross(up);
		setMatrixRow(frameMat, 0, tan.data());
		setMatrixRow(frameMat, 1, up.data());
		setMatrixRow(frameMat, 2, normalZ.data());
		return s;
	}

	inline Status& makeFrame(
		Status& s,
		Eigen::Affine3f& frameMat,
		const Eigen::Vector3f& pos,
		const Eigen::Vector3f& tan,
		const Eigen::Vector3f& normal
	) {
		/* x is tangent,
		y is up,
		z is normal
		*/
		Eigen::Vector3f up = tan.cross(normal);
		Eigen::Vector3f normalZ = tan.cross(up);
		setMatrixRow(frameMat, 0, tan.normalized().data());
		setMatrixRow(frameMat, 1, up.normalized().data());
		setMatrixRow(frameMat, 2, normalZ.normalized().data());
		return s;
	}

	inline Status& makeFrame(
		Status& s,
		Eigen::Affine3f& frameMat,
		const Eigen::Vector3f& pos,
		const Eigen::Vector3f& tan
	) {// default {0, 1, 0} normal
		return makeFrame(s, frameMat, pos, tan, { 0.0, 1.0, 0.0 });
	}

	//inline Eigen::Vector3f splineTan(const Eigen::Spline3d& sp, const double u) {
	//	//sp.derivatives(u, 0);

	//	sp.derivatives<1>(u);
	//	
	//	return sp.derivatives(u, 1).matrix().row(0);
	//}

	template<typename T>
	inline Eigen::MatrixX3<T> cubicTangentPointsForBezPoints(
		const Eigen::MatrixX3<T>& inPoints,
		const bool closed,
		float* inContinuities = nullptr
	); 

	template<typename T>
	inline Eigen::MatrixX3<T> resampleVectorArray(
		const Eigen::MatrixX3<T>& inVs,
		float start, float end,
		int nSamples,
		bool normalise = true
	) {
		Eigen::MatrixX3<T> result(nSamples, 3);
		float range = end - start;
		float step = range / float(nSamples - 1);
		for (int i = 0; i < nSamples; i++) {
			float newU = step * i;
			int lowIndex;
			int highIndex;
			float sampleT = getArrayIndicesTForU(static_cast<int>(inVs.rows()), newU, lowIndex, highIndex);
			result.row(i) = lerp(Vector3f(inVs.row(lowIndex)), Vector3f(inVs.row(highIndex)), sampleT);
			if (normalise) {
				result.row(i) = result.row(i).normalized();
			}
		}
		return result;
	}

	//inline double closestParamOnSpline(const Eigen::Spline3d& sp, const Eigen::Vector3f pt,
	//	int nSamples =10,
	//	int iterations=2
	//) {
	//	/* basic binary search?
	//	or we start by checking control points?
	//	
	//	or we just put this in maya
	//	
	//	seems like an initial scattered search is used even in some SVG libraries
	//	algebraic solutions are apdomainly possible?

	//	JUST BRUTE FORCE IT
	//	10 sparse, 10 dense, move on
	//	*/
	//	Eigen::Vector3f l, r;
	//	double a = 0.0;
	//	double b = 1.0;
	//	/*l = sp(a).matrix();
	//	r = sp(b).matrix();*/
	//	Eigen::ArrayX3d iterSamples(nSamples, 3);
	//	Eigen::ArrayXd lins(nSamples);
	//	Eigen::ArrayXd distances(nSamples);
	//	//int maxCol, maxRow = 0;
	//	int minIndex = 0;
	//	for (int i = 0; i < iterations; i++) {
	//		//iterSamples = Eigen::ArrayX3d::Zero(nSamples);
	//		lins = Eigen::ArrayXd::LinSpaced(nSamples, a, b);
	//		for (int n = 0; n < nSamples; n++) {
	//			//auto res = sp(lins[n]);
	//			//iterSamples(n) = res;
	//			//iterSamples.row(n) = res;
	//			iterSamples.row(n) = sp(lins(n));
	//			distances[n] = (sp(lins[n]).matrix() - pt).squaredNorm();
	//		}
	//		//distances.maxCoeff(maxRow, maxCol);
	//		//minIndex = distances.minCoeff();
	//		distances.minCoeff(&minIndex);
	//		if (minIndex == (nSamples - 1)) { // closest to right
	//			b = lins[nSamples - 2];
	//			continue;
	//		}
	//		if (minIndex == 0) { // closest to left
	//			a = lins[1];
	//			continue;
	//		}

	//		//if(distances[minIndex + 1] > distances)
	//		a = minIndex - 1;
	//		b = minIndex + 1;
	//		
	//	}
	//	// return last hit float coord
	//	return lins[minIndex];
	//}

	inline Status& matrixAtU(
		Status& s,
		Eigen::Affine3f& mat,
		const Eigen::Spline3f& posSpline,
		const Eigen::Spline3f& normalSpline,
		const float u
		) {
		
		s = makeFrame(s,
			mat,
			Eigen::Vector3f(posSpline(u).matrix()),
			Eigen::Vector3f(posSpline.derivatives(u, 1).col(0).matrix()),
			normalSpline(u).matrix()
		);
		return s;
	}

	//inline Status& matrixAtU(
	//	Status& s,
	//	Eigen::Affine3f& mat,
	//	const bez::CubicBezierPath,
	//	const float u
	//) {

	//	s = makeFrame(s,
	//		mat,
	//		Eigen::Vector3f(posSpline(u).matrix()),
	//		Eigen::Vector3f(posSpline.derivatives(u, 1).col(0).matrix()),
	//		normalSpline(u).matrix()
	//	);
	//	return s;
	//}


	/*
	curve functions from Pomax Bezier Primer

	getCubicDerivative(t, points) {
	let mt = (1 - t), a = mt*mt, b = 2*mt*t, c = t*t, d = [
		{
			x: 3 * (points[1].x - points[0].x),
			y: 3 * (points[1].y - points[0].y)
		},
		{
			x: 3 * (points[2].x - points[1].x),
			y: 3 * (points[2].y - points[1].y)
		},
		{
			x: 3 * (points[3].x - points[2].x),
			y: 3 * (points[3].y - points[2].y)
		}
	];

	return {
		x: a * d[0].x + b * d[1].x + c * d[2].x,
		y: a * d[0].y + b * d[1].y + c * d[2].y
	};
}

	*/
	//Eigen::MatrixX3d
	inline Status& getCubicDerivative(Status& s, Eigen::Vector3f& out, float t, Eigen::Matrix<float, 4, 3> points) {
		// TODO: if this works, rewrite last block as arrays
		auto mt = 1.0f - t;
		auto a = mt * mt;
		auto b = 2.0f * mt * t;
		auto c = t * t;
		Eigen::Matrix3f d;
		d << 3.0f * (points.row(1).x() - points.row(0).x()),
			3.0f * (points.row(1).y() - points.row(0).y()),
			3.0f * (points.row(1).z() - points.row(0).z()),

			3.0f * (points.row(2).x() - points.row(1).x()),
			3.0f * (points.row(2).y() - points.row(1).y()),
			3.0f * (points.row(2).z() - points.row(1).z()),

			3.0f * (points.row(3).x() - points.row(2).x()),
			3.0f * (points.row(3).y() - points.row(2).y()),
			3.0f * (points.row(3).z() - points.row(2).z());

		out(0) = a * d.row(0).x() + b * d.row(1).x() + c * d.row(2).x();
		out(1) = a * d.row(0).y() + b * d.row(1).y() + c * d.row(2).y();
		out(2) = a * d.row(0).z() + b * d.row(1).z() + c * d.row(2).z();
		return s;
	}

	//inline double closestParamOnSpline(const bez::BezierCurve& sp, const Eigen::Vector3f pt,
	//	int nSamples = 10,
	//	int iterations = 2
	//) {

	//	Eigen::Vector3f l, r;
	//	double a = 0.0;
	//	double b = 1.0;
	//	/*l = sp(a).matrix();
	//	r = sp(b).matrix();*/
	//	Eigen::ArrayX3d iterSamples(nSamples, 3);
	//	Eigen::ArrayXd lins(nSamples);
	//	Eigen::ArrayXd distances(nSamples);
	//	//int maxCol, maxRow = 0;
	//	int minIndex = 0;
	//	for (int i = 0; i < iterations; i++) {
	//		//iterSamples = Eigen::ArrayX3d::Zero(nSamples);
	//		lins = Eigen::ArrayXd::LinSpaced(nSamples, a, b);
	//		for (int n = 0; n < nSamples; n++) {
	//			//auto res = sp(lins[n]);
	//			//iterSamples(n) = res;
	//			//iterSamples.row(n) = res;
	//			iterSamples.row(n) = sp(lins(n));
	//			distances[n] = (sp(lins[n]).matrix() - pt).squaredNorm();
	//		}
	//		//distances.maxCoeff(maxRow, maxCol);
	//		//minIndex = distances.minCoeff();
	//		distances.minCoeff(&minIndex);
	//		if (minIndex == (nSamples - 1)) { // closest to right
	//			b = lins[nSamples - 2];
	//			continue;
	//		}
	//		if (minIndex == 0) { // closest to left
	//			a = lins[1];
	//			continue;
	//		}

	//		//if(distances[minIndex + 1] > distances)
	//		a = minIndex - 1;
	//		b = minIndex + 1;

	//	}
	//	// return last hit float coord
	//	return lins[minIndex];
	//}



	Eigen::ArrayXf arcLengthToParamMapping(const Eigen::Spline3f& sp, const int npoints = 20);

	Eigen::ArrayXf arcLengthToParamMapping(const bez::CubicBezierSpline& sp, const int npoints = 20);

	Eigen::ArrayXf arcLengthToParamMapping(const bez::CubicBezierPath& sp, const int npoints = 20);

	Status& splineUVN(
		Status& s,
		//Eigen::Matrix4f& outMat,
		Eigen::Affine3f& outMat,
		const Eigen::Spline3f& posSpline,
		const Eigen::Spline3f& normalSpline,
		float uvw[3]
	);

	/**************************************************************************//**
	@brief solve B = A^^power, where A is a matrix

	@ingroup geometry

	This version uses a Quadratic Bezier approximation.

	The first column of the matrix is presumed to be the tangent.

	The power should be a real number between 0 and 1.

	For a proper matrix power function, see MatrixPower.

*//***************************************************************************/
	template <typename MATRIX>
	class MatrixBezier
	{
	public:
		MatrixBezier(void) {}

		template <typename T>
		void	solve(MATRIX& B, const MATRIX& A, T a_power) const;
	};

	template <typename MATRIX>
	template <typename T>
	inline void MatrixBezier<MATRIX>::solve(MATRIX& B, const MATRIX& A,
		T a_power) const
	{
		const T linearity = 0.45;	//* TODO param
		//A = Eigen::Matrix4f();
		//const Vector<3, T>& rTranslation = A.translation();
		const Eigen::Vector3<T>& rTranslation = A.translation();
		//const T distance = magnitude(rTranslation);
		const T distance = rTranslation.norm();

		//const MATRIX rotation(angularlyScaled(SpatialQuaternion(A), a_power));
		
		const MATRIX rotation = angularlyScaled<T>(Eigen::Quaternion<T>(A), a_power).matrix() ;

		const T power1 = T(1) - a_power;
		const T xStart = distance * a_power;
		const T xFull = -distance * power1;

		//* transform of X-only vector
		//const Vector<3, T> locTip = xFull * A.column(0) + rTranslation;
		const Eigen::Vector3<T> locTip = xFull * A.col(0) + rTranslation;

		//const Vector<3, T> locBlend(
		const Eigen::Vector3<T> locBlend(
			locTip[0] * a_power + xStart * power1,
			locTip[1] * a_power,
			locTip[2] * a_power);

		/*const Vector<3, T> locLinear = rTranslation * a_power;
		const Vector<3, T> locMix = locLinear * linearity + locBlend * (1.0 - linearity);*/
		const Eigen::Vector3<T> = rTranslation * a_power;
		const Eigen::Vector3<T> = locLinear * linearity + locBlend * (1.0 - linearity);


		//* rotation of unit axes
		/*const Vector<3, T>& xArmBlend = rotation.column(0);
		const Vector<3, T>& yArmBlend = rotation.column(1);*/
		const Eigen::Vector3<T>& xArmBlend = rotation.column(0);
		const Eigen::Vector3<T>& yArmBlend = rotation.column(1);


		makeFrameTangentX(B, locMix, xArmBlend, yArmBlend);
	}



	template <typename T>
	inline T lerpSampleScalarArr(Eigen::VectorX<T> arr, T t) {
		// sample array at a certain interval
		//float& a;
		
		if (t >= 1.0) {
			return arr.tail<1>()[0];
		}
		if (t <= 0.0) {
			return arr[0];
		}
		int a;
		int b;
		a = floor(arr.size() * t);
		b = a + 1;
		return lerp<T, T>(arr[a], arr[b], t - (arr.size() * t));
	}

	template <typename T>
	inline T lerpSampleScalarArr(Eigen::ArrayX<T> arr, T t) {
		// sample array at a certain interval
		//float& a;

		if (t >= 1.0) {
			return arr.tail<1>()[0];
		}
		if (t <= 0.0) {
			return arr[0];
		}
		int a;
		int b;
		a = static_cast<int>(floor(arr.size() * t));
		b = a + 1;
		return lerp<T, T>(arr[a], arr[b], t - (arr.size() * t));
	}
	

	template <typename T>
	inline T lerpSampleScalarArr(Eigen::VectorX<T> arr, T t, int& lowI, int& highI) {
		// sample array at a certain interval
		//float& a;

		if (t >= 1.0) {
			lowI = arr.size() - 1;
			highI = arr.size() - 1;
			return arr.tail<1>()[0];
		}
		if (t <= 0.0) {
			lowI = 0;
			highI = 0;
			return arr[0];
		}
		int a;
		int b;
		a = floor(arr.size() * t);
		b = a + 1;
		lowI = a;
		highI = b;
		return lerp<T, T>(arr[a], arr[b], t - (arr.size() * t));
	}

	template <typename T, int N>
	inline Eigen::Vector<T, N> lerpSampleMatrix(const Eigen::MatrixX<T>& arr, T t) {
		// sample array at a certain interval - INTERVAL NOT NORMALISED
		if (t >= 1.0) {
			return arr.row(arr.rows()-1);
		}
		if (t <= 0.0) {
			return arr.row(0);
		}
		int a;
		int b;
		a = static_cast<int>(static_cast<float>(arr.size()) * t);
		b = a + 1;
		return lerp<Eigen::Vector<T, N>, T>(arr.row(a), arr.row(b), t - (arr.size() * t));
	}

	template <typename T, int N>
	inline Eigen::Vector<T, N> smoothSampleMatrix(const Eigen::MatrixX<T>& arr, T t) {
		// sample array at a certain interval - INTERVAL NOT NORMALISED
		if (t >= 1.0) {
			return arr.row(arr.rows() - 1);
		}
		if (t <= 0.0) {
			return arr.row(0);
		}
		int a;
		int b;
		a = static_cast<int>(static_cast<float>(arr.size()) * t);
		b = a + 1;
		return lerp<Eigen::Vector<T, N>, T>(arr.row(a), arr.row(b), 
			smoothstepCubic( t - (arr.size() * t)));
	}


	inline float getArrayIndicesTForU(const int nEntries, const float u, int& a, int& b) {
		/* get upper and lower indices and t value for final interpolation, 
		from a 0-1 global u*/
		a = static_cast<int>(u * float(nEntries));
		b = a + 1;
		if (u >= 1.0f) {
			a = nEntries - 1;
			b = nEntries - 1;
			return 1.0f;
		}
		if (u <= 0.0f) {
			a = 0;
			b = 0;
			return 0.0f;
		}
		return u * float(nEntries) - float(a);
	}

	template <typename T>
	inline T getArrayUValueNonNorm(const Eigen::VectorX<T>& arr, T searchVal) {
		// get U value for a float in an equally spaced array
		// brute force for now, not sure how you would do this kind of arg search in proper eigen
		float u = 0.0f;
		
		int i = 0;
		for ( i = 0; i < arr.size() - 1; i++) {
			if (arr[i] > searchVal) {
				break;
			}
			u += 1.0f;
		}
		if (i == 0) {
			return T(0.0f);
		}
		return mapTo01(u, arr[i - 1], arr[i], true);
	}

	template <typename T>
	inline T getArrayUValueNonNorm(const Eigen::VectorX<T>& arr, T searchVal, int& a, int& b, float& t) {
		/* return the U value between 2 indices for sampling a mono-increasing array
		at the given search val, interpolated linearly
		
		also populate lower and upper indices, so you can use this result to lerp a value
		between data points*/
		float u = 0.0f;

		int i = 0;
		for (i = 0; i < arr.size() - 1; i++) {
			if (arr[i] > searchVal) {
				break;
			}
			u += 1.0f;
		}
		if (i == 0) {
			a = 0;
			b = 0;
			return T(0.0f);
		}
		a = i - 1;
		b = i;
		return mapTo01(searchVal - arr[a], arr[a], arr[b], true);
	}

	template <typename T>
	inline T getArrayUValueNorm(const Eigen::VectorX<T>& arr, T searchVal) {
		return (getArrayUValueNonNorm(arr, searchVal) / T(arr.size()));
	}
	//template<typename T>
	//inline T getArrayIndicesTForUNonUniform(const Eigen::VectorX<T>& arr, T searchVal) {

	//}

	/* tFox on techartists.org discovered awesome way to blend multiple quats together evenly - 
	take log of each, add them up, then take exponential
	*/
	const float constE = 2.71828f;

	inline Eigen::Quaternionf quatLogarithm(const Eigen::Quaternionf& q) {
		Eigen::Vector3f v = (q.vec() / q.vec().norm()) * std::acos((q.w() / q.norm()));
		return Eigen::Quaternionf{
			std::log(q.norm()), v.x(), v.y(), v.z()
		};
	}

	inline Eigen::Quaternionf quatExponential(const Eigen::Quaternionf& q) {
		float ea = pow(constE, q.w());
		Eigen::Vector3f v = (q.vec() / q.vec().norm()) * std::sin(q.vec().norm()) * ea;
		return Eigen::Quaternionf{
			std::cos(v.norm()) * ea, v.x(), v.y(), v.z()
		};
	}

	//inline Quaternionf blend2Quater

	inline Quaternionf blendQuaternions(
		const std::vector<Quaternionf>& quats,
		const VectorXf& weights
	) {
		if (quats.size() == 1) return quats[0];
		if (quats.size() == 2) {
			float w = weights(1) / weights.sum();
			return quats[0].slerp(w, quats[1]);
		}

		// For N > 2: Iterative SLERP or quaternion averaging
		// (Log-space blending is overkill for most cases)
		Quaternionf result = quats[0];
		float accumulated_weight = weights(0);

		for (size_t i = 1; i < quats.size(); ++i) {
			float t = weights(i) / (accumulated_weight + weights(i));
			result = result.slerp(t, quats[i]);
			accumulated_weight += weights(i);
		}

		return result;
	}

	inline Quaternionf blendQuaternionsExact(
		const std::vector<Quaternionf>& quats,
		const VectorXf& weights
	) {
		Matrix4f M = Matrix4f::Zero();

		for (size_t i = 0; i < quats.size(); ++i) {
			Vector4f q = quats[i].coeffs();  // [x, y, z, w]
			M += weights(i) * (q * q.transpose());
		}

		// Largest eigenvector = average quaternion
		Eigen::SelfAdjointEigenSolver<Matrix4f> solver(M);
		return Quaternionf(solver.eigenvectors().col(3));
	}

	inline Eigen::Affine3f blendTransforms(
		//const Eigen::MatrixX4f& quats,
		std::vector<Eigen::Affine3f>& tfs,
		Eigen::VectorXf& weights
	) {
		std::vector<Quaternionf> quats(tfs.size());
		Vector3f pos = { 0, 0, 0 };
		for (size_t i = 0; i < quats.size(); i++) {
			quats[i] = Quaternionf(tfs[i].rotation());
			pos += tfs[i].translation() * weights(i);
		}
		Quaternionf qResult = blendQuaternions(quats, weights);
		Affine3f result(qResult);
		result.translation() = pos / weights.sum();
		return result;
	}

	inline float getAngleAroundAxis(
		const Eigen::Vector3f& frameNormal,
		const Eigen::Vector3f& frameUp,
		const Eigen::Vector3f& v
	) { // where is the axis?
		// the axis is implied - this returns 0-1 depending on v's rotation around frameNormal x frameUp
		// test using only dots
		// i think this is the counterpart to atan2
		return 0.5f - sus(frameNormal.dot(v)) * 0.5f + float(frameUp.dot(v) > 0.0f) * 0.5f;
		//if (frameUp.dot(v) > 0.0) { // on the frame up side, should vary between 0 and 0.5
		//	return 0.5 - sus(frameNormal.dot(v)) * 0.5;  // at dot == 0.99 (vector almost on normal), angle should be near 0.0, 
		//}
		//return 1.0 - sus(frameNormal.dot(v)) * 0.5;
	}




	/* subsampling, */

	struct BezSplitPts {
		Eigen::Matrix<float, 4, 3> aPts;
		Eigen::Matrix<float, 4, 3> bPts;
	};

	inline BezSplitPts splitBezSpline(
		const Eigen::Matrix<float, 4, 3>& cvs,
		const float z
	) { /* from the pomax bezier page(where else)
		using matrix formulation for control points - 
		for now just copying the final derivation for control points
		*/
		BezSplitPts result;
		
		result.aPts.row(0) = cvs.row(0);
		result.aPts.row(1) = z * cvs.row(1) - 
			(z - 1.0f) * cvs.row(0);
		result.aPts.row(2) = z * z * cvs.row(2) - 
			2.0f * z * (z - 1.0f) * cvs.row(1) + 
			(z - 1.0f) * (z - 1.0f) * cvs.row(0);
		result.aPts.row(3) = z * z * z * cvs.row(3) -
			3.0f * z * z * (z - 1.0f) * cvs.row(2) +
			3.0f * z * (z - 1.0f) * (z - 1.0f) * cvs.row(1) -
			(z - 1.0f) * (z - 1.0f) * (z - 1.0f) * cvs.row(0);
		
		result.bPts.row(0) = result.aPts.row(3);
		result.bPts.row(1) = z * z * cvs.row(3) - 
			2.0f * z * (z - 1.0f) * cvs.row(2) + 
			(z - 1.0f) * (z - 1.0f) * cvs.row(1);
		result.bPts.row(2) = z * cvs.row(3) - 
			(z - 1.0f) * cvs.row(2);
		result.bPts.row(3) = cvs.row(3);
		
		return result;
	}

	bez::CubicBezierPath splitBezPath(bez::CubicBezierPath& crv, float lowT, float highT);

	std::tuple<float, Vector3f, float, Vector3f> closestBezPointToRay(bez::CubicBezierPath& crv, Vector3f rayO, Vector3f raySpan,
		int initRaySamples=10, int mutualIters=3
	);

	//std::tuple<float, Vector3f> closestBezPointWeighted(bez::CubicBezierPath& crv, Vector3f pos,
	//	int samplesPerSpan)


	Eigen::MatrixX3f makeRMFNormals(
		Eigen::MatrixX3f& positions,
		Eigen::MatrixX3f& tangents,
		const Eigen::MatrixX3f& targetNormals,
		const int nSamples
	);


	Eigen::MatrixX3f makeRMFNormals(
		bez::CubicBezierPath& crv,
		const Eigen::MatrixX3f& targetNormals,
		const Eigen::VectorXf& targetNormalParams,
		const int nSamples
	);

	ArrayX3f makeRMFNormals(
		const ArrayX3f& positions,
		const ArrayX3f& targetNormals,
		const ArrayXf& normalParams,
		const ArrayXf& twistValues,
		const ArrayXf& normalWeights
	);

	static void covariance(const Ref<const MatrixXf> x, const Ref<const MatrixXf> y, Ref<MatrixXf> C)
	{
		const float num_observations = static_cast<float>(x.rows());
		const RowVectorXf x_mean = x.colwise().sum() / num_observations;
		const RowVectorXf y_mean = y.colwise().sum() / num_observations;
		C = (x.rowwise() - x_mean).transpose() * (y.rowwise() - y_mean) / num_observations;
	}

	/* and to call:
	MatrixXf m1, m2, m3
	cov(m1, m2, m3);
	cov(m1.leftCols<3>(), m2.leftCols<3>(), m3.topLeftCorner<3,3>());
*/

	template<typename vT, typename weightArrT>
	void weightedSum(std::vector<vT>& values, weightArrT& weights, int n, vT& result, float power = 1.0) {
		float weightSum = 0.0;
		for (int i = 0; i < n; i++) {
			float weight = EQ(power, 1.0) ? weights[i] : pow(weights[i], power);
			result += values[i] * weight;
			weightSum += weight;
		}
		result /= weightSum;
	}

	template<typename arrT, typename weightArrT, typename vT>
	void weightedSum(Eigen::MatrixXf& values, weightArrT& weights, int n, vT& result, float power = 1.0) {
		float weightSum = 0.0;
		for (int i = 0; i < n; i++) {
			float weight = EQ(power, 1.0) ? weights[i] : pow(weights[i], power);
			result += values.row(i) * weight;
			weightSum += weight;
		}
		result /= weightSum;
		//return result;
	}


	//template<typename DataT, typename ResultT>
	void interp1D(
		Eigen::VectorXf& x,
		Eigen::MatrixX3f& y,
		Eigen::VectorXf& xI,
		Eigen::MatrixX3f& yI
	) {
		/* given data points y at mono-increasing x-coords x,
		* sample at coords xI and store in matrix yI
		* 
		* of course this won't be as fast as numpy, but it seemed easier than bringing in
		* armadillo and converting between eigen and arma vectors, matrices etc
		* 
		* couldn't work out how to do the templating properly, so explicitly write function for each kind of value to use
		*/
		
		for (int sampleI = 0; sampleI < xI.size(); sampleI++) {
			int a, b;
			float t;
			float u = getArrayUValueNonNorm(
				x, 
				xI(sampleI), 
				a, b, t
			);
			//yI.row(sampleI) = lerp(y.row(a), y.row(b), t);
			//yI.row(sampleI) << lerp<Eigen::Vector3f, float>(y.row(a), y.row(b), t);
			yI.row(sampleI) = lerp(y.row(a), y.row(b), t);
		}
	}

	void interp1D(
		Eigen::VectorXf& x,
		Eigen::MatrixX3f& y,
		float& xI,
		Eigen::Vector3f& yI
	) {
			int a, b;
			float t;
			float u = getArrayUValueNonNorm(
				x,
				xI,
				a, b, t
			);
			yI = lerp(y.row(a), y.row(b), t);
	}

	template<typename arrVT, typename T>
	int closestArrayEntry(std::vector<arrVT>& arr, T val) {
		auto it = std::lower_bound(arr.begin(), arr.end(), static_cast<arrVT>(val));
		if (it == arr.end()) {
			return static_cast<int>(arr.size());
		}
		/* check if lower entry is closer than upper*/
		if ((val - static_cast<T>(*it)) < (static_cast<T>(*(it + 1)) - val)) {
			return static_cast<int>(std::distance(arr.begin(), it));
		}
		return static_cast<int>(std::distance(arr.begin(), it)) + 1;
	}

	std::vector<int> gridConnectivityTriIndexBuffer(
		int columns,
		int rows
	) {
		/* ASSUME THAT
		vertices laid out linearly, rows first,
		count goes back to start at each line

		no locality, no optimisation for now
		*/
		int nTris = (columns - 1) * (rows - 1) * 2;
		std::vector<int> result(nTris * 3);
		int triIndex = 0;
		for(int i = 0; i < nTris / 2; i++){
			result[i * 3 * 2] = i;
			result[i * 3 * 2 + 1] = i + 1;
			result[i * 3 * 2 + 2] = i + columns;

			result[i * 3 * 2 + 3] = i + 1;
			result[i * 3 * 2 + 4] = i + columns + 1;
			result[i * 3 * 2 + 5] = i + columns;
		}
		return result;
	}


	template<typename Derived, typename Scalar = Derived::Scalar >
	inline aabb::AABB< Derived::ColsAtCompileTime, Scalar> getAABB(const Eigen::DenseBase<Derived>& positions) {
		static_assert(Derived::ColsAtCompileTime == 3 || Derived::ColsAtCompileTime == Eigen::Dynamic,
			"getAABB requires 3-column array (xyz positions)");

		aabb::AABB< Derived::ColsAtCompileTime, Scalar> result;			
		// Handle empty input
		if (positions.rows() == 0) {
			result.min = Eigen::Matrix<Scalar, Derived::ColsAtCompileTime, 1>::Zero();
			result.max = Eigen::Matrix<Scalar, Derived::ColsAtCompileTime, 1>::Zero();
			return result;
		}

		// Compute min/max per column (x, y, z)
		result.min = positions.colwise().minCoeff();
		result.max = positions.colwise().maxCoeff();

		return result;
	}

	/**
	 * @brief Compute AABB with explicit row-major layout
	 */
	inline auto getAABB(const Eigen::MatrixX3f& positions) {
		return getAABB<Eigen::MatrixX3f>(positions);
	}

	/**
	 * @brief Compute AABB from ArrayX3f (common case)
	 */
	inline auto getAABB(const Eigen::ArrayX3f& positions) {
		return getAABB<Eigen::ArrayX3f>(positions);
	}

	/**
	 * @brief Compute AABB from vector of Vector3f
	 */
	inline auto getAABB(const std::vector<Eigen::Vector3f>& positions) {
		aabb::AABB<3, float> result;

		if (positions.empty()) {
			result.min = Eigen::Vector3f::Zero();
			result.max = Eigen::Vector3f::Zero();
			return result;
		}

		result.min = positions[0];
		result.max = positions[0];

		for (size_t i = 1; i < positions.size(); ++i) {
			result.min = result.min.cwiseMin(positions[i]);
			result.max = result.max.cwiseMax(positions[i]);
		}

		return result;
	}

	/**
	 * @brief Expand AABB to include another AABB
	 */
	template<typename AABBT>
	inline AABBT& expand(AABBT& a, const AABBT& b) {
		a.min = a.min.cwiseMin(b.min);
		a.max = a.max.cwiseMax(b.max);
		return a;
	}

	/**
	 * @brief Expand AABB to include a point
	 */
	template<typename AABBT, typename VectorT>
	inline AABBT& expand(AABBT& a, const VectorT& point) {
		a.min = a.min.cwiseMin(point);
		a.max = a.max.cwiseMax(point);
		return a;
	}

	/**
	 * @brief Compute center of AABB
	 */
	template<typename AABBT>
	inline Eigen::Matrix<typename AABBT::Scalar, AABBT::Dim, 1> center(const AABBT& aabb) {
		return (aabb.min + aabb.max) * 0.5f;
	}

	/**
	 * @brief Compute size/extents of AABB
	 */
	template<typename AABBT>
	inline Eigen::Matrix<typename AABBT::Scalar, AABBT::Dim, 1> size(const AABBT& aabb) {
		return aabb.max - aabb.min;
	}

	/**
	 * @brief Compute diagonal length of AABB
	 */
	template<typename AABBT>
	inline typename AABBT::Scalar diagonal(const AABBT& aabb) {
		return (aabb.max - aabb.min).norm();
	}

	/**
	 * @brief Check if point is inside AABB
	 */

	template<typename AABBT, typename VectorT>
	inline bool contains(const AABBT& aabb, const VectorT& point) {
		return (point.array() >= aabb.min.array()).all() &&
			(point.array() <= aabb.max.array()).all();
	}

	/**
	 * @brief Check if two AABBs intersect
	 */

	template<typename AABBT>
	inline bool intersects(const AABBT& a, const AABBT& b) {
		return (a.min.array() <= b.max.array()).all() &&
			(a.max.array() >= b.min.array()).all();
	}


}

/**
 * @brief Compute AABB from Eigen matrix/array of positions
 * 
 * Works with:
 * - MatrixX3f, MatrixX2f (dynamic rows, fixed columns)
 * - ArrayX3f, ArrayX2f
 * - Matrix<float, N, 3>, etc. (fixed-size)
 */
template<typename Derived>
inline auto getAABB(const Eigen::DenseBase<Derived>& positions) {
    using Scalar = typename Derived::Scalar;
    constexpr int Cols = Derived::ColsAtCompileTime;
    
    static_assert(Cols == 2 || Cols == 3,
        "getAABB requires 2D or 3D positions (2 or 3 columns)");

    using AABBType = aabb::AABB<Cols>;
    using VectorType = Eigen::Matrix<Scalar, Cols, 1>;
    
    // Handle empty input
    if (positions.rows() == 0) {
        VectorType zero = VectorType::Zero();
        return AABBType(zero, zero);
    }

    // Compute bounds
    VectorType min = positions.colwise().minCoeff().transpose();
    VectorType max = positions.colwise().maxCoeff().transpose();
    
    return AABBType(min, max);
}

/**
 * @brief Compute 3D AABB from vector of Vector3f
 */
template<typename Scalar = float>
inline aabb::AABB<3, Scalar> getAABB(const std::vector<Eigen::Matrix<Scalar, 3, 1>>& positions) {
    using VectorType = Eigen::Matrix<Scalar, 3, 1>;
    
    if (positions.empty()) {
        VectorType zero = VectorType::Zero();
        return aabb::AABB<3>(zero, zero);
    }

    VectorType minCoords = positions[0];
    VectorType maxCoords = positions[0];

    for (size_t i = 1; i < positions.size(); ++i) {
        minCoords = minCoords.cwiseMin(positions[i]);
        maxCoords = maxCoords.cwiseMax(positions[i]);
    }

    return aabb::AABB<3>(minCoords, maxCoords);
}

//// Convenience specializations
//inline aabb::AABB<3, float> getAABB(const Eigen::MatrixX3f& positions) {
//    return getAABB<Eigen::MatrixX3f>(positions);
//}
//
//inline aabb::AABB<3> getAABB(const Eigen::ArrayX3f& positions) {
//    return getAABB<Eigen::ArrayX3f>(positions);
//}
//
//inline aabb::AABB<2> getAABB(const Eigen::MatrixX2f& positions) {
//    return getAABB<Eigen::MatrixX2f>(positions);
//}
