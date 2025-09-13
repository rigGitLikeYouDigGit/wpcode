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

//#include "bezier/bezier.h"
/* would be nice to remove hard bezier dependency here, 
but couldn't work out the namespacing
*/

//#include <bezier/bezier.h>


namespace bez {
	class CubicBezierSpline;
	struct CubicBezierPath;
}

namespace strata {
	/* copying various from Free Electron,
	adapting to Eigen types
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
		bool operator() (const Vector3i& lhs, const Vector3i& rhs) const
		{
			return (lhs.x() < rhs.x()) || (lhs.y() < rhs.y()) || (lhs.z() < rhs.z());
		}
		bool operator() (const Vector3f& lhs, const Vector3i& rhs) const
		{
			Vector3i iLhs = toKey(lhs);
			return (iLhs.x() < rhs.x()) || (iLhs.y() < rhs.y()) || (iLhs.z() < rhs.z());
		}
		bool operator() (const Vector3i& lhs, const Vector3f& rhs) const
		{
			Vector3i iRhs = toKey(rhs);
			return (lhs.x() < iRhs.x()) || (lhs.y() < iRhs.y()) || (lhs.z() < iRhs.z());
		}
		bool operator() (const Vector3f& lhs, const Vector3f& rhs) const
		{
			Vector3i iLhs = toKey(lhs);
			Vector3i iRhs = toKey(rhs);
			return (iLhs.x() < iRhs.x()) || (iLhs.y() < iRhs.y()) || (iLhs.z() < iRhs.z());
		}
	};
	
	template <typename V>
	using Vector3iMap = std::map<Vector3i, V, Vector3iCompare>;
	template <typename V>
	using Vector3iUMap = std::unordered_map<Vector3i, V, Vector3iCompare>;

	/* eigen to string functions - a little spaghett*/
	inline std::string str(Vector3f& any) {
		std::stringstream ss;
		ss << any;
		return ss.str();
	}
	inline std::string str(Affine3f& any) {
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

	MMatrixArray curveMatricesFromDriverDatas(
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
	//	algebraic solutions are apparently possible?

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
		a = floor(arr.size() * t);
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
	inline T getArrayUValueNorm(const Eigen::VectorX<T>& arr, T searchVal) {
		return (getArrayUValueNonNorm(arr, searchVal) / T(arr.size()));
	}

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

	inline Eigen::Quaternionf blendQuaternions(
		//const Eigen::MatrixX4f& quats,
		std::vector<Eigen::Quaternionf>& quats,
		Eigen::VectorXf& weights
	) {
		/* convert each to log, add together, 
		take exponential of result
		
		if only 2 quats given, just slerp
		*/
		Eigen::Vector4f result(0, 0, 0, 0);

		if (quats.size() == 1) {
			return quats[0];
		}

		if (quats.size() == 2) {
			float weight = weights(1) / (weights(0) + weights(1));
			return quats[0].slerp(weight, quats[1]);
		}

		for (size_t i = 0; i < quats.size(); i++) {
			auto scaled = quatLogarithm(quats.at(i)).coeffs() * weights[i];
			result += scaled; // mixed matrices of diff sizes
		}
		return quatExponential(Eigen::Quaternionf(result / weights.sum()));
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
		return 0.5f - sus<float>(frameNormal.dot(v)) * 0.5f + float(frameUp.dot(v) > 0.0f) * 0.5f;
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

}
