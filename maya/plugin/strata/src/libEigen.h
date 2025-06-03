#pragma once

#include <vector>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include "MInclude.h"

#include <unsupported/Eigen/Splines>

#include "api.h"
#include "macro.h"
#include "status.h"
#include "lib.h"

//#include <bezier/bezier.h>

namespace ed {
	/* copying various from Free Electron,
	adapting to Eigen types
	*/

	using namespace Eigen;

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



	Status& closestPointInSegments(
		Status& s,
		const Eigen::MatrixX3f& pts,
		const Eigen::Vector3f& samplePos,
		const bool closed,
		float& distance,
		int& nearestPt,
		float& u
	) {
		float minDist = 1000000.0f;
		nearestPt = 0;

		for (int i = 0; i < pts.rows() - 1; i++) {

		}


		return s;
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
	) 
	{
		// treat open and closed the same, just discount final span if not closed

		Eigen::MatrixX3<T> result(inPoints.rows() * 3, 3);
		int nPoints = static_cast<int>(inPoints.rows());

		// set tangent vectors for each one (scaling done later)
		// also includes start and end, as if it were closed
		int nextInI, prevInI;
		int outI, nextOutI, prevOutI;
		for (int i = 0; i < nPoints; i++) {
			nextInI = (i + 1) % nPoints;
			prevInI = (i - 1) % nPoints;

			outI = i * 3;
			nextOutI = (i* 3 + 1) % nPoints;
			prevOutI = (i* 3 - 1) % nPoints;

			auto thisPos = inPoints.row(i);
			auto nextPos = inPoints.row(nextInI);
			auto prevPos = inPoints.row(prevInI);

			// set start point from originals
			result.row(outI) = thisPos;

			// vector from prev ctl pt to next
			auto tanVec = nextPos - prevPos;
			//thisDriver.baseTan = tanVec;

			auto toThisVec = thisPos - prevPos;
			// vector from this ctl pt to next
			auto toNextVec = nextPos - thisPos;
			//auto toPrevVec = nextPos - thisPos;

			// forwards tan scale factor
			T nextTanScale = tanVec.dot(toNextVec) - (tanVec.dot(toThisVec)) / T(3.0);
			nextTanScale = -sminQ<T>(-nextTanScale, T(0.0), T(0.2));

			// back tan scale factor
			T prevTanScale = tanVec.dot(-toThisVec) - (tanVec.dot(toThisVec)) / T(3.0);
			prevTanScale = -sminQ<T>(-prevTanScale, T(0.0), T(0.2));

			result.row(nextOutI) = tanVec.normalized() * nextTanScale;
			result.row(prevOutI) = -tanVec.normalized() * prevTanScale;
		}

		// if we don't care about continuities, return
		if (inContinuities == nullptr) {
			return result;
		}


		// set ends if not continuous
		// if not closed, ends are not continuous
		if (!closed) {
			inContinuities[0] = T(0.0);
			inContinuities[nPoints - 1] = T(0.0);
		}

		// check continuity
		for (int i = 0; i < nPoints; i++) {
			nextInI = (i + 1) % nPoints;
			prevInI = (i - 1) % nPoints;

			outI = i * 3;
			nextOutI = (i * 3 + 1) % nPoints;
			prevOutI = (i * 3 - 1) % nPoints;

			int nextPtI = ((i + 1) * 3) % nPoints;
			int nextPtPrevTanI = ((i + 1) * 3 - 1) % nPoints;
			int nextPtNextTanI = ((i + 1) * 3 + 1) % nPoints;

			int prevPtI = ((i - 1) * 3) % nPoints;
			int prevPtPrevTanI = ((i - 1) * 3 - 1) % nPoints;
			int prevPtNextTanI = ((i - 1) * 3 + 1) % nPoints;


			// blend between next tangent point and next point, based on continuity
			//double postTanLen = thisDriver.postTan.norm();
			auto postTanLen = result.row(nextOutI).norm();
			Eigen::Vector3<T> nextOutV(result.row(nextOutI));
			Eigen::Vector3<T> nextPtV(result.row(nextPtI));
			Eigen::Vector3<T> nextPtPlusPrevTanV(result.row(nextPtI) + result.row(nextPtPrevTanI));
			Eigen::Vector3<T> outV(result.row(outI));

			// use continuity of next point to check where sharp target should be
			Eigen::Vector3<T> targetLerpV = lerp(
				nextPtV, nextPtPlusPrevTanV,
				static_cast<T>(inContinuities[nextInI])
				);
			// use continuity of this point to check how strongly tangent should lerp to that target

			result.row(nextOutI) = lerp(
				nextOutV,
				targetLerpV,
				static_cast<T>(inContinuities[i])
			);

			//result.row(nextOutI) = lerp(
			//	Eigen::Vector3<T>(result.row(nextOutI)),
			//	Eigen::Vector3<T>(lerp(
			//			//nextDriver.pos(),
			//			Eigen::Vector3<T>(result.row(nextPtI)),
			//			//Eigen::Vector3f(nextDriver.pos() + nextDriver.prevTan),
			//			Eigen::Vector3<T>(result.row(nextPtI) + result.row(nextPtPrevTanI)),
			//			//Eigen::Vector3f(nextDriver.pos() + nextDriver.preTan).matrix(),
			//			//nextDriver.continuity
			//			T(inContinuities[nextInI])
			//		) - Eigen::Vector3<T>(result.row(outI))),
			//		T(inContinuities[i])
			//);
		
			//thisDriver.postTan = thisDriver.postTan.normalized() * postTanLen;
			result.row(nextOutI) = result.row(nextOutI).normalized() * postTanLen;

			// prev tan
			//double prevTanLen = thisDriver.prevTan.norm();
			//thisDriver.prevTan = lerp(
			//	thisDriver.postTan,
			//	(lerp(
			//		prevDriver.pos(),
			//		Eigen::Vector3f(prevDriver.pos() + prevDriver.postTan),
			//		prevDriver.continuity
			//	) - thisDriver.pos()).eval(),
			//	thisDriver.continuity
			//);
			//thisDriver.prevTan = thisDriver.prevTan.normalized() * prevTanLen;

			float prevTanLen = result.row(prevOutI).norm();
			result.row(prevOutI) = lerp(
				Eigen::Vector3<T>(result.row(prevOutI)),
				Eigen::Vector3<T>(
					lerp(
					
					Eigen::Vector3<T>(result.row(prevPtI)),
					Eigen::Vector3<T>(result.row(prevPtI) + result.row(prevPtNextTanI)),
					T(inContinuities[prevInI])

				) - Eigen::Vector3<T>(result.row(outI))),
				T(inContinuities[i])

			);

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



	inline Eigen::ArrayXf arcLengthToParamMapping(const Eigen::Spline3f& sp, const int npoints = 20) {
		// return an array of equally-spaced points giving the 0-1 arc length to each point
		Eigen::ArrayXf result = Eigen::ArrayXf::Constant(npoints, 0.0f);
		//Eigen::ArrayXXd data = Eigen::ArrayXXd::Constant(nRow, nCol, 1.0);
		Eigen::Vector3f prevpt = sp(0.0f);
		Eigen::Vector3f thispt;
		for (int i = 1; i < npoints; i++) {
			float u = 1.0f / float(npoints - 1) * i;
			thispt = sp(u);
			result[i] = result[i-1] + (thispt - prevpt).norm();
			prevpt = thispt;
		}
		return result;
	}

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
	 T lerpSampleScalarArr(Eigen::VectorX<T> arr, T t) {
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
		a = floor(u * float(nEntries));
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

	inline Eigen::Quaternionf blendQuaternions(
		//const Eigen::MatrixX4f& quats,
		std::vector<Eigen::Quaternionf>& quats,
		Eigen::VectorXf& weights
	) {
		/* convert each to log, add together, 
		take exponential of result*/
		Eigen::Vector4f result(0, 0, 0, 0);
		for (size_t i = 0; i < quats.size(); i++) {
			auto scaled = quatLogarithm(quats.at(i)).matrix() * weights[i];
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

	// TODO: template this?
	inline Eigen::MatrixX3f subSampleBezSpline(bez::CubicBezierSpline& baseCrv, int nSpans) {
		/* create control points for given number of segments on bezier spline. 
		*/

		Eigen::MatrixX3f resultPoints(nSpans * 3 + 1);

		Eigen::Matrix<float, 4, 3> cvs = baseCrv.pointsAsMatrix();
		
		float step = (1.0f / float(nSpans * 3 + 1));
		BezSplitPts splitPts;
		for (int i = 0; i < nSpans - 1; i++) {
			
			splitPts = splitBezSpline(cvs, step * (i + 1));
			resultPoints.row(i * 3) = splitPts.aPts.row(0);
			resultPoints.row(i * 3 + 1) = splitPts.aPts.row(1);
			resultPoints.row(i * 3 + 2) = splitPts.aPts.row(2);
			
			cvs = splitPts.bPts;

			if (EQ(step * (i + 1), 1.0f)) {
				resultPoints.row((i + 1) * 3) = splitPts.bPts.row(0);
				resultPoints.row((i + 1) * 3 + 1) = splitPts.bPts.row(1);
				resultPoints.row((i + 1) * 3 + 2) = splitPts.bPts.row(2);
				resultPoints.row((i + 1) * 3 + 3) = splitPts.bPts.row(3);
			}
		}

		return resultPoints;
	}


	inline Eigen::MatrixX3f subSampleBezPathPts(bez::CubicBezierPath& baseCrv, int spansPerSpan) {
		/* create a new bezier path with given number of subspans, per-span
		* TODO: parallel
		*/
		Eigen::MatrixX3f result(baseCrv.splines_.size() * spansPerSpan * 3 + 1, 3);

		for (int i = 0; i < static_cast<int>(baseCrv.splines_.size()); i++) { // parallel
			int startIndex = i * spansPerSpan * 3;
			int indexSpan = 3 * spansPerSpan + 1;
			bez::CubicBezierSpline& splineRef = *baseCrv.splines_[i].get();
			Eigen::MatrixX3f segmentSpansResult = subSampleBezSpline(splineRef, spansPerSpan);
			DEBUGS("BEFORE STD COPY")
			std::copy(segmentSpansResult.data(), segmentSpansResult.data() + indexSpan,
				result.data() + startIndex
			) ;
			DEBUGS("AFTER STD COPY")
		}
		return result;
	}

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
			float param = (0.9999 / float(nSamples - 1) * float(i));
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
