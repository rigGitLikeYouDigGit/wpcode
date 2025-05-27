#pragma once

#include <vector>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include "MInclude.h"
#include "api.h"
#include "macro.h"
#include "status.h"

#include <unsupported/Eigen/Splines>

//#include <bezier/bezier.h>

namespace ed {
	/* copying various from Free Electron,
	adapting to Eigen types
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
		Eigen::Affine3d& frameMat,
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
		Eigen::Affine3d& frameMat,
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

		Eigen::MatrixX3<T> result(inPoints.rows() * 3);
		int nPoints = inPoints.rows();

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
			T nextTanScale = tanVec.dot(toNextVec) - (tanVec.dot(toThisVec)) / 3.0;
			nextTanScale = -sminQ(-nextTanScale, 0.0, 0.2);

			// back tan scale factor
			T prevTanScale = tanVec.dot(-toThisVec) - (tanVec.dot(toThisVec)) / 3.0;
			prevTanScale = -sminQ(-prevTanScale, 0.0, 0.2);

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
			inContinuities[0] = 0.0;
			inContinuities[nPoints - 1] = 0.0;
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
			float postTanLen = result.row(nextOutI).norm();
			result.row(nextOutI) = lerp(
				Eigen::Vector3<T>(result.row(nextOutI)),
				Eigen::Vector3<T>(lerp(
						//nextDriver.pos(),
						Eigen::Vector3<T>(result.row(nextPtI)),
						//Eigen::Vector3f(nextDriver.pos() + nextDriver.prevTan),
						Eigen::Vector3<T>(result.row(nextPtI) + result.row(nextPtPrevTanI)),
						//Eigen::Vector3f(nextDriver.pos() + nextDriver.preTan).matrix(),
						//nextDriver.continuity
						T(inContinuities[nextInI])
					) - result.row(outI)),
					T(inContinuities[i])
					
			);
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

				) - result.row(outI)),
				T(inContinuities[i])

			);

		}

		return result;

	}



	inline double closestParamOnSpline(const Eigen::Spline3d& sp, const Eigen::Vector3f pt,
		int nSamples =10,
		int iterations=2
	) {
		/* basic binary search?
		or we start by checking control points?
		
		or we just put this in maya
		
		seems like an initial scattered search is used even in some SVG libraries
		algebraic solutions are apparently possible?

		JUST BRUTE FORCE IT
		10 sparse, 10 dense, move on
		*/
		Eigen::Vector3f l, r;
		double a = 0.0;
		double b = 1.0;
		/*l = sp(a).matrix();
		r = sp(b).matrix();*/
		Eigen::ArrayX3d iterSamples(nSamples, 3);
		Eigen::ArrayXd lins(nSamples);
		Eigen::ArrayXd distances(nSamples);
		//int maxCol, maxRow = 0;
		int minIndex = 0;
		for (int i = 0; i < iterations; i++) {
			//iterSamples = Eigen::ArrayX3d::Zero(nSamples);
			lins = Eigen::ArrayXd::LinSpaced(nSamples, a, b);
			for (int n = 0; n < nSamples; n++) {
				//auto res = sp(lins[n]);
				//iterSamples(n) = res;
				//iterSamples.row(n) = res;
				iterSamples.row(n) = sp(lins(n));
				distances[n] = (sp(lins[n]).matrix() - pt).squaredNorm();
			}
			//distances.maxCoeff(maxRow, maxCol);
			//minIndex = distances.minCoeff();
			distances.minCoeff(&minIndex);
			if (minIndex == (nSamples - 1)) { // closest to right
				b = lins[nSamples - 2];
				continue;
			}
			if (minIndex == 0) { // closest to left
				a = lins[1];
				continue;
			}

			//if(distances[minIndex + 1] > distances)
			a = minIndex - 1;
			b = minIndex + 1;
			
		}
		// return last hit float coord
		return lins[minIndex];
	}

	inline Status& matrixAtU(
		Status& s,
		Eigen::Affine3d& mat,
		const Eigen::Spline3d& posSpline,
		const Eigen::Spline3d& normalSpline,
		const double u
		) {
		
		s = makeFrame(s,
			mat,
			posSpline(u).matrix(),
			posSpline.derivatives(u, 1).col(0).matrix(),
			normalSpline(u).matrix()
		);
		return s;
	}

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
	Status& getCubicDerivative(Status& s, Eigen::Vector3f& out, double t, Eigen::Matrix<double, 4, 3> points) {
		// TODO: if this works, rewrite last block as arrays
		auto mt = 1.0 - t;
		auto a = mt * mt;
		auto b = 2.0 * mt * t;
		auto c = t * t;
		Eigen::Matrix3d d;
		d << 3.0 * (points.row(1).x() - points.row(0).x()),
			3.0 * (points.row(1).y() - points.row(0).y()),
			3.0 * (points.row(1).z() - points.row(0).z()),

			3.0 * (points.row(2).x() - points.row(1).x()),
			3.0 * (points.row(2).y() - points.row(1).y()),
			3.0 * (points.row(2).z() - points.row(1).z()),

			3.0 * (points.row(3).x() - points.row(2).x()),
			3.0 * (points.row(3).y() - points.row(2).y()),
			3.0 * (points.row(3).z() - points.row(2).z());

		out(0) = a * d.row(0).x() + b * d.row(1).x() + c * d.row(2).x();
		out(1) = a * d.row(0).y() + b * d.row(1).y() + c * d.row(2).y();
		out(2) = a * d.row(0).z() + b * d.row(1).z() + c * d.row(2).z();
		return s;
	}

	inline double closestParamOnSpline(const bezier::BezierCurve& sp, const Eigen::Vector3f pt,
		int nSamples = 10,
		int iterations = 2
	) {

		Eigen::Vector3f l, r;
		double a = 0.0;
		double b = 1.0;
		/*l = sp(a).matrix();
		r = sp(b).matrix();*/
		Eigen::ArrayX3d iterSamples(nSamples, 3);
		Eigen::ArrayXd lins(nSamples);
		Eigen::ArrayXd distances(nSamples);
		//int maxCol, maxRow = 0;
		int minIndex = 0;
		for (int i = 0; i < iterations; i++) {
			//iterSamples = Eigen::ArrayX3d::Zero(nSamples);
			lins = Eigen::ArrayXd::LinSpaced(nSamples, a, b);
			for (int n = 0; n < nSamples; n++) {
				//auto res = sp(lins[n]);
				//iterSamples(n) = res;
				//iterSamples.row(n) = res;
				iterSamples.row(n) = sp(lins(n));
				distances[n] = (sp(lins[n]).matrix() - pt).squaredNorm();
			}
			//distances.maxCoeff(maxRow, maxCol);
			//minIndex = distances.minCoeff();
			distances.minCoeff(&minIndex);
			if (minIndex == (nSamples - 1)) { // closest to right
				b = lins[nSamples - 2];
				continue;
			}
			if (minIndex == 0) { // closest to left
				a = lins[1];
				continue;
			}

			//if(distances[minIndex + 1] > distances)
			a = minIndex - 1;
			b = minIndex + 1;

		}
		// return last hit float coord
		return lins[minIndex];
	}



	inline Eigen::ArrayXd arcLengthToParamMapping(const Eigen::Spline3d& sp, const int npoints = 20) {
		// return an array of equally-spaced points giving the 0-1 arc length to each point
		Eigen::ArrayXd result = Eigen::ArrayXd::Constant(npoints, 0.0);
		//Eigen::ArrayXXd data = Eigen::ArrayXXd::Constant(nRow, nCol, 1.0);
		Eigen::Vector3f prevpt = sp(0.0);
		Eigen::Vector3f thispt;
		for (int i = 1; i < npoints; i++) {
			double u = 1.0 / double(npoints - 1) * i;
			thispt = sp(u);
			result[i] = result[i-1] + (thispt - prevpt).norm();
			prevpt = thispt;
		}
		return result;
	}

	Status& splineUVN(
		Status& s,
		//Eigen::Matrix4d& outMat,
		Eigen::Affine3d& outMat,
		const Eigen::Spline3d& posSpline,
		const Eigen::Spline3d& normalSpline,
		double uvw[3]
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
	Eigen::Vector<T, N> lerpSampleMatrix(Eigen::MatrixX<T> arr, T t) {
		// sample array at a certain interval
		//float& a;

		if (t >= 1.0) {
			return arr.row(arr.rows()-1);
		}
		if (t <= 0.0) {
			return arr.row(0);
		}
		int a;
		int b;
		a = floor(arr.size() * t);
		b = a + 1;
		return lerp<Eigen::Vector<T, N>, T>(arr.row(a), arr.row(b), t - (arr.size() * t));
	}

	inline float getAngleAroundAxis(
		const Eigen::Vector3f& frameNormal,
		const Eigen::Vector3f& frameUp,
		const Eigen::Vector3f& v
	) { // where is the axis?
		// the axis is implied - this returns 0-1 depending on v's rotation around frameNormal x frameUp
		// test using only dots
		// i think this is the counterpart to atan2
		return 0.5 - sus(frameNormal.dot(v)) * 0.5 + float(frameUp.dot(v) > 0.0) * 0.5;
		//if (frameUp.dot(v) > 0.0) { // on the frame up side, should vary between 0 and 0.5
		//	return 0.5 - sus(frameNormal.dot(v)) * 0.5;  // at dot == 0.99 (vector almost on normal), angle should be near 0.0, 
		//}
		//return 1.0 - sus(frameNormal.dot(v)) * 0.5;
	}

}
