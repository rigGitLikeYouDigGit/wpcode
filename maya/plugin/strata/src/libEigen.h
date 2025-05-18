#pragma once

#include <vector>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include "MInclude.h"
#include "api.h"
#include "macro.h"
#include "status.h"

#include <unsupported/Eigen/Splines>

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

	template<typename MATRIX, typename T, int N=3>
	inline void setMatrixRow(MATRIX& mat, const int& rowIndex, const T* data) {
		for (int i = 0; i < N; i++) {
			mat(rowIndex, i) = data[i];
		}
	}

	template<typename T>
	inline bool makeFrame(Eigen::Matrix4<T>& frameMat,
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
		return true;
	}

	inline bool makeFrame(Eigen::Affine3d& frameMat,
		const Eigen::Vector3d& pos,
		const Eigen::Vector3d& tan,
		const Eigen::Vector3d& normal
	) {
		/* x is tangent,
		y is up,
		z is normal
		*/
		Eigen::Vector3d up = tan.cross(normal);
		Eigen::Vector3d normalZ = tan.cross(up);
		setMatrixRow(frameMat, 0, tan.data());
		setMatrixRow(frameMat, 1, up.data());
		setMatrixRow(frameMat, 2, normalZ.data());
		return true;
	}

	inline Eigen::Vector3d splineTan(const Eigen::Spline3d& sp, const double u) {
		return sp.derivatives(u, 1).col(0).matrix();
	}

	inline double closestParamOnSpline(const Eigen::Spline3d& sp, const Eigen::Vector3d pt,
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
		Eigen::Vector3d l, r;
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
		
		makeFrame(
			mat,
			posSpline(u).matrix(),
			posSpline.derivatives(u, 1).col(0).matrix(),
			normalSpline(u).matrix()
		);
		return s;
	}


	inline Eigen::ArrayXd arcLengthToParamMapping(const Eigen::Spline3d& sp, const int npoints = 20) {
		// return an array of equally-spaced points giving the 0-1 arc length to each point
		Eigen::ArrayXd result = Eigen::ArrayXd::Constant(npoints, 0.0);
		//Eigen::ArrayXXd data = Eigen::ArrayXXd::Constant(nRow, nCol, 1.0);
		Eigen::Vector3d prevpt = sp(0.0);
		Eigen::Vector3d thispt;
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


}
