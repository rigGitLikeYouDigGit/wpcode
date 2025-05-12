#pragma once

#include <vector>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include "MInclude.h"
#include "api.h"
#include "macro.h"
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

	template<typename T>
	bool makeFrameTangentX(Eigen::Matrix4<T>& a_frame,
		const Eigen::Vector3<T>& a_location,
		const Eigen::Vector3<T>& a_tangentX, 
		const Eigen::Vector3<T>& a_normalY)
	{
		//Eigen::Vector3<T> sideZ = cross(a_tangentX, a_normalY);
		Eigen::Vector3<T> sideZ = a_tangentX.cross(a_normalY);
		//if (isZero(sideZ))
		if (sideZ)
		{
			return FALSE;
		}
		//normalize(sideZ);
		sideZ.normalize();
		//SpatialVector normalY = cross(sideZ, a_tangentX);
		Eigen::Vector3<T> normalY = sideZ.cross( a_tangentX);
		if (isZero(normalY))
		{
			return false;
		}
		//normalize(normalY);
		normalY.normalize();

		/*a_frame.direction() = a_tangentX;
		a_frame.left() = normalY;
		a_frame.up() = sideZ;
		a_frame.translation() = a_location;*/
		a_frame[0] = a_tangentX;
		a_frame[1] = normalY;
		a_frame[2] = sideZ;
		a_frame[3] = a_location;
		return true;
	}


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
