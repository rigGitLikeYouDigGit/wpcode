

#pragma once
#ifndef ED_LIB_EIGEN

#include <cstdint>
#include <chrono>
#include <thread>
#include <algorithm>
#include <vector>
#include <math.h>

#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <unsupported/Eigen/MatrixFunctions>
#include <unsupported/Eigen/Splines>


#include "libEigen.h"


using namespace ed;

template <typename T>
//inline void rotateVector(const Eigen::Matrix4<T>& lhs, const Eigen::Vector3<T>& in,
void rotateVector(const Eigen::Matrix4<T>& lhs, const Eigen::Vector3<T>& in,
	Eigen::Vector3<T>& out)
{
	//FEASSERT(N > 2);
	/*fe::set(out, in[0] * lhs(0, 0) + in[1] * lhs(0, 1) + in[2] * lhs(0, 2),
		in[0] * lhs(1, 0) + in[1] * lhs(1, 1) + in[2] * lhs(1, 2),
		in[0] * lhs(2, 0) + in[1] * lhs(2, 1) + in[2] * lhs(2, 2));*/
	//out(0) = in[0] * lhs[0][0] + in[1] * lhs[0][1] + in[2] * lhs[0][2];
	//out(1) = in[0] * lhs[1][0] + in[1] * lhs[1][1] + in[2] * lhs[1][2];
	//out(2) = in[0] * lhs[2][0] + in[1] * lhs[2][1] + in[2] * lhs[2][2];
	out(0) = in(0) * lhs(0, 0) + in(1) * lhs(0, 1) + in(2) * lhs(0, 2);
	out(1) = in(0) * lhs(1, 0) + in(1) * lhs(1, 1) + in(2) * lhs(1, 2);
	out(2) = in(0) * lhs(2, 0) + in(1) * lhs(2, 1) + in(2) * lhs(2, 2);
	//return out;
}

template <typename T>
//inline Eigen::Matrix4<T> & translate(
Eigen::Matrix4<T>& translate(
	Eigen::Matrix4<T>& lhs,
	const Eigen::Vector3<T>& translation)
{
	//FEASSERT(N > 2);
	//Vector<N, T> rotated;
	Eigen::Vector3<T> rotated;
	rotateVector<T>(lhs, translation, rotated);

	//lhs.translation() += rotated;
	//lhs[3] += rotated;
	lhs.block<1, 3>(3, 0, 1, 3) += rotated;
	return lhs;
}

Status& ed::splineUVN(
	Status& s,
	//Eigen::Matrix4d& outMat,
	Eigen::Affine3d& outMat,
	const Eigen::Spline3d& posSpline,
	const Eigen::Spline3d& normalSpline,
	double uvw[3]
) {
	/* polar-like coords around spline
	u is parametre
	v is rotation
	w is distance
	*/
	uvw[0] = std::min(std::max(uvw[0], 0.0), 1.0);

	Eigen::Vector3d pos = posSpline(uvw[0]).matrix(); // we assume that normal is actually normal, no need to redo double cross here
	//auto derivatives = posSpline.derivatives<1>(uvw[0], 1);
	Eigen::Vector3d tan = posSpline.derivatives(uvw[0], 1).col(1).matrix();
	//Eigen::Array3d n = posSpline(uvw[1]);
	Eigen::Vector3d n = normalSpline(uvw[1]).matrix();

	
	s = makeFrame(s, 
		outMat.matrix(), pos, tan, n);
	if (!EQ(uvw[1], 0.0)) {
		/* rotate n times clockwise about tangent*/
		Eigen::AngleAxisd orient(uvw[1], tan);
		outMat *= orient;
	}
	if (!EQ(uvw[2], 0.0)) {
		/* rotate n times clockwise about tangent*/
		outMat.translate(Eigen::Vector3d{ 0.0, 0.0, uvw[2] });
	}

	return s;
}


Status& ed::splineUVN(
	Status& s,
	//Eigen::Matrix4d& outMat,
	Eigen::Affine3d& outMat,
	const Eigen::Spline3d& posSpline,
	const Eigen::Spline3d& normalSpline,
	double uvw[3]
) {
	/* polar-like coords around spline
	u is parametre
	v is rotation
	w is distance
	*/
	uvw[0] = std::min(std::max(uvw[0], 0.0), 1.0);

	Eigen::Vector3d pos = posSpline(uvw[0]).matrix(); // we assume that normal is actually normal, no need to redo double cross here
	//auto derivatives = posSpline.derivatives<1>(uvw[0], 1);
	Eigen::Vector3d tan = posSpline.derivatives(uvw[0], 1).col(1).matrix();
	//Eigen::Array3d n = posSpline(uvw[1]);
	Eigen::Vector3d n = normalSpline(uvw[1]).matrix();


	s = makeFrame(s,
		outMat.matrix(), pos, tan, n);
	if (!EQ(uvw[1], 0.0)) {
		/* rotate n times clockwise about tangent*/
		Eigen::AngleAxisd orient(uvw[1], tan);
		outMat *= orient;
	}
	if (!EQ(uvw[2], 0.0)) {
		/* rotate n times clockwise about tangent*/
		outMat.translate(Eigen::Vector3d{ 0.0, 0.0, uvw[2] });
	}

	return s;
}


#define FE_MSQ_METHOD 0

template <typename MATRIX>
class MatrixSqrt
{
public:
	MatrixSqrt(void) :
		m_iterations(1)
	{}

	void	solve(MATRIX& B, const MATRIX& A) const;

	//void	setIterations(U32 iterations) { m_iterations = iterations; }
	void	setIterations(UINT32 iterations) { m_iterations = iterations; }

private:
	UINT32		m_iterations;
};

template <typename MATRIX>
inline void MatrixSqrt<MATRIX>::solve(MATRIX& B, const MATRIX& A) const
{
#if FE_MSQ_DEBUG
	feLog("\nA\n%s\n", c_print(A));
#endif
	DEBUGSL("sqrt solve")
	MATRIX AA = A;
	MATRIX correction;
	correction.setIdentity();

	//* cancellation protection
	const bool fix0 = (fabs(AA(0, 0) + 1) < 1e-3);
	const bool fix1 = (fabs(AA(1, 1) + 1) < 1e-3);
	const bool fix2 = (fabs(AA(2, 2) + 1) < 1e-3);
	const bool fix = (fix0 || fix1 || fix2);

	DEBUGS("fix:" + std::to_string(fix));
	if (fix)
	{
		//Matrix<3, 4, F64> doubleY = AA;
		//Eigen::Matrix<double, 3, 4> doubleY = AA;
		Eigen::Matrix4d doubleY = AA;

		//Quaternion<F64> quat = doubleY;
		//Eigen::Quaternion<double> quat = doubleY;
		//Eigen::Quaternion<double> quat(Eigen::AngleAxisd(doubleY));
		//Eigen::Quaternion<double> quat(doubleY.);
		//Eigen::Matrix3d doubleYRot(doubleY.reshaped(3, 3));
		Eigen::Matrix3d doubleYRot(doubleY.block<3, 3>(0,0, 3, 3));
		Eigen::Quaternion<double> quat(doubleYRot);

		//F64 radians;
		double radians;
		//Vector<3, F64> axis;
		Eigen::Vector3d axis;
		//quat.computeAngleAxis(radians, axis);
		//Eigen::AngleAxisd angleAxis(quat);
		//Eigen::AngleAxisd angleAxis{ quat };
		Eigen::AngleAxisd angleAxis;
		angleAxis = quat;
		
		//quat.computeAngleAxis(radians, axis);
		radians = angleAxis.angle();
		axis = angleAxis.axis();

		const double tiny = 0.03;
		const double tinyAngle = tiny * radians;

		//Quaternion<F64> forward(0.5 * tinyAngle, axis);
		Eigen::Quaterniond forward(Eigen::AngleAxis<double>(0.5 * tinyAngle, axis));
		//const Matrix<3, 4, F64> correct64(forward);
		//const Eigen::Matrix<double, 3, 4> correct64(forward);
		//correction = forward.toRotationMatrix();
		DEBUGQuat("forward", forward.coeffs());
		Eigen::Matrix4d correction;
		correction.setIdentity();
		correction.block<3,3>(0, 0, 3, 3) = forward.toRotationMatrix();

		DEBUGMMAT("correction rot", toMMatrix(correction));
		//correction = correct64;
		//const SpatialVector forwardT = 0.5 * tiny * doubleY.translation();
		Eigen::Vector3d forwardT;
		forwardT = 0.5 * tiny * doubleY.block<1, 3>(3, 0, 1, 3);
		//forwardT = 0.5 * tiny * doubleY.block<3,3>(0, 3, 3, 1);
		DEBUGS("forwardT");
		DEBUGMV(forwardT);

		translate<double>(correction, forwardT);
		DEBUGMMAT("correction end", toMMatrix(correction));

		//Quaternion<F64> reverse(-tinyAngle, axis);
		Eigen::Quaterniond reverse(Eigen::AngleAxisd(-tinyAngle, axis));
		//const Matrix<3, 4, F64> tweak64(reverse);
		//const Eigen::Matrix4d tweak64(reverse);
		//MATRIX tweak = tweak64;
		MATRIX tweak;
		tweak.setIdentity();
		tweak.block<3,3>(0, 0, 3, 3) = reverse.toRotationMatrix();
		DEBUGMMAT("tweak:", toMMatrix(tweak));
		//const SpatialVector reverseT = -tiny * doubleY.translation();
		//const Eigen::Vector3d reverseT = -tiny * Eigen::Vector3d(doubleY.row(3));
		//Eigen::Vector3d reverseT;
		//reverseT.block<1, 3>(0, 0, 1, 3) = (doubleY.block<1, 3>(3, 0, 1, 3));
		Eigen::Vector3d reverseT(
			doubleY(3, 0),
			doubleY(3, 1),
			doubleY(3, 2)
		);
		reverseT *= -tiny;
		translate<double>(tweak, reverseT);

#if FE_MSQ_METHOD!=0
		correction(3, 3) = 1;
		tweak(3, 3) = 1;
#endif

		//		feLog("ty\n%s\nyt\n%s\n",
		//				c_print(tweak*AA),
		//				c_print(AA*tweak));

		AA = tweak * AA;


#if FE_MSQ_DEBUG
		feLog("\ntweak\n%s\n", c_print(tweak));
		feLog("\ncorrection\n%s\n", c_print(correction));
#endif
	}

	MATRIX Y[2];
	MATRIX Z[2];
	//U32 current = 0;
	UINT32 current = 0;

	Y[0] = AA;
	Y[1].setIdentity();
	DEBUGMMAT("Y[0]:", toMMatrix(Y[0]));
	//setIdentity(Z[0]);
	Z[0] = MATRIX::Identity();
	Z[1].setIdentity();
	

#if FE_MSQ_METHOD==0
	//* Denman-Beavers
	MATRIX invY;
	invY.setIdentity();
	MATRIX invZ;
	invZ.setIdentity();
#elif FE_MSQ_METHOD==1
	//* Meini
	Y[1] = Y[0];
	Y[0] = Z[0] - Y[1];		//* I-A'
	Z[0] = 2 * (Z[0] + Y[1]);	//* 2(I+A')

	MATRIX invZ;
#else
	//* Schulz

	MATRIX I3;
	setIdentity(I3);
	I3 *= 3;
#endif

	//U32 iteration;
	UINT32 iteration;
	for (iteration = 0; iteration < m_iterations; iteration++)
	{
		//U32 last = current;
		UINT32 last = current;
		current = !current;

#if FE_MSQ_DEBUG
		feLog("\n>>>> iteration %d\nY\n%s\nZ\n%s\n", iteration,
			c_print(Y[last]),
			c_print(Z[last]));
		feLog("Y*Y\n%s\nZ*Z\n%s\n",
			c_print(Y[last] * Y[last]),
			c_print(Z[last] * Z[last]));
#endif

#if FE_MSQ_METHOD==0

		//* Denman-Beavers (1976)

		/*invert(invY, Y[last]);
		invert(invZ, Z[last]);*/
		invY = Y[last].inverse();
		invZ = Z[last].inverse();

#if FE_MSQ_DEBUG
		feLog("invY\n%s\n",
			c_print(invY));
		feLog("invZ\n%s\n",
			c_print(invZ));
		feLog("Y+invZ\n%s\nZ+invY\n%s\n",
			c_print(Y[last] + invZ),
			c_print(Z[last] + invY));
#endif

		Y[current] = 0.5 * (Y[last] + invZ);
		Z[current] = 0.5 * (Z[last] + invY);

#elif FE_MSQ_METHOD==1

		//* Meini (2004)

		invert(invZ, Z[last]);

#if FE_MSQ_DEBUG
		feLog("invZ\n%s\n",
			c_print(invZ));
#endif

		Y[current] = -1 * Y[last] * invZ * Y[last];
		Z[current] = Z[last] + 2 * Y[current];

#else

		//* Schulz

		MATRIX I3ZY = I3 - Z[last] * Y[last];

		Y[current] = 0.5 * Y[last] * I3ZY;
		Z[current] = 0.5 * I3ZY * Z[last];

#endif
	}

#if FE_MSQ_METHOD==0
	//* Denman-Beavers
	MATRIX& R = Y[current];
#elif FE_MSQ_METHOD==1
	//* Meini
	MATRIX R = 0.25 * Z[current];
	R(3, 3) = 1;
#else
	//* Schulz
	MATRIX& R = Y[current];
	R(3, 3) = 1;
#endif

#if FE_MSQ_DEBUG
	feLog("\nA\n%s\n", c_print(A));
	if (fix)
	{
		feLog("\nA'\n%s\n", c_print(AA));
	}
	feLog("\nB'\n%s\nB'*B'\n%s\n", c_print(R), c_print(R * R));
#endif

	if (fix)
	{
		B = correction * R;

#if FE_MSQ_DEBUG
		feLog("\ncorrection\n%s\n", c_print(correction));
		feLog("\ncorrected\n%s\n", c_print(B));
		feLog("\nsquared\n%s\n", c_print(B * B));
#endif
	}
	else
	{
		B = R;
	}

#if FE_MSQ_VERIFY
	bool invalid = FALSE;
	MATRIX diff = AA - R * R;
	F32 sumR = 0.0f;
	F32 sumT = 0.0f;
	for (U32 m = 0; m < width(diff); m++)
	{
		U32 n;
		for (n = 0; n < height(diff) - 1; n++)
		{
			if (FE_INVALID_SCALAR(diff(m, n)))
			{
				invalid = TRUE;
			}
			sumR += fabs(diff(m, n));
		}
		if (FE_INVALID_SCALAR(diff(m, n)))
		{
			invalid = TRUE;
		}
		sumT += fabs(diff(m, n));
	}
#endif
#if FE_MSQ_VERIFY && FE_MSQ_DEBUG
	feLog("\ndiff\n%s\ncomponent sumR=%.6G\n", c_print(diff), sumR);
#endif
#if FE_MSQ_VERIFY
	if (invalid || sumR > FE_MSQ_MAX_ERROR_R || sumT > FE_MSQ_MAX_ERROR_T)
	{
		feLog("MatrixSqrt< %s >::solve"
			" error of %.6G,%.6G exceeded limit of %.6G,%.6G\n",
			FE_TYPESTRING(MATRIX).c_str(),
			sumR, sumT, FE_MSQ_MAX_ERROR_R, FE_MSQ_MAX_ERROR_T);
		feLog("\nA'\n%s\n", c_print(AA));
		feLog("\nB'\n%s\nB'*B'\n%s\n", c_print(R), c_print(R * R));

		feX("MatrixSqrt<>::solve", "failed to converge");
	}
#endif
}

// NOTE fraction: 23 bits for single and 52 for double
/**************************************************************************//**
	@brief solve B = A^^power, where A is a matrix

	@ingroup geometry

	The power can be any arbitrary real number.

	Execution time is roughly proportional to the number of set bits in
	the integer portion of the floating point power and a fixed number
	of iterations for the fractional part.

	The number of iterations used to compute of the fractional portion
	of the power can be changed.  The maximum error after each iteration
	is half of the previous iteration, starting with one half.  The entire
	integer portion of the power is always computed.
*//***************************************************************************/
template <typename MATRIX>
class MatrixPower
{
public:
	MatrixPower(void) :
		m_iterations(16) {}

	template <typename T>
	void	solve(MATRIX& B, const MATRIX& A, T a_power) const;

	void	setIterations(UINT32 iterations) { m_iterations = iterations; }

private:
	MatrixSqrt<MATRIX>	m_matrixSqrt;
	UINT32					m_iterations;
};

template <typename MATRIX>
template <typename T>
inline void MatrixPower<MATRIX>::solve(MATRIX& B, const MATRIX& A,
	T a_power) const
{
	T absolute = a_power;

#if MRP_DEBUG
	feLog("\nA\n%s\npower=%.6G\n", print(A).c_str(), absolute);
#endif

	const bool inverted = (absolute < 0.0);
	if (inverted)
	{
		absolute = -absolute;
	}

	UINT32 whole = UINT32(absolute);
	T fraction = absolute - whole;

#if MRP_DEBUG
	feLog("\nwhole=%d\nfraction=%.6G\n", whole, fraction);
#endif

	MATRIX R;
	//setIdentity(R);

	MATRIX partial = A;
	float contribution = 1.0;
	UINT32 iteration;
	for (iteration = 0; iteration < m_iterations; iteration++)
	{
		m_matrixSqrt.solve(partial, partial);
		contribution *= 0.5;
#if MRP_DEBUG
		feLog("\ncontribution=%.6G\nfraction=%.6G\n", contribution, fraction);
#endif
		if (fraction >= contribution)
		{
			R *= partial;
			fraction -= contribution;
		}
	}

	partial = A;
	while (whole)
	{
#if MRP_DEBUG
		feLog("\nwhole=%d\n", whole);
#endif
		if (whole & 1)
		{
			R *= partial;
		}
		whole >>= 1;
		if (whole)
		{
			partial *= partial;
		}
	}

#if MRP_VALIDATE
	bool invalid = FALSE;
	for (U32 m = 0; m < width(R); m++)
	{
		for (U32 n = 0; n < height(R); n++)
		{
			if (FE_INVALID_SCALAR(R(m, n)))
			{
				invalid = TRUE;
			}
		}
	}
	if (invalid)
	{
		feLog("MatrixPower< %s >::solve invalid results power=%.6G\n",
			FE_TYPESTRING(MATRIX).c_str(), a_power);
		feLog("\nA\n%s\n", print(A).c_str());
		feLog("\nB\n%s\n", print(R).c_str());

		feX("MatrixPower<>::solve", "invalid result");
	}
#endif

	if (inverted)
	{
		//invert(B, R);
		B = R.inverse();
	}
	else
	{
		B = R;
	}
}

MMatrixArray ed::curveMatricesFromDriverDatas(
	MMatrixArray controlMats, int segmentPointCount,
	int rootIterations
	) {
	/* interpolate rational-root matrices between drivers, and then add drivers and interpolated mats to result*/
	MMatrixArray result;
	//result.reserve(controlMats.length() + segmentPointCount * (controlMats.length() - 1));

	/* TODO: parallelise segments here if matrix roots are costly*/
	rootIterations = std::min(rootIterations, 1);
	for (unsigned int i = 0; i < (controlMats.length() - 1); i++) {
		result.append(controlMats[i]);

		DEBUGMMAT("ctlMat:", controlMats[i]);

		// get relative matrix from this point to the next
		Eigen::Matrix4d relMat = toEigen(controlMats[i].inverse() * controlMats[i + 1]);


		// get square root of matrix, for single midpoint; cubic for 2, etc
		/*Eigen::MatrixPower<Eigen::Matrix4d> relMatPower(relMat);

		result.append(toMMatrix(relMatPower(2)));*/

		//auto matRoot = relMat.sqrt();

		/*result.append(toMMatrix(matRoot));
		continue;*/

		//Eigen::Matrix4d sqrtResult;
		//sqr.solve(sqrtResult, relMat);
		//result.append(controlMats[i] * toMMatrix<Eigen::Matrix4d>(sqrtResult));
		
		//continue;

		//Eigen::MatrixPower<Eigen::Matrix4d> relMatPower(relMat);
		////auto step = relMatPower(1.0 / float(segmentPointCount + 1));
		//auto step = 0.5;
		//Eigen::Matrix4d sqrtResult = relMatPower(step);
		//
		//MMatrix sqrtMMat = toMMatrix(sqrtResult);
		//sqrtMMat = controlMats[i] * sqrtMMat * MMatrix::identity;

		//result.append(sqrtMMat);


		MatrixSqrt<Eigen::Matrix4d> sq;
		sq.setIterations(rootIterations);
		auto matResult = toEigen(MMatrix(controlMats[i]));	
		sq.solve(matResult, relMat);

		result.append(toMMatrix(matResult));


		DEBUGMMAT("mid mat:", result[result.length() - 1]); // NANs :(
		continue;

		

		// raise that root matrix to the same power as its segment point index
		segmentPointCount = 1;
		for (size_t n = 0; n < segmentPointCount; n++) {

			//result.push_back(controlMats[i] * toMMatrix<Eigen::Matrix4d>(
			/*result.append(controlMats[i] * toMMatrix<Eigen::Matrix4d>(
				relMatPower(float(n + 1) / float(segmentPointCount + 1))
			));*/

			//MatrixSqrt<Eigen::Matrix4d> sq;
			//sq.setIterations(rootIterations);
			//auto matResult = toEigen(MMatrix(controlMats[i]));
			//
			////sq.solve(matResult, toEigen(relMat));
			//sq.solve(matResult, relMat);

			//result.append(toMMatrix(matResult));
			//DEBUGMMAT("mid mat:", toMMatrix(matResult)); // NANs :(
			//continue;



			//Eigen::Matrix4d matResult;
			//MatrixPower<Eigen::Matrix4d> mp;
			//double ratio = float(n + 1) / float(segmentPointCount + 1);
			//mp.solve<double>(matResult, relMat, ratio);
			//result.append(controlMats[i] * toMMatrix<Eigen::Matrix4d>(
			//	matResult
			//));

		}

	}
	DEBUGMMAT("endMat", controlMats[controlMats.length() - 1]);
	//result.push_back(controlMats.back());
	result.append(controlMats[controlMats.length() - 1]);
	return result;
}


MPointArray ed::curvePointsFromEditPoints(
	MMatrixArray controlMats, int segmentPointCount
	//int rootIterations
) {
	return MPointArray();
}

MPointArray ed::curvePointsFromEditPointsAndTangents(
	MMatrixArray controlMats, int segmentPointCount
	//int rootIterations
) {
	return MPointArray();
}







#endif // !ED_LIB_EIGEN