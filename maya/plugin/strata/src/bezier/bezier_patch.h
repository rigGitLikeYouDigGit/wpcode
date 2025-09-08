#pragma once


#include "bezier_curve.h"
#include <Eigen/Dense>


namespace bez {

	struct CubicBezierPatch {
		/* allow evaling position 
		and also passing in blocks of 16 float values.

		allow retrieving iso curves at U and V

		for exact normal take cross of tangent of iso curves at that U/V
		
		x(u, v) = U * Mb * Gbx * Mb^T * V^T
		y(u, v) = U * Mb * Gby * Mb^T * V^T
		z(u, v) = U * Mb * Gbz * Mb^T * V^T

		Mb = [
			-1,	3,	-3,	1
			3,	-6,	3,	0
			-3,	3,	0,	0
			1,	0,	0,	0
			]

		Gb is single dimension of each control point:
		[
			p00	p01	p02	p03
			p10	p11	p12	p13
			p20	p21	p22	p23
			p30	p31	32	p33
		]

		U = [ u^3, u^2, u, 1 ]
		V = [ v^3, v^2, v, 1 ]
		*/

		static const Eigen::Matrix4f Mb;


		//Eigen::Matrix<float, 16, 3> controlPoints;
		Eigen::Matrix4f controlPointsAxes[3];

		CubicBezierPatch(std::vector<Eigen::Vector3f>& controlPoints);
		CubicBezierPatch(Eigen::Matrix<float, 16, 3>& controlPoints);

		void setControlPoint(int id, Eigen::Vector3f& v);
		void setControlPoint(int row, int col, Eigen::Vector3f& v);

		Eigen::Vector3f pos(float u, float v);
		Eigen::Vector3f tanU(float u, float v);
		Eigen::Vector3f tanV(float u, float v);
		Eigen::Vector3f normal(float u, float v);
		//Eigen::Vector3f normal(float u, float v, Eigen::Vector3f prevPos);

		void frame(float u, float v, Eigen::Affine3f& mat);

	};

}

