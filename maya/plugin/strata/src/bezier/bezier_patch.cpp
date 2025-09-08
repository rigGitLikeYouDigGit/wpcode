
#include "bezier_patch.h"

using namespace bez;
const Eigen::Matrix4f CubicBezierPatch::Mb = (Eigen::Matrix4f() <<
	-1, 3, -3, 1,
	3, -6, 3, 0,
	-3, 3, 0, 0,
	1, 0, 0, 0).finished();

void CubicBezierPatch::setControlPoint(
	int id, Eigen::Vector3f& v
) {
	int r = id / 4;
	int c = id % 4;
	setControlPoint(r, c, v);

	return;
}

void CubicBezierPatch::setControlPoint(
	int row, int col, Eigen::Vector3f& v
) {
	controlPointsAxes[0](row, col) = v.x();
	controlPointsAxes[1](row, col) = v.y();
	controlPointsAxes[2](row, col) = v.z();
	return;
}

CubicBezierPatch::CubicBezierPatch(std::vector<Eigen::Vector3f>& controlPoints) {
	for (int i = 0; i < 16; i++) {
		setControlPoint(i, controlPoints[i]);
	}
}
CubicBezierPatch::CubicBezierPatch(Eigen::Matrix<float, 16, 3>& controlPoints){
	/* TODO: figure out how to write Eigen functions properly, accepting blocks, dense objects etc*/
	for (int i = 0; i < 16; i++) {
		setControlPoint(i, Eigen::Vector3f(controlPoints.row(i)));
	}
}

Eigen::Vector3f CubicBezierPatch::pos(float u, float v) {
	/* eval matrix form of patch
	* taken from (I believe) Robert Fisher's pages for University of Edinburgh
	*/
	Eigen::Vector3f pos;
	Eigen::RowVector4f U(u * u * u, u * u, u, 1);
	Eigen::RowVector4f V(v * v * v, v * v, v, 1);
	for (int i = 0; i < 3; i++) {

		Eigen::Matrix4f& Gb = controlPointsAxes[i];
		pos(i) = U * Mb * Gb * Mb.transpose() * V.transpose();
	}
	return pos;

}

Eigen::Vector3f CubicBezierPatch::tanU(float u, float v) {
	Eigen::Vector3f tan;
	Eigen::RowVector4f U(3 * u * u, 2* u, 1, 0); // derivative wrt u from above
	Eigen::RowVector4f V(v * v * v, v * v, v, 1);
	for (int i = 0; i < 3; i++) {

		Eigen::Matrix4f& Gb = controlPointsAxes[i];
		tan(i) = U * Mb * Gb * Mb.transpose() * V.transpose();
	}
	return tan;
}
Eigen::Vector3f CubicBezierPatch::tanV(float u, float v) {
	Eigen::Vector3f tan;
	Eigen::RowVector4f U(u * u * u, u * u, u, 1);
	Eigen::RowVector4f V(3 * v * v, 2 * v, 1, 0); // derivative wrt v from above
	for (int i = 0; i < 3; i++) {

		Eigen::Matrix4f& Gb = controlPointsAxes[i];
		tan(i) = U * Mb * Gb * Mb.transpose() * V.transpose();
	}
	return tan;
}
Eigen::Vector3f CubicBezierPatch::normal(float u, float v) { // not normalised
	return tanU(u, v).cross(tanV(u, v));
}

void CubicBezierPatch::frame(float u, float v, Eigen::Affine3f& mat) {
	mat.translate(pos(u, v)).rotate(
		Eigen::Quaternionf::FromTwoVectors(
			tanU(u, v),
			tanV(u, v)
		)
	);


}
