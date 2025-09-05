
#include "bezier_patch.h"

using namespace bez;
const Eigen::Matrix4f CubicBezierPatch::Mb = (Eigen::Matrix4f() <<
	-1, 3, -3, 1,
	3, -6, 3, 0,
	-3, 3, 0, 0,
	1, 0, 0, 0).finished();


Eigen::Vector3f CubicBezierPatch::pos(float u, float v) {
	/* might be faster to cache control point matrices*/
	Eigen::Vector3f pos;
	Eigen::Vector4f U(u * u * u, u * u, u, 1);
	Eigen::Vector4f V(v * v * v, v * v, v, 1);
	for (int i = 0; i < 3; i++) {
		Eigen::Matrix4f Gb = {
			controlPoints(0, i),
			controlPoints(1, i),
			controlPoints(2, i),
			controlPoints(3, i),
			controlPoints(4, i),
			controlPoints(5, i),
			controlPoints(6, i),
			controlPoints(7, i),
			controlPoints(8, i),
			controlPoints(9, i),
			controlPoints(10, i),
			controlPoints(11, i),
			controlPoints(12, i),
			controlPoints(13, i),
			controlPoints(14, i),
			controlPoints(15, i)
		};

		//float val = U.transpose().dot(Mb);
		//auto val = U.dot(Mb) ;
		//auto val = U * Mb ;
		//auto newU = U.transpose();
		
		//auto newU = U.reshaped<4, 1>();
		auto newU = U.reshaped(4, 1);
		//auto val = 

		//float val = U.dot(Mb.dot(Gb.dot(Mb.transpose().dot(V.transpose()))));
	}

}
