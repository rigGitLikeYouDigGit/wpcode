#pragma once


#include "element.h"


namespace strata {

	struct SElData {
		int index = -1;
		std::string creatorNode;
	};

	struct SPointDriverData {
		/* may not be entirely irrelevant*/
		int index = -1;
		Vector3f uvn; // coord on driver for point position
	};

	// parent datas always relative in parent space - when applied, recover the original shape of element
	struct SPointSpaceData { // parent data FOR a point, driver could be any type
		//int index = -1;
		std::string name; // name of parent space element
		// has to be robust to storing/retrieving between graph iterations
		float weight = 1.0;
		Vector3f uvn = { 0, 0, 0 }; // uvn separate to offset in case point goes outside parent space area - 
		// eg if point goes off edge of space surface
		Affine3f offset = Eigen::Affine3f::Identity(); // translation is UVN, rotation is relative rotation from that position

		std::string strInfo();
	};

	struct SPointData : SElData {
		SPointDriverData driverData;
		std::vector<SPointSpaceData> spaceDatas = {}; // datas for each driver
		//MMatrix finalMatrix = MMatrix::identity; // final evaluated matrix in world space
		Eigen::Affine3f finalMatrix = Eigen::Affine3f::Identity(); // final evaluated matrix in world space
		//std::vector<std::string> nodeHistory; // each node that has affected this point, starting with creator

		std::string strInfo();
	};

}
