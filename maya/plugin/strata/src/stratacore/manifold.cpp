

#include "manifold.h"

#include "../libEigen.h"

using namespace ed;

std::string ed::SPointSpaceData::strInfo() {
	return "<PSData " + name + " w:" + str(weight) + " uvn:" + str(uvn) + " offset:" + str(offset) + ">";
}

std::string ed::SPointData::strInfo() {
	std::string result = "<PData driver:" + str(driverData.index) + " ";
	for (auto& i : spaceDatas) {
		result += i.strInfo() + " ";
	}
	result += " mat: " + str(finalMatrix) + ">";
	return result;
}

//
//inline Float3Array StrataManifold::getWireframeVertexPositionArray(Status& s) {
//	/* return flat float3 array for vector positions on points and curves
//	*
//	* order as [point positions, dense curve positions]
//	*
//	* each point has 4 coords - point itself, and then 0.1 units in x, y, z of that point
//	*/
//
//	Float3Array result(pointDatas.size() * 4 + edgeDatas.size() * CURVE_SHAPE_RES);
//
//	for (size_t i = 0; i < pointDatas.size(); i++) {
//		result[i * 4] = pointDatas[i].finalMatrix * MVector::zero;
//		result[i * 4 + 1] = pointDatas[i].finalMatrix * MVector::xAxis;
//		result[i * 4 + 2] = pointDatas[i].finalMatrix * MVector::yAxis;
//		result[i * 4 + 3] = pointDatas[i].finalMatrix * MVector::zAxis;
//	}
//
//	auto uParams = Eigen::VectorXd::LinSpaced(CURVE_SHAPE_RES, 0.0, 1.0);
//
//	// TODO: spline-interpolate in eigen 
//	for (size_t i = 0; i < edgeDatas.size(); i++) {
//		size_t curveStartIndex = pointDatas.size() * 4 + i * CURVE_SHAPE_RES;
//
//		for (int n; n < CURVE_SHAPE_RES; n++) {
//			/*result[curveStartIndex + n] = matrixAt(edgeIndexGlobalIndexMap[i], {uParams[n], 0.0, 0.0},
//			)*/
//			float uvn[3] = { static_cast<float>(uParams[n]), 0.0, 0.0 };
//			MVector posOut;
//			s = posAt(edgeIndexGlobalIndexMap[static_cast<int>(i)], uvn, posOut, s);
//			if (s) {
//				DEBUGSL("error sampling curve " + std::to_string(i) + "at point : " + std::to_string(n));
//				return result;
//			}
//			result[curveStartIndex + n] = posOut;
//		}
//	}
//	return result;
//}


//inline std::vector<int> SPoint::otherNeighbourPoints(StrataManifold& manifold) {
//	// return list of pointers to neighbours of the same element type
//	// return all points connected to edges here
//	std::vector<int> result;
//	result.reserve(8);
//	for (int edgeId : edges) {
//		StrataElement* edgePtr = manifold.getEl(edgeId);
//		for (int ptId : edgePtr->points) {
//			if (ptId == globalIndex) { continue; }
//			result.push_back(ptId);
//		}
//	}
//	return result;
//}
//inline std::vector<int> SPoint::otherNeighbourEdges(StrataManifold& manifold) {
//	// return list of pointers to neighbours of the same element type
//	// return all points connected to edges here
//	std::vector<int> result;
//	for (int edgeId : edges) {
//		result.push_back(edgeId);
//	}
//	return result;
//}
//
////inline std::vector<int> SPoint::edgeStar(StrataManifold& manifold) {
////	return otherNeighbourEdges(manifold);
////}
//
//inline std::vector<int> SPoint::otherNeighbourFaces(StrataManifold& manifold) {
//	// return list of pointers to neighbours of the same element type
//	// return all points connected to edges here
//	std::vector<int> result(6, -1);
//	std::set<int> foundFaces;
//	for (int edgeId : edges) {
//		StrataElement* edge = manifold.getEl(edgeId);
//		foundFaces.insert(edge->faces.begin(), edge->faces.end());
//	}
//	for (int faceId : foundFaces) {
//		result.push_back(faceId);
//	}
//	return result;
//}
//
//
//inline std::vector<int> SEdge::otherNeighbourPoints(StrataManifold& manifold) {
//	// return list of pointers to neighbours of the same element type
//	// return end points of this edge only, for now
//	//SmallList<int, 32> result(2);
//	std::vector<int> result;
//
//	if (manifold.getEl(drivers[0])->elType == SElType::point) {
//		result.push_back(drivers[0]);
//	}
//	if (manifold.getEl(drivers[1])->elType == SElType::point) {
//		result.push_back(drivers[1]);
//	}
//	return result;
//}
//
//inline std::vector<int> SEdge::otherNeighbourEdges(StrataManifold& manifold) {
//	std::vector<int> result;
//	result.reserve(8);
//	for (int& ptId : otherNeighbourPoints(manifold)) {
//		for (int& edgeId : ((manifold.getEl(ptId))->otherNeighbourEdges(manifold))) {
//			result.push_back(edgeId);
//		}
//	}
//	auto removeResult = std::remove(
//		result.begin(), result.end(), globalIndex); // remove this edge if it's added
//	return result;
//}


//struct SFace : StrataElement {
//	SElType elType = SElType::face;
//
//	std::vector<bool> edgeOrients; // SURELY vector<bool> is cringe
//	// true for forwards, false for backwards
//	bool flipFace; // if face as a whole should be flipped, after edge winding
//
//	SFace(std::string elName) : StrataElement(elName) {
//	}
//};



