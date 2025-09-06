#pragma once

#include "element.h"
#include "pointData.h"
#include "edgeData.h"

namespace strata {

	/* TODO:
	probably unite all settings constants in a constexpr map?
	*/
	constexpr const int ST_FACE_CORNER_SQUARE = 0; /* close faces with rounded boundary curves*/
	constexpr const int ST_FACE_CORNER_ROUND = 1; /* close faces by adding a new point, maintaining a square corner*/

	struct SFaceDriverData {
		int index = -1; // index of driver component
		std::array<Vector3f, 2> uvns = { // full uvn vectors likely unnecessary
			Vector3f{0.0f, 0.0f, 0.0f},
			Vector3f{0.0f, 0.0f, 0.0f},
		};

	};

	struct SFaceSpaceData {

	};

	struct SubPatchData {
		/* save data for single subpatch -
		*/
		int subIndex = -1;
		int faceIndex = -1;
	};

	struct SFaceCreationParams {
		/* save options used to create a face/ face group
		*/
		std::string faceName;
		std::string creationStr; // source expression
		
		int cornerMode = ST_FACE_CORNER_ROUND;
	};

	struct SFaceData : SElData {
		//std::string name; // probably generated, but still needed for semantics?
		std::vector<SEdgeDriverData> driverDatas; // drivers of this edge
		std::vector<SEdgeSpaceData> spaceDatas; // curves in space of each driver

		Vector3f centrePos; // central point of this surface
		Vector3f centreNormal;
		/* normal at centre of face - all subpatch curves must end with this as their normal*/

		/* vector of corner vertex indices */
		//std::vector<std::vector<int>> vertices;
		std::vector<int> vertices;
		/* {index, startU, endU} of each edge defining face
		*/
		//std::vector<std::tuple<int, float, float>> edgeSpans;

		/* how much data should faces store in themselves? copy subcurves in here directly?
		otherwise we need to pass in a manifold object to look up edges from vertices - 
		is that fine?
		it's probably fine*/


		/* tangent vectors at midpoints of driver edges
		multiply and average to find centrePos
		multiply by how far?
		good question
		*/
		std::vector<Vector3f> midEdgeTangents;

		aabb::AABB getAABB();

		int nBorderEdges() {
			return static_cast<int>(vertices.size());
		}
		std::tuple<int, float, float> edgeSpan(StrataManifold& manifold, int borderIndex);
	};

}
