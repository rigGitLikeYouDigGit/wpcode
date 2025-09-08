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
		/* NAMING:
		
		edge : first-class Strata element
		border : section of edge contributing to face


		currently don't have any kind of SubCurve object decided,
		so sampling border sections of edges will be verbose
		*/


		//std::string name; // probably generated, but still needed for semantics?
		std::vector<SFaceDriverData> driverDatas; // drivers of this edge
		std::vector<SFaceSpaceData> spaceDatas; // curves in space of each driver

		Affine3f centre;
		/* normal at centre of face - all subpatch curves must end with this as their normal*/

		/* vector of corner vertex indices */
		//std::vector<std::vector<int>> vertices;
		std::vector<int> vertices;

		///* copying curves here so this struct is self contained
		//with everything we need to draw face.
		//might be excessive, change later if it helps
		//*/
		std::vector<bez::BezierSubPath> borderCurves = {};

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

		std::pair<Vertex*, Vertex*> vtxPair(StrataManifold& man, int borderIndex);

		SEdgeData& eDataForBorder(StrataManifold& man, int borderIndex);

		float map01UCoordToEdge(StrataManifold& man, float u, int vtxA, int vtxB);
		float map01UCoordToEdge(StrataManifold& man, float u, int borderIndex);
	};

	
	/* consider how we might store displacement - 
	* a regular grid for values would probably work, but could we just store "important"
	* details as individual data points?
	*/
	template<typename T>
	struct FacePrimVar {
		std::string name;

		Eigen::MatrixX2f coordPoints;

		std::vector<T> pointVals;
	};

}
