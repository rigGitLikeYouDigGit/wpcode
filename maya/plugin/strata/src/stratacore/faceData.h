#pragma once

#include "types.h"
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

	
	/* TODO: it would certainly be convenient here to hold pointers to other types -
	obviously those break when we copy the manifold object.
	is it worth having a "repair()" function to go and replace them all after copy?
	*/


	struct SFaceSpaceData {

	};

	struct SubPatchData {
		/* save data for single subpatch - subpatches guaranteed to be
		* 4-sided patch, with arbitrary paths as borders
		* 
		* single subpatch has 4 "simple" 4-point bezier splines for borders, with offsets to 
		* the more complex full paths saved there?
		* 
		* need subpaths for each half of each original edge, probably held at fData level
		*/
		int subIndex = -1;
		int faceIndex = -1;
		int fBorderIndex = -1;


		//std::array<bez::CubicBezierPath*, 2> midPaths;
		//std::array<bez::BezierSubPath*, 2> edgeSubPaths;

		/* final worldspace borders
		* 0 is u, 1 is v
		*/
		std::array<bez::CubicBezierPath, 2> worldMidPaths;
		std::array<bez::BezierSubPath, 2> worldEdgePaths; //2 subpatch edges always taken from original edge curves

		/* keep separate edge resolutions for later when we can upres
		parts selectively? */
		std::array<int, 2> uRes = { 8, 8 };
		std::array<int, 2> vRes = { 8, 8 };

		/* probably need some rules on where to place points on borders,
		so they match up across patches
		*/

		SElement* fEl(StrataManifold& man);
		SFaceData& fData(StrataManifold& man);
		//SFaceData& fData(StrataManifold& man);
		void syncConnections(StrataManifold& man);

		/* functions working on smooth surface defined by curves
		*/
		Eigen::Vector3f evalUVSmooth(Eigen::Vector3f& uvn);
		Eigen::Vector3f evalUVSmooth(float u, float v);

		/* later need same support for dense result mesh - 
		consider how we might handle different resolutions of that mesh too

		surface construction done through layers of interpolated points in UV - 
		basic level is borders, points having UV bounds and positions -
		then maybe try and account for tangents of those points? 
		*/

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

		Affine3f centre; /* 
		x axis U
		y axis V
		z axis N
		(where applicable for quad faces)
		*/

		/* vector of corner vertex indices */
		//std::vector<std::vector<int>> vertices;
		std::vector<int> vertices;

		///* copying curves here so this struct is self contained
		//with everything we need to draw face.
		//might be excessive, change later if it helps
		//*/
		std::vector<bez::BezierSubPath> borderCurves = {};

		/* 
		save simple bezier splines to each half of each border, so we interpolate between 
		complex paths in curve space
		*/
		std::vector<std::array<bez::CubicBezierSpline, 2>> borderHalfSplines = {};
		/* control points in curve space, with their params along a simple cubic spline
		*/
		std::vector<std::array<Eigen::MatrixX3f, 2>> borderHalfLocalControlPoints = {};
		std::vector<std::array<Eigen::VectorXf, 2>> borderHalfLocalControlPointParams = {};


		/* mid curves of each edge, connecting at face centre*/
		std::vector<bez::CubicBezierPath> borderMidCurves = {};

		/* tangent vectors at midpoints of driver edges
		multiply and average to find centrePos
		multiply by how far?
		good question

		should rely separately on original frames along borders, 
		scaling frames by edge/face crease value
		*/
		std::vector<Affine3f> midEdgeFrames;

		std::vector<SubPatchData> subPatchdatas;

		aabb::AABB getAABB();

		int nBorderEdges() {
			return static_cast<int>(vertices.size());
		}

		inline Vector3f centreNormal() {
			/* return normal direction vector*/
			return centre.rotation() * Vector3f(0, 1, 0);
		}

		std::tuple<int, float, float> edgeSpan(StrataManifold& manifold, int borderIndex);

		std::pair<Vertex*, Vertex*> vtxPair(StrataManifold& man, int borderIndex);

		SEdgeData& eDataForBorder(StrataManifold& man, int borderIndex);

		float map01UCoordToEdge(StrataManifold& man, float u, int vtxA, int vtxB);
		float map01UCoordToEdge(StrataManifold& man, float u, int borderIndex);


		void syncConnections(StrataManifold& man);
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
