#pragma once

#include "element.h"
#include "pointData.h"
#include "../AABB.h"

namespace strata {

	// data for discrete hard connections between elements
	struct SEdgeDriverData {
		/* struct for a single driver OF an edge - get tangent, normal and twist
		vectors for curve frame

		// do we just use NAN values to show arrays being unused?

		TODO: tension, param pinning etc?

		tangents are held in driver, not parent, since they affect

		*/
		int index = -1; // index of parent element
		Eigen::Vector3f uvn = { 0, 0, 0 }; // uvn coords of parent element to sample for points on this edge
		//float tan[3] = { NAN, 0, 0 }; // tangent of curve at this point - if left NAN is unused

		// tangents normally inline, unless continuity is not 1

		// tangents should be local to final matrix
		// they're NOT, they're GLOBAL for now, it was too complicated for first version

		Eigen::Vector3f baseTan = { 0, 0, 0 }; // vector from prev to next point
		Eigen::Vector3f prevTan = { -1, 0, 0 }; // tangent leading to point
		Eigen::Vector3f postTan = { 1, 0, 0 }; // tangent after point
		float normal[3] = { NAN, 0, 1 }; // normal of curve at this point - if left NAN is unused
		float orientWeight = 0.0; // how strongly matrix should contribute to curve tangent, vs auto behaviour
		float continuity = 1.0; // how sharply to break tangents - maybe just use this to scale tangents in?
		float twist = 0.0; // how much extra twist to add to point, on top of default curve frame
		Eigen::Affine3f finalMatrix = Eigen::Affine3f::Identity();

		float uOnEdge = 0.0; // u-value of this driver along output edge -
		/* can't always be inferred if one of the drivers is a surface*/

		inline Eigen::Vector3f pos() { return finalMatrix.translation(); }

		//inline Status& syncMatrix() {
		//	/* update final matrix from local driver tangents, offsets etc
		//	*/
		//}
	};

	// DRIVER datas are in space of the DRIVER
	// PARENT datas are in space of the PARENT
	// convert between them by multiplying out to world and back - 
	// parent datas always overlap drivers at some point

	/* so to generate a full curve,
	for each separate parent, we transpose driver points and vectors
	into that parent's space,
	then do the spline operations, get a dense vector of UVN parametres, and save that dense data as an SEdgeSpaceData.
	maybe we also cache the final result in world space.

	this is done on the op that first creates the edge and creates its data.

	on later iterations of the graph,
	if when a parent changes, we re-evaluate the UVNs of each SEdgeSpaceData.

	*/
	struct SEdgeSpaceData// : StaticClonable<SEdgeSpaceData> 
	{
		using thisT = SEdgeSpaceData;
		using T = SEdgeSpaceData;
		//DECLARE_DEFINE_CLONABLE_METHODS(thisT)

		int index = -1; // feels cringe to copy the index on all of these  
		// TEEECHNICALLLY this should be independent of any driver - 
		Eigen::ArrayXf weights; // per-dense-point weights for this parent
		Eigen::ArrayX3f cvs; // UVN bezier control points - ordered {pt, tanOut, tanIn, pt, tanOut...} etc
		bez::CubicBezierPath parentCurve; // curve in UVN space of parent, used for final interpolation

		Eigen::MatrixX3f finalNormals; // worldspace normals // hopefully smoothstep interpolation is good enough

		inline bez::ClosestPointSolver* closestSolver() {
			return parentCurve.getSolver();
		}

		void initEmpty() {
			/*initialise variables */
			weights = Eigen::ArrayXf();
			cvs = Eigen::ArrayX3f();
			//parentCurve = bez::CubicBezierPath();
			finalNormals = MatrixX3f();
		}
	};


	struct SEdgeData : SElData//, StaticClonable<SEdgeData> 
	{
		/* need dense final result to pick up large changes in
		parent space.

		Try "eA.bestP()" or "eA.p" in expressions to say "best-fitting point", leave it to operatioms to determine that point

		*/
		using thisT = SEdgeData;
		using T = SEdgeData;
		std::vector<SEdgeDriverData> driverDatas; // drivers of this edge
		std::vector<SEdgeSpaceData> spaceDatas; // curves in space of each driver

		//int denseCount = 10; // number of dense sub-spans in each segment
		/* TODO: adaptive by arc length? adaptive by screen size?
		*/

		bool closed = false;
		/* don't keep live splines, output from parent system etc -
		all temporary during construction
		posSpline is FINAL spline of all points on this edge
		*/

		/* for splitting edges and components, results will be
		clipped components with one master driver*/
		bool isClipped = false;

		Eigen::ArrayX3d uvnOffsets = {}; // final dense offsets should only be in space of final built curve?
		// maybe???? 

		//// IGNORE FOR NOW
		/// brain too smooth
		// surrender to ancestors
		// become caveman

		//Eigen::MatrixX3d finalPositions; // dense worldspace positions
		bez::CubicBezierPath finalCurve; // dense? final curve // DENSE
		Eigen::MatrixX3f finalNormals = {}; // worldspace normals 

		Eigen::MatrixX3f finalPoints = {}; // densely sampled final points in worldspace - use for querying

		int _bufferStartIndex = -1;


		inline bool isClosed() const {
			if (!driverDatas.size()) {
				return false;
			}
			return driverDatas[0].finalMatrix.translation().isApprox(
				driverDatas.back().finalMatrix.translation());
		}

		inline Eigen::Vector3f samplePos(const float t) {
			/* sample curve at
			*/
		}

		inline int densePointCount() {
			/* point count before resampling -
			curve has point at each driver, and (segmentPointCount) points
			in each span between them*/
			//return static_cast<int>(driverDatas.size() + denseCount * (driverDatas.size() - 1));
			if (isClosed()) {
				return static_cast<int>(ST_EDGE_DENSE_NPOINTS * (driverDatas.size()));
			}
			return static_cast<int>(ST_EDGE_DENSE_NPOINTS * (driverDatas.size() - 1));
		}

		inline int densePointCount() const {
			/* point count before resampling -
			curve has point at each driver, and (segmentPointCount) points
			in each span between them*/
			//return static_cast<int>(driverDatas.size() + denseCount * (driverDatas.size() - 1));
			if (isClosed()) {
				return static_cast<int>(ST_EDGE_DENSE_NPOINTS * (driverDatas.size()));
			}
			return static_cast<int>(ST_EDGE_DENSE_NPOINTS * (driverDatas.size() - 1));
		}

		inline int nSpans() {
			if (isClosed()) {
				return static_cast<int>(driverDatas.size());
			}
			return static_cast<int>(driverDatas.size()) - 1;
		}

		inline int nCVs() {
			// number of all cvs including tangent points
			return static_cast<int>((driverDatas.size()) * 3);
		}
		inline int nBezierCVs() {
			// number of cvs in use with bezier curves - basically shaving off start and end
			return static_cast<int>((driverDatas.size() - 1) * 3 + 2);
		}

		inline void rawBezierCVs(Eigen::Array3Xf& arr) {
			// ARRAY MUST BE CORRECTLY SIZED FIRST from nBezierCVs()

			//arr.resize(nBezierCVs());
			for (int i = 0; i < driverDatas.size(); i++) {
				if (i != 0) {
					arr.row(i * 3 - 1) = driverDatas[i].pos() + driverDatas[i].prevTan;
				}
				arr.row(i * 3) = driverDatas[i].pos();

				if (i != driverDatas.size() - 1) {
					arr.row(i * 3 + 1) = driverDatas[i].pos() + driverDatas[i].postTan;
				}
			}
		}

		inline void driversForSpan(const int spanIndex, SEdgeDriverData& lower, SEdgeDriverData& upper) {
			lower = driverDatas[spanIndex];
			upper = driverDatas[spanIndex + 1];
		}
		Status& buildFinalBuffers(Status& s);

		Status& getSubData(Status& s, SEdgeData& target, float lowU, float highU);


		aabb::AABB getAABB();
	};


}