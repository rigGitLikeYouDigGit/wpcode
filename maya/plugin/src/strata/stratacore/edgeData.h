#pragma once

#include "element.h"
#include "pointData.h"
#include "../AABB.h"


/* MESHING

why discuss polygons and meshing in the edge/curve file?
because this is where it starts. Each edge has to know all the discrete points required of it,
which may come from disparate smooth processes.
wow those sentences sure sound like AI, but this was actually pure man slop, no ai slop at all

"Requested" edge point : smooth process requests a discrete point at this parametre


this means final meshing has to be done all-at once;
THIS means that once one meshing pass is done, points of ALL EDGES THAT INFORM IT BECOME IMMUTABLE?

I think the right way is to do 2 passes - smooth, then discrete

*/


namespace strata {

	// data for discrete hard connections between elements

	/* 
	conventions:

	X : TANGENT

	Y : BINORMAL

	Z : NORMAL
	
	why?
	in surface setting, UVN still holds that Z is normal - 
	this way it's (somewhat) consistent between surface and edges


	need an overall fat thick juicy struct for general rich strata coord?


	*/
	struct SEdgeAnchorData {
		/* struct for a single anchor OF an edge - get tangent, normal and twist
		vectors for curve frame. saving exact tangent information may not be needed here.
		use in conjunction with SCoord describing single matrix

		anchors specified in world space, and for now exactly define upvector at this point

		TODO: tension, param pinning etc?

		*/
		int index = -1; // index of domain element


		// tangents inline, unless continuity is not 1

		Eigen::Vector3f prevTan = { -1, 0, 0 }; // tangent leading to point
		Eigen::Vector3f postTan = { 1, 0, 0 }; // tangent after point
		float normal[3] = { NAN, 0, 1 }; // normal of curve at this point - if left NAN is unused
		float orientWeight = 0.0; // how strongly matrix should contribute to curve tangent, vs auto behaviour
		float continuity = 1.0; // how sharply to break tangents - maybe just use this to scale tangents in?
		float twist = 0.0; // how much extra twist to add to point, on top of default curve frame

		float uOnEdge = 0.0; // u-value of this anchor along output edge -
		/* can't always be inferred if one of the anchors is a surface*/

	};

	// ANCHOR datas are in space of the ANCHOR
	// DOMAIN datas are in space of the DOMAIN
	// convert between them by multiplying out to world and back - 
	// domain datas always overlap anchors at some point

	/* so to generate a full curve,
	for each separate domain, we transpose anchor points and vectors
	into that domain's space,
	then do the spline operations, get a dense vector of UVN parametres, and save that dense data as an SEdgeSpaceData.
	maybe we also cache the final result in world space.

	this is done on the op that first creates the edge and creates its data.

	on later iterations of the graph,
	if when a domain changes, we re-evaluate the UVNs of each SEdgeSpaceData.

	if a curve spans multiple domains/anchors, we blend between separate copies in each space
	*/
	struct SEdgeSpaceData
	{
		using thisT = SEdgeSpaceData;
		using T = SEdgeSpaceData;

		ArrayXf weights; // per-dense-point weights for this domain
		//Eigen::ArrayX3f cvs; // UVN bezier control points - ordered {pt, tanOut, tanIn, pt, tanOut...} etc
		bez::CubicBezierPath domainCurve; // curve in UVN space of domain, used for final interpolation
		// sampling path 
		ArrayX3f positions; // UVN bezier control points - ordered {pt, tanOut, tanIn, pt, tanOut...} etc
		ArrayX3f normals; // worldspace normals // hopefully smoothstep interpolation is good enough

		inline bez::ClosestPointSolver* closestSolver() {
			return domainCurve.getSolver();
		}

		//void initEmpty() {
		//	/*initialise variables */
		//	weights = ArrayXf();
		//	///cvs = Eigen::ArrayX3f();
		//	//domainCurve = bez::CubicBezierPath();
		//	positions = ArrayX3f();
		//	normals = ArrayX3f();
		//}
	};

	struct DiscreteEdgePoint {
		/* probably make this its own element type*/
		int edge = -1;
		float u = 0.0;
		Eigen::Vector3f pos; /* should we duplicate / save positions here?*/
	};

	struct SEdge : SElement
	{
		/* need dense final result to pick up large changes in
		domain space.

		Try "eA.bestP()" or "eA.p" in expressions to say "best-fitting point", leave it to operations to determine that point

		so this final curve object is not actually a curve? just an array of matrices? 
		this tea is just hot leaf juice?

		*/
		using thisT = SEdge;
		using T = SEdge;
		std::vector<SEdgeAnchorData> anchorDatas; // anchors of this edge
		std::vector<SEdgeSpaceData> spaceDatas; // curves in space of each anchor



		bool closed = false;
		/* don't keep live splines, output from domain system etc -
		all temporary during construction
		posSpline is FINAL spline of all points on this edge
		*/

		ArrayX3f uvnOffsets = {}; // final dense offsets should only be in space of final built curve?
		// maybe???? 

		//// IGNORE FOR NOW
		/// brain too smooth
		// surrender to ancestors
		// become caveman

		Eigen::MatrixX3f finalPoints = {}; // densely sampled final points in worldspace - use for querying

		std::vector<int> vertices = {}; 
		bool _verticesSorted = false;
		/* LEAVE IT LIKE THIS,
		KEEP TRACK OF HOW IT'S USED IN TRAVERSAL.
		RESTRUCTURE IF BENEFICIAL
		*/
		/* TEST 
		should vertices be stored on edges by UVN? vertices mapping to output edges?
		iMap already has proper intersection information
		*/
		
		std::vector<DiscreteEdgePoint> discretePoints;
		bool discretePointsSorted = false;


		int subspaceAnchor = -1;

		int _bufferStartIndex = -1;

		/* space datas need direct access to scoords
		*/

		inline bool isClosed() const {
			if (!anchorDatas.size()) {
				return false;
			}
			return anchors[0].finalMat.translation().isApprox(
				anchors.end()->finalMat.translation());
		}

		inline Eigen::Vector3f samplePos(const float t) {
			/* sample curve at
			*/
		}

		inline int densePointCount() {
			/* point count before resampling -
			curve has point at each anchor, and (segmentPointCount) points
			in each span between them*/
			//return static_cast<int>(anchorDatas.size() + denseCount * (anchorDatas.size() - 1));
			if (isClosed()) {
				return static_cast<int>(ST_EDGE_DENSE_NPOINTS * (anchorDatas.size()));
			}
			return static_cast<int>(ST_EDGE_DENSE_NPOINTS * (anchorDatas.size() - 1));
		}

		inline int densePointCount() const {
			/* point count before resampling -
			curve has point at each anchor, and (segmentPointCount) points
			in each span between them*/
			//return static_cast<int>(anchorDatas.size() + denseCount * (anchorDatas.size() - 1));
			if (isClosed()) {
				return static_cast<int>(ST_EDGE_DENSE_NPOINTS * (anchorDatas.size()));
			}
			return static_cast<int>(ST_EDGE_DENSE_NPOINTS * (anchorDatas.size() - 1));
		}

		inline int nSpans() {
			if (isClosed()) {
				return static_cast<int>(anchorDatas.size());
			}
			return static_cast<int>(anchorDatas.size()) - 1;
		}

		inline int nCVs() {
			// number of all cvs including tangent points
			return static_cast<int>((anchorDatas.size()) * 3);
		}
		inline int nBezierCVs() {
			// number of cvs in use with bezier curves - basically shaving off start and end
			return static_cast<int>((anchorDatas.size() - 1) * 3 + 2);
		}

		inline void rawBezierCVs(Eigen::Array3Xf& arr) {
			// ARRAY MUST BE CORRECTLY SIZED FIRST from nBezierCVs()

			//arr.resize(nBezierCVs());
			for (int i = 0; i < anchorDatas.size(); i++) {
				if (i != 0) {
					arr.row(i * 3 - 1) = anchorDatas[i].pos() + anchorDatas[i].prevTan;
				}
				arr.row(i * 3) = anchorDatas[i].pos();

				if (i != anchorDatas.size() - 1) {
					arr.row(i * 3 + 1) = anchorDatas[i].pos() + anchorDatas[i].postTan;
				}
			}
		}

		inline void anchorsForSpan(const int spanIndex, SEdgeAnchorData& lower, SEdgeAnchorData& upper) {
			lower = anchorDatas[spanIndex];
			upper = anchorDatas[spanIndex + 1];
		}
		Status& buildFinalBuffers(Status& s);

		Status& getSubData(Status& s, SEdge& target, float lowU, float highU);

		void sortVertices(StrataManifold& manifold);

		aabb::AABB getAABB();
	};


}