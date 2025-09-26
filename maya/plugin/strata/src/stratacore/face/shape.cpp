
#include "shape.h"
#include "../manifold.h"
#include "../libManifold.h"
#include "../../libEigen.h"

using namespace strata;

using namespace Eigen;


Status& makeTessellatedPoints(
	Status& s,
	int startURes, int endURes,
	int startVRes, int endVRes,
	std::vector<Vector2f>& uvs,
	std::vector<std::array<int, 4>>& faces
) {
	/* for uneven resolutions of start-end borders, find
	* arrangement of points that increase resolution gracefully over span of face
	
	for each axis we have
	- number of rows
	- number of darts PER row
	
	if we move in alternating axes, since each step might reduce the number of subsequent
	steps we have in the opposite axis?
	
	OR we somehow radiate out from u0v0?

	do one entire axis first, using most sparse count (for now) for opposite axis

	output uv coords on subpatch
	output face connectivity - use -1 in last entry in case of triangle

	thinking ahead to space trees, try global register for all of them used in a single application - 
	indexed by frame/context, then by separate name?

	AT HIGHER LEVEL, check if startU==endU and startV==endV - if so, face is simple quad topology

	*/
	int uMax = std::max(startURes, endURes);
	int vMax = std::max(startVRes, endVRes);
	uvs.reserve(uMax * vMax);
	faces.reserve(uMax * vMax);
	int uMin = std::min(startURes, endURes);
	int vMin = std::min(startVRes, endVRes);

	/* require at least 3 points (2 face loops) on each edge 
	TODO: is it right to cart round this border exclusion through tessellation
	logic? could probably take care of it separately at higher level, not sure if that's correct*/
	if (uMin < 3) {
		STAT_ERROR(s, "min U resolution less than 3, CANNOT TESSELLATE");
	}
	if( vMin < 3) {
		STAT_ERROR(s, "min V resolution less than 3, CANNOT TESSELLATE");
	}

	/* u axis first */
	int uDartsNeeded = abs(startURes - endURes);
	int nUMaxDartsPerStep = ceil(float(uDartsNeeded) / float(vMin - 2)); /* max number of darts for each step - final row may be uneven? */
	int vDartsNeeded = abs(startVRes - endVRes);
	int nVMaxDartsPerStep = ceil(float(vDartsNeeded) / float(uMin - 2)); 
	bool uMoreToLess = startURes > endURes;
	bool vMoreToLess = startVRes > endVRes;

	/* test instead using grid representing face strips -
	start out with as dense a grid as possible
	-1 denotes free-flowing edge, quad face
	positive integer points to the point in the next row that this face strip should collapse to
	-2 denotes dead/collapsed face strip
	-3 denotes joining diagonal vertices for 2 strips joining directly as a 45deg quad
	-4 denotes a face previous to this effect

	trying the diagonal / outwards approach again, starting at uMax0vMax0
	*/
	MatrixXi uFaceGrid(uMax - 1, vMax - 1);
	uFaceGrid.fill(-1);

	int coordU = 0;
	int coordV = 0;

	int nUFaceRanks = uMax - 1;
	int nVFaceRanks = vMax - 1;

	/* ranks may decrease during iteration */
	while ((coordU < nUFaceRanks) && (coordV < nVFaceRanks)) {
		coordU += 1;
		coordV += 1;
	}


	std::vector<int> uRankNPts(vMin); /* number of points per rank */
	std::vector<std::vector<int>> uRankNextPointIndices(vMin); /* for each point of each rank, give the point in the next  rank that this point will connect to (we align this to the shrinking direction so each input only connects to one
	*/


	
	/* all processing here is done in more to less direction - flip at the end if needed
	*/

	
	for (int uStep = 0; uStep < vMin; uStep++) {
		/* bookend rank numbers with consistent numbers of points */
		if (uStep == 0) {
			uRankNPts[0] = startURes;
			continue;
		}
		if (uStep == (vMin - 1)) {
			uRankNPts[uStep] = endURes;
			continue;
		}
		/* with each rank, reduce the number of darts needed */
		uRankNPts[uStep] = std::min(nMaxDartsPerStep, uDartsNeeded);

		uDartsNeeded -= nMaxDartsPerStep;
	}
	
	
	/* make actual coordinates in U, using number of */
	for (int uStep = 0; uStep < uRankNPts.size() - 1; uStep++) {
	}

}

Affine3f makeNewFaceCentre(
	StrataManifold& man,
	SFaceData& fData,
	SElement* el
) {
	/* for each border edge, look at its 2 adjacent edges
	
		for each adjacent edge, get its midpoint frame in the space of the end of the original edge
			transport that adjacent midpoint along border edge, to border midpoint
	average all of those transforms, maybe weight them somehow idk

	breaks on the tube example
	*/

	std::vector<Affine3f> transportMidpoints(fData.nBorderEdges() * 2);
	/* TODO: experiment weighting by closeness to midpoint?
	*/
	Eigen::VectorXf transportWeights(fData.nBorderEdges() * 2); 
	for (int i = 0; i < fData.nBorderEdges(); i++) {
		Affine3f startFrame = fData.borderCurves[i].frame(0.0);
		Affine3f midFrame = fData.borderCurves[i].frame(0.5);
		Affine3f endFrame = fData.borderCurves[i].frame(1.0);


		Affine3f prevMidFrame = fData.borderCurves[iPrev(i, fData.nBorderEdges())].frame(0.5);
		Affine3f nextMidFrame = fData.borderCurves[iNext(i, fData.nBorderEdges())].frame(0.5);
		transportMidpoints[2 * i] = startFrame.inverse() * prevMidFrame * midFrame;
		transportMidpoints[2 * i + 1] = endFrame.inverse() * nextMidFrame * midFrame;

		transportWeights(i * 2) = 1.0f;
		transportWeights(i * 2 + 1) = 1.0f;
		
	}
	return blendTransforms(transportMidpoints, transportWeights);

}

bez::CubicBezierPath makeBorderMidEdge(
	StrataManifold& man,
	SFaceData& fData,
	SElement* el,
	int borderIndex
) {
	/* build mid curve in worldspace for border edge -
	need to be as high-res as highest-res adjacent edge
	
	get control points of each border in space of simple bezier splines - 
	half-interpolate each to here.

	for each U coord, interp half along control point hulls of each edge,
	then transform to space of simple bezier to centre

	*/
	int borderNext = (borderIndex + 1) % fData.nBorderEdges();
	int borderPrev = (borderIndex - 1) % fData.nBorderEdges();

	bez::CubicBezierSpline& prevSpline = fData.borderHalfSplines[borderPrev][1];
	bez::CubicBezierSpline& nextSpline = fData.borderHalfSplines[borderNext][0];

	Eigen::MatrixX3f& prevLocalCtlPts = fData.borderHalfLocalControlPoints[borderPrev][1]; // towards start of this border
	Eigen::MatrixX3f& nextLocalCtlPts = fData.borderHalfLocalControlPoints[borderNext][0]; // away from end of this border

	int nSamplePoints = static_cast<int>(std::max(prevLocalCtlPts.rows(), nextLocalCtlPts.rows())); 

	/* build middle simple spline
	TODO: see if we can avoid flipping here, if centre ever rotates past 180
	or rather:
		if normal vector of either end matrix points directly at other end, what should happen?
	*/

	/* here for simple splines we take the normals given by connections of previous and next edges - 
	this might be fine? see if it gives wildly different behaviour than the 
	baked-in frames of the full edge curves
	*/
	MatrixX3f borderNormals(2, 3);
	VectorXf borderNormalUs(2);
	borderNormalUs(0) = 0.0f;
	borderNormalUs(1) = 1.0f;
	borderNormals.row(0) = -prevSpline.tangentAt(1.0);
	borderNormals.row(1) = nextSpline.tangentAt(0.0);

	Affine3f startMat;
	fData.borderHalfSplines[borderIndex][0].frame(
		1.0,
		borderNormals,
		borderNormalUs,
		{0.0, 1.0},
		startMat
		);

	/* Z axis is normal vector out from face*/
	Vector3f directV = fData.centre.translation() - startMat.translation();
	Vector3f startT = (startMat.rotation() * Vector3f(0, 0, 1)).cross(directV).cross((startMat.rotation() * Vector3f(0, 0, 1)));
	Vector3f endT = (fData.centre.rotation() * Vector3f(0, 0, 1)).cross(-directV).cross((fData.centre.rotation() * Vector3f(0, 0, 1)));
	bez::CubicBezierSpline midSpline = bez::CubicBezierSpline::fromPointsTangents(
		startMat.translation(), startT, fData.centre.translation(), endT
	);


	/* VERY SILLY for now, take average of sample on both curve hulls for position of control point.*/

	VectorXf sampleUs = VectorXf::LinSpaced(nSamplePoints, 0.0, 1.0);
	Eigen::MatrixX3f blendedCtlPts(nSamplePoints, 3);

	for (int i = 0; i < nSamplePoints; i++) {
		float t = float(i) / float(nSamplePoints - 1);
		int reverseI = nSamplePoints - i - 1;
		float reverseT = float(reverseI) / float(nSamplePoints - 1);
		
		/* blend point from incoming previous control points, and outgoing next control points
		TODO: Eigen recommends collating big expressions like this as much as possible
		*/
		Eigen::Vector3f localPos = (lerpSampleMatrix<float, 3>(
			prevLocalCtlPts, reverseT
		) +
			lerpSampleMatrix<float, 3>(
				nextLocalCtlPts, t
			)) / 2.0f;
		
		Affine3f midMat;
		midSpline.frame(
			t,
			borderNormals, borderNormalUs,
			{ 0.0f, 1.0f },
			midMat
		);
		blendedCtlPts.row(i) = midMat * localPos;
	}
	return bez::CubicBezierPath(blendedCtlPts);
}

SubPatchData makeSubPatchData(
	StrataManifold& man,
	SFaceData& fData,
	SElement* el,
	int borderIndex/*
	bez::BezierSubPath& uSubPath,
	bez::BezierSubPath& vSubPath,
	bez::CubicBezierPath& uMidPath,
	bez::CubicBezierPath& vMidPath*/
) {
	/* build the lowest half of subPatch for each border - 
	keep flags on whether we need to flip orientation
	of face later*/
	int nextBorder = (borderIndex + 1) % (fData.nBorderEdges());
	bez::BezierSubPath& u1SubPath = fData.borderCurves[borderIndex];  // forwards
	bez::CubicBezierPath& v1MidPath = fData.borderMidCurves[borderIndex]; // forwards
	bez::CubicBezierPath& u2MidPath = fData.borderMidCurves[nextBorder]; // backwards
	bez::BezierSubPath& v2SubPath = fData.borderCurves[nextBorder]; // backwards

	SubPatchData sData;
	sData.faceIndex = el->globalIndex;
	sData.fBorderIndex = borderIndex;

	return sData;

}

Status& strata::makeNewFaceData( /* */
	Status& s,
	StrataManifold& man,
	std::vector<int>& vertexPath,
	SFaceCreationParams& faceCreateParams, /* should this be packed in some other way?*/
	SElement*& el
) {
	SFaceData& fData = man.fDataMap[el->name];
	fData.vertices = vertexPath;
	
	/* reserve vectors to fit */
	fData.borderHalfLocalControlPoints.resize(fData.nBorderEdges());
	fData.borderHalfLocalControlPointParams.resize(fData.nBorderEdges());
	fData.borderHalfSplines.resize(fData.nBorderEdges());
	fData.midEdgeFrames.resize(fData.nBorderEdges());
	fData.borderMidCurves.resize(fData.nBorderEdges());
	fData.subPatchdatas.resize(fData.nBorderEdges());

	/* copy all edge paths and normal vectors */
	for (int i = 0; i < fData.nBorderEdges(); i++) {
		SEdgeData& eData = fData.eDataForBorder(man, i);
		fData.borderCurves.push_back(bez::BezierSubPath(eData.finalCurve));
		Vertex* vtxA = man.getVertex(fData.vertices[i]);
		Vertex* vtxB = man.getVertex(fData.vertices[iNext(i, fData.nBorderEdges())]);
		fData.borderCurves.back().uBounds[0] = vtxA->edgeUs[1];
		fData.borderCurves.back().uBounds[1] = vtxB->edgeUs[0];
		fData.borderCurves.back().reverse = !vtxA->edgeDirs[1];

		/* TODO: trim normal vectors at u values as well*/
		VectorXf normalUs = VectorXf::LinSpaced(eData.finalNormals.rows(), 0.0, 1.0);


		/* split each border in half to localise control points to curves later
		* 
		*/
		float midU = (vtxA->edgeUs[1] + vtxB->edgeUs[0]) / 2.0f;
		Vector3f midPos = eData.finalCurve.eval(midU);
		Vector3f midTan = eData.finalCurve.tangentAt(midU);

		/* get simple bezier curve between border end points*/
		Vector3f startPos = eData.finalCurve.eval(vtxA->edgeUs[1]);
		Vector3f startTan = eData.finalCurve.tangentAt(vtxA->edgeUs[1]);
		Vector3f endTan = -eData.finalCurve.tangentAt(vtxB->edgeUs[0]);
		Vector3f endPos = eData.finalCurve.eval(vtxB->edgeUs[0]);

		if (!vtxA->edgeDirs[1]) {
			Vector3f temp = startPos;
			startPos = endPos;
			endPos = temp;
			startTan = -startTan;
			endTan = -endTan;
			midTan = -midTan;
		}

		/* TEST having half-splines going TOWARDS midpoint of edge?
		or leave it to later functions to reverse
		*/

		/* first half of border edge */
		fData.borderHalfSplines[i][0] = bez::CubicBezierSpline::fromPointsTangents(
			startPos, startTan, midPos, -midTan);

		/* get control points of full-res half-path, in space of that simple bezier
		*/
		splitBezPath(
			eData.finalCurve, vtxA->edgeUs[1], midU
		).pointsInOtherCurveSpace(
				fData.borderHalfSplines[i][0], eData.finalNormals, normalUs,
				fData.borderHalfLocalControlPoints[i][0],
				fData.borderHalfLocalControlPointParams[i][0],
				bez::K_U_PARAM
			);

		/* second half of border edge*/
		fData.borderHalfSplines[i][1] = bez::CubicBezierSpline::fromPointsTangents(
			midPos, midTan, endPos, -endTan);
		splitBezPath(
			eData.finalCurve, midU, vtxB->edgeUs[0]
		).pointsInOtherCurveSpace(
				fData.borderHalfSplines[i][1], eData.finalNormals, normalUs,
				fData.borderHalfLocalControlPoints[i][1],
				fData.borderHalfLocalControlPointParams[i][1],
				bez::K_U_PARAM
			);
	}

	/* get centre point for face*/
	fData.centre = makeNewFaceCentre(
		man,
		fData,
		el
	);

	for (int i = 0; i < fData.nBorderEdges(); i++) {
		fData.borderMidCurves[i] = (
			makeBorderMidEdge(
				man,
				fData,
				el,
				i
			)
		);
	}
	for (int i = 0; i < fData.nBorderEdges(); i++) {
		fData.subPatchdatas[i] = (
			makeSubPatchData(
				man,
				fData,
				el,
				i
			)
		);
	}
	return s;
}