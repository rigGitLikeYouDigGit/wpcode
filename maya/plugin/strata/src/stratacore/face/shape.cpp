
#include "shape.h"
#include "../manifold.h"
#include "../libManifold.h"
#include "../../libEigen.h"

using namespace strata;

using namespace Eigen;

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
	half-interpolate each to here

	*/
	int borderNext = (borderIndex + 1) % fData.nBorderEdges();
	int borderPrev = (borderIndex - 1) % fData.nBorderEdges();
	
	bez::BezierSubPath& prevCrv = fData.borderCurves[borderPrev];
	bez::BezierSubPath& nextCrv = fData.borderCurves[borderNext];
	/* VERY SILLY for now, take average of sample on both curve hulls for position of control point.*/

	int samplePoints = std::max(prevCrv.nSplines() * 3 + 1, nextCrv.nSplines() * 3 + 1) * 2; /* sample more densely*/
	MatrixX3f midCtlPts(samplePoints, 3);
	for (int i = 0; i < samplePoints; i++) {
		float u = float(i) / float(samplePoints - 1);
		midCtlPts.row(i) = (prevCrv.eval(u) + nextCrv.eval(u)) / 2.0;
	}
	return bez::CubicBezierPath(midCtlPts);
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
	/* copy all edge paths and normal vectors */
	for (int i = 0; i < fData.nBorderEdges(); i++) {
		SEdgeData& eData = fData.eDataForBorder(man, i);
		fData.borderCurves.push_back(bez::BezierSubPath(eData.finalCurve));
		Vertex* vtxA = man.getVertex(fData.vertices[i]);
		Vertex* vtxB = man.getVertex(fData.vertices[iNext(i, fData.nBorderEdges())]);
		fData.borderCurves.back().uBounds[0] = vtxA->edgeUs[1];
		fData.borderCurves.back().uBounds[1] = vtxB->edgeUs[0];
		fData.borderCurves.back().reverse = !vtxA->edgeDirs[1];

		/* split each border in half to localise control points to curves later
		* 
		*/
		float midU = (vtxA->edgeUs[1] + vtxB->edgeUs[0]) / 2.0f;
		Vector3f midPos = eData.finalCurve.eval(midU);
		Vector3f midTan = eData.finalCurve.tangentAt(midU);
		/* get simple bezier curve between border end points*/
		fData.borderHalfPaths[i][0] = splitBezPath(
			eData.finalCurve, vtxA->edgeUs[1], midU);
		fData.borderHalfPaths[i][1] = splitBezPath(
			eData.finalCurve,  midU, vtxB->edgeUs[0] );
	}

	/* get centre point for face*/
	fData.centre = makeNewFaceCentre(
		man,
		fData,
		el
	);

	for (int i = 0; i < fData.nBorderEdges(); i++) {
		fData.borderMidCurves.push_back(
			makeBorderMidEdge(
				man,
				fData,
				el,
				i
			)
		);
	}

}