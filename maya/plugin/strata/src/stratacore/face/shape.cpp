
#include "shape.h"
#include "../manifold.h"
#include "../libManifold.h"

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
	}

	/* get centre point for face*/
	fData.centre = makeNewFaceCentre(
		man,
		fData,
		el
	);

}