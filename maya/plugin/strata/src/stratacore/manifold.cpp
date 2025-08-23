

#include "manifold.h"

//#include "../exp/expParse.h"

#include "../libEigen.h"

using namespace strata;


IntersectionPoint* IntersectionRecord::getPointByVectorPosition(Vector3f worldPos, bool create) {
	create = false;	// create explicitly outside this function
	Vector3i vecKey = toKey(worldPos);
	auto found = posPointMap.find(vecKey);
	if (found != posPointMap.end()) {
		return &points[found->second];
	}
	if (create) {
		points.emplace_back();
		posPointMap[vecKey] = points.size() - 1;
		auto ptr = &points.back();
		ptr->pos = worldPos;
		return ptr;
	}
	return nullptr;
}
IntersectionPoint* IntersectionRecord::getPointByElUVN(int gId, Vector3f uvn, bool create) {
	create = false; // don't have enough information to create passed into this function
	Vector3i vecKey = toKey(uvn);
	auto found = elUVNPointMap.find(gId);
	if (found != elUVNPointMap.end()) {
		auto foundPt = found->second.find(vecKey);
		if (foundPt != found->second.end()) {
			return &points[foundPt->second];
		}
	}
	if (create) {
		int newIdx = static_cast<int>(points.size());
		points.emplace_back();
		elUVNPointMap[gId][vecKey] = newIdx;
		return &points[newIdx];
	}
	return nullptr;
}
std::vector<
	std::pair<IntersectionPoint*, IntersectionCurve*>
> IntersectionRecord::getIntersectionsBetweenEls(int gIdA, int gIdB) {
	std::vector < std::pair<IntersectionPoint*, IntersectionCurve*>> result;
	auto foundA = elMap.find(gIdA);
	if (foundA == elMap.end()) {
		return result;
	}
	auto foundB = foundA->second.find(gIdB);
	if (foundB == foundA->second.end()) {
		return result;
	}

	result.resize(foundB->second.size());
	/* for each found int pair, replace with valid pointer or nullptr*/
	int i = 0;
	for (auto& p : foundB->second) {
		switch (p.second) {
		case Intersection::POINT: {
			result[i] = std::make_pair(&points[p.first], nullptr);
			break;
		}
		case Intersection::EDGE: {
			result[i] = std::make_pair(nullptr, &curves[p.first]);
			break;
		}
		}
		i += 1;
	}
	return result;
}


std::string strata::SPointSpaceData::strInfo() {
	return "<PSData " + name + " w:" + str(weight) + " uvn:" + str(uvn) + " offset:" + str(offset) + ">";
}

std::string strata::SPointData::strInfo() {
	std::string result = "<PData driver:" + str(driverData.index) + " ";
	for (auto& i : spaceDatas) {
		result += i.strInfo() + " ";
	}
	result += " mat: " + str(finalMatrix) + ">";
	return result;
}


Status& StrataManifold::intersectingElements(
	Status& s,
	SElement* el,
	std::vector<SElement*>& outEls
) {
	return s;
}

bool StrataManifold::elementsCouldIntersect(
	SElement* elA,
	SElement* elB
) {
	/* return true if it's topologically possible that some elements
	could intersect - still need to do fine checking later.
	
	Think of this like the broad phase of a collision check.
	Efficiency is not on the cards

	There are probably rules around dimensions here - 
	like an intersection is only possible if one is a direct driver,
	OR if a shared driver descends in dimension with each step

	IE
	curveA -> curveB
	curveA -> pt1 -> curveC // still possible

	curveA -> pt1 -> curveD -> curve C // not possible? unless they all align at exactly the same coords, do we allow that?
	*/
	
	/* if histories of elements do not intersect at all, then clearly
	* neither can they */

	if (std::find(elA->drivers.begin(), elA->drivers.end(), elB->globalIndex) != elA->drivers.end()) {
		return true;
	}
	if (std::find(elB->drivers.begin(), elB->drivers.end(), elA->globalIndex) != elB->drivers.end()) {
		return true;
	}
	std::vector<int> result;
	result.reserve(std::min(elA->drivers.size(), elB->drivers.size()));
	auto intersect = std::set_intersection(
		elA->drivers.begin(), elA->drivers.end(),
		elB->drivers.begin(), elB->drivers.end(),
		std::back_inserter(result)
	);
	if (!result.size()) { return false; }
	
	/* check if either appears in the history of the other*/

	// guaranteed to be sorted in order?
	/* higher dimensions need to be direct driver of end elements to guarantee
	intersection - 
	ie if 2 curves don't DIRECTLY lie on the same surface, there's no guarantee
	*/

	return true;
}

void getIntersections(
	SElement* elA, SElement elB
) {
	/* what forms could an intersection even take?
	
	point/point - 
		bool, yes or no
	edge/edge - 
		one or more points
		shorter curve, if a subcurve or combined curve

	face/face - 
		one or more edges
		one or more shared feature points?
		subsurface
	
	*/
}

Float3Array strata::StrataManifold::getWireframePointGnomonVertexPositionArray(Status& s) {
	/* return flat float3 array for gnomon positions only on points
	* each point has 4 coords - point itself, and then 0.1 units in x, y, z of that point
	*/
	//LOG("Wireframe gnomon pos array " + str(pDataMap.size()));
	Float3Array result(pDataMap.size() * 4);
	int i = 0;
	std::string name;
	for (auto& p : pointIndexGlobalIndexMap) {
		name = getEl(p.second)->name;
		Affine3f mat = pDataMap.at(name).finalMatrix;
		result[i * 4] = mat.translation();
		result[i * 4 + 1] = mat * Vector3f{ 1, 0, 0 };
		result[i * 4 + 2] = mat * Eigen::Vector3f{ 0, 1, 0 };
		result[i * 4 + 3] = mat * Eigen::Vector3f{ 0, 0, 1 };
		i += 1;
	}
	return result;
}
Status& strata::StrataManifold::getWireframePointGnomonVertexPositionArray(Status& s, Float3* outArr, int startIndex) {
	int i = 0;
	for (auto& p : pDataMap) {
		Affine3f mat = p.second.finalMatrix;
		outArr[i * 4 + startIndex] = mat.translation();
		outArr[i * 4 + 1 + startIndex] = mat * Vector3f{ 1, 0, 0 };
		outArr[i * 4 + 2 + startIndex] = mat * Eigen::Vector3f{ 0, 1, 0 };
		outArr[i * 4 + 3 + startIndex] = mat * Eigen::Vector3f{ 0, 0, 1 };
		i += 1;
	}
	return s;
}

IndexList strata::StrataManifold::getWireframePointIndexArray(Status& s) {
	/* return index array for point gnomons
	* intended to emit as separate lines, so half is duplication
	*/
	//LOG("Wireframe point index list array: " + str(pDataMap.size()));
	IndexList result(pDataMap.size() * 3 * 2);
	//int i = 0;
	std::string name;
	for (int i = 0; i < static_cast<int>(pDataMap.size()); i++) {
		result[i * 6] = i * 4;
		result[i * 6 + 1] = i * 4 + 1;
		result[i * 6 + 2] = i * 4;
		result[i * 6 + 3] = i * 4 + 2;
		result[i * 6 + 4] = i * 4;
		result[i * 6 + 5] = i * 4 + 3;
	}
	return result;
}

Status& strata::StrataManifold::getWireframeSingleEdgeGnomonVertexPositionArray(Status& s, Float3Array& outArr, SElement* el, int arrStartIndex) {
	/* TODO: cache this ffs, shouldn't sample every subcurve while drawing
	*/
	SEdgeData& d = eDataMap[el->name];
	int n;
	float u;
	Eigen::Affine3f aff;
	Eigen::Vector3f uvn;
	for (int i = 0; i < d.densePointCount(); i++) {
		n = arrStartIndex + (i * 4);
		u = 1.0f / float(d.densePointCount() - 1) * float(i);
		uvn[0] = u;
		uvn[1] = 0; uvn[2] = 0;
		s = edgeDataMatrixAt(s, aff, eDataMap[el->name], uvn);

		outArr[n] = d.finalCurve.eval(u);
		outArr[n + 1] = (aff * Eigen::Vector3f{ 1, 0, 0 }).data();
		outArr[n + 2] = aff * Eigen::Vector3f{ 0, 1, 0 };
		outArr[n + 3] = aff * Eigen::Vector3f{ 0, 0, 1 };
	}
	return s;
}

void strata::StrataManifold::setGnomonIndexList(unsigned int* result, unsigned int i) {
	result[i * 4] = i * 4;
	result[i * 4 + 1] = i * 4 + 1;
	result[i * 4 + 2] = i * 4;
	result[i * 4 + 3] = i * 4 + 2;
	result[i * 4 + 4] = i * 4;
	result[i * 4 + 5] = i * 4 + 3;
}

Float3Array strata::StrataManifold::getWireframeEdgeGnomonVertexPositionArray(Status& s) {
	// return all edge 
	int totalPositionEntries = 0;
	int totalIndexEntries = 0;
	for (auto& p : eDataMap) {
		SEdgeData& edata = p.second;
		totalPositionEntries += edata.densePointCount();
		totalIndexEntries += edata.densePointCount() * 4;
	}
	Float3Array posResult(totalPositionEntries);
	IndexList indexResult(totalIndexEntries);
	int posStartIndex = 0;
	for (auto& p : eDataMap) {
		SEdgeData& edata = p.second;
		SElement& el = elements[edata.index];
		getWireframeSingleEdgeGnomonVertexPositionArray(
			s,
			posResult,
			&el,
			posStartIndex
		);
		posStartIndex += edata.densePointCount() * 4;
		//setGnomonIndexList()
	}
	return posResult;
}

IndexList strata::StrataManifold::getWireframeEdgeGnomonVertexIndexList(Status& s) {
	int totalIndexEntries = 0;
	for (auto& p : eDataMap) {
		SEdgeData& edata = p.second;
		totalIndexEntries += edata.densePointCount() * 4;
	}
	IndexList indexResult(totalIndexEntries);
	int posStartIndex = 0;
	for (auto& p : eDataMap) {
		SEdgeData& edata = p.second;
		SElement& el = elements[edata.index];
		for (unsigned int n = 0; n < static_cast<unsigned int>(edata.densePointCount()); n++) {
			setGnomonIndexList(indexResult.data(), posStartIndex);
			posStartIndex += 6;
		}
		///posStartIndex += edata.densePointCount() * 6;
	}
	return indexResult;
}

Float3Array strata::StrataManifold::getWireframeEdgeVertexPositionArray(Status& s) {
	/* TODO: get positions of all edges in parallel
	is there a world where we go full SOA and just have a big array of edge positions, normals etc?
	
	*/
	int totalIndexEntries = 0;
	for (auto& p : eDataMap) {
		SEdgeData& edata = p.second;
		edata._bufferStartIndex = totalIndexEntries;
		totalIndexEntries += edata.densePointCount();
	}
	Float3Array result(totalIndexEntries);

	/* TODO: I don't like that you can't properly access a map by index - should we change the
	* storage for edge data?
	*/
	std::map<int, std::string> indexMap;
	int posStartIndex = 0;
	for(auto& p : eDataMap){
		//indexMap[p.first] = posStartIndex;
		indexMap[posStartIndex] = p.first;
		posStartIndex += 1;
	}
	// TODO: parallel this
	for (int i = 0; i < static_cast<int>(eDataMap.size()); i++) {
		/* get all positions for single edge*/
		const SEdgeData& eData = eDataMap.at(indexMap[i]);
		//const int arrStartIndex = i * eData.densePointCount();
		const int arrStartIndex = eData._bufferStartIndex;
		for (int pt = 0; pt < eData.densePointCount(); pt++) { // could also parallel this by curve segment
			float u = (1.0f / (eData.densePointCount() - 1)) * pt;
			const Vector3f uvn{ u, 0.0, 0.0 };
			Vector3f posVec; // converting between eigen and normal float3 is sad
			s = edgePosAt(s, posVec, eData, uvn);
			/* TODO: should we cache the final sampled points? seems like it might be
			worth it, computing this much just to draw probably isn't great */
			result[arrStartIndex + pt] = posVec;
		}
	}
	return result;
}

Status& strata::StrataManifold::getWireframeEdgeVertexPositionArray(Status& s, Float3* outArr, int startIndex) {
	std::vector<int> indexStartMap(eDataMap.size() + 1);
	indexStartMap[0] = 0;
	int edgeStartIndex = 0;
	int i = 0;
	for (auto& p : eDataMap) {
		SEdgeData& edata = p.second;
		edata._bufferStartIndex = edgeStartIndex + startIndex; // store start index on data for later index list
		indexStartMap[i] = edgeStartIndex; 
		edgeStartIndex += edata.densePointCount();
		/*indexStartMap[i] = edgeStartIndex;*/
		i += 1;
	}
	
	// cache indices on object
	_nEdgeVertexBufferEntries = edgeStartIndex;
	_edgeVertexBufferEntryStart = startIndex;

	//Float3Array result(totalIndexEntries);

	// TODO: parallel this
	//for (int i = 0; i < static_cast<int>(eDataMap.size()); i++) {
	//i = 0;
	for (auto& p : eDataMap){
		/* get all positions for single edge*/
		const SEdgeData& eData = p.second;
		SElement* el = getEl(eData.index);

		const int thisEdgeStartIndex = indexStartMap[el->elIndex];
		for (int pt = 0; pt < eData.densePointCount(); pt++) { // could also parallel this by curve segment
			float u = (1.0f / (eData.densePointCount() - 1)) * pt;
			const Vector3f uvn{ u, 0.0, 0.0 };
			Vector3f posVec; // converting between eigen and normal float3 is sad

			s = edgePosAt(s, posVec, eData, uvn);

			/// TEST DEBUG
			auto pair = eData.finalCurve.globalToLocalParam(u);
			//posVec = toEigConst(eData.finalCurve.splines_[pair.first].control_points_[pt % 4]);

			/* TODO: should we cache the final sampled points? seems like it might be
			worth it, computing this much just to draw probably isn't great */
			outArr[startIndex + thisEdgeStartIndex + pt] = posVec;
		}
		//i += 1;
	}
	return s;
}


IndexList strata::StrataManifold::getWireframeEdgeVertexIndexList(Status& s) {
	/* assume we emit each edge as a continuous line
	* 
	* seems line indices need to be dense - 
	*/
	// add entry for -1 after each curve
	//IndexList result(_nEdgeVertexBufferEntries + static_cast<int>(eDataMap.size()));
	IndexList result(_nEdgeVertexBufferEntries);
	int i = 0;
	for (auto& p : eDataMap) {
		SEdgeData& eData = p.second;
		int n = 0;
		for (n; n < eData.densePointCount(); n++) {
			//result[eData._bufferStartIndex - _edgeVertexBufferEntryStart + n + i] = eData._bufferStartIndex + n;
			result[eData._bufferStartIndex - _edgeVertexBufferEntryStart + n] = eData._bufferStartIndex + n;
		}
		i += 1;
	}

	return result;
}

IndexList strata::StrataManifold::getWireframeEdgeVertexIndexList(Status& s, SEdgeData& eData) {
	/*separate buffer for each edge - 
	* seems super wasteful
	*/
	IndexList result(eData.densePointCount());
	for (int n = 0; n < eData.densePointCount(); n++) {
		result[n] = eData._bufferStartIndex + n;
	}
	return result;
}

IndexList strata::StrataManifold::getWireframeEdgeVertexIndexListPATCH(Status& s) {
	/* assume we emit each edge as a continuous line
	*
	* seems line indices need to be dense -
	*/
	// add entry for -1 after each curve
	//IndexList result(_nEdgeVertexBufferEntries + static_cast<int>(eDataMap.size()));
	//IndexList result(_nEdgeVertexBufferEntries / ST_EDGE_DENSE_NPOINTS);
	//int i = 0;

	//for (auto& p : eDataMap) {
	//	
	//	SEdgeData& eData = p.second;
	//	result[i] = eData._bufferStartIndex;
	//	//int n = 0;
	//	//for (n; n < eData.densePointCount(); n++) {
	//	//	//result[eData._bufferStartIndex - _edgeVertexBufferEntryStart + n + i] = eData._bufferStartIndex + n;
	//	//	result[eData._bufferStartIndex - _edgeVertexBufferEntryStart + n] = eData._bufferStartIndex + n;
	//	//}
	//	
	//	//result[eData._bufferStartIndex - _edgeVertexBufferEntryStart + n + i ] = -1;
	//	//result[eData._bufferStartIndex - _edgeVertexBufferEntryStart + n + i ] = UINT_MAX;
	//	//result[eData._bufferStartIndex - _edgeVertexBufferEntryStart + n + i ] = UINT_MAX;
	//	//result[eData._bufferStartIndex - _edgeVertexBufferEntryStart + n + i ] = nan;
	//	i += 1;

	//	//result[eData._bufferStartIndex - _edgeVertexBufferEntryStart + n - 1] = -1;
	//	/*result[i * 2] = eData._bufferStartIndex;
	//	result[i * 2 + 1] = eData._bufferStartIndex + eData.densePointCount()-1;*/
	//}

	IndexList result(ST_EDGE_DENSE_NPOINTS);
	for (int i = 0; i < result.size(); i++) {
		result[i] = i;
	}

	return result;
}

Status& strata::StrataManifold::pointSpaceMatrix(Status& s, Affine3f& outMat, SPointData& data) {
	/* get space matrix for point space datas
	if space data EXISTS, it means something

	todo: do we actually need to store names in space datas?
	seems like we can just use integers, won't be dynamic unless we add a way
	to edit structure of graph in history
	*/
	LOG("pointSpaceMatrix");

	if (data.spaceDatas.size() == 0) {
		l("no space datas, space matrix is identity");
		outMat = Affine3f::Identity();
		return s;
	}
	if (data.spaceDatas.size() == 1) {
		l("single space data, look up UVN");
		auto spaceEl = getEl(data.spaceDatas[0].name);
		s = matrixAt(s, outMat, spaceEl, data.spaceDatas[0].uvn);
		outMat = outMat * data.spaceDatas[0].offset; // add space-specific, child-specific offset
		return s;
	}
	VectorXf weights(data.spaceDatas.size());
	std::vector<Affine3f> tfs(data.spaceDatas.size());
	for (int i = 0; static_cast<int>(data.spaceDatas.size()); i++) {
		auto spaceEl = getEl(data.spaceDatas[i].name);
		weights(i) = data.spaceDatas[0].weight;
		s = matrixAt(s, tfs[i], spaceEl, data.spaceDatas[i].uvn);
		tfs[i] = tfs[i] * data.spaceDatas[i].offset;
	}
	outMat = blendTransforms(tfs, weights);
	return s;
}

Status& strata::StrataManifold::computePointData(Status& s, SPointData& data//, bool doProjectToDrivers=false
) {
	/* given all space data is built, find final matrix
	*
	* DO WE PROJECT TO DRIVERS HERE???
	* no, it's only one call outside - be explicit in calling code
	*/
	LOG("compute point data: " + getEl(data.index)->name);
	l(data.strInfo());
	if (data.spaceDatas.size()) { // no parent, just literal data
		Affine3f spaceMat = Affine3f::Identity();
		s = pointSpaceMatrix(s, spaceMat, data);
		l("found space mat:" + str(spaceMat));
		//data.finalMatrix = spaceMat * data.finalMatrix;
		data.finalMatrix = spaceMat; // oh god oh god is this correct
		/* each space holds its own offset, so there should be no reason to
		pull from any existing data on main pData object */
	}
	//if (doProjectToDrivers) {
	//	l("projectingToDrivers");
	//	s = pointProjectToDrivers(s, data.finalMatrix, getEl(data.index));
	//}
	l("final matrix: " + str(data.finalMatrix));
	return s;
}

Status& strata::StrataManifold::pointProjectToDrivers(Status& s, Affine3f& mat, SElement* el) {
	/* project/snap given matrix to driver of point
	(there should of course be a maximum of 1 driver for a point)
	*/
	LOG("point project to drivers: " + el->name);
	SElement* driverEl = getEl(el->drivers[0]);
	if (driverEl == nullptr) {
		l("driver ptr not found, skipping");
		return s;
	}
	switch (driverEl->elType) {
	case SElType::point: {
		SPointData& driverData = pDataMap[driverEl->name];
		mat.translation() = driverData.finalMatrix.translation();
		break;

	}
	}
	return s;

}

Status& strata::StrataManifold::edgeParentDataFromDrivers(Status& s, SEdgeData& eData, SEdgeSpaceData& pData)
{
	/*Assumes edge data already has final drivers set up
	*
	*/
	int parentElIndex = getEl(pData.index)->elIndex;
	Eigen::Array3Xf cvs(eData.nBezierCVs(), 3);
	eData.rawBezierCVs(cvs);
	Eigen::Affine3f outMat;
	for (int i = 0; i < cvs.size(); i++) {

		//s = edgeInSpaceOf(s, cvs.row(i), parentElIndex, cvs.row(i));
	}

	return s;

}

Status& strata::StrataManifold::buildEdgeDrivers(Status& s, SEdgeData& eData) {
	/* build driver matrices for this edge -
	* sample driver elements, get worldspace matrices
	* DO NOT sample into parent space, that will be done when curve is eval'd
	*/
	Eigen::MatrixX3f driverPoints(static_cast<int>(eData.driverDatas.size()), 3);
	std::vector<float> inContinuities(driverPoints.rows());


	// set base matrices on all points, eval driver at saved UVN
	for (int i = 0; i < static_cast<int>(eData.driverDatas.size()); i++) {
		matrixAt(s,
			eData.driverDatas[i].finalMatrix,
			getEl(eData.driverDatas[i].index),
			eData.driverDatas[i].uvn
		);
		driverPoints.row(i) = eData.driverDatas[i].finalMatrix.translation();
		inContinuities[i] = eData.driverDatas[i].continuity;
	}

	Eigen::MatrixX3f pointsAndTangents = cubicTangentPointsForBezPoints(
		driverPoints,
		eData.closed,
		inContinuities.data()
	);

	eData.finalCurve = bez::CubicBezierPath(pointsAndTangents);

	/// TODO //// 
	//// resample these back into driver's space? or no point since they'll be sampled into PARENT's space anyway
	// what is man talkin about
	int nCVs = static_cast<int>(eData.driverDatas.size());
	for (int i = 0; i < nCVs; i++) {
		int thisI = (i * 3) % nCVs;
		int prevI = (i * 3 - 1 + nCVs) % nCVs;
		int nextI = (i * 3 + 1) % nCVs;
		eData.driverDatas[thisI].prevTan = pointsAndTangents.row(prevI);
		eData.driverDatas[thisI].postTan = pointsAndTangents.row(nextI);
	}


	return s;
}

Status& strata::StrataManifold::buildEdgeData(Status& s, SEdgeData& eData) {
	/* construct final dense array for data, assuming all parents and driver indices are set in data

	build base curve matrices in worldspace,
	then get into space of each driver

	but we can only work in worldspace when curve is freshly added, otherwise
	we can only save driver-space versions
	*/

	// Bezier control points for each span

	s = buildEdgeDrivers(s, eData);

	s = eData.buildFinalBuffers(s);

	return s;
}


Status& strata::StrataManifold::buildFaceDrivers(Status& s, SFaceData& fData) {
	return s;
}

Status& strata::StrataManifold::buildFaceData(Status& s, SFaceData& fData) {
	return s;
}


//std::vector<int> strata::StrataManifold::expValuesToElements(std::vector<ExpValue>& values, Status& s) {
//	/* resolve all possible values to elements */
//
//	std::vector<int> result;
//	if (!values.size()) { return result; }
//	LOG("expValuesToElements: " + str(values.size()));
//	for (size_t vi = 0; vi < values.size(); vi++) {
//		//for (auto& v : values) {
//		ExpValue& v = values[vi];
//		//l("check value:" + v.printInfo());
//		for (auto& f : v.numberVals) { // check for integer indices
//			int id = fToInt(f);
//			//SElement* ptr = manifold->getEl(id);
//			SElement* ptr = getEl(id);
//			if (ptr == nullptr) { // index not found in manifold
//				continue;
//			}
//			if (!seqContains(result, id)) { // add unique value found
//				result.push_back(id);
//			}
//		}
//		for (auto& s : v.stringVals) { // check for string names
//			// patterns will already have been expanded by top level
//			SElement* ptr = manifold->getEl(s);
//			if (ptr == nullptr) {
//				continue;
//			}
//			if (!seqContains(result, ptr->globalIndex)) { // add unique value found
//				result.push_back(ptr->globalIndex);
//			}
//		}
//	}
//	return result;
//}