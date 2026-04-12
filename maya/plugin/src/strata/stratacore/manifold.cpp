

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
		posPointMap[vecKey] = static_cast<int>(points.size() - 1);
		auto ptr = &points.back();
		ptr->pos = worldPos;
		return ptr;
	}
	return nullptr;
}
//IntersectionPoint* IntersectionRecord::getPointByElUVN(int gId, Vector3f uvn, bool create) {
//	create = false; // don't have enough information to create passed into this function
//	Vector3i vecKey = toKey(uvn);
//	auto found = elUVNPointMap.find(gId);
//	if (found != elUVNPointMap.end()) {
//		auto foundPt = found->second.find(vecKey);
//		if (foundPt != found->second.end()) {
//			return &points[foundPt->second];
//		}
//	}
//	if (create) {
//		int newIdx = static_cast<int>(points.size());
//		points.emplace_back();
//		elUVNPointMap[gId][vecKey] = newIdx;
//		return &points[newIdx];
//	}
//	return nullptr;
//}

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

std::vector<
	std::pair<IntersectionPoint*, IntersectionCurve*>
> IntersectionRecord::getIntersectionsBetweenEls(int gIdA, std::vector<int> gIdsB) {
	/* return composite list of all intersections between gIdA and 
	all other ids given*/
	std::vector < std::pair<IntersectionPoint*, IntersectionCurve*>> result;
	for (auto gIdB : gIdsB) {
		/* for now, we don't allow self-intersections here, logic is too complicated for me*/
		if (gIdA == gIdB) {
			continue;
		}
		auto elResult = getIntersectionsBetweenEls(gIdA, gIdB);
		result.insert(result.end(), elResult.begin(), elResult.end());
	}
	return result;
}
void IntersectionRecord::_sortElMap() {
	/* sort el intersection map, at least for curves
	sort second layer of each based on their U coord on first layer edge
	
	*/

}

//std::string strata::SPoint::strInfo() {
//	return "<PSData " + name + " w:" + str(weight) + " uvn:" + str(uvn) + " offset:" + str(offset) + ">";
//}

std::string strata::SPoint::strInfo() {
	std::string result = "<PData anchor:" + str(globalIndex) + " ";
	for (auto& i : spaceDatas) {
		result += i.strInfo() + " ";
	}
	result += " mat: " + str(finalMatrix) + ">";
	return result;
}

Status& StrataManifold::mergeOther(StrataManifold& other, int mergeMode, Status& s) {
	/*given another manifold, merge it into this one
	* if names are found, update according to merge mode -
	*	MERGE_OVERWRITE - overwrite this graph's data with matching names in other
	*	MERGE_LEAVE - leave matching names as they are
	*
	*
	*/

	// add any elements not already known by name
	for(auto& otherElTie : other.nameGlobalIdMap) {
		SElement* otherEl = other.getEl(otherElTie.first);
		SElement* outEl = nullptr;
		switch (otherEl->elType) {
		case SElType::point: {
			addElement(*otherEl, outEl);
			break;
		}
		case SElType::edge: {
			addElement(*otherEl, outEl);
			break;
		}
		/*case SElType::face: {
			addElement(other.faces[elIdx], nullptr);
			break;
		}*/
		}
	}

	return s;
}

/* groups */
SGroup* StrataManifold::addGroup(StrataName& groupName, SElType elType=SElType::point) {
	SGroup* result = &groupMap[groupName];
	result->elT = elType;
	return result;
}

SGroup* StrataManifold::getGroup(StrataName& groupName) {
	auto found = groupMap.find(groupName);
	if (found == groupMap.end()) {
		return nullptr;
	}
	return &groupMap[groupName];
}

/* should groups hold a set of strings? */
//void StrataManifold::addToGroup(SGroup* grp, SElement* el) {
//	grp->elNames.insert(el->name);
//	el->groups.insert(grp->name);
//}
//void StrataManifold::removeFromGroup(SGroup* grp, SElement* el) {
//	grp->elNames.erase(el->name);
//	el->groups.erase(grp->name);
//}


//Status& StrataManifold::deleteGroup(Status& s, std::string& groupName) {
//	auto found = groupMap.find(groupName);
//	if (found == groupMap.end()) {
//		return s;
//	}
//	for (auto& elName : found->second.elNames) {
//		SElement* el = getEl(elName);
//		//el->groups.erase(std::find(el->groups.begin(), el->groups.end(), groupName));
//		el->groups.erase(groupName);
//	}
//	groupMap.erase(groupName);
//	return s;
//}


//Status& StrataManifold::renameGroup(Status& s, std::string& startName, std::string& newName, int mergeMode=MERGE_UNION) {
//	/* look for group name in map - 
//	if mode is union, elements are merged into original group */
//	auto found = groupMap.find(startName);
//	if (found == groupMap.end()) {
//		STAT_ERROR(s, "no base group found with name " + startName);
//	}
//	auto foundNew = groupMap.find(newName);
//	if (foundNew != groupMap.end()) {
//		switch (mergeMode) {
//		case MERGE_LEAVE: { /* don't overwrite anything, delete group we're trying to rename?*/
//			s = deleteGroup(s, startName);
//			return s;
//		}
//		case MERGE_OVERWRITE: { /* remove original group*/
//			s = deleteGroup(s, newName );
//			break;
//		}
//		case MERGE_UNION: { /* merge elements with original group */
//			for (auto& elName : found->second.elNames) {
//				SElement* el = getEl(elName);
//				//*(std::find(el->groups.begin(), el->groups.end(), startName)) = newName;
//				el->groups.erase(startName);
//				el->groups.insert(newName);
//			}
//			groupMap.erase(startName);
//			return s;
//		}
//		}
//	}
//	for (auto& elName : found->second.elNames) {
//		SElement* el = getEl(elName);
//		//*(std::find(el->groups.begin(), el->groups.end(), startName)) = newName;
//		el->groups.insert(newName);
//	}
//	found->second.name = newName;
//	groupMap.insert({ newName, found->second });
//	groupMap.erase(startName);
//	return s;
//}


static Status& pointPosAt(Status& s, Eigen::Vector3f& out, const SPoint& d, const Eigen::Vector3f& uvn) {
	out = d.finalMatrix * Eigen::Vector3f(uvn);
	return s;
}
static Status& edgePosAt(Status& s, Eigen::Vector3f& out, const SEdge& d, const Eigen::Vector3f& uvn) {
	/* as above, but just return position -
	may allow faster sampling in future

	UVN is (curve param, rotation from normal, distance from curve)
	*/

	// check if we need full matrix
	if (EQ(uvn[1], 0.0f) && EQ(uvn[2], 0.0f)) {
		if (d.spaceDatas.size()) {
			out = d.spaceDatas[0].domainCurve.eval(uvn[0]);
			return s;
		}
		out = d.finalCurve.eval(uvn[0]);
		return s;
	}
	Eigen::Affine3f curveMat;
	s = edgeDataMatrixAt(s, curveMat, d, uvn);
	out = curveMat.translation();
	return s;
}

Status& posAt(Status& s, Eigen::Vector3f& out, int globalIndex, Eigen::Vector3f& uvn) {
	/* as above, but just return position -
	may allow faster sampling in future*/
	SElement* el = getEl(globalIndex);
	switch (el->elType) {
	case SElType::point: {
		//SPointData& d = pointDatas.at(el->elIndex);
		SPoint& d = pDataMap.at(el->name);
		return pointPosAt(s, out, d, uvn);
		break;
	}
	case SElType::edge: {
		SEdge& d = eDataMap.at(el->name);
		return edgePosAt(s, out, d, uvn);
		break;
	}
	}
	return s;
}

static Status& pointMatrixAt(Status& s, Eigen::Affine3f& out, const SPoint& d, const Eigen::Vector3f& uvn) {
	LOG("pointMatrixAt: ");
	out = d.finalMatrix;
	//out.translate(uvn);
	out.translation() = out * uvn;
	return s;
}
static inline Status& edgeDataMatrixAt(Status& s, Eigen::Affine3f& out, const SEdge& d, const Eigen::Vector3f& uvn
) {/* if we don't cache a final dense curve for edge data,
	we would have to eval all domains here, then blend between them.

	we could also output the exact curve position? to save a sample?
	*/

	// first make frame
	Vector3f pos = d.finalCurve.eval(uvn(0));
	Vector3f tan = d.finalCurve.tangentAt(uvn(0), pos);

	// get normals
	int a; int b;
	float t = getArrayIndicesTForU(static_cast<int>(d.finalNormals.rows()), uvn(0), a, b);
	Vector3f normal = lerp<Vector3f>(d.finalNormals.row(a), d.finalNormals.row(b), smoothstepCubic(t));

	// make base frame
	s = makeFrame(s, out, pos,
		tan.normalized(),
		normal.normalized()
	);
	out.translation() = pos;

	// make angle axis describing twist around X
	AngleAxisf baseAA(uvn(1), Vector3f::UnitX());
	out.rotate(baseAA);
	// translate out along twisted normal in Z
	out.translate(Vector3f::UnitZ() * uvn(2));

	return s;
}
Status& matrixAt(Status& s, Eigen::Affine3f& outMat, SElement* el, Eigen::Vector3f uvn) {
	/* interpolate a spatial element to get a matrix in world space -
	look up UVN on given element, return that matrix
	*/
	LOG("matrixAt: " + el->name + " " + str(uvn));
	if (el == nullptr) {
		l("el is null");
		outMat = Affine3f::Identity();
		return s;
	}
	//SElement* el = getEl(globalIndex);
	switch (el->elType) {
	case (SElType::point): {
		//SPointData& d = pointDatas[el->elIndex];
		SPoint& d = pDataMap.at(el->name);
		return pointMatrixAt(s, outMat, d, uvn);
	}
	case (SElType::edge): {
		SEdge& d = eDataMap[el->name];
		//return edgeMatrixAt(s, out, el->elIndex, uvn);
		return edgeDataMatrixAt(s, outMat, d, uvn);
	}
	default: STAT_ERROR(s, "Cannot eval matrix at UVN for type " + std::to_string(el->elType));
	}
	return s;
}
Status& pointClosestMatrix(Status& s, Eigen::Affine3f& outMat, SPoint& d, const Eigen::Vector3f& worldVec) {
	//outMat = pointDatas[elIndex].finalMatrix;
	outMat = d.finalMatrix;

	return s;
}
Status& edgeClosestMatrix(Status& s, Eigen::Affine3f& outMat, SElement* el, const Eigen::Vector3f& worldVec) {
	/* localise matrix by point domain -
	how do we handle local rotations?
	get nearest point to curve
	*/
	//SEdgeData& d = edgeDatas[elIndex];
	SEdge& d = eDataMap[el->name];

	float u;
	Eigen::Vector3f tan;
	Eigen::Vector3f pos = d.finalCurve.getClosestPoint(
		worldVec,
		d.finalCurve.getSolver(), u,
		tan)
		;

	Eigen::Vector3f normal = lerpSampleMatrix<float, 3>(d.finalNormals, u);

	s = makeFrame<float>(s, outMat, pos, tan, normal);

	return s;
}
Status& edgeClosestMatrix(Status& s, Eigen::Affine3f& outMat, SElement* el, const Eigen::Affine3f& worldMat) {
	return edgeClosestMatrix(s, outMat, el, worldMat.translation());
}

Status& StrataManifold::pointGetUVN(Status& s, Eigen::Vector3f& outUVN, SPoint& d, const Eigen::Vector3f worldPos) {
	/* return UVN displacement from point matrix
	*/
	//LOG("pointGetUVN");
	outUVN = (d.finalMatrix.inverse() * worldPos);
	return s;
}

Status& StrataManifold::edgeGetUVN(Status& s, Eigen::Vector3f& uvn, SElement* el, const Eigen::Vector3f& worldVec) {
	/* NEED POLAR / CYLINDRICAL conversion for UVN
	*
	* for blending, we should probably blend along shortest paths positive or negative,
	or everything will move in spirals
	but let's save that for when we actually do blending
	*/

	// first get closest matrix on curve
	Eigen::Affine3f curveMat;
	s = edgeClosestMatrix(s, curveMat, el, worldVec);

	//SEdgeData& d = edgeDatas[elIndex];
	SEdge& d = eDataMap[el->name];

	float u;
	Eigen::Vector3f tan;
	Eigen::Vector3f pos = d.finalCurve.getClosestPoint(
		worldVec,
		d.finalCurve.getSolver(), u,
		tan)
		;
	uvn(0) = u;

	Eigen::Vector3f normal = lerpSampleMatrix<float, 3>(d.finalNormals, u);

	s = makeFrame<float>(s, curveMat, pos, tan, normal);

	uvn(1) = getAngleAroundAxis(
		curveMat * Eigen::Vector3f(0, 0, 1),
		curveMat * Eigen::Vector3f(0, 1, 0),
		(worldVec - curveMat.translation()).normalized()
	);
	uvn(2) = (worldVec - curveMat.translation()).norm();
	return s;
}

Status& closestMatrix(Status& s, Eigen::Affine3f& outMat, SElement* el, const Eigen::Vector3f closePos) {
	// localise a world transform into UVN coordinates in the space of given domain
	// make another function to return a full transform, for point domains
	if (el == nullptr) {
		outMat = Affine3f::Identity();
		return s;
	}
	//SElement* el = getEl(globalIndex);
	switch (el->elType) {
	case (SElType::point): {
		return pointClosestMatrix(s, outMat, pDataMap.at(el->name), closePos);
	}
	case (SElType::edge): {
		return edgeClosestMatrix(s, outMat, el, closePos);
	}
	default: STAT_ERROR(s, "Cannot eval matrix at UVN for type " + std::to_string(el->elType));
	}
	return s;
}



Status& pointGetUVN(Status& s, Eigen::Vector3f& outUVN, SPoint& d, const Eigen::Vector3f worldPos);


Status& edgeGetUVN(Status& s, Eigen::Vector3f& uvn, SElement* el, const Eigen::Vector3f& worldVec);


Status& getUVN(Status& s, Eigen::Vector3f& uvn, SElement* el, const Eigen::Vector3f closePos) {
	// localise a world transform into UVN coordinates in the space of given domain
	// make another function to return a full transform, for point domains

	switch (el->elType) {
	case (SElType::point): {
		return pointGetUVN(s, uvn, pDataMap.at(el->name), closePos);
	}
	case (SElType::edge): {
		return edgeGetUVN(s, uvn, el, closePos);
	}
	default: STAT_ERROR(s, "Cannot get UVN for type " + std::to_string(el->elType));
	}
	return s;
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
	SEdge& d = eDataMap[el->name];
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
		SEdge& edata = p.second;
		totalPositionEntries += edata.densePointCount();
		totalIndexEntries += edata.densePointCount() * 4;
	}
	Float3Array posResult(totalPositionEntries);
	IndexList indexResult(totalIndexEntries);
	int posStartIndex = 0;
	for (auto& p : eDataMap) {
		SEdge& edata = p.second;
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
		SEdge& edata = p.second;
		totalIndexEntries += edata.densePointCount() * 4;
	}
	IndexList indexResult(totalIndexEntries);
	int posStartIndex = 0;
	for (auto& p : eDataMap) {
		SEdge& edata = p.second;
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
		SEdge& edata = p.second;
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
		const SEdge& eData = eDataMap.at(indexMap[i]);
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
		SEdge& edata = p.second;
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
		const SEdge& eData = p.second;
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
		SEdge& eData = p.second;
		int n = 0;
		for (n; n < eData.densePointCount(); n++) {
			//result[eData._bufferStartIndex - _edgeVertexBufferEntryStart + n + i] = eData._bufferStartIndex + n;
			result[eData._bufferStartIndex - _edgeVertexBufferEntryStart + n] = eData._bufferStartIndex + n;
		}
		i += 1;
	}

	return result;
}

IndexList strata::StrataManifold::getWireframeEdgeVertexIndexList(Status& s, SEdge& eData) {
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

Status& strata::StrataManifold::pointSpaceMatrix(Status& s, Affine3f& outMat, SPoint& data) {
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

Status& strata::StrataManifold::computePointData(Status& s, SPoint& data//, bool doProjectToAnchors=false
) {
	/* given all space data is built, find final matrix
	*
	* DO WE PROJECT TO ANCHORS HERE???
	* no, it's only one call outside - be explicit in calling code
	*/
	LOG("compute point data: " + getEl(data.index)->name);
	l(data.strInfo());
	if (data.spaceDatas.size()) { // no domain, just literal data
		Affine3f spaceMat = Affine3f::Identity();
		s = pointSpaceMatrix(s, spaceMat, data);
		l("found space mat:" + str(spaceMat));
		//data.finalMatrix = spaceMat * data.finalMatrix;
		data.finalMatrix = spaceMat; // oh god oh god is this correct
		/* each space holds its own offset, so there should be no reason to
		pull from any existing data on main pData object */
	}
	//if (doProjectToAnchors) {
	//	l("projectingToAnchors");
	//	s = pointProjectToAnchors(s, data.finalMatrix, getEl(data.index));
	//}
	l("final matrix: " + str(data.finalMatrix));
	return s;
}

Status& strata::StrataManifold::pointProjectToAnchors(Status& s, Affine3f& mat, SElement* el) {
	/* project/snap given matrix to anchor of point
	(there should of course be a maximum of 1 anchor for a point)
	*/
	LOG("point project to anchors: " + el->name);
	SElement* anchorEl = getEl(el->anchors[0]);
	if (anchorEl == nullptr) {
		l("anchor ptr not found, skipping");
		return s;
	}
	switch (anchorEl->elType) {
	case SElType::point: {
		SPoint& anchorData = pDataMap[anchorEl->name];
		mat.translation() = anchorData.finalMatrix.translation();
		break;

	}
	}
	return s;

}

Status& strata::StrataManifold::edgeDomainDataFromAnchors(Status& s, SEdge& eData, SEdgeSpaceData& pData)
{
	/*Assumes edge data already has final anchors set up
	*
	*/
	int domainElIndex = getEl(pData.index)->elIndex;
	Eigen::Array3Xf cvs(eData.nBezierCVs(), 3);
	eData.rawBezierCVs(cvs);
	Eigen::Affine3f outMat;
	for (int i = 0; i < cvs.size(); i++) {

		//s = edgeInSpaceOf(s, cvs.row(i), domainElIndex, cvs.row(i));
	}

	return s;

}

Status& strata::StrataManifold::buildEdgeAnchors(Status& s, SEdge& eData) {
	/* build anchor matrices for this edge -
	* sample anchor elements, get worldspace matrices
	* DO NOT sample into domain space, that will be done when curve is eval'd
	*/
	Eigen::MatrixX3f anchorPoints(static_cast<int>(eData.anchorDatas.size()), 3);
	std::vector<float> inContinuities(anchorPoints.rows());


	// set base matrices on all points, eval anchor at saved UVN
	for (int i = 0; i < static_cast<int>(eData.anchorDatas.size()); i++) {
		matrixAt(s,
			eData.anchorDatas[i].finalMatrix,
			getEl(eData.anchorDatas[i].index),
			eData.anchorDatas[i].uvn
		);
		anchorPoints.row(i) = eData.anchorDatas[i].finalMatrix.translation();
		inContinuities[i] = eData.anchorDatas[i].continuity;
	}

	Eigen::MatrixX3f pointsAndTangents = cubicTangentPointsForBezPoints(
		anchorPoints,
		eData.closed,
		inContinuities.data()
	);

	eData.finalCurve = bez::CubicBezierPath(pointsAndTangents);

	/// TODO //// 
	//// resample these back into anchor's space? or no point since they'll be sampled into DOMAIN's space anyway
	// what is man talkin about
	int nCVs = static_cast<int>(eData.anchorDatas.size());
	for (int i = 0; i < nCVs; i++) {
		int thisI = (i * 3) % nCVs;
		int prevI = (i * 3 - 1 + nCVs) % nCVs;
		int nextI = (i * 3 + 1) % nCVs;
		eData.anchorDatas[thisI].prevTan = pointsAndTangents.row(prevI);
		eData.anchorDatas[thisI].postTan = pointsAndTangents.row(nextI);
	}


	return s;
}

Status& strata::StrataManifold::buildEdgeData(Status& s, SEdge& eData) {
	/* construct final dense array for data, assuming all domains and anchor indices are set in data

	build base curve matrices in worldspace,
	then get into space of each anchor

	but we can only work in worldspace when curve is freshly added, otherwise
	we can only save anchor-space versions
	*/

	// Bezier control points for each span

	s = buildEdgeAnchors(s, eData);

	s = eData.buildFinalBuffers(s);

	return s;
}


Status& strata::StrataManifold::buildFaceAnchors(Status& s, SFaceData& fData) {
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