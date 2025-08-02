

#include "manifold.h"

#include "../libEigen.h"

using namespace ed;

std::string ed::SPointSpaceData::strInfo() {
	return "<PSData " + name + " w:" + str(weight) + " uvn:" + str(uvn) + " offset:" + str(offset) + ">";
}

std::string ed::SPointData::strInfo() {
	std::string result = "<PData driver:" + str(driverData.index) + " ";
	for (auto& i : spaceDatas) {
		result += i.strInfo() + " ";
	}
	result += " mat: " + str(finalMatrix) + ">";
	return result;
}
inline Float3Array ed::StrataManifold::getWireframePointGnomonVertexPositionArray(Status& s) {
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
		//COUT << mat.matrix() << std::endl;
		result[i * 4] = mat.translation();
		//result[i * 4 + 1] = pDataMap.at(name).finalMatrix * Eigen::Vector3f{ 1, 0, 0 };
		result[i * 4 + 1] = mat * Vector3f{ 1, 0, 0 };
		result[i * 4 + 2] = mat * Eigen::Vector3f{ 0, 1, 0 };
		result[i * 4 + 3] = mat * Eigen::Vector3f{ 0, 0, 1 };
		i += 1;
	}
	return result;
}
inline IndexList ed::StrataManifold::getWireframePointIndexArray(Status& s) {
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

Status& ed::StrataManifold::getWireframeSingleEdgeGnomonVertexPositionArray(Status& s, Float3Array& outArr, SElement* el, int arrStartIndex) {
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

void ed::StrataManifold::setGnomonIndexList(unsigned int* result, unsigned int i) {
	result[i * 4] = i * 4;
	result[i * 4 + 1] = i * 4 + 1;
	result[i * 4 + 2] = i * 4;
	result[i * 4 + 3] = i * 4 + 2;
	result[i * 4 + 4] = i * 4;
	result[i * 4 + 5] = i * 4 + 3;
}

inline Float3Array ed::StrataManifold::getWireframeEdgeGnomonVertexPositionArray(Status& s) {
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

inline IndexList ed::StrataManifold::getWireframeEdgeGnomonVertexIndexList(Status& s) {
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

Float3Array ed::StrataManifold::getWireframeEdgeVertexPositionArray(Status& s) {
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
			float u = (1.0 / (eData.densePointCount() - 1)) * pt;
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

IndexList ed::StrataManifold::getWireframeEdgeVertexIndexList(Status& s) {
	/* assume we emit each edge as a continuous line
	*/
	IndexList result(eDataMap.size() * 2);
	
	for (auto& p : eDataMap) {
		SEdgeData& eData = p.second;
		result[i] = eData._bufferStartIndex;
		result[i + 1] = eData._bufferStartIndex + eData.densePointCount();
	}

}
