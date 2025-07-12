#pragma once

#include <stdio.h>
#include "../MInclude.h"
#include "strataMayaLib.h"
#include "../strataop/mergeOp.h"
#include "strataOpNodeBase.h"
#include "../logger.h"
#include "../macro.h"
#include "../lib.h"
#include "../exp/expParse.h"

#include "strataShapeNode.h"


class StrataShapeUI : public MPxSurfaceShapeUI {
	/* this class is apparently only for viewport1,
	and deprecated in newer versions, but we still
	need a creator function to register it
	*/
public:
	static void* creator() {
		LOG("STRATA SHAPE UI CREATOR");
		return new StrataShapeUI;
	}
	StrataShapeUI() {}
	~StrataShapeUI() {}
	void getDrawRequests(const MDrawInfo& info,
		bool objectAndActiveOnly,
		MDrawRequestQueue& queue) override
	{
		LOG("STRATA UI DRAW REQUESTS");
		return;
	};

	virtual bool canDrawUV() const {
		return false;
	}
};

class StrataShapeGeometryOverride : public MHWRender::MPxGeometryOverride {
	/* we copy the drawn value here,
	so it can draw while a new value computes*/

public:

	// examples do this so I guess it's legal? Hold pointer to shape node on its draw override
	StrataShapeNode* shapeNodePtr = nullptr;
	MObjectHandle shapeNodeObjHdl;
	ed::StrataManifold manifold;

	bool selectablePoints = false;
	bool selectableEdges = false;
	bool selectableFaces = false;


	/*static const char* sActiveWireframeRenderItemName;
	static const char* sDormantWireframeRenderItemName;
	static const char* sShadedRenderItemName;*/

	static constexpr char* sStPointRenderItemName = "stPRI";
	static constexpr char* sStEdgeRenderItemName = "stERI";

	inline Status getShapeMObj(MObject& result) {
		Status s;
		if (!shapeNodeObjHdl.isAlive()) {
			result = MObject::kNullObj;
			STAT_ERROR(s, "shapeGeoOverride could not retrieve shape MObject");
		}
		result = shapeNodeObjHdl.object();
		return s;
	}

	//MString initialize(const MInitContext& initContext, MSharedPtr<MUserData>&)


	inline Status syncManifold() {
		/* eval manifold if its final out node is dirty*/
		LOG("SYNC MANIFOLD");
		Status s;
		ed::StrataOpGraph* graphP = shapeNodePtr->opGraphPtr.get();
		l("geo o got graphP, nNodes: " + ed::str(graphP->nodes.size()));
		if (graphP == nullptr) {
			STAT_ERROR(s, "graphPtr is null, returning");
		}
		//return s; // doesn't hang when we return here
		// if graph has no nodes in it, exit
		if (!graphP->hasOutputNode()) {
			l("geo o EXIT empty graph");
			manifold.clear();
				return s;
		}
		
		ed::StrataMergeOp* opPtr = shapeNodePtr->getStrataOp<StrataShapeNode>(shapeNodePtr->thisMObject());
		if (opPtr == nullptr) {
			STAT_ERROR(s, "geo o opPtr is null, returning");
		}


		l("geo o opPtr index: " + ed::str(opPtr->index) + "dirty: " + ed::str(opPtr->anyDirty())); 
		//return s; //doesn't hang
		graphP = shapeNodePtr->opGraphPtr.get();
		int outIndex = graphP->getOutputIndex();
		l("geo o outindex:" + std::to_string(outIndex));
		if (outIndex < 0) {
			l("geo o EXIT empty graph after adding op (how)");
				manifold.clear();
			return s;
		}

		if (opPtr->anyDirty()) {
			l("op ptr dirty, eval-ing op");
			s = graphP->evalGraph(s, outIndex);
			CWRSTAT(s, "error eval-ing manifold to draw with strataShape node");
			l("dirty eval done");
		}

		l("before get other manifold");
		l("geo o outindex after eval:" + std::to_string(graphP->_outputIndex));
		ed::StrataManifold& otherManifold = graphP->results[graphP->getOutputIndex()];
		l("got manifold after eval: " + ed::str(otherManifold.elements.size()) + " " + ed::str(otherManifold.pDataMap.size()));
		manifold = otherManifold;
		/*manifold.clear();
		l("geo o outindex after eval:" + std::to_string(graphP->_outputIndex));
		manifold = graphP->results[graphP->getOutputIndex()];*/
		return s;
	}

	void updateDG() override {
		/* check here if the linked shape node has dirty items*/
		LOG("UPDATE DG");
		if (!shapeNodePtr) {
			l("no shape node ptr in geo O, returning");
			return;
		}

		/* pull op output to ensure it gets eval'd*/
		MFnDependencyNode depFn(shapeNodePtr->thisMObject());

		/* directly getting input plug DOES force eval - 
		why doesn't this work with data handles?*/
		/*l("before get input plug ints");
		volatile int inInt = depFn.findPlug(StrataShapeNode::aStInput, true).elementByPhysicalIndex(0).asInt();
		l("input val:" + ed::str(inInt));*/
		l("before get output plug int");
		//volatile int outInt = depFn.findPlug(StrataShapeNode::aStOutput, false).asInt();
		volatile int outInt = depFn.findPlug(StrataShapeNode::aStOutput, true).asInt();


		l("got outInt:" + ed::str(outInt));
		Status s = syncManifold();
		l("synced manifold");
		if (s) {
			CWMSG(s, "error syncing manifold");
			return;
		}
		l("shape node manifold after sync: " + ed::str(manifold.elements.size()));
		//CWMSG(s, "Error on syncManifold in updateDG() for strataShape");
		return;
	}

	static MHWRender::MPxGeometryOverride* Creator(const MObject& obj)
	{
		LOG("GEO O CREATOR()");
		return new StrataShapeGeometryOverride(obj);
	}

	~StrataShapeGeometryOverride() override {}
	MHWRender::DrawAPI supportedDrawAPIs() const override {
		return MHWRender::kOpenGLCoreProfile;
	}


	void updateSelectionGranularity(const MDagPath& path,
		MSelectionContext& selectionContext
	) {
		/* this is actually really complicated, skip for now*/
	}



	virtual bool supportsEvaluationManagerParallelUpdate()	const {
		return true;
		//return false;
	}

	bool requiresGeometryUpdate() const {
		LOG("GEO O REQ GEOMETRY UPDATE()");
		return true;
	}  

	/* UI drawing - for now just dots on points, as well as the gnomon lines -
	just an experiment*/
	virtual bool hasUIDrawables()	const {
		/* only render ui points if in selection/component mode?
		*/
		//getFrameContext()->getSelectionInfo()
		return false;
	}
	virtual void addUIDrawables(const MDagPath& path,
		MUIDrawManager& drawManager,
		const MFrameContext& frameContext
	) {
		Status s;
		LOG("ADD UI DRAWABLES");
		return;
		drawManager.beginDrawable();
		drawManager.setPointSize(1.0);
		drawManager.points(manifold.getPointPositionArray<MPointArray>(s), false);
		CWMSG(s, "error getting point position array for addUiDrawables");
		drawManager.endDrawable();
	}

	inline MIndexBufferDescriptor getCurveIndexBufferDescriptor() {
		/* todo:
		could probably drop down to uint16 here, unlikely we'll have so many points for edges alone*/
		LOG("get curve buffer descriptor()")
			return MIndexBufferDescriptor(MIndexBufferDescriptor::kEdgeLine, // may also be kHullEdgeLine
				"stEdgeIBD",
				MGeometry::kLineStrip,
				ed::StrataManifold::CURVE_SHAPE_RES
			);
	}

	inline MVertexBufferDescriptor getCurvePositionVertexBufferDescriptor() {
		LOG("get curve buffer descriptor()");
		return MVertexBufferDescriptor(
			"stEdgePosVBD",
			MGeometry::kPosition,
			MGeometry::kFloat,
			3
		);
	}

	inline std::string paramTypeName(MShaderInstance::ParameterType t) {
		switch (t)
		{
		case MHWRender::MShaderInstance::kInvalid:
			return ("'Invalid', ");
			break;
		case MHWRender::MShaderInstance::kBoolean:
			return ("'Boolean', ");
			break;
		case MHWRender::MShaderInstance::kInteger:
			return ("'Integer', ");
			break;
		case MHWRender::MShaderInstance::kFloat:
			return ("'Float', ");
			break;
		case MHWRender::MShaderInstance::kFloat2:
			return ("'Float2', ");
			break;
		case MHWRender::MShaderInstance::kFloat3:
			return ("'Float3', ");
			break;
		case MHWRender::MShaderInstance::kFloat4:
			return ("'Float4', ");
			break;
		case MHWRender::MShaderInstance::kFloat4x4Row:
			return ("'Float4x4Row', ");
			break;
		case MHWRender::MShaderInstance::kFloat4x4Col:
			return ("'Float4x4Col', ");
			break;
		case MHWRender::MShaderInstance::kTexture1:
			return ("'1D Texture', ");
			break;
		case MHWRender::MShaderInstance::kTexture2:
			return ("'2D Texture', ");
			break;
		case MHWRender::MShaderInstance::kTexture3:
			return ("'3D Texture', ");
			break;
		case MHWRender::MShaderInstance::kTextureCube:
			return ("'Cube Texture', ");
			break;
		case MHWRender::MShaderInstance::kSampler:
			return ("'Sampler', ");
			break;
		default:
			return ("'Unknown', ");
			break;
		}
	}

	void updateRenderItems(const MDagPath& path, MHWRender::MRenderItemList& renderItems) override {
		/* largely copied from the geometryOverrideExample2 in the maya devkit
		*
		* we have 3 kinds of render items for now:
		* all points
		* each edge
		* each sub-patch
		*/
		LOG("UPDATE RENDER ITEMS: " + std::string(path.fullPathName().asChar()) + " " + ed::str(renderItems.length()));
		if (!path.isValid()) {
			l("returning invalid path");
			return;
		}
		MRenderer* renderer = MRenderer::theRenderer();
		if (!renderer) {
			l("returning no renderer");
			return;
		}
		const MShaderManager* shaderManager = renderer->getShaderManager();
		if (!shaderManager) {
			l("returning no shader manager");
			return;
		}
		MStatus ms(MS::kSuccess);

		// Get the inherited DAG display properties.
		auto wireframeColor = MHWRender::MGeometryUtilities::wireframeColor(path);
		auto displayStatus = MHWRender::MGeometryUtilities::displayStatus(path);
		// Update the wireframe render item used when the object will be selected
		bool isWireFrameRenderItemEnabled = displayStatus == MHWRender::kLead || displayStatus == MHWRender::kActive;

		MGeometry::DrawMode drawMode = MGeometry::kAll;
		unsigned int depthPriority = MHWRender::MRenderItem::sSelectionDepthPriority;
		MColor color = wireframeColor;
		bool isEnable = true; // isWireFrameRenderItemEnabled

		///////// render item for points
		MHWRender::MRenderItem* renderItem = nullptr;
		// Try to find the active wireframe render item.
		// If the returning index is smaller than 0, that means 
		// the render item does't exists yet. So, create it.
		int renderItemIndex = renderItems.indexOf(sStPointRenderItemName);
		l("point render item index:" + std::to_string(renderItemIndex));
		if (renderItemIndex < 0)
		{
			l("create render item: " + ed::str(renderItemIndex));
			// Create the new render item with the given name.
			// We designate this item as a UI "decoration" and will not be
			// involved in rendering aspects such as casting shadows
			renderItem = MHWRender::MRenderItem::Create(sStPointRenderItemName,
				MHWRender::MRenderItem::DecorationItem,
				MHWRender::MGeometry::kLines
			);
			// We want this render item to show up when in all mode ( Wireframe, Shaded, Textured and BoundingBox)
			//renderItem->setDrawMode(MGeometry::kWireframe);
			renderItem->setDrawMode(MGeometry::kAll);
			// Set selection priority: on top of everything
			renderItem->depthPriority(depthPriority);
			// Get an instance of a 3dSolidShader from the shader manager.
			//MShaderInstance* shader = shaderManager->getStockShader(MShaderManager::k3dSolidShader);
			MShaderInstance* shader = shaderManager->getStockShader(MShaderManager::k3dCPVSolidShader);
			//MShaderInstance* shader = shaderManager->getStockShader(MShaderManager::k3dFloat3NumericShader);
			if (shader != nullptr)
			{
				l("found shader");
				renderItem->setShader(shader);

				/* get some info from the CPV shader:
				*    |found shader
   |param: C_4F, COLOR0 t:'Float4', 
   |param: dimmer,  t:'Float', 
   |param: selectionHiddenColor,  t:'Float4', 
   |param: isSelectionHighlightingON,  t:'Boolean', 
   |param: Pm, POSITION t:'Float3', 
   |param: WorldViewProj, worldviewprojection t:'Float4x4Row', 
   |param: DepthPriority, DepthPriority t:'Float', 
   |param: orthographic, isorthographic t:'Boolean', 
   |param: depthPriorityThreshold, mayadepthprioritythreshold t:'Float', 
   |param: depthPriorityScale, mayadepthprioirtyscale t:'Float', 
   |param: Instanced,  t:'Boolean', 
   |param: MultiDraw,  t:'Integer',

   color is float4
				*/
				//MStringArray paramList;
				//shader->parameterList(paramList);
				//for (int pi = 0; pi < paramList.length(); pi++) {
				//	MString paramName = paramList[pi];
				//	MString paramSemantic = shader->parameterSemantic(paramName, ms);
				//	MShaderInstance::ParameterType paramType = shader->parameterType(paramName);
				//	l("param: " + paramName + ", " + paramSemantic + " t:" + paramTypeName(paramType).c_str());
				//}

				renderItem->enable(true);
				// Once assigned, no need to hold on to shader instance
				shaderManager->releaseShader(shader);
			}
			else {
				l("no shader found");
			}
			// The item must be added to the persistent list to be considered
			// for update / rendering
			renderItems.append(renderItem);
		}
		else
		{
			l("retrieving render item");
			renderItem = renderItems.itemAt(renderItemIndex);
		}
		if (renderItem != nullptr)
		{
			l("found render item, updating it");
			MHWRender::MShaderInstance* shader = renderItem->getShader();
			if (shader)
			{
				// Set the shader color parameter
				//shader->setParameter("solidColor", &color.r);
			}
			//renderItem->enable(isEnable); 
			renderItem->enable(true);
		}
		else {
			l("renderItem is still null");
		}
	}
	/* where do we actually build the requirements,
	need to properly call for vertex / index buffers there*/
	void populateGeometry(
		const MHWRender::MGeometryRequirements& requirements, const MHWRender::MRenderItemList& renderItems, MHWRender::MGeometry& data)
	{
		/* we deviate a bit from the example here:
		all edges are curves, sampled at ed::CURVE_SHAPE_RES intervals

		edge is line strip
		point is 3 lines

		memcpy is a thrill

		maybe we don't do the full res polygon drawing here?
		do a separate set of methods to get the proper positions and normals for polygons, should
		be parallelised per-patch at least

		for positions, order [point positions, dense edge positions]

		mgeometryrequirements:
		union of all vertex requirements from all shaders assigned to the object

		How exactly are we meant to know which buffers and requirements here correspond to
		which items are created in updateRenderItems

		names are set on RENDER ITEMS before this
		*/

		LOG("POPULATE_GEOMETRY on render items:");
		for (int i = 0; i < renderItems.length(); i++) {
			const MRenderItem* item = renderItems.itemAt(i);
			l(ed::str(item->name().asChar()) + " ");

		}
		//return;
		Status s;
		MS ms(MS::kSuccess);
		const MVertexBufferDescriptorList& vertexBufferDescriptorList = requirements.vertexRequirements();
		for (int i = 0; i < vertexBufferDescriptorList.length(); i++)
		{
			l("vertexBuffer index: " + ed::str(i));
			MVertexBufferDescriptor desc{};
			if (!vertexBufferDescriptorList.getDescriptor(i, desc)) {
				l("descriptor not found, continuing");
				continue;
			}
			l(desc.semanticName().asChar());

			l("VertexBufferDescriptor in list: " + desc.name()); // name seems empty here

			switch (desc.semantic())
			{
			case MGeometry::kPosition:
			{ // descriptor names aren't set by default - 
				/* unsure if this function might be called with point and edge renderItems in the same list,
				*/
				l("vertex buffer position");
				// Create and fill the vertex position buffer
				MHWRender::MVertexBuffer* positionBuffer = data.createVertexBuffer(desc);
				if (!positionBuffer) {
					l("could not create positionBuffer for vertex data");
					return;
				}
				ed::Float3Array positions = manifold.getWireframePointGnomonVertexPositionArray(s);
				if (!positions.size()) {
					l("returning empty manifold");
					return;
				}
				if (s) {
					l("ERROR getting position vertex buffer for manifold points");
					return;
				}
				l("position array:" + ed::str(positions.size()) + " : ");

				void* buffer = positionBuffer->acquire(static_cast<unsigned int>(positions.size()), true /*writeOnly */);
				if (buffer)
				{
					const std::size_t bufferSizeInByte = sizeof(ed::Float3Array::value_type) * positions.size();
					memcpy(buffer, positions.data(), bufferSizeInByte);
					// Transfer from CPU to GPU memory.
					positionBuffer->commit(buffer);
					l("committed position buffer");
				}
				else {
					l("could not acquire point position buffer");
				}
				break;


			}
			case MGeometry::kNormal:
			{
				l("vertex buffer normal");

				break;
				////
				//// Create and fill the vertex normal buffer
				////
				//MHWRender::MVertexBuffer* normalsBuffer = data.createVertexBuffer(desc);
				//if (normalsBuffer)
				//{
				//	GeometryOverrideExample2_shape::Float3Array normals = fMesh->getNormals();
				//	void* buffer = normalsBuffer->acquire(normals.size(), true /*writeOnly*/);
				//	if (buffer)
				//	{
				//		const std::size_t bufferSizeInByte =
				//			sizeof(GeometryOverrideExample2_shape::Float3Array::value_type) * normals.size();
				//		memcpy(buffer, normals.data(), bufferSizeInByte);
				//		// Transfer from CPU to GPU memory.
				//		normalsBuffer->commit(buffer);
				//	}
				//}
			}
			case MGeometry::kTangent:
			{
				l("vertex buffer tangent");

				break;
				//MHWRender::MVertexBuffer* tangentBuffer = data.createVertexBuffer(desc);
				//if (tangentBuffer)
				//{
				//	GeometryOverrideExample2_shape::Float3Array tangents = fMesh->getTangents();
				//	void* buffer = tangentBuffer->acquire(tangents.size(), true /*writeOnly*/);
				//	if (buffer)
				//	{
				//		const std::size_t bufferSizeInByte =
				//			sizeof(GeometryOverrideExample2_shape::Float3Array::value_type) * tangents.size();
				//		memcpy(buffer, tangents.data(), bufferSizeInByte);
				//		// Transfer from CPU to GPU memory.
				//		tangentBuffer->commit(buffer);
				//	}
				//}
			}
			case MGeometry::kBitangent:
			{
				l("vertex buffer bitangent");

				break;
				//MHWRender::MVertexBuffer* tangentBuffer = data.createVertexBuffer(desc);
				//if (tangentBuffer)
				//{
				//	GeometryOverrideExample2_shape::Float3Array tangents = fMesh->getBiTangents();
				//	void* buffer = tangentBuffer->acquire(tangents.size(), true /*writeOnly*/);
				//	if (buffer)
				//	{
				//		const std::size_t bufferSizeInByte =
				//			sizeof(GeometryOverrideExample2_shape::Float3Array::value_type) * tangents.size();
				//		memcpy(buffer, tangents.data(), bufferSizeInByte);
				//		// Transfer from CPU to GPU memory.
				//		tangentBuffer->commit(buffer);
				//	}
				//}
			}
			case MGeometry::kTexture:
			{
				l("vertex buffer texture");

				////
				//// Create and fill the vertex texture coords buffer
				////
				//MHWRender::MVertexBuffer* texCoordsBuffer = data.createVertexBuffer(desc);
				//if (texCoordsBuffer)
				//{
				//	GeometryOverrideExample2_shape::Float2Array texCoords = fMesh->getTexCoords();
				//	void* buffer = texCoordsBuffer->acquire(texCoords.size(), true /*writeOnly*/);
				//	if (buffer)
				//	{
				//		const std::size_t bufferSizeInByte =
				//			sizeof(GeometryOverrideExample2_shape::Float2Array::value_type) * texCoords.size();
				//		memcpy(buffer, texCoords.data(), bufferSizeInByte);
				//		// Transfer from CPU to GPU memory.
				//		texCoordsBuffer->commit(buffer);
				//	}
				//}
				break;

			}
			case MGeometry::kColor: {
				l("vertex buffer colour");
				MVertexBufferDescriptor desc{};
				if (!vertexBufferDescriptorList.getDescriptor(i, desc)) {
					l("descriptor not found, continuing");
					continue;
				}
				l("descriptor: " + ed::str(desc.semanticName().asChar()));
				MHWRender::MVertexBuffer* colourBuffer = data.createVertexBuffer(desc);
				if (!colourBuffer) {
					l("could not create colourBuffer for vertex data");
					return;
				}
				ed::Float4Array colours(manifold.pDataMap.size() * 4);
				float selectedFloat = 0.5f;
				for (int n = 0; n < static_cast<int>(manifold.pDataMap.size()); n++) {
					colours[n * 4] = ed::Float4(0.0f, 0.0f, 0.0f, 1.0f); // centre point black when not selected
					colours[n * 4 + 1] = ed::Float4(1.0f, 0.0f, 0.0f, selectedFloat);
					colours[n * 4 + 2] = ed::Float4(0.0f, 1.0f, 0.0f, selectedFloat);
					colours[n * 4 + 3] = ed::Float4(0.0f, 0.0f, 1.0f, selectedFloat);
				}
				void* buffer = colourBuffer->acquire(static_cast<unsigned int>(colours.size()), true /*writeOnly */);
				if (buffer)
				{
					const std::size_t bufferSizeInByte = sizeof(ed::Float4Array::value_type) * colours.size();
					memcpy(buffer, colours.data(), bufferSizeInByte);
					// Transfer from CPU to GPU memory.
					colourBuffer->commit(buffer);
					l("committed colour buffer");
				}
				else {
					l("could not acquire point colour buffer");
				}
				break;
			}
			case MGeometry::kTangentWithSign: {
				l("vertex buffer tangentWithSign");
			}
			case MGeometry::kInvalidSemantic: {
				l("vertex buffer invalidSemantic");
				// avoid compiling error
								//
								// In this example, we don't need to used those vertex informantions.
								//
				break;
			}
			}
				
		}


		// what are we actually meant to do here?
		const MIndexBufferDescriptorList& indexBufferDescriptorList = requirements.indexingRequirements();
		l("get indexBufferDescriptors : " + ed::str(indexBufferDescriptorList.length()));
		for (int i = 0; i < indexBufferDescriptorList.length(); i++) {

			MIndexBufferDescriptor desc{};
			if (!indexBufferDescriptorList.getDescriptor(i, desc)) {
				l("descriptor: " + ed::str(i) + " not found, skipping");

				continue;
			}

			l("IndexBufferDescriptor in list: " + desc.name());

		}

		//   Update indexing data for all appropriate render items
		const int numItems = renderItems.length();
		l("update indexing all render items: " + ed::str(numItems));
		for (int i = 0; i < numItems; i++)
		{
			const MHWRender::MRenderItem* item = renderItems.itemAt(i);
			if (!item) {
				l("item: " + ed::str(i) + "not found, skipping");
				continue;

			}
				

			l("RenderItem name: " + item->name());

			if (item->name() == sStPointRenderItemName) {
				l("point renderItem");
				// make index buffer to render point gnomons
				MHWRender::MIndexBuffer* indexBuffer = data.createIndexBuffer(MHWRender::MGeometry::kUnsignedInt32);
				if (indexBuffer == nullptr) {
					l("invalid semantic used to create index buffer, aborting");
					continue;
				}
				ed::IndexList indices = manifold.getWireframePointIndexArray(s);
				if (s) {
					l("ERROR getting wireframe point index array, aborting");
					continue;
				}
				l("got point indices: ");
				DEBUGVI(indices);
				if (!indices.size()) {
					l("got zero-length point indices, skipping");
					continue;
				}
				void* buffer = indexBuffer->acquire(static_cast<unsigned int>(indices.size()), true /*writeOnly*/);
				if (!buffer) {
					l("could not acquire index buffer for points");
					continue;
				}

				const std::size_t bufferSizeInByte =
					sizeof(ed::IndexList::value_type) * indices.size();
				l("before mcpy");
				memcpy(buffer, indices.data(), bufferSizeInByte);
				l("before commit");
				// Transfer from CPU to GPU memory.
				indexBuffer->commit(buffer);
				l("before associate");
				// Associate index buffer with render item
				item->associateWithIndexBuffer(indexBuffer);
				l("render item done");
			}
		}
	}
	void cleanUp() override {
		/* so it seems the sequence goes:
		geo o create
		cleanUp()
		updateDG()
		cleanUp()

		seems cleanup() is explicitly only to wipe changes made in updateDG
		testing just doing nothing here
		*/
		LOG("CLEANUP()")
		//shapeNodePtr = nullptr;
		//manifold.clear();
	}

	StrataShapeGeometryOverride(const MObject& obj)
		: MHWRender::MPxGeometryOverride(obj)
	{
		// get the real mesh object from the MObject
		LOG("GEO O INIT");
		MStatus status;
		MFnDependencyNode node(obj, &status);
		if (status)
		{
			shapeNodeObjHdl = obj;

			shapeNodePtr = dynamic_cast<StrataShapeNode*>(node.userNode());
			if (shapeNodePtr == nullptr) {
				l("could not dynamic cast shapeNodePtr for shapeGeometryOverride");
			}
		}
	}

};