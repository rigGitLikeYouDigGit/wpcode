#pragma once


#include "../MInclude.h"
#include "strataMayaLib.h"
#include "../strataop/mergeOp.h"
#include "strataOpNodeBase.h"

#include "../exp/expParse.h"

/*
shape node to display end result of strata graph.
also allows interacting with points as if a normal maya shape,
creating data overrides,
or outputting spatial data back into maya

registering a new interactive shape class in maya, how hard can it be?

*/

# define NODE_STATIC_MEMBERS(prefix, nodeT) \
prefix MObject nodeT aStDataIn;\
prefix MObject nodeT aStExpIn;\
prefix MObject nodeT aStSpaceModeIn;\
prefix MObject nodeT aStSpaceIndexIn;\
prefix MObject nodeT aStSpaceNameIn;\
prefix MObject nodeT aStMatrixIn;\
prefix MObject nodeT aStUVNIn;\
\
prefix MObject nodeT aStDataOut;\
prefix MObject nodeT aStExpOut;\
prefix MObject nodeT aStMatrixOut;\
prefix MObject nodeT aStCurveOut;\
\
prefix MObject nodeT aStShowPoints;\


/* per-space stuff - 
if space is -1, default, use either no space (if el has none)
or the first space of the element -
this should suffice for most of them.

if space name is given, use that if found; no effect if not
if space index is given, use that if found; no effect if not
*/

/* todo:
for visibility, enabled, allow setting expression based overrides
to show/hide groups and faces more precisely*/

namespace ed {



}

class StrataShapeUI : public MPxSurfaceShapeUI {
	/* this class is apparently only for viewport1, 
	and deprecated in newer versions, but we still
	need a creator function to register it
	*/
public:
	static void* creator() {
		return new StrataShapeUI;
	}
	StrataShapeUI() {}
	~StrataShapeUI() {}
	void getDrawRequests(const MDrawInfo& info,
		bool objectAndActiveOnly,
		MDrawRequestQueue& queue) override 
	{
		return;
	};

	virtual bool canDrawUV() const {
		return false;
	}
};

class StrataShapeNode : public MPxNode, public StrataOpNodeTemplate<ed::StrataMergeOp> {
//class StrataShapeNode : public MPxNode, public StrataOpNodeBase {
//class StrataShapeNode : public MPxComponentShape, public StrataOpNodeBase {
public:
	
	using superT = StrataOpNodeBase;
	using thisT = StrataShapeNode;
	using thisStrataOpT = ed::StrataMergeOp;

	// cached values used for drawing
	float pointOpacity = 1.0;


	StrataShapeNode() {}
	virtual ~StrataShapeNode() {}

	static void* creator() {
		StrataShapeNode* newObj = new StrataShapeNode();
		return newObj;
	}

	DECLARE_STATIC_NODE_H_MEMBERS(STRATABASE_STATIC_MEMBERS);
	DECLARE_STATIC_NODE_H_MEMBERS(NODE_STATIC_MEMBERS);

	//virtual void postConstructor();

	//static MStatus legalConnection(
	//	const MPlug& plug,
	//	const MPlug& otherPlug,
	//	bool 	asSrc,
	//	bool& isLegal
	//);

	static MTypeId kNODE_ID;// = const MTypeId(0x00122C1C);
	static MString kNODE_NAME;// = MString("curveFrame");

	static  MString     drawDbClassification;
	static  MString     drawRegistrantId;


	static MStatus initialize();

	MStatus addDeltaTarget(
		MObject& nodeObj, MDataBlock& data, MDataHandle& elDH,
		ed::StrataManifold& manifold, ed::SElement* finalEl, ed::SAtomBackDeltaGroup& deltaGrp
	);

	MStatus runShapeBackPropagation(MObject& nodeObj, MDataBlock& data);

	MStatus populateOutputs(MDataBlock& data);

	virtual MStatus syncStrataParams(MObject& nodeObj, MDataBlock& data);

	virtual MStatus compute(const MPlug& plug, MDataBlock& data);

	// override base class static strata objects, so each leaf class still has attributes
	// initialised separately to the base
	//DECLARE_STRATA_STATIC_MEMBERS;

	/*DECLARE_STATIC_NODE_MEMBERS(
		STRATAADDPOINTSOPNODE_STATIC_MEMBERS)*/

	void postConstructor();

	MStatus legalConnection(
		const MPlug& plug,
		const MPlug& otherPlug,
		bool 	asSrc,
		bool& isLegal
	) const;

	virtual MStatus connectionMade(const MPlug& plug,
		const MPlug& otherPlug,
		bool 	asSrc
	);

	virtual MStatus connectionBroken(const MPlug& plug,
		const MPlug& otherPlug,
		bool 	asSrc
	);

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

	inline Status syncManifold() {
		/* eval manifold if its final out node is dirty*/
		Status s;
		int outIndex = shapeNodePtr->opGraphPtr.get()->outNodeIndex;
		if (shapeNodePtr->opGraphPtr.get()->getNode(outIndex)->anyDirty()) {
			s = shapeNodePtr->opGraphPtr.get()->evalGraph(s, outIndex);
			CWRSTAT(s, "error eval-ing manifold to draw with strataShape node");
		}
		manifold = shapeNodePtr->opGraphPtr.get()->results[outIndex];
		return s;
	}

	/*static const char* sActiveWireframeRenderItemName;
	static const char* sDormantWireframeRenderItemName;
	static const char* sShadedRenderItemName;*/

	static constexpr char* sStPointRenderItemName = "stPRI";
	static constexpr char* sStEdgeRenderItemName = "stERI";

	inline Status getShapeMObj(MObject& result){
		Status s;
		if (!shapeNodeObjHdl.isAlive()) {
			result = MObject::kNullObj;
			STAT_ERROR(s, "shapeGeoOverride could not retrieve shape MObject");
		}
		result = shapeNodeObjHdl.object();
		return s;
	}

	

	static MHWRender::MPxGeometryOverride* Creator(const MObject& obj)
	{
		return new StrataShapeGeometryOverride(obj);
	}

	~StrataShapeGeometryOverride() override {}
	MHWRender::DrawAPI supportedDrawAPIs() const override {
		return MHWRender::kOpenGL;
	}


	void updateSelectionGranularity(const MDagPath& path,
		MSelectionContext& selectionContext
	) {
		/* this is actually really complicated, skip for now*/
	}

	void updateDG() override {
		/* check here if the linked shape node has dirty items*/
		Status s = syncManifold();
		CWMSG(s, "Error on syncManifold in updateDG() for strataShape");
	}

	virtual bool supportsEvaluationManagerParallelUpdate()	const {
		return true;
	}

	/* UI drawing - for now just dots on points, as well as the gnomon lines -
	just an experiment*/
	virtual bool hasUIDrawables()	const {
		/* only render ui points if in selection/component mode?
		*/
		//getFrameContext()->getSelectionInfo()
		return true;
	}
	virtual void addUIDrawables(const MDagPath& path,
		MUIDrawManager& drawManager,
		const MFrameContext& frameContext
	) {
		Status s;
		drawManager.beginDrawable();
		drawManager.setPointSize(1.0);
		drawManager.points(manifold.getPointPositionArray<MPointArray>(s), false);
		CWMSG(s, "error getting point position array for addUiDrawables");
		drawManager.endDrawable();
	}

	inline MIndexBufferDescriptor getCurveIndexBufferDescriptor() {
		/* todo:
		could probably drop down to uint16 here, unlikely we'll have so many points for edges alone*/
		return MIndexBufferDescriptor(MIndexBufferDescriptor::kEdgeLine, // may also be kHullEdgeLine
			"stEdgeIBD",
			MGeometry::kLineStrip,
			ed::StrataManifold::CURVE_SHAPE_RES
		);
	}

	inline MVertexBufferDescriptor getCurvePositionVertexBufferDescriptor() {
		return MVertexBufferDescriptor(
			"stEdgePosVBD",
			MGeometry::kPosition,
			MGeometry::kFloat,
			3
		);
	}

	void updateRenderItems(const MDagPath& path, MHWRender::MRenderItemList& renderItems) override {
		/* largely copied from the geometryOverrideExample2 in the maya devkit
		* 
		* we have 3 kinds of render items for now:
		* all points
		* each edge
		* each sub-patch
		*/
		if (!path.isValid())
			return;
		MRenderer* renderer = MRenderer::theRenderer();
		if (!renderer)
			return;
		const MShaderManager* shaderManager = renderer->getShaderManager();
		if (!shaderManager)
			return;
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
		auto renderItemIndex = renderItems.indexOf(sStPointRenderItemName);
		if (renderItemIndex < 0)
		{
			// Create the new render item with the given name.
			// We designate this item as a UI "decoration" and will not be
			// involved in rendering aspects such as casting shadows
			renderItem = MHWRender::MRenderItem::Create(sStPointRenderItemName,
				MHWRender::MRenderItem::DecorationItem,
				MHWRender::MGeometry::kLines
			);
			// We want this render item to show up when in all mode ( Wireframe, Shaded, Textured and BoundingBox)
			renderItem->setDrawMode(drawMode);
			// Set selection priority: on top of everything
			renderItem->depthPriority(depthPriority);
			// Get an instance of a 3dSolidShader from the shader manager.
			MShaderInstance* shader = shaderManager->getStockShader(MShaderManager::k3dSolidShader);
			if (shader)
			{
				renderItem->setShader(shader);
				// Once assigned, no need to hold on to shader instance
				shaderManager->releaseShader(shader);
			}
			// The item must be added to the persistent list to be considered
			// for update / rendering
			renderItems.append(renderItem);
		}
		else
		{
			renderItem = renderItems.itemAt(renderItemIndex);
		}
		if (renderItem)
		{
			MHWRender::MShaderInstance* shader = renderItem->getShader();
			if (shader)
			{
				// Set the shader color parameter
				shader->setParameter("solidColor", &color.r);
			}
			//renderItem->enable(isEnable); 
			renderItem->enable(true);
		}	
	}

	void populateGeometry(const MHWRender::MGeometryRequirements& requirements, const MHWRender::MRenderItemList& renderItems, MHWRender::MGeometry& data)
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

		DEBUGSL("PopulateGeometry");
		Status s;
		MS ms(MS::kSuccess);
		const MVertexBufferDescriptorList& vertexBufferDescriptorList = requirements.vertexRequirements();
		for (int i = 0; i < vertexBufferDescriptorList.length(); i++)
		{
			MVertexBufferDescriptor desc{};
			if (!vertexBufferDescriptorList.getDescriptor(i, desc))
				continue;
			std::cout << desc.semanticName().asChar() << std::endl;
			
			DEBUGSL( "VertexBufferDescriptor in list: " + desc.name());

			switch (desc.semantic())
			{
			case MGeometry::kPosition:
			{
				// Create and fill the vertex position buffer
				MHWRender::MVertexBuffer* positionBuffer = data.createVertexBuffer(desc);
				if (!positionBuffer) {
					DEBUGSL("could not create positionBuffer for vertex data");
					return;
				}

				// check if this is for point positions
				if (desc.name() == sStPointRenderItemName) {

					ed::Float3Array positions = manifold.getWireframePointGnomonVertexPositionArray(s);
					if (s) {
						DEBUGSL("ERROR getting position vertex buffer for manifold points");
						return;
					}
					void* buffer = positionBuffer->acquire(static_cast<unsigned int>(positions.size()), true /*writeOnly */);
					if (buffer)
					{
						const std::size_t bufferSizeInByte =
							sizeof(ed::Float3Array::value_type) * positions.size();
						memcpy(buffer, positions.data(), bufferSizeInByte);
						// Transfer from CPU to GPU memory.
						positionBuffer->commit(buffer);
					}
					else {
						DEBUGSL("could not acquire point position buffer")
					}

				}
				else {
					DEBUGSL("unknown desc name: " + desc.name() + " requested position vertex buffer");
				}
				
			}
			break;
			case MGeometry::kNormal:
			{
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
			break;
			case MGeometry::kTangent:
			{
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
			break;
			case MGeometry::kBitangent:
			{
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
			break;
			case MGeometry::kTexture:
			{
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
			}
			break;
			case MGeometry::kColor:
			case MGeometry::kTangentWithSign:
			case MGeometry::kInvalidSemantic:   // avoid compiling error
				//
				// In this example, we don't need to used those vertex informantions.
				//
				break;
			}
		}



		const MIndexBufferDescriptorList& indexBufferDescriptorList = requirements.indexingRequirements();
		for (int i = 0; i < indexBufferDescriptorList.length(); i++) {

			MIndexBufferDescriptor desc{};
			if (!indexBufferDescriptorList.getDescriptor(i, desc))
				continue;

			DEBUGSL("IndexBufferDescriptor in list: " + desc.name());

		}

		//   Update indexing data for all appropriate render items
		const int numItems = renderItems.length();
		for (int i = 0; i < numItems; i++)
		{
			const MHWRender::MRenderItem* item = renderItems.itemAt(i);
			if (!item)
				continue;

			DEBUGSL("RenderItem name: " + item->name());

			if (item->name() == sStPointRenderItemName) {
				// make index buffer to render point gnomons
				MHWRender::MIndexBuffer* indexBuffer = data.createIndexBuffer(MHWRender::MGeometry::kUnsignedInt32);
				if (indexBuffer == nullptr) {
					DEBUGSL("invalid semantic used to create index buffer, aborting");
					return;
				}
				ed::IndexList indices = manifold.getWireframePointIndexArray(s);
				if (s) {
					DEBUGSL("ERROR getting wireframe point index array, aborting");
					return;
				}

				void* buffer = indexBuffer->acquire(static_cast<unsigned int>(indices.size()), true /*writeOnly*/);
				if (!buffer) {
					DEBUGSL("could not acquire index buffer for points");
					continue;
				}
				
				const std::size_t bufferSizeInByte =
					sizeof(ed::IndexList::value_type) * indices.size();
				memcpy(buffer, indices.data(), bufferSizeInByte);
				// Transfer from CPU to GPU memory.
				indexBuffer->commit(buffer);
				// Associate index buffer with render item
				item->associateWithIndexBuffer(indexBuffer);
			}
		}
	}
	void cleanUp() override {
		/* null the pointers just in case?*/
		shapeNodePtr = nullptr;
		manifold.clear();
	}





private:
	StrataShapeGeometryOverride(const MObject& obj)
		: MHWRender::MPxGeometryOverride(obj)
	{
		// get the real mesh object from the MObject
		MStatus status;
		MFnDependencyNode node(obj, &status);
		if (status)
		{
			shapeNodeObjHdl = obj;

			shapeNodePtr = dynamic_cast<StrataShapeNode*>(node.userNode());
			if (shapeNodePtr == nullptr) {
				DEBUGSL("could not dynamic cast shapeNodePtr for shapeGeometryOverride");
			}
		}
	}

};

