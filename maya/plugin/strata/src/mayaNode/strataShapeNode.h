#pragma once


#include "../MInclude.h"
#include "strataMayaLib.h"
#include "../strataop/elementOp.h"
#include "strataOpNodeBase.h"

#include "../exp/expParse.h"

#include <maya/MPxSurfaceShapeUI.h>
#include <maya/MPxGeometryOverride.h>
#include <maya/MDrawContext.h>
#include <maya/MDrawRegistry.h>
#include <maya/MShaderManager.h>
#include <maya/MSelectionMask.h>

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
prefix MObject nodeT aStMatrixIn;\
\
prefix MObject nodeT aStDataOut;\
prefix MObject nodeT aStExpOut;\
prefix MObject nodeT aStMatrixOut;\
prefix MObject nodeT aStCurveOut;\

namespace ed {

	/* handy way to work more easily with vertex buffer memory - cast 
	it to vector of float types like this */
	struct Float2
	{
		Float2() {}
		Float2(float x, float y)
			: x(x), y(y) {}
		float x;
		float y;
	};
	struct Float3
	{
		Float3() {}
		Float3(float x, float y, float z)
			: x(x), y(y), z(z) {}
		float x;
		float y;
		float z;
	};
	typedef std::vector<Float3>       Float3Array;
	typedef std::vector<Float2>       Float2Array;
	typedef std::vector<unsigned int> IndexList;


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

//class StrataShapeNode : public MPxNode, public StrataOpNodeTemplate<ed::StrataElementOp> {
//class StrataShapeNode : public MPxNode, public StrataOpNodeBase {
class StrataShapeNode : public MPxComponentShape, public StrataOpNodeBase {
public:
	//using thisStrataOpT = ed::StrataElementOp;
	//using superT = StrataOpNodeTemplate<ed::StrataElementOp>;
	using superT = StrataOpNodeBase;
	using thisT = StrataShapeNode;
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

public:
	
	// examples do this so I guess it's legal? Hold pointer to shape node on its draw override
	StrataShapeNode* shapeNodePtr = nullptr; 
	MObjectHandle shapeNodeObjHdl;


	static const char* sActiveWireframeRenderItemName;
	static const char* sDormantWireframeRenderItemName;
	static const char* sShadedRenderItemName;

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
	void updateDG() override {
		/* check here if the linked shape node has dirty items*/
	}
	void updateRenderItems(const MDagPath& path, MHWRender::MRenderItemList& renderItems) override {
		/* largely copied from the geometryOverrideExample2 in the maya devkit
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
		updateWireframeItems(sActiveWireframeRenderItemName,
			MHWRender::MGeometry::kAll,
			MHWRender::MRenderItem::sSelectionDepthPriority,
			wireframeColor,
			isWireFrameRenderItemEnabled,
			renderItems,
			*shaderManager);
		// Update the wireframe render item used when the object will not be selected
		isWireFrameRenderItemEnabled = displayStatus == MHWRender::kDormant;
		updateWireframeItems(sDormantWireframeRenderItemName,
			MHWRender::MGeometry::kWireframe,
			MHWRender::MRenderItem::sDormantWireDepthPriority,
			wireframeColor,
			isWireFrameRenderItemEnabled,
			renderItems,
			*shaderManager);
	
	}
	void populateGeometry(const MHWRender::MGeometryRequirements& requirements, const MHWRender::MRenderItemList& renderItems, MHWRender::MGeometry& data)
	{
		if (!shapeNodePtr)
			return;
		const MVertexBufferDescriptorList& vertexBufferDescriptorList = requirements.vertexRequirements();
		for (int i = 0; i < vertexBufferDescriptorList.length(); i++)
		{
			MVertexBufferDescriptor desc{};
			if (!vertexBufferDescriptorList.getDescriptor(i, desc))
				continue;
			std::cout << desc.semanticName().asChar() << std::endl;
			switch (desc.semantic())
			{
			case MGeometry::kPosition:
			{
				//
				// Create and fill the vertex position buffer
				//
				MHWRender::MVertexBuffer* positionBuffer = data.createVertexBuffer(desc);
				if (positionBuffer)
				{
					StrataShapeNode::Float3Array positions = fMesh->getPositions();
					void* buffer = positionBuffer->acquire(positions.size(), true /*writeOnly */);
					if (buffer)
					{
						const std::size_t bufferSizeInByte =
							sizeof(GeometryOverrideExample2_shape::Float3Array::value_type) * positions.size();
						memcpy(buffer, positions.data(), bufferSizeInByte);
						// Transfer from CPU to GPU memory.
						positionBuffer->commit(buffer);
					}
				}
			}
			break;
			case MGeometry::kNormal:
			{
				//
				// Create and fill the vertex normal buffer
				//
				MHWRender::MVertexBuffer* normalsBuffer = data.createVertexBuffer(desc);
				if (normalsBuffer)
				{
					GeometryOverrideExample2_shape::Float3Array normals = fMesh->getNormals();
					void* buffer = normalsBuffer->acquire(normals.size(), true /*writeOnly*/);
					if (buffer)
					{
						const std::size_t bufferSizeInByte =
							sizeof(GeometryOverrideExample2_shape::Float3Array::value_type) * normals.size();
						memcpy(buffer, normals.data(), bufferSizeInByte);
						// Transfer from CPU to GPU memory.
						normalsBuffer->commit(buffer);
					}
				}
			}
			break;
			case MGeometry::kTangent:
			{
				MHWRender::MVertexBuffer* tangentBuffer = data.createVertexBuffer(desc);
				if (tangentBuffer)
				{
					GeometryOverrideExample2_shape::Float3Array tangents = fMesh->getTangents();
					void* buffer = tangentBuffer->acquire(tangents.size(), true /*writeOnly*/);
					if (buffer)
					{
						const std::size_t bufferSizeInByte =
							sizeof(GeometryOverrideExample2_shape::Float3Array::value_type) * tangents.size();
						memcpy(buffer, tangents.data(), bufferSizeInByte);
						// Transfer from CPU to GPU memory.
						tangentBuffer->commit(buffer);
					}
				}
			}
			break;
			case MGeometry::kBitangent:
			{
				MHWRender::MVertexBuffer* tangentBuffer = data.createVertexBuffer(desc);
				if (tangentBuffer)
				{
					GeometryOverrideExample2_shape::Float3Array tangents = fMesh->getBiTangents();
					void* buffer = tangentBuffer->acquire(tangents.size(), true /*writeOnly*/);
					if (buffer)
					{
						const std::size_t bufferSizeInByte =
							sizeof(GeometryOverrideExample2_shape::Float3Array::value_type) * tangents.size();
						memcpy(buffer, tangents.data(), bufferSizeInByte);
						// Transfer from CPU to GPU memory.
						tangentBuffer->commit(buffer);
					}
				}
			}
			break;
			case MGeometry::kTexture:
			{
				//
				// Create and fill the vertex texture coords buffer
				//
				MHWRender::MVertexBuffer* texCoordsBuffer = data.createVertexBuffer(desc);
				if (texCoordsBuffer)
				{
					GeometryOverrideExample2_shape::Float2Array texCoords = fMesh->getTexCoords();
					void* buffer = texCoordsBuffer->acquire(texCoords.size(), true /*writeOnly*/);
					if (buffer)
					{
						const std::size_t bufferSizeInByte =
							sizeof(GeometryOverrideExample2_shape::Float2Array::value_type) * texCoords.size();
						memcpy(buffer, texCoords.data(), bufferSizeInByte);
						// Transfer from CPU to GPU memory.
						texCoordsBuffer->commit(buffer);
					}
				}
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
		//   Update indexing data for all appropriate render items
		const int numItems = renderItems.length();
		for (int i = 0; i < numItems; i++)
		{
			const MHWRender::MRenderItem* item = renderItems.itemAt(i);
			if (!item)
				continue;
			if (item->primitive() == MHWRender::MGeometry::kTriangles)
			{
				//
				// Create and fill the index buffer used to render triangles
				//
				MHWRender::MIndexBuffer* indexBuffer = data.createIndexBuffer(MHWRender::MGeometry::kUnsignedInt32);
				if (indexBuffer)
				{
					GeometryOverrideExample2_shape::IndexList indices = fMesh->getShadedIndices();
					void* buffer = indexBuffer->acquire(indices.size(), true /*writeOnly*/);
					if (buffer)
					{
						const std::size_t bufferSizeInByte =
							sizeof(GeometryOverrideExample2_shape::IndexList::value_type) * indices.size();
						memcpy(buffer, indices.data(), bufferSizeInByte);
						// Transfer from CPU to GPU memory.
						indexBuffer->commit(buffer);
						// Associate index buffer with render item
						item->associateWithIndexBuffer(indexBuffer);
					}
				}
			}
			else if (item->primitive() == MHWRender::MGeometry::kLines)
			{
				//
				// Create and fill the index buffer used to render lines (Wireframe)
				//
				MHWRender::MIndexBuffer* indexBuffer = data.createIndexBuffer(MHWRender::MGeometry::kUnsignedInt32);
				if (indexBuffer)
				{
					GeometryOverrideExample2_shape::IndexList indices = fMesh->getWireFrameIndices();
					void* buffer = indexBuffer->acquire(indices.size(), true /*writeOnly*/);
					if (buffer)
					{
						const std::size_t bufferSizeInByte =
							sizeof(GeometryOverrideExample2_shape::IndexList::value_type) * indices.size();
						memcpy(buffer, indices.data(), bufferSizeInByte);
						// Transfer from CPU to GPU memory.
						indexBuffer->commit(buffer);
						// Associate index buffer with render item
						item->associateWithIndexBuffer(indexBuffer);
					}
				}
			}
		}
	}
	void cleanUp() override {
		/* null the pointers just in case?*/
		shapeNodePtr = nullptr;
	}
	bool requiresGeometryUpdate() const override
	{
		/* check with cached node */
		return false;
	}
	bool supportsEvaluationManagerParallelUpdate() const override {
		return true;
	}


	/* lord I do not understand these mystic moon runes*/
	void updateWireframeItems(const char* renderItemName, MGeometry::DrawMode drawMode,
		unsigned int depthPriority, MColor color, bool isEnable,
		MHWRender::MRenderItemList& renderItemList,
		const MHWRender::MShaderManager& shaderManager)
	{
		/* I THINK we also need to add new MGeometry::Points items here to draw unbound strata transforms - 
		or we can just draw 3 basis lines for each
		*/


		MHWRender::MRenderItem* renderItem = nullptr;
		// Try to find the active wireframe render item.
		// If the returning index is smaller than 0, that means 
		// the render item does't exists yet. So, create it.
		auto renderItemIndex = renderItemList.indexOf(renderItemName);
		if (renderItemIndex < 0)
		{
			// Create the new render item with the given name.
			// We designate this item as a UI "decoration" and will not be
			// involved in rendering aspects such as casting shadows
			// The "topology" for the render item is a line list.
			renderItem = MHWRender::MRenderItem::Create(renderItemName,
				MHWRender::MRenderItem::DecorationItem,
				MHWRender::MGeometry::kLines
			);
			// We want this render item to show up when in all mode ( Wireframe, Shaded, Textured and BoundingBox)
			renderItem->setDrawMode(drawMode);
			// Set selection priority: on top of everything
			renderItem->depthPriority(depthPriority);
			// Get an instance of a 3dSolidShader from the shader manager.
			// The shader tells the graphics hardware how to draw the geometry. 
			// The MShaderInstance is a reference to a shader along with the values for the shader parameters.
			MShaderInstance* shader = shaderManager.getStockShader(MShaderManager::k3dSolidShader);
			if (shader)
			{
				// Assign the shader to the render item. This adds a reference to that
				// shader.
				renderItem->setShader(shader);
				// Once assigned, no need to hold on to shader instance
				shaderManager.releaseShader(shader);
			}
			// The item must be added to the persistent list to be considered
			// for update / rendering
			renderItemList.append(renderItem);
		}
		else
		{
			renderItem = renderItemList.itemAt(renderItemIndex);
		}
		if (renderItem)
		{
			MHWRender::MShaderInstance* shader = renderItem->getShader();
			if (shader)
			{
				// Set the shader color parameter
				shader->setParameter("solidColor", &color.r);
			}
			renderItem->enable(isEnable);
		}
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

