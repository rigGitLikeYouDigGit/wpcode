#pragma once

#include <maya/MPxSkinCluster.h>
#include <maya/MPxGPUDeformer.h>
#include <maya/MGPUDeformerRegistry.h>
#include <maya/MOpenCLInfo.h>
#include <maya/MOpenCLAutoPtr.h>
#include <maya/MTypeId.h>
#include <maya/MString.h>
#include <maya/MDataBlock.h>
#include <maya/MPlug.h>
#include <maya/MArrayDataBuilder.h>
#include <maya/MItGeometry.h>
#include <maya/MMatrix.h>
#include <maya/MMatrixArray.h>
#include <maya/MFloatArray.h>
#include <maya/MPointArray.h>
#include <maya/MFnNumericAttribute.h>
#include <maya/MFnMatrixAttribute.h>
#include <maya/MFnCompoundAttribute.h>
#include <maya/MGlobal.h>

#include <vector>
#include <Eigen/Dense>

// Forward declarations for GPU deformer
#ifdef USE_CUDA
#include <cuda_runtime.h>
#endif

namespace wp {

// Forward declaration
class WpSkinClusterGPUDeformer;


class WpSkinGPURegistrationInfo : public MGPUDeformerRegistrationInfo {
public:
    WpSkinGPURegistrationInfo() {}
    ~WpSkinGPURegistrationInfo() override {}
    MPxGPUDeformer* createGPUDeformer() override;
    bool validateNodeInGraph(MDataBlock& block, const MEvaluationNode& evaluationNode,
        const MPlug& plug, MStringArray* messages) override
    {
        //return WpSkinClusterGPUDeformer::validateNodeInGraph(block, evaluationNode, plug, messages);
        return true;
    }
    bool validateNodeValues(MDataBlock& block, const MEvaluationNode& evaluationNode,
        const MPlug& plug, MStringArray* messages) override
    {
        //return WpSkinClusterGPUDeformer::validateNodeValues(block, evaluationNode, plug, messages);
        return true;
    }
};



class WpSkinCluster : public MPxSkinCluster {
public:
    WpSkinCluster();
    virtual ~WpSkinCluster();

    // Maya plugin requirements
    static void* creator();
    static MStatus initialize();
    
    
    // Deformation
    virtual MStatus deform(MDataBlock& block,
                          MItGeometry& iter,
                          const MMatrix& mat,
                          unsigned int multiIndex) override;

    // Node type info
    static const MString kNODE_NAME;
    static const MTypeId kNODE_ID;
    static const MString typeName;

    // Attribute objects
    static MObject aMaxInfluences;
    static MObject aLinearizedWeights;
    static MObject aLinearizedIndices;
    static MObject aLinearizedMatrices;
    static MObject aLinearizedActiveMatrices;
    static MObject aUseGPU;
    static MObject aGPUBackend;
    static MObject aDebugMode;
    static MObject aRefMesh;

    // Callbacks
    virtual MStatus setDependentsDirty(const MPlug& plugBeingDirtied,
                                      MPlugArray& affectedPlugs) override;

    //// Accessor for linearized data (used by GPU deformer)
    //const std::vector<float>& getLinearWeights() const { return m_linearWeights; }
    //const std::vector<int>& getLinearInfluenceIndices() const { return m_linearInfluenceIndices; }
    //const std::vector<float>& getLinearMatrices() const { return m_linearMatrices; }

private:
    // CPU deformation
    MStatus deformCPU(MDataBlock& block,
                     MItGeometry& iter,
                     const MMatrix& mat,
                     unsigned int multiIndex);

    // GPU deformation paths
    MStatus deformGPU(MDataBlock& block,
                     MItGeometry& iter,
                     const MMatrix& mat,
                     unsigned int multiIndex);

#ifdef USE_CUDA
    MStatus deformCUDA(MDataBlock& block,
                      MItGeometry& iter,
                      const MMatrix& mat,
                      unsigned int multiIndex);
#endif

    MStatus deformMayaGPU(MDataBlock& block,
                         MItGeometry& iter,
                         const MMatrix& mat,
                         unsigned int multiIndex);

    // Weight and matrix management
    std::vector<float> updateLinearizedWeights(
        MDataBlock& block,
        int nMaxInfluences,
        int nVertices);
    std::vector<float> updateLinearizedMatrices(
        MDataBlock& block,
        MObject& matrixAttr
    );
    //MStatus linearizeWeights(const MArrayDataHandle& weightListArray,
    //                        std::vector<float>& outWeights,
    //                        std::vector<int>& outInfluenceIndices);
    //MStatus linearizeMatrices(const MArrayDataHandle& matrixArray,
    //                         std::vector<float>& outMatrices);

    // GPU resource management
#ifdef USE_CUDA
    void initCUDA();
    void cleanupCUDA();
    
    float* d_vertices;
    float* d_weights;
    float* d_matrices;
    int*   d_influenceIndices;
    float* d_output;
    size_t d_allocatedVertices;
    size_t d_allocatedWeights;
    bool   m_cudaInitialized;
#endif

    // Cache flags
    bool m_weightsNeedUpdate;
    bool m_restMatricesNeedUpdate;
	bool m_restPositionsNeedUpdate;
    
    // Cached linearized data
    std::vector<float> m_linearWeights;
    std::vector<int> m_linearInfluenceIndices;
    std::vector<float> m_restMatrices;
    std::vector<float> m_activeMatrices;
    std::vector<float> m_restPositions;

    friend class WpSkinClusterGPUDeformer;
};

// ============================================================================
// GPU Deformer Implementation using Maya's MPxGPUDeformer
// ============================================================================

class WpSkinClusterGPUDeformer : public MPxGPUDeformer {
public:
    static MGPUDeformerRegistrationInfo* getGPUDeformerInfo();
    static MPxGPUDeformer* creator();
    
    WpSkinClusterGPUDeformer();
    virtual ~WpSkinClusterGPUDeformer();

    // GPU deformer interface
    virtual MPxGPUDeformer::DeformerStatus evaluate(
        MDataBlock& block,
        const MEvaluationNode& evaluationNode,
        const MPlug& outputPlug,
        unsigned int numElements,
        const MAutoCLMem inputBuffer,
        const MAutoCLEvent inputEvent,
        MAutoCLMem outputBuffer,
        MAutoCLEvent& outputEvent) override;

    virtual void terminate() override;

    // OpenCL kernel compilation
    static const char* getOpenCLKernelCode();

    // OpenCL resources
    MOpenCLInfo fOpenCLInfo;
    MAutoCLKernel fKernel;
    
    // GPU buffers for skinning data
	int fWeightsBufferSize;
    MAutoCLMem fWeightsBuffer;
	int fIndicesBufferSize;
    MAutoCLMem fIndicesBuffer;
	int fMatricesBufferSize;
    MAutoCLMem fMatricesBuffer;
    
    // Buffer sizes
    unsigned int fNumVertices;
    unsigned int fNumInfluences;
    
    bool fBuffersInitialized;

    // Helper methods
    MStatus initializeKernel(MOpenCLInfo& openCLInfo);
    MStatus updateGPUBuffers(MDataBlock& block, const WpSkinCluster* cpuNode);
};

} // namespace wp