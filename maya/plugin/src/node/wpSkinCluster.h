#pragma once

#include <maya/MPxSkinCluster.h>
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
    static MObject aLinearizedWeights;      // Linearized weight array
    static MObject aLinearizedIndices;      // Linearized weight index array
    static MObject aLinearizedMatrices;     // Linearized bind matrix array
    static MObject aLinearizedActiveMatrices;     // Linearized active matrix array
    static MObject aUseGPU;                 // Toggle GPU computation
    static MObject aGPUBackend;             // 0=CUDA, 1=Maya GPU
    static MObject aDebugMode;              // Debug output toggle

    // Callbacks
    virtual MStatus setDependentsDirty(const MPlug& plugBeingDirtied,
                                      MPlugArray& affectedPlugs) override;

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
    MStatus updateLinearizedWeights(MDataBlock& block);
    MStatus updateLinearizedMatrices(MDataBlock& block);
    MStatus linearizeWeights(const MArrayDataHandle& weightListArray,
                            std::vector<float>& outWeights,
                            std::vector<int>& outInfluenceIndices);
    MStatus linearizeMatrices(const MArrayDataHandle& matrixArray,
                             std::vector<float>& outMatrices);

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
    bool m_matricesNeedUpdate;
    
    // Cached linearized data
    std::vector<float> m_linearWeights;
    std::vector<int> m_linearInfluenceIndices;
    std::vector<float> m_linearMatrices;
};

} // namespace wp