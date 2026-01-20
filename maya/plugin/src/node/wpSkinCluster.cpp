#include "wpSkinCluster.h"
#include "../macro.h"

#include <maya/MFnMatrixData.h>
#include <maya/MFnNumericData.h>
#include <maya/MFnEnumAttribute.h>
#include <maya/MFnTypedAttribute.h>
#include <maya/MFnData.h>
#include <maya/MVector.h>
#include <maya/MPoint.h>

#ifdef USE_CUDA
#include <cuda_runtime.h>

// CUDA kernel for skinning
__global__ void skinDeformKernel(
    const float* vertices,
    const float* weights,
    const int* influenceIndices,
    const float* matrices,
    float* output,
    int numVertices,
    int maxInfluences)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numVertices) return;

    float outX = 0.0f, outY = 0.0f, outZ = 0.0f;
    
    int weightOffset = idx * maxInfluences;
    
    for (int i = 0; i < maxInfluences; ++i) {
        int influenceIdx = influenceIndices[weightOffset + i];
        if (influenceIdx < 0) break;
        
        float weight = weights[weightOffset + i];
        if (weight < 0.0001f) continue;
        
        // Matrix is stored in row-major order
        int matrixOffset = influenceIdx * 16;
        
        float vx = vertices[idx * 3 + 0];
        float vy = vertices[idx * 3 + 1];
        float vz = vertices[idx * 3 + 2];
        
        // Transform vertex by weighted matrix
        outX += weight * (matrices[matrixOffset + 0] * vx +
                         matrices[matrixOffset + 1] * vy +
                         matrices[matrixOffset + 2] * vz +
                         matrices[matrixOffset + 3]);
        outY += weight * (matrices[matrixOffset + 4] * vx +
                         matrices[matrixOffset + 5] * vy +
                         matrices[matrixOffset + 6] * vz +
                         matrices[matrixOffset + 7]);
        outZ += weight * (matrices[matrixOffset + 8] * vx +
                         matrices[matrixOffset + 9] * vy +
                         matrices[matrixOffset + 10] * vz +
                         matrices[matrixOffset + 11]);
    }
    
    output[idx * 3 + 0] = outX;
    output[idx * 3 + 1] = outY;
    output[idx * 3 + 2] = outZ;
}

#endif // USE_CUDA

namespace wp {

// Static member initialization
const MString WpSkinCluster::kNODE_NAME("wpSkinCluster");
const MTypeId WpSkinCluster::kNODE_ID(0x00090001); // Replace with your ID
const MString WpSkinCluster::typeName("wpSkinCluster");

MObject WpSkinCluster::aLinearizedWeights;
MObject WpSkinCluster::aLinearizedMatrices;
MObject WpSkinCluster::aLinearizedActiveMatrices;
MObject WpSkinCluster::aUseGPU;
MObject WpSkinCluster::aGPUBackend;
MObject WpSkinCluster::aDebugMode;

WpSkinCluster::WpSkinCluster() 
    : m_weightsNeedUpdate(true)
    , m_matricesNeedUpdate(true)
#ifdef USE_CUDA
    , d_vertices(nullptr)
    , d_weights(nullptr)
    , d_matrices(nullptr)
    , d_influenceIndices(nullptr)
    , d_output(nullptr)
    , d_allocatedVertices(0)
    , d_allocatedWeights(0)
    , m_cudaInitialized(false)
#endif
{
}

WpSkinCluster::~WpSkinCluster() {
#ifdef USE_CUDA
    cleanupCUDA();
#endif
}

void* WpSkinCluster::creator() {
    return new WpSkinCluster();
}

MStatus WpSkinCluster::initialize() {
    MStatus status;
    MFnNumericAttribute nAttr;
    MFnMatrixAttribute mAttr;
    MFnEnumAttribute eAttr;
    MFnCompoundAttribute cAttr;
    MFnTypedAttribute tAttr;

    // Linearized weights array
    aLinearizedWeights = tAttr.create(
        "linearWeight", "linearWeight",
        MFnData::kFloatArray, &status);
    MCHECK(status, "create linearizedWeights");
    nAttr.setStorable(true);
    nAttr.setKeyable(false);
    addAttribute(aLinearizedWeights);

    // Linear index array
    aLinearizedIndices = tAttr.create(
        "linearIndices", "linearIndices",
        MFnData::kFloatArray, &status);
    MCHECK(status, "create linearIndices");
    nAttr.setStorable(true);
    nAttr.setKeyable(false);
    addAttribute(aLinearizedWeights);

    // Linearized matrices array
    aLinearizedMatrices = tAttr.create(
        "linearRestMatrix", "linearRestMatrix",
        MFnData::kMatrixArray, &status
    );
    MCHECK(status, "create linearizedMatrices");
    mAttr.setStorable(true);
    mAttr.setKeyable(false);
    addAttribute(aLinearizedMatrices);

    // linear active matrix array
    aLinearizedActiveMatrices = tAttr.create(
        "linearActiveMatrix", "linearActiveMatrix",
        MFnData::kMatrixArray, &status
    );
    MCHECK(status, "create linearizedActiveMatrices");
    mAttr.setStorable(true);
    mAttr.setKeyable(false);
    addAttribute(aLinearizedActiveMatrices);


    // Use GPU toggle
    aUseGPU = nAttr.create("useGPU", "gpu", MFnNumericData::kBoolean, false, &status);
    MCHECK(status, "create useGPU");
    nAttr.setStorable(true);
    nAttr.setKeyable(true);
    addAttribute(aUseGPU);

    // GPU backend selection
    aGPUBackend = eAttr.create("gpuBackend", "gpub", 0, &status);
    MCHECK(status, "create gpuBackend");
#ifdef USE_CUDA
    eAttr.addField("CUDA", 0);
    eAttr.addField("Maya GPU Deformer", 1);
#else
    eAttr.addField("Maya GPU Deformer", 0);
#endif
    eAttr.setStorable(true);
    eAttr.setKeyable(true);
    addAttribute(aGPUBackend);

    // Debug mode
    aDebugMode = nAttr.create("debugMode", "dbg", MFnNumericData::kBoolean, false, &status);
    MCHECK(status, "create debugMode");
    nAttr.setStorable(true);
    nAttr.setKeyable(true);
    addAttribute(aDebugMode);

    // Set up attribute dependencies
    attributeAffects(weightList, outputGeom);
    attributeAffects(matrix, outputGeom);
    attributeAffects(bindPreMatrix, outputGeom);
    attributeAffects(aLinearizedWeights, outputGeom);
    attributeAffects(aLinearizedIndices, outputGeom);
    attributeAffects(aLinearizedMatrices, outputGeom);
    attributeAffects(aLinearizedActiveMatrices, outputGeom);
    attributeAffects(aUseGPU, outputGeom);

    return MS::kSuccess;
}

MStatus WpSkinCluster::setDependentsDirty(const MPlug& plugBeingDirtied,
                                          MPlugArray& affectedPlugs) {
    MObject thisNode = thisMObject();
    
    // Mark linearized data as needing update when weights or matrices change
    if (plugBeingDirtied == weightList || 
        plugBeingDirtied.isChild() && plugBeingDirtied.parent() == weightList) {
        m_weightsNeedUpdate = true;
    }
    
    if (plugBeingDirtied == matrix || plugBeingDirtied == bindPreMatrix) {
        m_matricesNeedUpdate = true;
    }
    
    return MPxSkinCluster::setDependentsDirty(plugBeingDirtied, affectedPlugs);
}

MStatus WpSkinCluster::deform(MDataBlock& block,
                              MItGeometry& iter,
                              const MMatrix& mat,
                              unsigned int multiIndex) {
    MStatus status;
    
    // Update linearized data if needed
    if (m_weightsNeedUpdate) {
        updateLinearizedWeights(block);
        m_weightsNeedUpdate = false;
    }
    
    if (m_matricesNeedUpdate) {
        updateLinearizedMatrices(block);
        m_matricesNeedUpdate = false;
    }
    
    // Check if GPU should be used
    bool useGPU = block.inputValue(aUseGPU, &status).asBool();
    MCHECK(status, "get useGPU");
    
    if (useGPU) {
        return deformGPU(block, iter, mat, multiIndex);
    } else {
        return deformCPU(block, iter, mat, multiIndex);
    }
}

MStatus WpSkinCluster::deformCPU(MDataBlock& block,
                                 MItGeometry& iter,
                                 const MMatrix& mat,
                                 unsigned int multiIndex) {
    MStatus status;
    
    // Get envelope (overall deformer weight)
    MDataHandle envData = block.inputValue(envelope, &status);
    MCHECK(status, "get envelope");
    float env = envData.asFloat();
    
    if (env == 0.0f) {
        return MS::kSuccess;
    }
    
    // Get weight list
    MArrayDataHandle weightListHandle = block.inputArrayValue(weightList, &status);
    MCHECK(status, "get weightList");
    
    // Get matrices
    MArrayDataHandle matrixHandle = block.inputArrayValue(matrix, &status);
    MCHECK(status, "get matrix");
    
    MArrayDataHandle bindPreMatrixHandle = block.inputArrayValue(bindPreMatrix, &status);
    MCHECK(status, "get bindPreMatrix");
    
    // Use Eigen for efficient matrix operations
    using Matrix4f = Eigen::Matrix4f;
    using Vector4f = Eigen::Vector4f;
    
    // Iterate vertices
    for (iter.reset(); !iter.isDone(); iter.next()) {
        int vertexIndex = iter.index();
        MPoint pt = iter.position();
        
        // Get weights for this vertex
        status = weightListHandle.jumpToElement(vertexIndex);
        if (status != MS::kSuccess) continue;
        
        MDataHandle weightsHandle = weightListHandle.inputValue(&status).child(weights);
        MArrayDataHandle weightsArray(weightsHandle, &status);
        
        Vector4f originalPos(pt.x, pt.y, pt.z, 1.0f);
        Vector4f finalPos = Vector4f::Zero();
        
        float totalWeight = 0.0f;
        
        // Iterate influences for this vertex
        for (unsigned int i = 0; i < weightsArray.elementCount(); ++i) {
            weightsArray.jumpToArrayElement(i);
            
            int influenceIndex = weightsArray.elementIndex();
            float weight = weightsArray.inputValue().asFloat();
            
            if (weight < 0.0001f) continue;
            
            totalWeight += weight;
            
            // Get matrices for this influence
            matrixHandle.jumpToElement(influenceIndex);
            MMatrix skinMatrix = matrixHandle.inputValue().asMatrix();
            
            bindPreMatrixHandle.jumpToElement(influenceIndex);
            MMatrix bindMatrix = bindPreMatrixHandle.inputValue().asMatrix();
            
            // Combine matrices
            MMatrix fullMatrix = bindMatrix * skinMatrix;
            
            // Convert to Eigen
            Matrix4f eigenMat;
            for (int row = 0; row < 4; ++row) {
                for (int col = 0; col < 4; ++col) {
                    eigenMat(row, col) = fullMatrix(row, col);
                }
            }
            
            // Weighted transform
            finalPos += weight * (eigenMat * originalPos);
        }
        
        // Normalize if needed
        if (totalWeight > 0.0f && totalWeight != 1.0f) {
            finalPos /= totalWeight;
        }
        
        // Apply envelope
        MPoint newPos(finalPos.x(), finalPos.y(), finalPos.z());
        newPos = pt + env * (newPos - pt);
        
        iter.setPosition(newPos);
    }
    
    return MS::kSuccess;
}

MStatus WpSkinCluster::deformGPU(MDataBlock& block,
                                 MItGeometry& iter,
                                 const MMatrix& mat,
                                 unsigned int multiIndex) {
    MStatus status;
    
    int backend = block.inputValue(aGPUBackend, &status).asInt();
    MCHECK(status, "get gpuBackend");
    
#ifdef USE_CUDA
    if (backend == 0) {
        return deformCUDA(block, iter, mat, multiIndex);
    }
#endif
    
    return deformMayaGPU(block, iter, mat, multiIndex);
}

#ifdef USE_CUDA
void WpSkinCluster::initCUDA() {
    if (m_cudaInitialized) return;
    
    cudaError_t err = cudaSetDevice(0);
    if (err != cudaSuccess) {
        MGlobal::displayError("Failed to initialize CUDA device");
        return;
    }
    
    m_cudaInitialized = true;
}

void WpSkinCluster::cleanupCUDA() {
    if (d_vertices) cudaFree(d_vertices);
    if (d_weights) cudaFree(d_weights);
    if (d_matrices) cudaFree(d_matrices);
    if (d_influenceIndices) cudaFree(d_influenceIndices);
    if (d_output) cudaFree(d_output);
    
    d_vertices = nullptr;
    d_weights = nullptr;
    d_matrices = nullptr;
    d_influenceIndices = nullptr;
    d_output = nullptr;
    d_allocatedVertices = 0;
    d_allocatedWeights = 0;
}

MStatus WpSkinCluster::deformCUDA(MDataBlock& block,
                                  MItGeometry& iter,
                                  const MMatrix& mat,
                                  unsigned int multiIndex) {
    if (!m_cudaInitialized) {
        initCUDA();
    }
    
    // Count vertices
    unsigned int numVertices = iter.count();
    if (numVertices == 0) return MS::kSuccess;
    
    // Allocate GPU memory if needed
    size_t vertexBytes = numVertices * 3 * sizeof(float);
    if (d_allocatedVertices < numVertices) {
        cleanupCUDA();
        
        cudaMalloc(&d_vertices, vertexBytes);
        cudaMalloc(&d_output, vertexBytes);
        
        // Allocate for weights and indices (conservative estimate)
        size_t weightBytes = numVertices * 8 * sizeof(float); // max 8 influences per vertex
        cudaMalloc(&d_weights, weightBytes);
        cudaMalloc(&d_influenceIndices, numVertices * 8 * sizeof(int));
        
        d_allocatedVertices = numVertices;
    }
    
    // Copy vertex data to GPU
    std::vector<float> h_vertices(numVertices * 3);
    iter.reset();
    for (unsigned int i = 0; i < numVertices; ++i, iter.next()) {
        MPoint pt = iter.position();
        h_vertices[i * 3 + 0] = pt.x;
        h_vertices[i * 3 + 1] = pt.y;
        h_vertices[i * 3 + 2] = pt.z;
    }
    cudaMemcpy(d_vertices, h_vertices.data(), vertexBytes, cudaMemcpyHostToDevice);
    
    // Copy weights and matrices (using linearized data)
    cudaMemcpy(d_weights, m_linearWeights.data(), 
               m_linearWeights.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_influenceIndices, m_linearInfluenceIndices.data(),
               m_linearInfluenceIndices.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_matrices, m_linearMatrices.data(),
               m_linearMatrices.size() * sizeof(float), cudaMemcpyHostToDevice);
    
    // Launch kernel
    int threadsPerBlock = 256;
    int blocks = (numVertices + threadsPerBlock - 1) / threadsPerBlock;
    
    skinDeformKernel<<<blocks, threadsPerBlock>>>(
        d_vertices, d_weights, d_influenceIndices, d_matrices,
        d_output, numVertices, 8);
    
    cudaDeviceSynchronize();
    
    // Copy results back
    std::vector<float> h_output(numVertices * 3);
    cudaMemcpy(h_output.data(), d_output, vertexBytes, cudaMemcpyDeviceToHost);
    
    // Update vertex positions
    iter.reset();
    for (unsigned int i = 0; i < numVertices; ++i, iter.next()) {
        MPoint newPt(h_output[i * 3 + 0],
                     h_output[i * 3 + 1],
                     h_output[i * 3 + 2]);
        iter.setPosition(newPt);
    }
    
    return MS::kSuccess;
}
#endif // USE_CUDA

MStatus WpSkinCluster::deformMayaGPU(MDataBlock& block,
                                     MItGeometry& iter,
                                     const MMatrix& mat,
                                     unsigned int multiIndex) {
    // Placeholder for Maya's built-in GPU deformer framework
    // This would use MPxGPUDeformer if needed
    MGlobal::displayWarning("Maya GPU deformer not yet implemented, falling back to CPU");
    return deformCPU(block, iter, mat, multiIndex);
}

MStatus WpSkinCluster::updateLinearizedWeights(MDataBlock& block) {
    MStatus status;
    MArrayDataHandle weightListHandle = block.inputArrayValue(weightList, &status);
    MCHECK(status, "get weightList for linearization");
    
    return linearizeWeights(weightListHandle, m_linearWeights, m_linearInfluenceIndices);
}

MStatus WpSkinCluster::updateLinearizedMatrices(MDataBlock& block) {
    MStatus status;
    MArrayDataHandle matrixHandle = block.inputArrayValue(matrix, &status);
    MCHECK(status, "get matrix for linearization");
    
    return linearizeMatrices(matrixHandle, m_linearMatrices);
}

MStatus WpSkinCluster::linearizeWeights(const MArrayDataHandle& weightListArray,
                                        std::vector<float>& outWeights,
                                        std::vector<int>& outInfluenceIndices) {
    // Implementation depends on your specific weight storage format
    // This is a simplified example
    outWeights.clear();
    outInfluenceIndices.clear();
    
    // Store weights in linear format for GPU
    // Format: [vert0_weight0, vert0_weight1, ..., vert1_weight0, ...]
    
    return MS::kSuccess;
}

MStatus WpSkinCluster::linearizeMatrices(const MArrayDataHandle& matrixArray,
                                         std::vector<float>& outMatrices) {
    outMatrices.clear();
    
    // Linearize matrices for GPU (row-major float arrays)
    
    return MS::kSuccess;
}

} // namespace wp