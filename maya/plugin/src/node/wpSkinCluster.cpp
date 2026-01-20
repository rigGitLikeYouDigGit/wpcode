#include "wpSkinCluster.h"
#include "../macro.h"

#include <maya/MFnMatrixData.h>
#include <maya/MFnNumericData.h>
#include <maya/MFnEnumAttribute.h>
#include <maya/MFnTypedAttribute.h>
#include <maya/MFnData.h>
#include <maya/MFnFloatArrayData.h>
#include <maya/MFnMatrixArrayData.h>
#include <maya/MFnMeshData.h>
#include <maya/MFnVectorArrayData.h>
#include <maya/MVector.h>
#include <maya/MPoint.h>
#include <maya/MFnDependencyNode.h>

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

// ============================================================================
// WpSkinCluster Implementation
// ============================================================================

// Static member initialization
const MString WpSkinCluster::kNODE_NAME("wpSkinCluster");
const MTypeId WpSkinCluster::kNODE_ID(0x00090001);
const MString WpSkinCluster::typeName("wpSkinCluster");

MObject WpSkinCluster::aLinearizedWeights;
MObject WpSkinCluster::aLinearizedIndices;
MObject WpSkinCluster::aLinearizedMatrices;
MObject WpSkinCluster::aLinearizedActiveMatrices;
MObject WpSkinCluster::aUseGPU;
MObject WpSkinCluster::aGPUBackend;
MObject WpSkinCluster::aDebugMode;
MObject WpSkinCluster::aMaxInfluences;
MObject WpSkinCluster::aRefMesh;

WpSkinCluster::WpSkinCluster() 
    : m_weightsNeedUpdate(true)
    , m_restMatricesNeedUpdate(true)
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

	aMaxInfluences = nAttr.create("maxInfluences", "maxInfluences",
        MFnNumericData::kInt, 4, &status);
	addAttribute(aMaxInfluences);

    // Linearized weights array
    aLinearizedWeights = tAttr.create(
        "linearWeight", "linW",
        MFnData::kFloatArray, &status);
    MCHECK(status, "create linearizedWeights");
    tAttr.setStorable(true);
    tAttr.setKeyable(false);
    tAttr.setHidden(false);
    addAttribute(aLinearizedWeights);

    // Linear index array
    aLinearizedIndices = tAttr.create(
        "linearIndices", "linI",
        MFnData::kIntArray, &status);
    MCHECK(status, "create linearIndices");
    tAttr.setStorable(true);
    tAttr.setKeyable(false);
    tAttr.setHidden(false);
    addAttribute(aLinearizedIndices);

    // Linearized rest matrices array
    aLinearizedMatrices = tAttr.create(
        "linearRestMatrix", "linRM",
        MFnData::kFloatArray, &status);
    MCHECK(status, "create linearizedMatrices");
    tAttr.setStorable(true);
    tAttr.setKeyable(false);
    tAttr.setHidden(false);
    addAttribute(aLinearizedMatrices);

    // Linear active matrix array
    aLinearizedActiveMatrices = tAttr.create(
        "linearActiveMatrix", "linAM",
        MFnData::kFloatArray, &status);
    MCHECK(status, "create linearizedActiveMatrices");
    tAttr.setStorable(true);
    tAttr.setKeyable(false);
    tAttr.setHidden(false);
    addAttribute(aLinearizedActiveMatrices);

	// ref mesh to save positions (and maybe normals later)
    aRefMesh = tAttr.create(
        "refMesh", "refm",
		MFnData::kMesh, &status);
	tAttr.setStorable(false);
	tAttr.setKeyable(false);
	tAttr.setReadable(false);
	addAttribute(aRefMesh);

    // Use GPU toggle
    aUseGPU = nAttr.create("useGPU", "gpu", MFnNumericData::kBoolean, false, &status);
    MCHECK(status, "create useGPU");
    nAttr.setStorable(true);
    nAttr.setKeyable(true);
    addAttribute(aUseGPU);

    // GPU backend selection
    aGPUBackend = eAttr.create("gpuBackend", "gpub", 0, &status);
    MCHECK(status, "create gpuBackend");
    eAttr.addField("Maya OpenCL", 0);
#ifdef USE_CUDA
    eAttr.addField("CUDA", 1);
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

	attributeAffects(aMaxInfluences, outputGeom);

    return MS::kSuccess;
}

MStatus WpSkinCluster::setDependentsDirty(const MPlug& plugBeingDirtied,
                                          MPlugArray& affectedPlugs) {
    MObject thisNode = thisMObject();
    
    // Mark linearized data as needing update when weights or matrices change
    if (plugBeingDirtied == weightList || 
        (plugBeingDirtied.isChild() && plugBeingDirtied.parent() == weightList)) {
        m_weightsNeedUpdate = true;
    }
    
    if (plugBeingDirtied == matrix || plugBeingDirtied == bindPreMatrix) {
        m_restMatricesNeedUpdate = true;
    }

    if(plugBeingDirtied == aMaxInfluences) {
        m_weightsNeedUpdate = true;
	}

    if(plugBeingDirtied == aRefMesh) {
		m_restPositionsNeedUpdate = true;
	}
    
    return MPxSkinCluster::setDependentsDirty(plugBeingDirtied, affectedPlugs);
}

MStatus WpSkinCluster::deform(MDataBlock& block,
                              MItGeometry& iter,
                              const MMatrix& mat,
                              unsigned int multiIndex) {
    MStatus status;
    
    // Update linearized data if needed
	int nMaxInfluences = block.inputValue(aMaxInfluences, &status).asInt();

    if (m_weightsNeedUpdate) {
        m_linearWeights = updateLinearizedWeights(block, 
			nMaxInfluences, iter.count()
            );
        m_weightsNeedUpdate = false;
    }
    
    if (m_restMatricesNeedUpdate) {
        m_restMatrices = updateLinearizedMatrices(block, bindPreMatrix);
        m_restMatricesNeedUpdate = false;
    }

	if (m_restPositionsNeedUpdate) {
		MFnMesh meshFn(
            block.inputValue(aRefMesh, &status).data()
        );
        if (!meshFn.numVertices()) {
			meshFn.setObject(block.inputValue(inputGeom, &status).data());
		}
		std::vector<float> restPositions(
            meshFn.numVertices() * 3);
        std::copy(meshFn.getRawPoints(&status), 
                  meshFn.getRawPoints(&status) + meshFn.numVertices() * 3,
			restPositions.data());

		m_restPositionsNeedUpdate = false;
	}

    /* update active joint matrices every frame*/
	m_activeMatrices = updateLinearizedMatrices(block, matrix);
    
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

	int maxInfs = block.inputValue(aMaxInfluences, &status).asInt();
    
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
    if (backend == 1) {
        return deformCUDA(block, iter, mat, multiIndex);
    }
#endif
    
    // Maya OpenCL GPU path is handled automatically by MPxGPUDeformer
    // if registered - fallback to CPU for safety
    return deformCPU(block, iter, mat, multiIndex);
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
        size_t weightBytes = numVertices * 8 * sizeof(float);
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
    cudaMemcpy(d_matrices, m_restMatrices.data(),
               m_restMatrices.size() * sizeof(float), cudaMemcpyHostToDevice);
    
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

MStatus getSkinWeightArrays(
    MDataBlock& block,
    unsigned int nVertices,
    unsigned int maxInfluences,
    std::vector<float>& outWeights,
    std::vector<int>& outInfluenceIndices) 
{
    MStatus status;

    MArrayDataHandle weightListHandle = block.inputArrayValue(WpSkinCluster::weightList, &status);

    for (int vtx = 0; vtx < nVertices; ++vtx) {
     
        MCHECK(status, "get weightList");
        
        status = weightListHandle.jumpToElement(vtx);
        MCHECK(status, "jump to vertex weight element");
        
        MDataHandle weightsHandle = weightListHandle.inputValue(&status).child(WpSkinCluster::weights);
        MCHECK(status, "get weights handle");
        
        MArrayDataHandle weightsArray(weightsHandle, &status);
        MCHECK(status, "get weights array handle");
        
        /* very simple for now, smaller influences beyond the max will cause others to be trimmed
        */
        float total = 0.0f;
		unsigned int nEls = std::min(maxInfluences, weightsArray.elementCount());
        for (unsigned int i = 0; i < nEls; ++i) {
            outWeights[vtx * maxInfluences + i] = weightsArray.inputValue().asFloat();
            outInfluenceIndices[vtx * maxInfluences + i] = weightsArray.elementIndex();
            total += outWeights[vtx * maxInfluences + i];
            weightsArray.next();
        }
        for(unsigned int i = 0; i < nEls; ++i) {
            outWeights[vtx * maxInfluences + i] /= total;
		}
	}
    return status;
}


std::vector<float> WpSkinCluster::updateLinearizedWeights(
    MDataBlock& block,
	int nMaxInfluences,
    int nVertices
    ) {
    MStatus status;

    std::vector<float> weights(nVertices * nMaxInfluences);
    std::vector<int> indices(nVertices * nMaxInfluences);
	getSkinWeightArrays(block, nVertices, nMaxInfluences, weights, indices);
    return weights;
    /*MCHECK(status, "get weightList for linearization");
    return status;*/
    /*MArrayDataHandle weightListHandle = block.inputArrayValue(weightList, &status);
    return linearizeWeights(weightListHandle, m_linearWeights, m_linearInfluenceIndices);*/
}

std::vector<float> WpSkinCluster::updateLinearizedMatrices(
    MDataBlock& block,
	MObject& matrixAttr
) {
    /* unsure if it's worth trimming to float array here*/
    MStatus status;
    MMatrixArray& matrixArr = MFnMatrixArrayData(
        block.inputValue(matrixAttr).data()).array(&status);
	std::vector<float> result(matrixArr.length() * 16);
    for (int i = 0; i < matrixArr.length(); ++i) {
        const MMatrix& mat = matrixArr[i];
        for(int row = 0; row < 4; ++row) {
            for(int col = 0; col < 4; ++col) {
                result[i * 16 + row * 4 + col] = static_cast<float>(mat(row, col));
            }
        }
	}
    return result;
}


// ============================================================================
// WpSkinClusterGPUDeformer Implementation (Maya OpenCL)
// ============================================================================

const char* WpSkinClusterGPUDeformer::getOpenCLKernelCode() {
    return 
R"(
__kernel void wpSkinDeform(
    unsigned int numVertices,
    unsigned int maxInfluences,
    __global float* inputPositions,
    __global float* outputPositions,
    __global float* weights,
    __global int* influenceIndices,
    __global float* matrices)
{
    unsigned int vertexId = get_global_id(0);
    if (vertexId >= numVertices) return;
    
    // Read input position
    float3 inPos = (float3)(
        inputPositions[vertexId * 3 + 0],
        inputPositions[vertexId * 3 + 1],
        inputPositions[vertexId * 3 + 2]
    );
    
    float3 outPos = (float3)(0.0f, 0.0f, 0.0f);
    float totalWeight = 0.0f;
    
    unsigned int weightOffset = vertexId * maxInfluences;
    
    // Iterate through influences
    for (unsigned int i = 0; i < maxInfluences; ++i) {
        int influenceIdx = influenceIndices[weightOffset + i];
        if (influenceIdx < 0) break;
        
        float weight = weights[weightOffset + i];
        if (weight < 0.0001f) continue;
        
        totalWeight += weight;
        
        // Get matrix for this influence (stored row-major, 16 floats)
        unsigned int matrixOffset = influenceIdx * 16;
        
        // Transform vertex by weighted matrix
        // Matrix is 4x4, but we only need upper 3x4 for point transform
        float3 transformed;
        transformed.x = matrices[matrixOffset + 0] * inPos.x +
                       matrices[matrixOffset + 1] * inPos.y +
                       matrices[matrixOffset + 2] * inPos.z +
                       matrices[matrixOffset + 3];
        transformed.y = matrices[matrixOffset + 4] * inPos.x +
                       matrices[matrixOffset + 5] * inPos.y +
                       matrices[matrixOffset + 6] * inPos.z +
                       matrices[matrixOffset + 7];
        transformed.z = matrices[matrixOffset + 8] * inPos.x +
                       matrices[matrixOffset + 9] * inPos.y +
                       matrices[matrixOffset + 10] * inPos.z +
                       matrices[matrixOffset + 11];
        
        outPos += weight * transformed;
    }
    
    // Normalize if needed
    if (totalWeight > 0.0f && fabs(totalWeight - 1.0f) > 0.0001f) {
        outPos /= totalWeight;
    }
    
    // Write output
    outputPositions[vertexId * 3 + 0] = outPos.x;
    outputPositions[vertexId * 3 + 1] = outPos.y;
    outputPositions[vertexId * 3 + 2] = outPos.z;
}
)";
}

WpSkinClusterGPUDeformer::WpSkinClusterGPUDeformer()
    : fNumVertices(0)
    , fNumInfluences(0)
    , fBuffersInitialized(false)
{
}

WpSkinClusterGPUDeformer::~WpSkinClusterGPUDeformer() {
    terminate();
}

MPxGPUDeformer* WpSkinClusterGPUDeformer::creator() {
    return new WpSkinClusterGPUDeformer();
}


MGPUDeformerRegistrationInfo* WpSkinClusterGPUDeformer::getGPUDeformerInfo() {
    static WpSkinGPURegistrationInfo theOne;
	return &theOne;
   /* static MGPUDeformerRegistrationInfo* sInfo = nullptr;
    if (!sInfo) {
        sInfo = new MGPUDeformerRegistrationInfo(
            WpSkinCluster::kNODE_ID,
            WpSkinCluster::kNODE_NAME,
            creator
        );
    }
    return sInfo;*/
}

void WpSkinClusterGPUDeformer::terminate() {
    fKernel.reset();
    fWeightsBuffer.reset();
    fIndicesBuffer.reset();
    fMatricesBuffer.reset();
    fBuffersInitialized = false;
    
    //MOpenCLInfo::releaseOpenCLInfo(fOpenCLInfo);
    fOpenCLInfo.releaseOpenCLKernel(fKernel);
	MPxGPUDeformer::terminate();
}

MStatus WpSkinClusterGPUDeformer::initializeKernel(MOpenCLInfo& openCLInfo) {
    MStatus status;
    
    // Get OpenCL context
    cl_context clContext = openCLInfo.getOpenCLContext();
    cl_device_id clDevice = openCLInfo.getOpenCLDeviceId();
    
    if (!clContext || !clDevice) {
        return MS::kFailure;
    }
    
    // Compile kernel
    const char* kernelSource = getOpenCLKernelCode();
    size_t sourceLength = strlen(kernelSource);
    
    cl_int err;
    cl_program program = clCreateProgramWithSource(
        clContext, 1, &kernelSource, &sourceLength, &err);
    
    if (err != CL_SUCCESS) {
        MGlobal::displayError("Failed to create OpenCL program");
        return MS::kFailure;
    }
    
    err = clBuildProgram(program, 1, &clDevice, nullptr, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        // Get build log
        size_t logSize;
        clGetProgramBuildInfo(program, clDevice, CL_PROGRAM_BUILD_LOG,
                             0, nullptr, &logSize);
        
        std::vector<char> buildLog(logSize);
        clGetProgramBuildInfo(program, clDevice, CL_PROGRAM_BUILD_LOG,
                             logSize, buildLog.data(), nullptr);
        
        MGlobal::displayError(MString("OpenCL build failed: ") + buildLog.data());
        clReleaseProgram(program);
        return MS::kFailure;
    }
    
    // Create kernel
    cl_kernel kernel = clCreateKernel(program, "wpSkinDeform", &err);
    clReleaseProgram(program);
    
    if (err != CL_SUCCESS) {
        MGlobal::displayError("Failed to create OpenCL kernel");
        return MS::kFailure;
    }
    
    fKernel.attach(kernel);
    
    return MS::kSuccess;
}

MStatus WpSkinClusterGPUDeformer::updateGPUBuffers(
    MDataBlock& block,
    const WpSkinCluster* cpuNode)
{
    if (!cpuNode) return MS::kFailure;
    
    MStatus status;
    cl_int err;
    
    // Get linearized data from CPU node
    const std::vector<float>& weights = cpuNode->m_linearWeights;
    const std::vector<int>& indices = cpuNode->m_linearInfluenceIndices;
    const std::vector<float>& restMatrices = cpuNode->m_restMatrices;
    const std::vector<float>& activeMatrices = cpuNode->m_restMatrices;
    
    if (weights.empty() || indices.empty() || restMatrices.empty()) {
        return MS::kFailure;
    }
    
    cl_context clContext = fOpenCLInfo.getOpenCLContext();
    
    // Create or update weights buffer
    size_t weightsSize = weights.size() * sizeof(float);
    if (!fWeightsBuffer.get() || fWeightsBuffer.getSize() < weightsSize) {
        cl_mem weightsMem = clCreateBuffer(
            clContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
            weightsSize, (void*)weights.data(), &err);
        
        if (err != CL_SUCCESS) return MS::kFailure;
        fWeightsBuffer.attach(weightsMem);
    }
    
    // Create or update indices buffer
    size_t indicesSize = indices.size() * sizeof(int);
    if (!fIndicesBuffer.get() || fIndicesBuffer.getSize() < indicesSize) {
        cl_mem indicesMem = clCreateBuffer(
            clContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
            indicesSize, (void*)indices.data(), &err);
        
        if (err != CL_SUCCESS) return MS::kFailure;
        fIndicesBuffer.attach(indicesMem);
    }
    
    // Create or update matrices buffer
    size_t matricesSize = matrices.size() * sizeof(float);
    if (!fMatricesBuffer.get() || fMatricesBuffer.getSize() < matricesSize) {
        cl_mem matricesMem = clCreateBuffer(
            clContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
            matricesSize, (void*)matrices.data(), &err);
        
        if (err != CL_SUCCESS) return MS::kFailure;
        fMatricesBuffer.attach(matricesMem);
    }
    
    fBuffersInitialized = true;
    
    return MS::kSuccess;
}

MPxGPUDeformer::DeformerStatus WpSkinClusterGPUDeformer::evaluate(
    MDataBlock& block,
    const MEvaluationNode& evaluationNode,
    const MPlug& outputPlug,
    unsigned int numElements,
    const MAutoCLMem inputBuffer,
    const MAutoCLEvent inputEvent,
    MAutoCLMem outputBuffer,
    MAutoCLEvent& outputEvent)
{
    MStatus status;
    
    // Get OpenCL info
    //if (!fOpenCLInfo) {
    //    //fOpenCLInfo = MOpenCLInfo::getOpenCLInfo();
    //    fOpenCLInfo = MOpenCLInfo();
    //    if (!fOpenCLInfo.isValid()) {
    //        return MPxGPUDeformer::kDeformerFailure;
    //    }
    //}
    fOpenCLInfo = MOpenCLInfo();


    // Initialize kernel if needed
    if (!fKernel.get()) {
        status = initializeKernel(fOpenCLInfo);
        if (status != MS::kSuccess) {
            return MPxGPUDeformer::kDeformerFailure;
        }
    }
    
    // Get CPU node to access linearized data
    MFnDependencyNode depNode(evaluationNode.dependencyNode());
    WpSkinCluster* cpuNode = static_cast<WpSkinCluster*>(depNode.userNode());
    
    // Update GPU buffers with latest skinning data
    status = updateGPUBuffers(block, cpuNode);
    if (status != MS::kSuccess || !fBuffersInitialized) {
        return MPxGPUDeformer::kDeformerFailure;
    }
    
    fNumVertices = numElements;
    
    // Set kernel arguments
    cl_int err;
    unsigned int argIdx = 0;
    unsigned int maxInfluences = 8; // Match your max influences
    
    err  = clSetKernelArg(fKernel.get(), argIdx++, sizeof(cl_uint), &fNumVertices);
    err |= clSetKernelArg(fKernel.get(), argIdx++, sizeof(cl_uint), &maxInfluences);
    err |= clSetKernelArg(fKernel.get(), argIdx++, sizeof(cl_mem), inputBuffer.getReadOnlyRef());
    err |= clSetKernelArg(fKernel.get(), argIdx++, sizeof(cl_mem), outputBuffer.getReadOnlyRef());
    err |= clSetKernelArg(fKernel.get(), argIdx++, sizeof(cl_mem), fWeightsBuffer.getReadOnlyRef());
    err |= clSetKernelArg(fKernel.get(), argIdx++, sizeof(cl_mem), fIndicesBuffer.getReadOnlyRef());
    err |= clSetKernelArg(fKernel.get(), argIdx++, sizeof(cl_mem), fMatricesBuffer.getReadOnlyRef());
    
    if (err != CL_SUCCESS) {
        MGlobal::displayError("Failed to set kernel arguments");
        return MPxGPUDeformer::kDeformerFailure;
    }
    
    // Execute kernel
    //cl_command_queue queue = fOpenCLInfo.getCommandQueue();
    cl_command_queue queue = fOpenCLInfo.getMayaDefaultOpenCLCommandQueue();
    size_t globalWorkSize = fNumVertices;
    size_t localWorkSize = 64; // Tune this for your GPU
    
    cl_event events[1] = { inputEvent.get() };
    cl_event outputCLEvent;
    
    err = clEnqueueNDRangeKernel(
        queue, fKernel.get(),
        1, nullptr, &globalWorkSize, &localWorkSize,
        inputEvent.get() ? 1 : 0, inputEvent.get() ? events : nullptr,
        &outputCLEvent);
    
    if (err != CL_SUCCESS) {
        MGlobal::displayError(MString("Failed to execute kernel: ") + err);
        return MPxGPUDeformer::kDeformerFailure;
    }
    
    outputEvent.attach(outputCLEvent);
    
    return MPxGPUDeformer::kDeformerSuccess;
}

} // namespace wp