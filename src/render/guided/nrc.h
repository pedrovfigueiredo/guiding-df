#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <memory>
#include <tiny-cuda-nn/common.h>
#include <tiny-cuda-nn/network.h>
#include <tiny-cuda-nn/gpu_matrix.h>
#include <tiny-cuda-nn/gpu_memory.h>
#include <tiny-cuda-nn/optimizer.h>
#include <tiny-cuda-nn/trainer.h>
#include <tiny-cuda-nn/network_with_input_encoding.h>
#include "json.hpp"

#include "parameters.h"
#include "train.h"
#include "device/context.h"
#include "device/cuda.h"
#include "backend.h"
#include "common.h"
#include "workqueue.h"
#include "guided.h"
#include "learnedDistribution.h"

KRR_NAMESPACE_BEGIN

using nlohmann::json;
using precision_t = tcnn::network_precision_t;

class NRC {
public:
    NRC() = default;
    ~NRC() = default;

    KRR_HOST void beginFrame();
    KRR_HOST void initialize();
    bool hasInitializedNetwork() const { return mNetwork != nullptr; }

    KRR_HOST void renderUI();

    float trainStep(const GuidedPixelStateBuffer* guidedState, 
					int maxQueueSize, 
					const AABB sceneAABB,
					size_t& numTrainingSamples);

	void inferenceStep(float* inputData, float* outputData, int batchSize);

    void inferenceStepFromTrainingData(GuidedPixelStateBuffer* guidedState,
        const int maxQueueSize, const AABB sceneAABB, bool skipDataGeneration = false);

	void mouseOverInference(const MouseOverInfoOutput* mouseOverInfo, 
        float* outputPDF, Vector2i outputRes, RayQueue* rayQueue, 
        ScatterRayQueue* scatterRayQueue, const PixelStateBuffer* pixelState, 
        OptiXGuidedBackend* backend, const AABB& sceneAABB);
	
    KRR_HOST void saveNetwork(const std::string& save_path);
    KRR_HOST void loadNetwork(const std::string& load_path);
    
    // TODO: implement when training NRC
    KRR_HOST void resetNetwork(json config);
    void resetTraining() { mTrainer->initialize_params(); }

    void clearInferenceBuffer() { GPUCall(KRR_DEVICE_LAMBDA() { mInferenceBufferFromTrainingData->clear(); }); }

    template <typename F>
	void ParallelFor(int nElements, F&& func, CUstream stream = 0) {
		DCHECK_GT(nElements, 0);
		GPUParallelFor(nElements, func, stream);
	}

    uint trainPixelOffset{ 0 };
	uint trainPixelStride{ 1 };

private:
    PCGSampler mSampler;
    cudaStream_t mStream = nullptr;

	TrainBuffer<GuidedInputNRC, GuidedOutputNRC>* mTrainBuffer;

    size_t batchPerFrame{ NRC_ITER_COUNT };
    size_t batchSize {TRAIN_BATCH_SIZE_NRC};

    json config;
    std::shared_ptr<tcnn::NetworkWithInputEncoding<precision_t>> mNetwork;
    std::shared_ptr<tcnn::Optimizer<precision_t>> mOptimizer;
    std::shared_ptr<tcnn::Loss<precision_t>> mLoss;
    std::shared_ptr<tcnn::Trainer<float, precision_t, precision_t>> mTrainer;

    static GPUMemory<float> inferenceInputBuffer;
    static GPUMemory<float> inferenceOutputBuffer;
    static GPUMemory<float> trainOutputInferenceBuffer;
    static GPUMemory<precision_t> trainOutputBuffer;
    static GPUMemory<precision_t> gradientBuffer;
    static GPUMemory<float> lossBuffer;

    NRCInputBuffer<GuidedInputNRC>* mInferenceBufferFromTrainingData;
};

KRR_NAMESPACE_END