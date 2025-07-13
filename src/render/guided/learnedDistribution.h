#pragma once
#include <cuda.h>
#include <cuda_runtime.h>
#include <memory>
#include <tiny-cuda-nn/gpu_matrix.h>
#include <tiny-cuda-nn/gpu_memory.h>
#include <tiny-cuda-nn/common.h>
#include <tiny-cuda-nn/network.h>
#include <tiny-cuda-nn/optimizer.h>
#include <tiny-cuda-nn/trainer.h>
#include <tiny-cuda-nn/network_with_input_encoding.h>
#include "json.hpp"

#include "device/cuda.h"
#include "util/math_utils.h"
#include "common.h"
#include "window.h"
#include "workqueue.h"
#include "math.h"
#include "guided.h"
#include "parameters.h"
#include "train.h"
#include "device/context.h"

#include "AR.h"

KRR_NAMESPACE_BEGIN

using nlohmann::json;
using precision_t = tcnn::network_precision_t;
using namespace tcnn;

template <typename T>
using GPUMatrix = tcnn::GPUMatrix<T, tcnn::MatrixLayout::ColumnMajor>;


class LearnedDistribution{
public:
    LearnedDistribution() = default;
    ~LearnedDistribution() = default;

	// Reset funcs
	void resetNetwork(json config);
	void resetTraining() { mTrainerTheta->initialize_params(); mTrainerPhi->initialize_params(); }

    // Training funcs
	void initialize();
	float trainStep(const GuidedPixelStateBuffer* guidedState, 
				int maxQueueSize, 
				const AABB sceneAABB,
				size_t& numTrainingSamples);
	void inferenceStep(ScatterRayQueue* rayQueue, 
		const PixelStateBuffer* pixelState, const AuxInfo* auxData, const AABB sceneAABB);

#if (RENDER_NOISY_ONLY == 0)
	void mouseOverInference(const MouseOverInfoOutput* mouseOverInfo, float* outputPDF, Vector2i outputRes);
	void mouseOverInferenceOptimized(const MouseOverInfoOutput* mouseOverInfo, float* outputPDF, Vector2i outputRes);
	void mouseOverSampling(const MouseOverInfoOutput* mouseOverInfo, float* outputDirections, uint numSamples, PCGSampler* samplers);
#endif

	// RenderUI funcs
	KRR_HOST void renderUI();

	bool hasInitializedNetwork() const { return mNetworkTheta != nullptr && mNetworkPhi != nullptr; }

	KRR_HOST void beginFrame();
	KRR_HOST void endFrame() { mGuiding.endFrame(); }

	KRR_HOST const float* guidingDataPtr() { return LearnedDistribution::inferenceWiBuffer.data(); }
	KRR_HOST static float* inferenceWiPtr() { return LearnedDistribution::inferenceWiBuffer.data(); }

	template <typename T, bool isNEE = true, bool isTrain = true>
	KRR_CALLABLE static AutoRegressive getDistribution(T* dataPtr, size_t idx, uint padding = 0) {
		uint nDim = 0;
		uint offset = 0;
		if constexpr (isTrain)
			nDim = N_DIM_WI;
		else
			nDim = N_WI_PER_PIXEL * N_DIM_WI;

		return AutoRegressive(dataPtr + idx * nDim + offset);
	}

	void saveNetwork(const std::string& save_path);
	void loadNetwork(const std::string& load_path);


	template <typename F>
	void ParallelFor(int nElements, F&& func, CUstream stream = 0) {
		DCHECK_GT(nElements, 0);
		GPUParallelFor(nElements, func, stream);
	}
	
	// guiding parameters
	class Guidance {
	public:
		KRR_CALLABLE bool isEnableGuiding() const { return trainState.isEnableGuiding(); }
		
		KRR_CALLABLE bool isEnableTraining() const { return trainState.isEnableTraining(); }
		
		KRR_CALLABLE bool isEnableGuiding(uint depth) const {
			return trainState.isEnableGuiding() && depth < maxGuidedDepth;
		}

		KRR_CALLABLE bool isEnableTraining(uint depth) const { 
			return trainState.isEnableTraining() && depth < maxTrainDepth;
		}

		KRR_CALLABLE bool isOneStep(uint depth) const { return trainState.isOneStep() && depth < maxTrainDepth; }

		KRR_HOST void beginFrame() {
			trainState.trainPixelOffset = trainState.trainPixelStride <= 1 ?
				0 : sampler.get1D() * trainState.trainPixelStride;
			
			trainState.enableTraining =
				(isEnableTraining() || autoTrain) && !isTrainingFinished;
			trainState.enableGuiding = trainState.enableGuiding || isEnableTraining() || autoTrain;
		};
		KRR_HOST void endFrame() {}
		
		KRR_HOST void renderUI();
		
		KRR_CALLABLE bool isTrainingPixel(uint pixelId) const {
			return trainState.isTrainingPixel(pixelId);
		}
		
		TrainState trainState;
		PCGSampler sampler;
		cudaStream_t stream{};
		bool cosineAware{ true };
		bool misAware{ false };
		bool productWithCosine{ false };
		float bsdfSamplingFraction{ 0.5 };
		bool autoTrain{ false };
		bool isTrainingFinished{ false };
		uint maxGuidedDepth{ 10 };
		uint maxTrainDepth{ 3 };
		uint batchPerFrame{ 5 };
		uint batchSize{ TRAIN_BATCH_SIZE };

		json config;
	} mGuiding;

private:
	static GPUMemory<precision_t> trainOutputBuffer;
	static GPUMemory<float> inferenceOutputBuffer;
	static GPUMemory<precision_t> gradientBuffer;
	static GPUMemory<float> lossBuffer;
	static GPUMemory<float> inferenceInputBuffer;
	static GPUMemory<float> inferenceWiBuffer;

	static GPUMemory<float> inferencePhiInputBuffer;
	static GPUMemory<float> inferencePhiOutputBuffer;
	static GPUMemory<float> trainWiBuffer;
	static GPUMemory<precision_t> gradientPhiBuffer;
	static GPUMemory<precision_t> trainOutputPhiBuffer;

	cudaStream_t mStream = nullptr;
	TrainBufferDuoInputs<GuidedInput, GuidedInputPhi, GuidedOutput>* mTrainBuffer;

	std::shared_ptr<tcnn::NetworkWithInputEncoding<precision_t>> mNetworkTheta;
    std::shared_ptr<tcnn::NetworkWithInputEncoding<precision_t>> mNetworkPhi;
    std::shared_ptr<tcnn::Optimizer<precision_t>> mOptimizerTheta;
	std::shared_ptr<tcnn::Optimizer<precision_t>> mOptimizerPhi;
	std::shared_ptr<tcnn::Loss<precision_t>> mLoss;
	std::shared_ptr<tcnn::Trainer<float, precision_t, precision_t>> mTrainerTheta;
    std::shared_ptr<tcnn::Trainer<float, precision_t, precision_t>> mTrainerPhi;
};

KRR_NAMESPACE_END