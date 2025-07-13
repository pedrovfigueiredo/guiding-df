#include <thrust/reduce.h>
#include <thrust/execution_policy.h>
#include <cmath>
#include "nrc.h"

KRR_NAMESPACE_BEGIN


GPUMemory<float> NRC::inferenceInputBuffer;
GPUMemory<float> NRC::inferenceOutputBuffer;
GPUMemory<float> NRC::trainOutputInferenceBuffer;
GPUMemory<precision_t> NRC::trainOutputBuffer;
GPUMemory<precision_t> NRC::gradientBuffer;
GPUMemory<float> NRC::lossBuffer;


KRR_HOST void NRC::initialize() {
    Allocator& alloc = *gpContext->alloc;
    if (!mTrainBuffer) mTrainBuffer = alloc.new_object<TrainBuffer<GuidedInputNRC, GuidedOutputNRC>>(TRAIN_BUFFER_SIZE);

    inferenceInputBuffer = GPUMemory<float>(MAX_INFERENCE_NUM * N_DIM_INPUT_NRC);
    inferenceOutputBuffer = GPUMemory<float>(MAX_INFERENCE_NUM * N_DIM_OUTPUT_NRC);
    trainOutputInferenceBuffer = GPUMemory<float>(N_DIM_PADDED_OUTPUT_NRC * TRAIN_BUFFER_SIZE);
    trainOutputBuffer = GPUMemory<precision_t>(N_DIM_PADDED_OUTPUT_NRC * TRAIN_BATCH_SIZE_NRC);
    gradientBuffer = GPUMemory<precision_t>(N_DIM_PADDED_OUTPUT_NRC * TRAIN_BATCH_SIZE_NRC);
	lossBuffer = GPUMemory<float>(N_DIM_PADDED_OUTPUT_NRC * TRAIN_BATCH_SIZE_NRC);
    if (!mInferenceBufferFromTrainingData) mInferenceBufferFromTrainingData = alloc.new_object<NRCInputBuffer<GuidedInputNRC>>(TRAIN_BUFFER_SIZE);
}

KRR_HOST void NRC::beginFrame() {
    GPUCall(KRR_DEVICE_LAMBDA() { mTrainBuffer->clear(); });
    trainPixelOffset = trainPixelStride <= 1 ? 0 : mSampler.get1D() * trainPixelStride;
}

void NRC::resetNetwork(json config) {
    if (!mStream) cudaStreamCreate(&mStream);

    json& encoding_config = config["encoding"];
	json& optimizer_config = config["optimizer"];
	json& network_config = config["network"];
	json& loss_config = config["loss"];

    mOptimizer.reset(tcnn::create_optimizer<precision_t>(optimizer_config));
    mLoss.reset(tcnn::create_loss<precision_t>(loss_config));
    mNetwork = std::make_shared<tcnn::NetworkWithInputEncoding<precision_t>>(
		N_DIM_INPUT_NRC, N_DIM_OUTPUT_NRC, encoding_config, network_config
    );
    mTrainer = std::make_shared<tcnn::Trainer<float, precision_t, precision_t>>(
		mNetwork, mOptimizer, mLoss, KRR_DEFAULT_RND_SEED);
    
    Log(Info, "Network has a padded output width of %d. Total params: %d", mNetwork->padded_output_width(), mNetwork->n_params());
	CHECK_LOG(tcnn::next_multiple(N_DIM_OUTPUT_NRC, 16u) == N_DIM_PADDED_OUTPUT_NRC, 
		"Padded network output width seems wrong!");
	CHECK_LOG(mNetwork->padded_output_width() == N_DIM_PADDED_OUTPUT_NRC,
		 "Padded network output width seems wrong!");
    
    mTrainer->initialize_params();
	mSampler.setSeed(KRR_DEFAULT_RND_SEED);
	CUDA_SYNC_CHECK();
}

void NRC::loadNetwork(const std::string& load_path){
	if (mTrainer == nullptr || mNetwork == nullptr){
		logWarning("Network or trainer not initialized, cannot load network weights");
		return;
	}
	
	// Loading network
	std::filesystem::path weight_filename = load_path;
	weight_filename += std::filesystem::path("NRC_network.bin");
	std::ifstream network_weight_file(weight_filename.string(), std::ios::binary);
	if (!network_weight_file.good()){
		auto error_msg = fmt::format("Failed to open file for loading network weights {}", weight_filename.string());
		logWarning(error_msg);
		return;
	}
	
	std::vector<float> host_weights_net(mNetwork->n_params());
	network_weight_file.read(reinterpret_cast<char*>(host_weights_net.data()), mNetwork->n_params() * sizeof(float));
	network_weight_file.close();
	mTrainer->set_params_full_precision(host_weights_net.data(), mNetwork->n_params(), false);
	CUDA_CHECK_THROW(cudaPeekAtLastError());
	CUDA_CHECK_THROW(cudaDeviceSynchronize());

	auto logMsg = fmt::format("Network weights loaded successfully from {}", weight_filename.string());
    logInfo(logMsg);
}

void NRC::saveNetwork(const std::string& save_path) {
	if (mTrainer == nullptr || mNetwork == nullptr){
        logWarning("Network or trainer not initialized, cannot save network weights");
        return;
    }
    
    // Saving Network
    std::filesystem::path weight_filename = save_path;
	weight_filename += std::filesystem::path("NRC_network.bin");
    std::ofstream network_weight_file(weight_filename.string(), std::ios::binary);
    if (!network_weight_file.good()){
        auto error_msg = fmt::format("Failed to open file for saving network weights {}", weight_filename.string());
        logWarning(error_msg);
        return;
    }
    
    std::vector<float> host_weights_net(mNetwork->n_params());
    CUDA_CHECK_THROW(cudaMemcpy(host_weights_net.data(), mTrainer->params_full_precision(), mNetwork->n_params() * sizeof(float), cudaMemcpyDeviceToHost));
    network_weight_file.write(reinterpret_cast<const char*>(host_weights_net.data()), mNetwork->n_params() * sizeof(float));
    network_weight_file.close();

	auto logMsg = fmt::format("Network weights saved successfully to {}", weight_filename.string());
    logInfo(logMsg);
}

float NRC::trainStep(const GuidedPixelStateBuffer* guidedState, 
					int maxQueueSize, 
					const AABB sceneAABB,
					size_t& numTrainingSamples)
{
    PROFILE("NRC trainStep");
    if (!hasInitializedNetwork()) logFatal("Network not initialized!");
    const cudaStream_t& stream = mStream;
    uint numTrainPixels = maxQueueSize / trainPixelStride;
    LinearKernel(generate_training_data_nrc_with_inference, stream, numTrainPixels,
        trainPixelOffset, trainPixelStride,
        mTrainBuffer, mInferenceBufferFromTrainingData, guidedState, sceneAABB);
    
    numTrainingSamples = mTrainBuffer->size();
    
    const uint numTrainBatches = max((uint)numTrainingSamples / batchSize + 1, batchPerFrame);
    const size_t effectiveBatchSize = min(numTrainingSamples / numTrainBatches, batchSize);
    float loss_sum = 0.0f;
    for (int iter = 0; iter < numTrainBatches; iter++) {
		size_t localBatchSize = min(numTrainingSamples - iter * effectiveBatchSize, effectiveBatchSize);
		localBatchSize -= localBatchSize % 128;
		if (localBatchSize < MIN_TRAIN_BATCH_SIZE) break;
		float* inputData = (float*)(mTrainBuffer->inputs() + iter * localBatchSize);
		float* outputData = (float*)(mTrainBuffer->outputs() + iter * localBatchSize);

		GPUMatrix<float> networkInputs(inputData, N_DIM_INPUT_NRC, localBatchSize);
		GPUMatrix<float> targets(outputData, N_DIM_OUTPUT_NRC, localBatchSize);
		
        // Trainer will do forward and backward pass, and update the network parameters
        GPUMatrix<precision_t> networkOutputs(trainOutputBuffer.data(), N_DIM_PADDED_OUTPUT_NRC, localBatchSize);
        GPUMatrix<precision_t> dL_doutput(gradientBuffer.data(), N_DIM_PADDED_OUTPUT_NRC, localBatchSize);
        GPUMatrix<float> L(lossBuffer.data(), N_DIM_PADDED_OUTPUT_NRC, localBatchSize);
        {
            PROFILE("Training step NRC");
            // auto ctx = mTrainer->training_step(stream, networkInputs, targets);
            CHECK_THROW(networkInputs.n() == targets.n());
            std::unique_ptr<tcnn::Context> ctx = mNetwork->forward(stream, networkInputs, &networkOutputs, false, false);

            mLoss->evaluate(
                stream,
                N_DIM_PADDED_OUTPUT_NRC,
                N_DIM_OUTPUT_NRC,
                TRAIN_LOSS_SCALE_NRC,
                networkOutputs,
                targets,
                L,
                dL_doutput,
                nullptr
            );

            mNetwork->backward(stream, *ctx, networkInputs, networkOutputs, dL_doutput, nullptr, false, EGradientMode::Overwrite);
            mTrainer->optimizer_step(stream, TRAIN_LOSS_SCALE_NRC);
            loss_sum += thrust::reduce(thrust::device, L.data(), L.data() + L.n_elements(), 0.f, thrust::plus<float>()) / localBatchSize;
        }
	}
    return loss_sum / numTrainBatches;
}

void NRC::inferenceStep(float* inputData, float* outputData, int batchSize){
    if (!hasInitializedNetwork()) logFatal("Network not initialized!");
    PROFILE("Network inference");
    const int paddedBatchSize = next_multiple(batchSize, 128);
    GPUMatrix<float> networkInputs(inputData, N_DIM_INPUT_NRC, paddedBatchSize);
	GPUMatrix<float> networkOutputs(outputData, N_DIM_OUTPUT_NRC, paddedBatchSize);
	mNetwork->inference(mStream, networkInputs, networkOutputs);
    cudaDeviceSynchronize();
	
}

void NRC::inferenceStepFromTrainingData(GuidedPixelStateBuffer* guidedState,
    const int maxQueueSize, const AABB sceneAABB, bool skipDataGeneration){
    PROFILE("NRC inferenceStepFromTrainingData");

    // 1. Generate NRC input from training data
    const uint numTrainPixels = maxQueueSize / trainPixelStride;
    const cudaStream_t& stream = mStream;

    if (!skipDataGeneration){
        LinearKernel(generate_NRCinput_from_training_data, stream, numTrainPixels,
            trainPixelOffset, trainPixelStride,
            mInferenceBufferFromTrainingData, guidedState, sceneAABB);
        cudaDeviceSynchronize();
    }

    // // 2. Run NRC Inference
    inferenceStep((float*) (mInferenceBufferFromTrainingData->data()), trainOutputInferenceBuffer.data(), mInferenceBufferFromTrainingData->size());

    // 3. Copy NRC output to guidedState's L output
    const int* nrcDepthsPtr = mInferenceBufferFromTrainingData->depths();
    const int* nrcPixelIdsPtr = mInferenceBufferFromTrainingData->pixelIds();
    const float* nrcOutputPtr = trainOutputInferenceBuffer.data();
    const uint nrcInputSize = mInferenceBufferFromTrainingData->size();
    
    LinearKernel(apply_NRCoutput_to_training_data, stream, nrcInputSize,
        nrcDepthsPtr, nrcPixelIdsPtr, nrcOutputPtr, guidedState);

    // If we're doing 4 or 5, we need to divide the radiance by F' (3) after applying (1) or (2)
    #if (OPTIMIZE_ON_NRC == 4 || OPTIMIZE_ON_NRC == 5)
        LinearKernel(divide_training_data_by_NRCoutput, stream, nrcInputSize,
            nrcDepthsPtr, nrcPixelIdsPtr, nrcOutputPtr, guidedState);
    #endif

    GPUCall(KRR_DEVICE_LAMBDA() { mInferenceBufferFromTrainingData->clear(); });
}

void NRC::mouseOverInference(const MouseOverInfoOutput* mouseOverInfo, 
    float* outputPDF, Vector2i outputRes, RayQueue* rayQueue, 
    ScatterRayQueue* scatterRayQueue, const PixelStateBuffer* pixelState, 
    OptiXGuidedBackend* backend, const AABB& sceneAABB)
{
    cudaDeviceSynchronize();
    PROFILE("NRC mouseOverInference");
    if (!hasInitializedNetwork()) logFatal("Network not initialized!");
    
    float* inferenceInput = inferenceInputBuffer.data();
    float* inferenceOutput = inferenceOutputBuffer.data();

    const cudaStream_t& stream = mStream;
    const size_t maxQueueSize = outputRes[0] * outputRes[1];

    GPUCall(KRR_DEVICE_LAMBDA() { rayQueue->reset(); scatterRayQueue->reset();});

    // To visualize NRC's Ls, we must spawn rays from the mouseOver viewpoint
    {
        PROFILE("Generate camera rays");
        RayQueue* cameraRayQueue = rayQueue;
        ParallelFor(maxQueueSize, KRR_DEVICE_LAMBDA(int pixelId){
            Sampler sampler = &pixelState->sampler[pixelId];
            Vector2i pixelCoord = { pixelId % outputRes[0], pixelId / outputRes[0] };
            
            Vector2f pixelCoordNormalized = Vector2f(pixelCoord) / Vector2f(outputRes); // [0, 1]
            Vector3f wi = normalize(squareToUniformSphere(pixelCoordNormalized));
            Vector3f p = offsetRayOrigin(mouseOverInfo->raw_position, mouseOverInfo->raw_normal, wi);
            Ray cameraRay = {p, wi};

            cameraRayQueue->pushCameraRay(cameraRay, pixelId);
        });
        cudaDeviceSynchronize();
    }
    {
        PROFILE("Tracing OneHit rays");
        backend->traceOneHit(
            maxQueueSize,
            rayQueue,
            scatterRayQueue);
        cudaDeviceSynchronize();
    }
    const int numInferenceSamples = scatterRayQueue->size();
    if (numInferenceSamples == 0) return;
    {
        PROFILE("Data Preparation");
        ParallelFor(maxQueueSize, KRR_DEVICE_LAMBDA(int i){
            if (i >= numInferenceSamples) return;
            const ShadingData &sd = scatterRayQueue->operator[](i).operator krr::ScatterRayWorkItem().sd;
            const uint data_idx = i * N_DIM_INPUT_NRC;

            *(Vector3f *) &inferenceInput[data_idx] = normalizeSpatialCoord(sd.pos, sceneAABB);
            *(Vector3f *) &inferenceInput[data_idx + 3] = warp_direction_for_sh(sd.wo.normalized());
            *(Vector2f *) &inferenceInput[data_idx + 6] = utils::cartesianToSphericalNormalized(sd.frame.N);
            inferenceInput[8] = warp_roughness_for_ob(sd.roughness);
            *(Vector3f *) &inferenceInput[9] = sd.diffuse;
            *(Vector3f *) &inferenceInput[12] = sd.specular;
        });
    }
    cudaDeviceSynchronize();
    inferenceStep(inferenceInput, inferenceOutput, numInferenceSamples);
    
    // Write output PDF
    ParallelFor(maxQueueSize, KRR_DEVICE_LAMBDA(int i){
        if (i >= maxQueueSize) return;
        outputPDF[i] = 0.0f;
    });
    ForAllQueued(scatterRayQueue, maxQueueSize,
        KRR_DEVICE_LAMBDA(ScatterRayWorkItem & w) {
        const ShadingData& sd = w.sd;
        const uint pixelId = w.pixelId;
        const int tid = blockIdx.x * blockDim.x + threadIdx.x;
        Color L = Color(inferenceOutput[tid * N_DIM_OUTPUT_NRC], 
            inferenceOutput[tid * N_DIM_OUTPUT_NRC + 1], 
            inferenceOutput[tid * N_DIM_OUTPUT_NRC + 2]);
        outputPDF[pixelId] = L.mean();
    });
}

KRR_NAMESPACE_END