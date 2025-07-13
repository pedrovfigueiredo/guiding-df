#include <thrust/reduce.h>
#include <thrust/execution_policy.h>
#include <cmath>
#include "learnedDistribution.h"


KRR_NAMESPACE_BEGIN

GPUMemory<precision_t> LearnedDistribution::trainOutputBuffer;
GPUMemory<precision_t> LearnedDistribution::gradientBuffer;
GPUMemory<float> LearnedDistribution::lossBuffer;
GPUMemory<float> LearnedDistribution::inferenceInputBuffer;
GPUMemory<float> LearnedDistribution::inferenceOutputBuffer;
GPUMemory<float> LearnedDistribution::inferenceWiBuffer;

GPUMemory<float> LearnedDistribution::trainWiBuffer;
GPUMemory<precision_t> LearnedDistribution::gradientPhiBuffer;
GPUMemory<precision_t> LearnedDistribution::trainOutputPhiBuffer;
GPUMemory<float> LearnedDistribution::inferencePhiInputBuffer;
GPUMemory<float> LearnedDistribution::inferencePhiOutputBuffer;

KRR_HOST void LearnedDistribution::beginFrame()
{
    mGuiding.beginFrame();
    GPUCall(KRR_DEVICE_LAMBDA() { mTrainBuffer->clear(); });
}

void LearnedDistribution::Guidance::renderUI() {
    trainState.renderUI();
	ui::DragFloat("Bsdf sampling fraction", &bsdfSamplingFraction, 0.001f, 0, 1, "%.3f");
	ui::Checkbox("Cosine aware sampling", &cosineAware);
	
	if (ui::CollapsingHeader("Advanced training options")) {
		if (ui::InputInt("Max guided depth", (int*)&maxGuidedDepth, 1))
			maxGuidedDepth = max(0U, (uint)maxGuidedDepth);
		if (ui::InputInt("Max train depth", (int*)&maxTrainDepth, 1))
			maxTrainDepth = max(0U, min(maxTrainDepth, (uint) MAX_TRAIN_DEPTH));
		if (ui::InputInt("Train pixel stride", (int*)&trainState.trainPixelStride, 1))
			trainState.trainPixelStride = max(1U, trainState.trainPixelStride);
		if (ui::InputInt("Train batch size", (int*)&batchSize, 1))
			batchSize = max(1U, min(batchSize, (uint)TRAIN_BATCH_SIZE));
		if (ui::InputInt("Batch per frame", (int*)&batchPerFrame, 1, 1))
			batchPerFrame = max(0U, batchPerFrame);
	}
}

void LearnedDistribution::renderUI(){
	mGuiding.renderUI();
	float lr_theta = mOptimizerTheta->learning_rate();
    if (ui::DragFloat("Theta Network Learning rate", &lr_theta, 1e-6f, 0, 1e-1, "%.6f")) {
        mOptimizerTheta->set_learning_rate(lr_theta);
    }
    float lr_phi = mOptimizerPhi->learning_rate();
    if (ui::DragFloat("Phi Network Learning rate", &lr_phi, 1e-6f, 0, 1e-1, "%.6f")) {
        mOptimizerPhi->set_learning_rate(lr_phi);
    }
}

void LearnedDistribution::initialize() {
    Allocator& alloc = *gpContext->alloc;
    if (!mTrainBuffer) mTrainBuffer = alloc.new_object<TrainBufferDuoInputs<GuidedInput, GuidedInputPhi, GuidedOutput>>(TRAIN_BUFFER_SIZE);
}

void LearnedDistribution::resetNetwork(json config){
	mGuiding.config = config;
	if (!mGuiding.stream) cudaStreamCreate(&mGuiding.stream);
	mGuiding.sampler.setSeed(KRR_DEFAULT_RND_SEED);

	LearnedDistribution::trainOutputBuffer = GPUMemory<precision_t>(N_DIM_PADDED_OUTPUT * TRAIN_BATCH_SIZE);
	LearnedDistribution::gradientBuffer = GPUMemory<precision_t>(N_DIM_PADDED_OUTPUT * TRAIN_BATCH_SIZE);
	LearnedDistribution::lossBuffer = GPUMemory<float>(TRAIN_BATCH_SIZE);
	LearnedDistribution::inferenceInputBuffer = GPUMemory<float>(N_DIM_INPUT * MAX_INFERENCE_NUM);
	LearnedDistribution::inferenceOutputBuffer = GPUMemory<float>(N_DIM_OUTPUT * MAX_INFERENCE_NUM);
	LearnedDistribution::inferenceWiBuffer = GPUMemory<float>(N_WI_PER_PIXEL * N_DIM_WI * MAX_INFERENCE_NUM);


	if (!mStream) cudaStreamCreate(&mStream);

    LearnedDistribution::inferencePhiInputBuffer = GPUMemory<float>(N_DIM_INPUT_PHI * MAX_INFERENCE_NUM);
    LearnedDistribution::inferencePhiOutputBuffer = GPUMemory<float>(N_DIM_OUTPUT_PHI * MAX_INFERENCE_NUM);
    LearnedDistribution::trainWiBuffer = GPUMemory<float>(N_DIM_WI * TRAIN_BATCH_SIZE);
    LearnedDistribution::gradientPhiBuffer = GPUMemory<precision_t>(N_DIM_OUTPUT_PHI * TRAIN_BATCH_SIZE);
    LearnedDistribution::trainOutputPhiBuffer = GPUMemory<precision_t>(N_DIM_OUTPUT_PHI * TRAIN_BATCH_SIZE);

    json& theta_encoding_config = config.contains("theta_encoding") ? config["theta_encoding"] : config["encoding"];
    json& theta_optimizer_config = config.contains("theta_optimizer") ? config["theta_optimizer"] : config["optimizer"];
    json& theta_network_config = config.contains("theta_network") ? config["theta_network"] : config["network"];

    json& phi_encoding_config = config.contains("phi_encoding") ? config["phi_encoding"] : config["encoding"];
    json& phi_optimizer_config = config.contains("phi_optimizer") ? config["phi_optimizer"] : config["optimizer"];
    json& phi_network_config = config.contains("phi_network") ? config["phi_network"] : config["network"];

    json& loss_config = config["loss"];			// just a dummy loss used to make trainer happy, will be by-passed

    // // Reset loss
    mLoss.reset(tcnn::create_loss<precision_t>(loss_config));

    // Reset optimizers
    mOptimizerTheta.reset(tcnn::create_optimizer<precision_t>(theta_optimizer_config));
    mOptimizerPhi.reset(tcnn::create_optimizer<precision_t>(phi_optimizer_config));

    // Reset networks
    // N_DIM_INPUT already accounts for extra 1dim for theta input on mNetworkPhi.
    // mNetworkTheta sets last float to zero to ignore the theta input.
    mNetworkTheta = std::make_shared<NetworkWithInputEncoding<precision_t>>(
			N_DIM_INPUT, N_DIM_OUTPUT, theta_encoding_config, theta_network_config);
    mNetworkPhi = std::make_shared<NetworkWithInputEncoding<precision_t>>(
			N_DIM_INPUT_PHI, N_DIM_OUTPUT_PHI, phi_encoding_config, phi_network_config);

    // Reset trainers
    mTrainerTheta = std::make_shared<Trainer<float, precision_t, precision_t>>(
			mNetworkTheta, mOptimizerTheta, mLoss, KRR_DEFAULT_RND_SEED);
    mTrainerPhi = std::make_shared<Trainer<float, precision_t, precision_t>>(
            mNetworkPhi, mOptimizerPhi, mLoss, KRR_DEFAULT_RND_SEED);
    
    Log(Info, "Theta Network has a padded output width of %d. Total params: %d", mNetworkTheta->padded_output_width(), mNetworkTheta->n_params());
    Log(Info, "Phi Network has a padded output width of %d. Total params: %d", mNetworkPhi->padded_output_width(), mNetworkPhi->n_params());
    CHECK_LOG(next_multiple(N_DIM_OUTPUT, 16u) == N_DIM_PADDED_OUTPUT, 
        "Padded network output width seems wrong!");
    CHECK_LOG(next_multiple(N_DIM_OUTPUT_PHI, 16u) == N_DIM_PADDED_OUTPUT_PHI, 
        "Padded network output width seems wrong!");
    CHECK_LOG(mNetworkTheta->padded_output_width() == N_DIM_PADDED_OUTPUT,
        "Padded mNetworkTheta output width seems wrong!");
    CHECK_LOG(mNetworkPhi->padded_output_width() == N_DIM_PADDED_OUTPUT_PHI,
        "Padded mNetworkPhi output width seems wrong!");
    
    mTrainerTheta->initialize_params();
    mTrainerPhi->initialize_params();
    CUDA_SYNC_CHECK();

    // [DEBUG] Load network weights from file 
    // std::vector<float> host_weights_theta_net(mNetworkTheta->n_params());
    // std::vector<float> host_weights_phi_net(mNetworkPhi->n_params());
    // const char* theta_network_weight_path = "network_theta_2500.bin";
    // const char* phi_network_weight_path = "network_phi_2500.bin";
    // std::ifstream theta_network_weight_file(theta_network_weight_path, std::ios::binary);
    // std::ifstream phi_network_weight_file(phi_network_weight_path, std::ios::binary);
    // if (theta_network_weight_file.good() && phi_network_weight_file.good()) {
    //     theta_network_weight_file.read(reinterpret_cast<char*>(host_weights_theta_net.data()), mNetworkTheta->n_params() * sizeof(float));
    //     phi_network_weight_file.read(reinterpret_cast<char*>(host_weights_phi_net.data()), mNetworkPhi->n_params() * sizeof(float));
    //     // std::cout << "Loaded network weights from file." << std::endl;
    //     logInfo("Loaded network weights from file.");
    // }
    // else {
    //     // std::cout << "Failed to load network weights from file. Using random initialization." << std::endl;
    //     logFatal("Failed to load network weights from file. Using random initialization.");
    // }
    // mTrainerTheta->set_params_full_precision(host_weights_theta_net.data(), mNetworkTheta->n_params(), false);
    // mTrainerPhi->set_params_full_precision(host_weights_phi_net.data(), mNetworkPhi->n_params(), false);
    // CUDA_SYNC_CHECK();

    // Verify that parameters are loaded correctly
    // std::vector<float> host_weights_theta_net_verify(mNetworkTheta->n_params());
    // std::vector<float> host_weights_phi_net_verify(mNetworkPhi->n_params());
    // CUDA_CHECK_THROW(cudaMemcpy(host_weights_theta_net_verify.data(), mTrainerTheta->params_full_precision(), 100 * sizeof(float), cudaMemcpyDeviceToHost));
    // CUDA_CHECK_THROW(cudaMemcpy(host_weights_phi_net_verify.data(), mTrainerPhi->params_full_precision(), 100 * sizeof(float), cudaMemcpyDeviceToHost));

    // for (int i = 0; i < 100; i++){
    //     Log(Info, "%d: Weights: Theta: %f, Phi: %f", i, host_weights_theta_net_verify[i], host_weights_phi_net_verify[i]);
    // }

    // Log(Info, "NetworkTheta params: %d", mNetworkTheta->n_params());
    // Log(Info, "NetworkPhi params: %d", mNetworkPhi->n_params());
    // exit(1);
}

void LearnedDistribution::inferenceStep(ScatterRayQueue* rayQueue, 
    const PixelStateBuffer* pixelState, const AuxInfo* auxData, const AABB sceneAABB) {
	cudaDeviceSynchronize();
	const cudaStream_t& stream = mGuiding.stream;
    const cudaStream_t& streamPhi = mStream;

	if (!hasInitializedNetwork()) logFatal("Network not initialized!");
	int numInferenceSamples = rayQueue->size();
	if (numInferenceSamples == 0) return;

	{
		PROFILE("Data preparation (theta)");
		LinearKernel(generate_inference_data_AR, stream, MAX_INFERENCE_NUM,
			rayQueue, inferenceWiBuffer.data(), inferenceInputBuffer.data(), 
            inferencePhiInputBuffer.data(), sceneAABB);
	}

	int paddedBatchSize = next_multiple(numInferenceSamples, 128);
	GPUMatrix<float> networkInputs(inferenceInputBuffer.data(), N_DIM_INPUT, paddedBatchSize);
	GPUMatrix<float> networkOutputs(inferenceOutputBuffer.data(), N_DIM_OUTPUT, paddedBatchSize);
    GPUMatrix<float> networkInputsPhi(inferencePhiInputBuffer.data(), N_DIM_INPUT_PHI, paddedBatchSize);
    GPUMatrix<float> networkOutputsPhi(inferencePhiOutputBuffer.data(), N_DIM_OUTPUT_PHI, paddedBatchSize);

	{
		PROFILE("Network inference (theta)");
		mNetworkTheta->inference(stream, networkInputs, networkOutputs);

        LinearKernel(thetaPostInference<float, N_DIM_OUTPUT, N_DIM_OUTPUT, N_DIM_OUTPUT>, stream, paddedBatchSize, networkOutputs.data(), 
            inferencePhiInputBuffer.data(), rayQueue, pixelState, auxData, inferenceWiBuffer.data());
    }

    // Network inference (MIS-BSDF, phi)
    {
		PROFILE("Network inference (MIS-BSDF-phi)");
		mNetworkPhi->inference(stream, networkInputsPhi, networkOutputsPhi);
        LinearKernel(phiPostInference<float, N_DIM_OUTPUT, N_DIM_OUTPUT, N_DIM_OUTPUT_PHI, N_DIM_OUTPUT_PHI, N_DIM_OUTPUT_PHI>, stream, paddedBatchSize, networkOutputs.data(), 
            networkOutputsPhi.data(), rayQueue, pixelState, auxData, inferenceWiBuffer.data());

	}
}

float LearnedDistribution::trainStep(const GuidedPixelStateBuffer* guidedState, 
    int maxQueueSize, 
    const AABB sceneAABB,
    size_t& numTrainingSamples) 
{
    PROFILE("Training");
	const cudaStream_t& stream = mGuiding.stream;
	const cudaStream_t& streamPhi = mStream;
    if (!hasInitializedNetwork()) logFatal("Network not initialized!");

    uint numTrainPixels = maxQueueSize / mGuiding.trainState.trainPixelStride;
    if (mGuiding.trainState.accumulateTrainingData){
        LinearKernel(generate_training_data_AR, stream, numTrainPixels,
            mGuiding.trainState.trainPixelOffset, mGuiding.trainState.trainPixelStride,
            mTrainBuffer, guidedState, sceneAABB);
    }
    
    cudaDeviceSynchronize();
    numTrainingSamples = mTrainBuffer->size();
    const uint numTrainBatches = min((uint)numTrainingSamples / mGuiding.batchSize + 1, mGuiding.batchPerFrame);

    float loss = 0.f;
    for (int iter = 0; iter < numTrainBatches; iter++) {
		size_t localBatchSize = min(numTrainingSamples - iter * mGuiding.batchSize, (size_t)mGuiding.batchSize);
		localBatchSize -= localBatchSize % 128;
		if (localBatchSize < MIN_TRAIN_BATCH_SIZE) break;
		float* inputDataTheta = (float*)(mTrainBuffer->inputs1() + iter * mGuiding.batchSize);
        float* inputDataPhi = (float*)(mTrainBuffer->inputs2() + iter * mGuiding.batchSize);
		GuidedOutput* outputData = mTrainBuffer->outputs() + iter * mGuiding.batchSize;

		GPUMatrix<float> networkInputs(inputDataTheta, N_DIM_INPUT, localBatchSize);
        GPUMatrix<float> networkInputsPhi(inputDataPhi, N_DIM_INPUT_PHI, localBatchSize);
		GPUMatrix<precision_t> networkOutputs(trainOutputBuffer.data(), N_DIM_PADDED_OUTPUT, localBatchSize);
        GPUMatrix<precision_t> networkOutputsPhi(trainOutputPhiBuffer.data(), N_DIM_PADDED_OUTPUT_PHI, localBatchSize);
		GPUMatrix<precision_t> dL_doutput_theta(gradientBuffer.data(), N_DIM_PADDED_OUTPUT, localBatchSize);
        GPUMatrix<precision_t> dL_doutput_phi(gradientPhiBuffer.data(), N_DIM_PADDED_OUTPUT_PHI, localBatchSize);

		// the dim of inputs need not be padded
		{
			PROFILE("Train step");
            std::unique_ptr<tcnn::Context> ctx_theta = nullptr, ctx_phi = nullptr;
			// go forward pass
            {
                PROFILE("Theta Forward Pass");
                ctx_theta = mNetworkTheta->forward(stream, networkInputs, &networkOutputs, false, false);

                precision_t* networkOutputsPtr = networkOutputs.data();
                LinearKernel(softmaxMultFactor<precision_t, N_DIM_OUTPUT, N_DIM_PADDED_OUTPUT, N_DIM_OUTPUT>, stream, localBatchSize, networkOutputs.data());
            }
            {
                PROFILE("Phi Forward Pass");
                ctx_phi = mNetworkPhi->forward(streamPhi, networkInputsPhi, &networkOutputsPhi, false, false);

                precision_t* networkOutputsPhiPtr = networkOutputsPhi.data();
                LinearKernel(softmaxMultFactor<precision_t, N_DIM_OUTPUT_PHI, N_DIM_PADDED_OUTPUT_PHI, N_DIM_OUTPUT_PHI>, streamPhi, localBatchSize, networkOutputsPhi.data());
            }

			cudaDeviceSynchronize();
			// Combined PDF compute and dl doutput
			LinearKernel(combine_AR_pdf_and_dl_doutput<LearnedDistribution>, stream, localBatchSize,
				outputData, networkOutputs.data(), networkOutputsPhi.data(), dL_doutput_theta.data(), dL_doutput_phi.data(), lossBuffer.data(), 
				TRAIN_LOSS_SCALE);
			cudaDeviceSynchronize();

			// go backward (gradient backprop) pass
            if (!ctx_theta || !ctx_phi) logFatal("Context not initialized!");
            {
                PROFILE("Phi Backward Pass");
                mNetworkTheta->backward(stream, *ctx_theta, networkInputs, networkOutputs, dL_doutput_theta, nullptr, false, EGradientMode::Overwrite);
                mTrainerTheta->optimizer_step(stream, TRAIN_LOSS_SCALE);
            }
            
            {
                PROFILE("Theta Backward Pass");
                mNetworkPhi->backward(streamPhi, *ctx_phi, networkInputsPhi, networkOutputsPhi, dL_doutput_phi, nullptr, false, EGradientMode::Overwrite);
                mTrainerPhi->optimizer_step(streamPhi, TRAIN_LOSS_SCALE);
            }
            

			// loss compute, logging and plotting
			loss += thrust::reduce(thrust::device, lossBuffer.data(), lossBuffer.data() + localBatchSize, 0.f, thrust::plus<float>()) / localBatchSize;
		}
	}

    return loss / numTrainBatches;
}

void LearnedDistribution::saveNetwork(const std::string& save_path){
    if (mTrainerTheta == nullptr || mNetworkTheta == nullptr){
        logWarning("Theta Network or trainer not initialized, cannot save network weights");
        return;
    }
    if (mTrainerPhi == nullptr || mNetworkPhi == nullptr){
        logWarning("Phi Network or trainer not initialized, cannot save network weights");
        return;
    }
    
    // Saving Theta
    std::filesystem::path theta_weight_filename = save_path;
	theta_weight_filename += std::filesystem::path("AR_network_theta.bin");
    std::ofstream theta_network_weight_file(theta_weight_filename.string(), std::ios::binary);
    if (!theta_network_weight_file.good()){
        auto error_msg = fmt::format("Failed to open file for saving network weights {}", theta_weight_filename.string());
        logWarning(error_msg);
        return;
    }
    
    std::vector<float> host_weights_theta_net(mNetworkTheta->n_params());
    CUDA_CHECK_THROW(cudaMemcpy(host_weights_theta_net.data(), mTrainerTheta->params_full_precision(), mNetworkTheta->n_params() * sizeof(float), cudaMemcpyDeviceToHost));
    theta_network_weight_file.write(reinterpret_cast<const char*>(host_weights_theta_net.data()), mNetworkTheta->n_params() * sizeof(float));
    theta_network_weight_file.close();


    // Saving Phi
    std::filesystem::path phi_weight_filename = save_path;
	phi_weight_filename += std::filesystem::path("AR_network_phi.bin");
    std::ofstream phi_network_weight_file(phi_weight_filename.string(), std::ios::binary);
    if (!phi_network_weight_file.good()){
        auto error_msg = fmt::format("Failed to open file for saving network weights {}", phi_weight_filename.string());
        logWarning(error_msg);
        return;
    }

    std::vector<float> host_weights_phi_net(mNetworkPhi->n_params());
    CUDA_CHECK_THROW(cudaMemcpy(host_weights_phi_net.data(), mTrainerPhi->params_full_precision(), mNetworkPhi->n_params() * sizeof(float), cudaMemcpyDeviceToHost));
    phi_network_weight_file.write(reinterpret_cast<const char*>(host_weights_phi_net.data()), mNetworkPhi->n_params() * sizeof(float));
    phi_network_weight_file.close();

    auto logMsg = fmt::format("Network weights saved successfully to {} and {}", theta_weight_filename.string(), phi_weight_filename.string());
    logInfo(logMsg);
}

void LearnedDistribution::loadNetwork(const std::string& load_path){
    if (mTrainerTheta == nullptr || mNetworkTheta == nullptr){
        logWarning("Theta Network or trainer not initialized, cannot load network weights");
        return;
    }
    if (mTrainerPhi == nullptr || mNetworkPhi == nullptr){
        logWarning("Phi Network or trainer not initialized, cannot load network weights");
        return;
    }

    // Loading Theta
    std::filesystem::path theta_weight_filename = load_path;
    theta_weight_filename += std::filesystem::path("AR_network_theta.bin");
    std::ifstream theta_network_weight_file(theta_weight_filename.string(), std::ios::binary);
    if (!theta_network_weight_file.good()){
        auto error_msg = fmt::format("Failed to open file for loading network weights {}", theta_weight_filename.string());
        logWarning(error_msg);
        return;
    }

    std::vector<float> host_weights_theta_net(mNetworkTheta->n_params());
    theta_network_weight_file.read(reinterpret_cast<char*>(host_weights_theta_net.data()), mNetworkTheta->n_params() * sizeof(float));
    mTrainerTheta->set_params_full_precision(host_weights_theta_net.data(), mNetworkTheta->n_params(), false);
    theta_network_weight_file.close();

    // Loading Phi
    std::filesystem::path phi_weight_filename = load_path;
    phi_weight_filename += std::filesystem::path("AR_network_phi.bin");
    std::ifstream phi_network_weight_file(phi_weight_filename.string(), std::ios::binary);
    if (!phi_network_weight_file.good()){
        auto error_msg = fmt::format("Failed to open file for loading network weights {}", phi_weight_filename.string());
        logWarning(error_msg);
        return;
    }

    std::vector<float> host_weights_phi_net(mNetworkPhi->n_params());
    phi_network_weight_file.read(reinterpret_cast<char*>(host_weights_phi_net.data()), mNetworkPhi->n_params() * sizeof(float));
    mTrainerPhi->set_params_full_precision(host_weights_phi_net.data(), mNetworkPhi->n_params(), false);
    phi_network_weight_file.close();

    auto logMsg = fmt::format("Network weights loaded successfully from {} and {}", theta_weight_filename.string(), phi_weight_filename.string());
    logInfo(logMsg);
}

#if (RENDER_NOISY_ONLY == 0)

void LearnedDistribution::mouseOverInferenceOptimized(const MouseOverInfoOutput* mouseOverInfo, float* outputPDF, Vector2i outputRes){
    cudaDeviceSynchronize();
    if (!hasInitializedNetwork()) logFatal("Network not initialized!");

    float* wiPtr = LearnedDistribution::inferenceWiPtr();
    
    const cudaStream_t& stream = mGuiding.stream;
    const cudaStream_t& streamPhi = mStream;

    const int thetaNumInferenceSamples = 1;
    const int phiNumInferenceSamples = outputRes[0];
	const int thetaPaddedBatchSize = next_multiple(thetaNumInferenceSamples, 128);
    const int phiPaddedBatchSize = next_multiple(phiNumInferenceSamples, 128);
	{
        // Theta input (1 entry)
		float* inputDataTheta = inferenceInputBuffer.data();
		GPUCall(KRR_DEVICE_LAMBDA() {
			*(Vector3f *) &inputDataTheta[0] = mouseOverInfo->position;
			#if GUIDED_PRODUCT_SAMPLING
				*(Vector3f *) &inputDataTheta[N_DIM_SPATIAL_INPUT] = (mouseOverInfo->wo + Vector3f::Ones()) * 0.5f;
			#endif
			#if GUIDED_AUXILIARY_INPUT
				*(Vector2f *) &inputDataTheta[N_DIM_SPATIAL_INPUT + N_DIM_DIRECTIONAL_INPUT] =
					mouseOverInfo->normal;
                inputDataTheta[N_DIM_SPATIAL_INPUT + N_DIM_DIRECTIONAL_INPUT + 2] =
					1 - expf(-mouseOverInfo->roughness);
			#endif
		});

        cudaDeviceSynchronize();
        // Phi input (phiNumInferenceSamples entries)
        float* inputDataPhi = inferencePhiInputBuffer.data();
        ParallelFor(phiNumInferenceSamples, KRR_DEVICE_LAMBDA (int idx) {
            float* inputDataPhi_i = inputDataPhi + idx * N_DIM_INPUT_PHI;
            const float* wiPtr_i = wiPtr + idx * (N_WI_PER_PIXEL * N_DIM_WI);

            for (int i = 0; i < N_DIM_INPUT; i++)
                inputDataPhi_i[i] = inputDataTheta[i];
            
            // Adds normalized theta as input to phi network
            const float theta = wiPtr_i[0];
            #if (AR_LINEAR_INTERPOLATION == 1)
                const int theta_floor_idx = (float) safe_floor(theta * N_DIM_OUTPUT, N_DIM_OUTPUT);
                const float weight = theta * N_DIM_OUTPUT - theta_floor_idx;
                const float theta_linear = (theta_floor_idx + weight) / N_DIM_OUTPUT;
                inputDataPhi_i[N_DIM_INPUT_PHI - 1] = theta_linear; // Adds theta as last input to phi network
            #else
                const float theta_floor = (float) safe_floor(theta * N_DIM_OUTPUT, N_DIM_OUTPUT) / N_DIM_OUTPUT;
                inputDataPhi_i[N_DIM_INPUT_PHI - 1] = theta_floor; // Adds theta as last input to phi network
            #endif
        }, streamPhi);
	}

    {
        GPUMatrix<float> networkInputs(inferenceInputBuffer.data(), N_DIM_INPUT, thetaPaddedBatchSize);
		GPUMatrix<float> networkOutputs(inferenceOutputBuffer.data(), N_DIM_OUTPUT, thetaPaddedBatchSize);

        GPUMatrix<float> networkPhiInputs(inferencePhiInputBuffer.data(), N_DIM_INPUT_PHI, phiPaddedBatchSize);
		GPUMatrix<float> networkPhiOutputs(inferencePhiOutputBuffer.data(), N_DIM_OUTPUT_PHI, phiPaddedBatchSize);

        // Network inference (theta)
        mNetworkTheta->inference(stream, networkInputs, networkOutputs);
        float* networkOutputsPtr = networkOutputs.data();
        LinearKernel(softmaxMultFactor<float, N_DIM_OUTPUT, N_DIM_OUTPUT, N_DIM_OUTPUT>, stream, thetaNumInferenceSamples, networkOutputs.data());

        // Computing ThetaPDF for outputRes[0] based off the single thetaPDF inference (thetaPDFs at index [0-(N_DIM_OUTPUT-1)])
        const float* thetaPDFs = networkOutputs.data();
        ParallelFor(outputRes[0], KRR_DEVICE_LAMBDA (int idx) {
            float* wiPtr_i = wiPtr + idx * (N_WI_PER_PIXEL * N_DIM_WI);
            const float pdf_i = sample_pdf<float, true>(thetaPDFs, wiPtr_i[0] * N_DIM_OUTPUT, N_DIM_OUTPUT);
            wiPtr_i[2] = pdf_i;
        }, stream);

        // Network inference (phi)
        mNetworkPhi->inference(streamPhi, networkPhiInputs, networkPhiOutputs);
        LinearKernel(softmaxMultFactor<float, N_DIM_OUTPUT_PHI, N_DIM_OUTPUT_PHI, N_DIM_OUTPUT_PHI>, streamPhi, phiNumInferenceSamples, networkPhiOutputs.data());

        const float* phiPDFs = networkPhiOutputs.data();
        ParallelFor(outputRes[0] * outputRes[1], KRR_DEVICE_LAMBDA (int idx) {
            float* wiPtr_i = wiPtr + idx * (N_WI_PER_PIXEL * N_DIM_WI);
            const float* phiPDF_i = phiPDFs + (idx % outputRes[0]) * N_DIM_OUTPUT_PHI; // Different PDF inference output per column. Repeats for each row.
            const float pdf_i = sample_pdf<float, false>(phiPDF_i, wiPtr_i[1] * N_DIM_OUTPUT_PHI, N_DIM_OUTPUT_PHI);
            wiPtr_i[3] = pdf_i;
        }, streamPhi);


        // Compute Final PDF
        cudaDeviceSynchronize();
        ParallelFor(outputRes[0] * outputRes[1], KRR_DEVICE_LAMBDA (int idx) {
            const float* thetaPtr_i = wiPtr + (idx % outputRes[0]) * (N_WI_PER_PIXEL * N_DIM_WI);
            const float* phiPtr_i = wiPtr + idx * (N_WI_PER_PIXEL * N_DIM_WI);

            const float thetaPdf = thetaPtr_i[2];
            const float phiPdf = phiPtr_i[3];
            const float pdf_transform = squareToUniformSpherePdf();

            outputPDF[idx] = thetaPdf * phiPdf * pdf_transform;
        }, stream);
        cudaDeviceSynchronize();
    }
}

void LearnedDistribution::mouseOverInference(const MouseOverInfoOutput* mouseOverInfo, float* outputPDF, Vector2i outputRes){
    cudaDeviceSynchronize();
    if (!hasInitializedNetwork()) logFatal("Network not initialized!");

    float* wiPtr = LearnedDistribution::inferenceWiPtr();
    
    const cudaStream_t& stream = mGuiding.stream;
    const cudaStream_t& streamPhi = mStream;

    const int thetaNumInferenceSamples = 1;
    const int phiNumInferenceSamples = outputRes[0] * outputRes[1];
	const int thetaPaddedBatchSize = next_multiple(thetaNumInferenceSamples, 128);
    const int phiPaddedBatchSize = next_multiple(phiNumInferenceSamples, 128);
	{
        // Theta input (1 entry)
		float* inputDataTheta = inferenceInputBuffer.data();
		GPUCall(KRR_DEVICE_LAMBDA() {
			*(Vector3f *) &inputDataTheta[0] = mouseOverInfo->position;
			#if GUIDED_PRODUCT_SAMPLING
				*(Vector3f *) &inputDataTheta[N_DIM_SPATIAL_INPUT] = (mouseOverInfo->wo + Vector3f::Ones()) * 0.5f;
			#endif
			#if GUIDED_AUXILIARY_INPUT
				*(Vector2f *) &inputDataTheta[N_DIM_SPATIAL_INPUT + N_DIM_DIRECTIONAL_INPUT] =
					mouseOverInfo->normal;
                inputDataTheta[N_DIM_SPATIAL_INPUT + N_DIM_DIRECTIONAL_INPUT + 2] =
					1 - expf(-mouseOverInfo->roughness);
			#endif
		});

        cudaDeviceSynchronize();
        // Phi input (phiNumInferenceSamples entries)
        float* inputDataPhi = inferencePhiInputBuffer.data();
        ParallelFor(phiNumInferenceSamples, KRR_DEVICE_LAMBDA (int idx) {
            float* inputDataPhi_i = inputDataPhi + idx * N_DIM_INPUT_PHI;
            const float* wiPtr_i = wiPtr + idx * (N_WI_PER_PIXEL * N_DIM_WI);

            for (int i = 0; i < N_DIM_INPUT; i++)
                inputDataPhi_i[i] = inputDataTheta[i];
            
            // Adds normalized theta as input to phi network
            const float theta = wiPtr_i[0];
            #if (AR_LINEAR_INTERPOLATION == 1)
                const int theta_floor_idx = (float) safe_floor(theta * N_DIM_OUTPUT, N_DIM_OUTPUT);
                const float weight = theta * N_DIM_OUTPUT - theta_floor_idx;
                const float theta_linear = (theta_floor_idx + weight) / N_DIM_OUTPUT;
                inputDataPhi_i[N_DIM_INPUT_PHI - 1] = theta_linear; // Adds theta as last input to phi network
            #else
                const float theta_floor = (float) safe_floor(theta * N_DIM_OUTPUT, N_DIM_OUTPUT) / N_DIM_OUTPUT;
                inputDataPhi_i[N_DIM_INPUT_PHI - 1] = theta_floor; // Adds theta as last input to phi network
            #endif
        }, streamPhi);
	}

    {
        GPUMatrix<float> networkInputs(inferenceInputBuffer.data(), N_DIM_INPUT, thetaPaddedBatchSize);
		GPUMatrix<float> networkOutputs(inferenceOutputBuffer.data(), N_DIM_OUTPUT, thetaPaddedBatchSize);

        GPUMatrix<float> networkPhiInputs(inferencePhiInputBuffer.data(), N_DIM_INPUT_PHI, phiPaddedBatchSize);
		GPUMatrix<float> networkPhiOutputs(inferencePhiOutputBuffer.data(), N_DIM_OUTPUT_PHI, phiPaddedBatchSize);

        // Network inference (theta)
        mNetworkTheta->inference(stream, networkInputs, networkOutputs);
        float* networkOutputsPtr = networkOutputs.data();
        LinearKernel(softmaxMultFactor<float, N_DIM_OUTPUT, N_DIM_OUTPUT, N_DIM_OUTPUT>, stream, thetaNumInferenceSamples, networkOutputs.data());

        // Computing ThetaPDF for outputRes[0] based off the single thetaPDF inference (thetaPDFs at index [0-(N_DIM_OUTPUT-1)])
        const float* thetaPDFs = networkOutputs.data();
        ParallelFor(outputRes[0] * outputRes[1], KRR_DEVICE_LAMBDA (int idx) {
            float* wiPtr_i = wiPtr + idx * (N_WI_PER_PIXEL * N_DIM_WI);
            const float pdf_i = sample_pdf<float, true>(thetaPDFs, wiPtr_i[0] * N_DIM_OUTPUT, N_DIM_OUTPUT);
            wiPtr_i[2] = pdf_i;
        }, stream);

        // Network inference (phi)
        mNetworkPhi->inference(streamPhi, networkPhiInputs, networkPhiOutputs);
        LinearKernel(softmaxMultFactor<float, N_DIM_OUTPUT_PHI, N_DIM_OUTPUT_PHI, N_DIM_OUTPUT_PHI>, streamPhi, phiNumInferenceSamples, networkPhiOutputs.data());

        const float* phiPDFs = networkPhiOutputs.data();
        ParallelFor(outputRes[0] * outputRes[1], KRR_DEVICE_LAMBDA (int idx) {
            float* wiPtr_i = wiPtr + idx * (N_WI_PER_PIXEL * N_DIM_WI);
            const float* phiPDF_i = phiPDFs + idx * N_DIM_OUTPUT_PHI;
            const float pdf_i = sample_pdf<float, false>(phiPDF_i, wiPtr_i[1] * N_DIM_OUTPUT_PHI, N_DIM_OUTPUT_PHI);
            wiPtr_i[3] = pdf_i;
        }, streamPhi);


        // Compute Final PDF
        cudaDeviceSynchronize();
        ParallelFor(outputRes[0] * outputRes[1], KRR_DEVICE_LAMBDA (int idx) {
            const float* thetaPtr_i = wiPtr + idx * (N_WI_PER_PIXEL * N_DIM_WI);
            const float* phiPtr_i = wiPtr + idx * (N_WI_PER_PIXEL * N_DIM_WI);

            const float thetaPdf = thetaPtr_i[2];
            const float phiPdf = phiPtr_i[3];
            const float pdf_transform = squareToUniformSpherePdf();

            outputPDF[idx] = thetaPdf * phiPdf * pdf_transform;
        }, stream);
        cudaDeviceSynchronize();
    }
}

void LearnedDistribution::mouseOverSampling(const MouseOverInfoOutput* mouseOverInfo, float* outputDirections, uint numSamples, PCGSampler* samplers){
    cudaDeviceSynchronize();
    if (!hasInitializedNetwork()) logFatal("Network not initialized!");
    
    const cudaStream_t& stream = mGuiding.stream;
    const cudaStream_t& streamPhi = mStream;

    const int thetaNumInferenceSamples = 1;
	const int thetaPaddedBatchSize = next_multiple(thetaNumInferenceSamples, 128);
    GPUMatrix<float> networkInputs(inferenceInputBuffer.data(), N_DIM_INPUT, thetaPaddedBatchSize);
    GPUMatrix<float> networkOutputs(inferenceOutputBuffer.data(), N_DIM_OUTPUT, thetaPaddedBatchSize);
    {
        // Theta input (1 entry)
        float* inputDataTheta = inferenceInputBuffer.data();
        GPUCall(KRR_DEVICE_LAMBDA() {
            *(Vector3f *) &inputDataTheta[0] = mouseOverInfo->position;
            #if GUIDED_PRODUCT_SAMPLING
                *(Vector3f *) &inputDataTheta[N_DIM_SPATIAL_INPUT] = (mouseOverInfo->wo + Vector3f::Ones()) * 0.5f;
            #endif
            #if GUIDED_AUXILIARY_INPUT
                *(Vector2f *) &inputDataTheta[N_DIM_SPATIAL_INPUT + N_DIM_DIRECTIONAL_INPUT] =
                    mouseOverInfo->normal;
                inputDataTheta[N_DIM_SPATIAL_INPUT + N_DIM_DIRECTIONAL_INPUT + 2] =
                    1 - expf(-mouseOverInfo->roughness);
            #endif
        });
        cudaDeviceSynchronize();


        float* networkOutputsPtr = networkOutputs.data();
        // Network inference (theta)
        mNetworkTheta->inference(stream, networkInputs, networkOutputs);
        CUDA_SYNC_CHECK();

        LinearKernel(softmaxMultFactor<float, N_DIM_OUTPUT, N_DIM_OUTPUT, N_DIM_OUTPUT>, stream, thetaNumInferenceSamples, networkOutputs.data());
        LinearKernel(sample_dir_guided_inputs_mouseover_sampling<true, N_DIM_OUTPUT, N_DIM_OUTPUT>, stream, numSamples, samplers, networkOutputsPtr, outputDirections);
        CUDA_SYNC_CHECK();
    }
    

    const uint numPhiBatches = (uint)numSamples / mGuiding.batchSize + 1;
    Log(Info, "Num Phi Batches: %d\n", numPhiBatches);
    for (int iter = 0; iter < numPhiBatches; iter++){
        const int offset = iter * mGuiding.batchSize;
        size_t localBatchSize = min((size_t)numSamples - offset, (size_t)mGuiding.batchSize);
        const int phiPaddedBatchSize = next_multiple((int) localBatchSize, 128);

        Log(Info, "%d: Offset: %d. BatchSize: %d. Padded Batchsize: %d\n", iter, offset, localBatchSize, phiPaddedBatchSize);
		
        GPUMatrix<float> networkInputsPhi(inferencePhiInputBuffer.data(), N_DIM_INPUT_PHI, phiPaddedBatchSize);
        GPUMatrix<float> networkOutputsPhi(inferencePhiOutputBuffer.data(), N_DIM_OUTPUT_PHI, phiPaddedBatchSize);
        // Phi
        LinearKernel(preparePhiInputs_mouseover_sampling<float>, stream, localBatchSize, outputDirections + offset * 2, networkInputs.data(), networkInputsPhi.data());

        mNetworkPhi->inference(stream, networkInputsPhi, networkOutputsPhi);
        CUDA_SYNC_CHECK();

        float* networkOutputsPtr = networkOutputsPhi.data();
        LinearKernel(softmaxMultFactor<float, N_DIM_OUTPUT_PHI, N_DIM_OUTPUT_PHI, N_DIM_OUTPUT_PHI>, stream, localBatchSize, networkOutputsPhi.data());

        LinearKernel(sample_dir_guided_inputs_mouseover_sampling<false, N_DIM_OUTPUT_PHI, N_DIM_OUTPUT_PHI>, stream, localBatchSize, samplers + (offset % MOUSEOVER_SAMPLING_N_SAMPLERS), 
                    networkOutputsPtr, outputDirections + offset * 2);
    }
    
    CUDA_SYNC_CHECK();
}
#endif

KRR_NAMESPACE_END
