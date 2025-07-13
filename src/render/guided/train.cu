#include "train.h"

KRR_NAMESPACE_BEGIN


__global__ void generate_training_data_AR(const size_t nElements,
	uint trainPixelOffset, uint trainPixelStride,
	TrainBufferDuoInputs<GuidedInput, GuidedInputPhi, GuidedOutput>* trainBuffer,
	const GuidedPixelStateBuffer* guidedState,
	const AABB sceneAABB) {
	// this costs about 0.5ms
	const size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
	int pixelId = trainPixelOffset + tid * trainPixelStride;
	int depth = guidedState->curDepth[pixelId];
	if (tid >= nElements) return;
	
	
	for (int curDepth = 0; curDepth < depth; curDepth++) {
		GuidedInput inputTheta = {};
		GuidedInputPhi inputPhi = {};
		GuidedOutput output = {};
		
		const RadianceRecordItem& record = guidedState->records[curDepth][pixelId];
		if (record.delta) continue;	// do not incorporate samples that from a delta lobe.

		const Vector3f pos = normalizeSpatialCoord(record.pos, sceneAABB);
		inputTheta.pos = pos;
		inputPhi.pos = pos;

#if GUIDED_PRODUCT_SAMPLING
		const Vector3f dir = warp_direction_for_sh(sphericalToCartesian(record.wo[0], record.wo[1]));
		inputTheta.dir = dir;
		inputPhi.dir = dir;
#endif
#if GUIDED_AUXILIARY_INPUT > 0
		const float normal_theta = record.normal[0] * M_INV_PI;
		const float normal_phi = record.normal[1] * M_INV_2PI;

		inputTheta.auxiliary[0] = normal_theta;
		inputPhi.auxiliary[0] = normal_theta;

		inputTheta.auxiliary[1] = normal_phi;
		inputPhi.auxiliary[1] = normal_phi;

		const float roughness_transformed = warp_roughness_for_ob(record.roughness);
		inputTheta.auxiliary[2] = roughness_transformed;
		inputPhi.auxiliary[2] = roughness_transformed;
#endif
		const Vector3f cartesianWi = sphericalToCartesian(record.dir[0], record.dir[1]);
		const Vector2f wi = utils::uniformSphereToSquare(cartesianWi);

		#if (AR_LINEAR_INTERPOLATION == 1)
			inputPhi.theta = wi[0];
			const int theta_floor_idx = (float) safe_floor(wi[0] * N_DIM_OUTPUT, N_DIM_OUTPUT);
			const float weight = wi[0] * N_DIM_OUTPUT - theta_floor_idx;
			const float theta_linear = (theta_floor_idx + weight) / N_DIM_OUTPUT;
			inputPhi.theta = theta_linear;
		#else
			inputPhi.theta = (((float)safe_floor(wi[0] * N_DIM_OUTPUT, N_DIM_OUTPUT)) / N_DIM_OUTPUT);
		#endif

		Color L = Color::Zero();
		for (int ch = 0; ch < Color::dim; ch++) {
			if (record.thp[ch] > M_EPSILON)
				L[ch] = record.L[ch] / record.thp[ch];
		}

		L *= record.misWeight; /*@addition MIS-aware distribution*/
		output.L = L;
		output.dir = wi;

#if GUIDED_PRODUCT_SAMPLING
		output.L *= record.bsdfVal;
#endif
		output.wiPdf = record.wiPdf;

		if (!(inputTheta.pos.hasNaN() ||output.dir.hasNaN() 
#if GUIDED_PRODUCT_SAMPLING
			|| inputTheta.dir.hasNaN()
#endif
			|| isnan(output.wiPdf) || output.wiPdf == 0 || L.hasNaN())){
			
			// if (output.L.mean() > 0.0f)
				trainBuffer->push(inputTheta, inputPhi, output);
		}	
		else printf("Find invalid training sample! (quite not expected...\n Pos: %f %f %f\n Dir: %f %f \n record.dir: %f %f \n L: %f %f %f\n record.L: %f %f %f\n record.thp[ch]: %f %f %f\n record.misWeight %f\n wiPdf: %f\n",
			inputTheta.pos[0], inputTheta.pos[1], inputTheta.pos[2],
			output.dir[0], output.dir[1],
			record.dir[0], record.dir[1],
			L[0], L[1], L[2],
			record.L[0], record.L[1], record.L[2],
			record.thp[0], record.thp[1], record.thp[2],
			record.misWeight,
			output.wiPdf);
	}
}


__global__ void generate_training_data_nrc(const size_t nElements,
	uint trainPixelOffset, uint trainPixelStride,
	TrainBuffer<GuidedInputNRC, GuidedOutputNRC>* trainBuffer,
	const GuidedPixelStateBuffer* guidedState,
	const AABB sceneAABB) {
	// this costs about 0.5ms
	const size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
	int pixelId = trainPixelOffset + tid * trainPixelStride;
	/* 
		Depth here is subtracted by 1, since the last bounce is only for inference at the next stage. 
		We do not have valid L estimate for the last bounce. 
	*/ 
	int depth = guidedState->curDepth[pixelId] - 1; 
	// int depth = guidedState->curDepth[pixelId]; 
	if (tid >= nElements) return;
	
	for (int curDepth = 0; curDepth < depth; curDepth++) {
		GuidedInputNRC input = {};
		
		const RadianceRecordItem& record = guidedState->records[curDepth][pixelId];
		if (record.delta || record.miss) continue;	// do not incorporate samples that from a delta lobe.

		input.pos = normalizeSpatialCoord(record.pos, sceneAABB);
		input.dir = warp_direction_for_sh(sphericalToCartesian(record.wo[0], record.wo[1]));
		input.auxiliary[0] = record.normal[0] * M_INV_PI;
		input.auxiliary[1] = record.normal[1] * M_INV_2PI;
		input.auxiliary[2] = warp_roughness_for_ob(record.roughness);
		input.auxiliary[3] = record.diffuse[0];
		input.auxiliary[4] = record.diffuse[1];
		input.auxiliary[5] = record.diffuse[2];
		input.auxiliary[6] = record.specular[0];
		input.auxiliary[7] = record.specular[1];
		input.auxiliary[8] = record.specular[2];

		Color L = Color::Zero();
		for (int ch = 0; ch < Color::dim; ch++) {
			if (record.thp[ch] > M_EPSILON)
				L[ch] = record.L[ch] / record.thp[ch];
		}
		L *= record.misWeight; /*@addition MIS-aware distribution*/
		L *= record.bsdfVal;
		L /= record.wiPdf + M_EPSILON; /* Include division by wiPdf, since grad is computed in trainer*/

		if (!(input.pos.hasNaN()
			|| input.dir.hasNaN()
			|| L.hasNaN())){
			trainBuffer->push(input, {L});
		}	
		else printf("Find invalid training sample! (quite not expected...\n Pos: %f %f %f\n L: %f %f %f\n record.L: %f %f %f\n record.thp[ch]: %f %f %f\n record.misWeight %f\n",
			input.pos[0], input.pos[1], input.pos[2],
			L[0], L[1], L[2],
			record.L[0], record.L[1], record.L[2],
			record.thp[0], record.thp[1], record.thp[2],
			record.misWeight);
	}
}

__global__ void generate_training_data_nrc_with_inference(const size_t nElements,
	uint trainPixelOffset, uint trainPixelStride,
	TrainBuffer<GuidedInputNRC, GuidedOutputNRC>* trainBuffer,
	NRCInputBuffer<GuidedInputNRC>* NRCinputBuffer,
	const GuidedPixelStateBuffer* guidedState,
	const AABB sceneAABB){
	
	const size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
	int pixelId = trainPixelOffset + tid * trainPixelStride; 
	int depth = guidedState->curDepth[pixelId]; 
	if (tid >= nElements) return;
	
	for (int curDepth = 0; curDepth < depth; curDepth++) {
		GuidedInputNRC input = {};
		
		const RadianceRecordItem& record = guidedState->records[curDepth][pixelId];
		if (record.delta || record.miss) continue;	// do not incorporate samples that from a delta lobe.

		input.pos = normalizeSpatialCoord(record.pos, sceneAABB);
		input.dir = warp_direction_for_sh(sphericalToCartesian(record.wo[0], record.wo[1]));
		input.auxiliary[0] = record.normal[0] * M_INV_PI;
		input.auxiliary[1] = record.normal[1] * M_INV_2PI;
		input.auxiliary[2] = warp_roughness_for_ob(record.roughness);
		input.auxiliary[3] = record.diffuse[0];
		input.auxiliary[4] = record.diffuse[1];
		input.auxiliary[5] = record.diffuse[2];
		input.auxiliary[6] = record.specular[0];
		input.auxiliary[7] = record.specular[1];
		input.auxiliary[8] = record.specular[2];

		Color L = Color::Zero();
		for (int ch = 0; ch < Color::dim; ch++) {
			if (record.thp[ch] > M_EPSILON)
				L[ch] = record.L[ch] / record.thp[ch];
		}
		L *= record.misWeight; /*@addition MIS-aware distribution*/
		L *= record.bsdfVal;
		L /= record.wiPdf + M_EPSILON; /* Include division by wiPdf, since grad is computed in trainer*/

		if (!(input.pos.hasNaN()
			|| input.dir.hasNaN()
			|| L.hasNaN())){
			NRCinputBuffer->push(input, curDepth, pixelId);
			if (curDepth < depth - 1) // Do not push the last depth to the training buffer
				trainBuffer->push(input, {L});
		}	
		else printf("Find invalid training sample! (quite not expected...\n Pos: %f %f %f\n L: %f %f %f\n record.L: %f %f %f\n record.thp[ch]: %f %f %f\n record.misWeight %f\n",
			input.pos[0], input.pos[1], input.pos[2],
			L[0], L[1], L[2],
			record.L[0], record.L[1], record.L[2],
			record.thp[0], record.thp[1], record.thp[2],
			record.misWeight);
	}


	// Decrement current depth of each training pixel by 1, since we don't have NRC ref for last depth
	guidedState->curDepth[pixelId]--;
}


__global__ void generate_NRCinput_from_training_data(const size_t nElements, 
	uint trainPixelOffset, uint trainPixelStride, 
	NRCInputBuffer<GuidedInputNRC>* NRCinputBuffer,
	GuidedPixelStateBuffer* guidedState, 
	const AABB sceneAABB) {
	
	const size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
	int pixelId = trainPixelOffset + tid * trainPixelStride;
	int depth = guidedState->curDepth[pixelId];
	if (tid >= nElements) return;

	int curDepth = 0;
#if (OPTIMIZE_ON_NRC == 1 || OPTIMIZE_ON_NRC == 2)
	curDepth = 1;
#elif (OPTIMIZE_ON_NRC == 3)
	depth--;
#endif

	for (; curDepth < depth; curDepth++) {
		GuidedInputNRC inputNRC = {};
		
		const RadianceRecordItem& record = guidedState->records[curDepth][pixelId];
		if (record.delta || record.miss) continue;	// do not incorporate samples that from a delta lobe.

		inputNRC.pos = normalizeSpatialCoord(record.pos, sceneAABB);
		inputNRC.dir = warp_direction_for_sh(sphericalToCartesian(record.wo[0], record.wo[1]));
		inputNRC.auxiliary[0] = record.normal[0] * M_INV_PI;
		inputNRC.auxiliary[1] = record.normal[1] * M_INV_2PI;
		inputNRC.auxiliary[2] = warp_roughness_for_ob(record.roughness);
		inputNRC.auxiliary[3] = record.diffuse[0];
		inputNRC.auxiliary[4] = record.diffuse[1];
		inputNRC.auxiliary[5] = record.diffuse[2];
		inputNRC.auxiliary[6] = record.specular[0];
		inputNRC.auxiliary[7] = record.specular[1];
		inputNRC.auxiliary[8] = record.specular[2];

		if (!(inputNRC.pos.hasNaN() || inputNRC.dir.hasNaN())){
			NRCinputBuffer->push(inputNRC, curDepth, pixelId);
		}	
		else printf("Find invalid training sample as NRC input! (quite not expected...\n Pos: %f %f %f\n Dir: %f %f %f\n",
			inputNRC.pos[0], inputNRC.pos[1], inputNRC.pos[2],
			inputNRC.dir[0], inputNRC.dir[1], inputNRC.dir[2]
		);
	}

	// Decrement current depth of each training pixel by 1, since we don't have NRC ref for last depth
	guidedState->curDepth[pixelId]--;
}

__global__ void apply_NRCoutput_to_training_data(const size_t nElements, 
	const int* depths, const int* pixelIds, const float* NRCoutput,
	GuidedPixelStateBuffer* guidedState) {
	const size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (tid >= nElements) return;

	int depth_i = depths[tid];
	int pixelId_i = pixelIds[tid];
	const Color nrcOutput_i = Color(NRCoutput[tid * N_DIM_OUTPUT_NRC], NRCoutput[tid * N_DIM_OUTPUT_NRC + 1], NRCoutput[tid * N_DIM_OUTPUT_NRC + 2]);

	#if (OPTIMIZE_ON_NRC == 1 || OPTIMIZE_ON_NRC == 4)
		// Replaces indirect radiance of the previous depth since NRC input is aligned with current Record, not RecordNext
		// Function takes care of depth == 0 edge case
		guidedState->replaceIndirectRadianceSkipDelta(
			pixelId_i, depth_i - 1, nrcOutput_i
		);
	#elif (OPTIMIZE_ON_NRC == 2 || OPTIMIZE_ON_NRC == 5)
		// Propagates radiance only at the last bounce inference
		int curDepth = guidedState->getCurDepth(pixelId_i);
		if (depth_i == curDepth) 
			guidedState->propagateIndirectRadiance(
				pixelId_i, depth_i - 1, nrcOutput_i
			);
	#elif (OPTIMIZE_ON_NRC == 3)
		guidedState->divideRadiance(
			pixelId_i, depth_i, nrcOutput_i
		);
	#endif
}

__global__ void divide_training_data_by_NRCoutput(const size_t nElements, 
	const int* depths, const int* pixelIds, const float* NRCoutput,
	GuidedPixelStateBuffer* guidedState) {
	const size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (tid >= nElements) return;

	int depth_i = depths[tid];
	int pixelId_i = pixelIds[tid];
	const Color nrcOutput_i = Color(NRCoutput[tid * N_DIM_OUTPUT_NRC], NRCoutput[tid * N_DIM_OUTPUT_NRC + 1], NRCoutput[tid * N_DIM_OUTPUT_NRC + 2]);

	guidedState->divideRadiance(
		pixelId_i, depth_i, nrcOutput_i
	);
}


__global__ void generate_inference_data(const size_t nElements,
	ScatterRayQueue* scatterRayQueue, float* data, const AABB sceneAABB) {
	
	const size_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= scatterRayQueue->size()) return;
	uint data_idx		  = i * N_DIM_INPUT;
	const ShadingData &sd = scatterRayQueue->operator[](i).operator krr::ScatterRayWorkItem().sd;
	Vector3f pos		  = normalizeSpatialCoord(sd.pos, sceneAABB);
	
	*(Vector3f *) &data[data_idx] = pos;

#if GUIDED_PRODUCT_SAMPLING
	*(Vector3f *) &data[data_idx + N_DIM_SPATIAL_INPUT] = warp_direction_for_sh(sd.wo.normalized());
#endif

#if GUIDED_AUXILIARY_INPUT
	*(Vector2f *) &data[data_idx + N_DIM_SPATIAL_INPUT + N_DIM_DIRECTIONAL_INPUT] =
		utils::cartesianToSphericalNormalized(sd.frame.N);
	data[data_idx + N_DIM_SPATIAL_INPUT + N_DIM_DIRECTIONAL_INPUT + 2] =
		warp_roughness_for_ob(sd.roughness);
#endif
}

__global__ void generate_inference_data_AR(const size_t nElements,
	ScatterRayQueue* scatterRayQueue, const float* wiBuffer, float* data, float* dataPhi, const AABB sceneAABB) {
	
	const size_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= scatterRayQueue->size()) return;
	uint data_idx		  = i * N_DIM_INPUT;
	uint dataPhi_idx	  = i * N_DIM_INPUT_PHI;
	const ShadingData &sd = scatterRayQueue->operator[](i).operator krr::ScatterRayWorkItem().sd;
	Vector3f pos		  = normalizeSpatialCoord(sd.pos, sceneAABB);
	
	*(Vector3f *) &data[data_idx] = pos;
	*(Vector3f *) &dataPhi[dataPhi_idx] = pos;

#if GUIDED_PRODUCT_SAMPLING
	const Vector3f wo_input = warp_direction_for_sh(sd.wo.normalized());
	*(Vector3f *) &data[data_idx + N_DIM_SPATIAL_INPUT] = wo_input;
	*(Vector3f *) &dataPhi[dataPhi_idx + N_DIM_SPATIAL_INPUT] = wo_input;
#endif

#if GUIDED_AUXILIARY_INPUT
	const Vector2f normal_input = utils::cartesianToSphericalNormalized(sd.frame.N);
	*(Vector2f *) &data[data_idx + N_DIM_SPATIAL_INPUT + N_DIM_DIRECTIONAL_INPUT] = normal_input;
	*(Vector2f *) &dataPhi[dataPhi_idx + N_DIM_SPATIAL_INPUT + N_DIM_DIRECTIONAL_INPUT] = normal_input;

	const float roughness_input = warp_roughness_for_ob(sd.roughness);		
	data[data_idx + N_DIM_SPATIAL_INPUT + N_DIM_DIRECTIONAL_INPUT + 2] = roughness_input;
	dataPhi[dataPhi_idx + N_DIM_SPATIAL_INPUT + N_DIM_DIRECTIONAL_INPUT + 2] = roughness_input;
#endif
}	


__global__ void compute_AR_pdf_train_combined(const size_t nElements, float* wiBuffer, const precision_t* predThetaPDFBuffer, const precision_t* predPhiPDFBuffer, const GuidedOutput* outputReference) {
	const size_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= nElements) return;

	// Save wi from outputRef to wiBuffer
	GuidedOutput output_i = outputReference[i];
	float* wi_Theta_i = wiBuffer + i * N_DIM_WI;
	float* wi_Phi_i = wiBuffer + i * N_DIM_WI + 1;
	float* savePDF_Theta_i = wiBuffer + i * N_DIM_WI + 2;
	float* savePDF_Phi_i = wiBuffer + i * N_DIM_WI + 3;

	const precision_t* pdf_Theta_i = predThetaPDFBuffer + i * N_DIM_PADDED_OUTPUT;
	const precision_t* pdf_Phi_i = predPhiPDFBuffer + i * N_DIM_PADDED_OUTPUT_PHI;

	*wi_Theta_i = output_i.dir[0];
	*wi_Phi_i = output_i.dir[1];

	const float theta_pdf = sample_pdf<precision_t, true>(pdf_Theta_i, output_i.dir[0] * N_DIM_OUTPUT, N_DIM_OUTPUT);
	*savePDF_Theta_i = theta_pdf;
	const float phi_pdf = sample_pdf<precision_t, false>(pdf_Phi_i, output_i.dir[1] * N_DIM_OUTPUT_PHI, N_DIM_OUTPUT_PHI);
	*savePDF_Phi_i = phi_pdf;
}


KRR_NAMESPACE_END