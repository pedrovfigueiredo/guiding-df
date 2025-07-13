/*This file should only be included in CUDA cpp files*/

#pragma once
#include <cuda.h>
#include <cuda_runtime.h>

#include "device/cuda.h"
#include "device/atomic.h"
#include "common.h"
#include "guideditem.h"
#include "workqueue.h"
#include "interop.h"

#include "util/math_utils.h"
#include "AR.h"

using precision_t = tcnn::network_precision_t;

// #define GRADIENT_CHECK

KRR_NAMESPACE_BEGIN

template <typename T>
class DeviceBuffer {
public:
	DeviceBuffer() = default;

	KRR_HOST DeviceBuffer(int n):
		mMaxSize(n) {
		cudaMalloc(&mData, n * sizeof(T));
	}

	KRR_CALLABLE int push(const T& w) {
		int index = allocateEntry();
		DCHECK_LT(index, mMaxSize);
		(*this)[index % mMaxSize] = w;
		return index;
	}

	KRR_CALLABLE void clear() {
		mSize.store(0);
	}

	KRR_CALLABLE int size() const {
		return mSize.load();
	}

	KRR_CALLABLE T* data() { return mData; }

	KRR_CALLABLE T& operator [] (int index) {
		DCHECK_LT(index, mMaxSize);
		return mData[index];
	}

	KRR_CALLABLE DeviceBuffer& operator=(const DeviceBuffer& w) {
		mSize.store(w.mSize);
		mMaxSize = w.mMaxSize;
		return *this;
	}

protected:
	KRR_CALLABLE int allocateEntry() {
		return mSize.fetch_add(1);
	}

private:
	atomic<int> mSize;
	T* mData;
	int mMaxSize{ 0 };
};

template <typename T>
class NRCInputBuffer{
public:
	struct CombinedOutput {
		T* data;
		int* depth;
		int* pixelId;
	};

	NRCInputBuffer() = default;

	KRR_HOST NRCInputBuffer(int n):
		mMaxSize(n) {
		cudaMalloc(&mData, n * sizeof(T));
		cudaMalloc(&mDepths, n * sizeof(int));
		cudaMalloc(&mPixelIds, n * sizeof(int));
	}

	KRR_CALLABLE int push(const T& w, int depth, int pixelId){
		int index = allocateEntry();
		DCHECK_LT(index, mMaxSize);
		mData[index % mMaxSize] = w;
		mDepths[index % mMaxSize] = depth;
		mPixelIds[index % mMaxSize] = pixelId;
		return index;
	}

	KRR_CALLABLE void clear() {
		mSize.store(0);
	}

	KRR_CALLABLE int size() const {
		return mSize.load();
	}

	KRR_CALLABLE T* data() { return mData; }
	KRR_CALLABLE int* depths() { return mDepths; }
	KRR_CALLABLE int* pixelIds() { return mPixelIds; }

	KRR_CALLABLE CombinedOutput operator [] (int index) {
		DCHECK_LT(index, mMaxSize);
		CombinedOutput output;
		output.data = &(mData[index]);
		output.depth = &(mDepths[index]);
		output.pixelId = &(mPixelIds[index]);
		return output;
	}

	KRR_CALLABLE NRCInputBuffer& operator=(const NRCInputBuffer& w) {
		mSize.store(w.mSize);
		mMaxSize = w.mMaxSize;
		return *this;
	}

protected:
	KRR_CALLABLE int allocateEntry() {
		return mSize.fetch_add(1);
	}

private:
	atomic<int> mSize;
	T* mData;
	int* mDepths;
	int* mPixelIds;
	int mMaxSize{ 0 };
};

template <typename Tin, typename Tout>
class TrainBuffer {
public:
	TrainBuffer() = default;

	KRR_HOST TrainBuffer(int n) :
		mMaxSize(n) {
		cudaMalloc(&mInputs, n * sizeof(Tin));
		cudaMalloc(&mOutputs, n * sizeof(Tout));
	}

	KRR_CALLABLE int push(const Tin& input, 
		const Tout& output) {
		int index = allocateEntry();
		DCHECK_LT(index, mMaxSize);
		mInputs[index] = input;
		mOutputs[index] = output;
		return index;
	}

	KRR_CALLABLE void clear() { 
		mSize.store(0);		
	}

	KRR_CALLABLE int size() const {
#ifndef KRR_DEVICE_CODE
		CUDA_SYNC_CHECK();
		cudaDeviceSynchronize();
#endif
		return mSize.load();
	}

	KRR_CALLABLE void resize(int n) {
		if (mMaxSize) {
			cudaFree(mInputs);
			cudaFree(mOutputs);
		}
		cudaMalloc(&mInputs, n * sizeof(Tin));
		cudaMalloc(&mOutputs, n * sizeof(Tout));
	}

	KRR_CALLABLE Tin* inputs() const { return mInputs; }

	KRR_CALLABLE Tout* outputs() const { return mOutputs; }

	KRR_CALLABLE TrainBuffer& operator=(const TrainBuffer& w) {
		mSize.store(w.mSize);
		mMaxSize = w.mMaxSize;
		return *this;
	}

private:
	KRR_CALLABLE int allocateEntry() {
		return mSize.fetch_add(1);
	}

	atomic<int> mSize;
	Tin* mInputs;
	Tout* mOutputs;
	int mMaxSize{ 0 };
};


template <typename Tin1, typename Tin2, typename Tout>
class TrainBufferDuoInputs {
public:
	TrainBufferDuoInputs() = default;

	KRR_HOST TrainBufferDuoInputs(int n) :
		mMaxSize(n) {
		cudaMalloc(&mInputs1, n * sizeof(Tin1));
		cudaMalloc(&mInputs2, n * sizeof(Tin2));
		cudaMalloc(&mOutputs, n * sizeof(Tout));
	}

	KRR_CALLABLE int push(const Tin1& input1, const Tin2& input2,
		const Tout& output) {
		int index = allocateEntry();
		DCHECK_LT(index, mMaxSize);
		mInputs1[index] = input1;
		mInputs2[index] = input2;
		mOutputs[index] = output;
		return index;
	}

	KRR_CALLABLE void clear() { 
		mSize.store(0);		
	}

	KRR_CALLABLE int size() const {
#ifndef KRR_DEVICE_CODE
		CUDA_SYNC_CHECK();
		cudaDeviceSynchronize();
#endif
		return mSize.load();
	}

	KRR_CALLABLE void resize(int n) {
		if (mMaxSize) {
			cudaFree(mInputs1);
			cudaFree(mInputs2);
			cudaFree(mOutputs);
		}
		cudaMalloc(&mInputs1, n * sizeof(Tin1));
		cudaMalloc(&mInputs2, n * sizeof(Tin2));
		cudaMalloc(&mOutputs, n * sizeof(Tout));
	}

	KRR_CALLABLE Tin1* inputs1() const { return mInputs1; }

	KRR_CALLABLE Tin2* inputs2() const { return mInputs2; }

	KRR_CALLABLE Tout* outputs() const { return mOutputs; }

	KRR_CALLABLE TrainBufferDuoInputs& operator=(const TrainBufferDuoInputs& w) {
		mSize.store(w.mSize);
		mMaxSize = w.mMaxSize;
		return *this;
	}

private:
	KRR_CALLABLE int allocateEntry() {
		return mSize.fetch_add(1);
	}

	atomic<int> mSize;
	Tin1* mInputs1;
	Tin2* mInputs2;
	Tout* mOutputs;
	int mMaxSize{ 0 };
};


KRR_CALLABLE Vector3f normalizeSpatialCoord(const Vector3f& coord, AABB aabb) {
	constexpr float inflation = 0.005f;
	aabb.inflate(aabb.diagonal().norm() * inflation);
	return Vector3f{ 0.5 } + (coord - aabb.center()) / aabb.diagonal();
}

__global__ void generate_training_data_AR(const size_t nElements, 
	uint trainPixelOffset, uint trainPixelStride, 
	TrainBufferDuoInputs<GuidedInput, GuidedInputPhi, GuidedOutput>* trainBuffer,
	const GuidedPixelStateBuffer* guidedState, 
	const AABB sceneAABB);

__global__ void generate_training_data_nrc(const size_t nElements,
	uint trainPixelOffset, uint trainPixelStride,
	TrainBuffer<GuidedInputNRC, GuidedOutputNRC>* trainBuffer,
	const GuidedPixelStateBuffer* guidedState,
	const AABB sceneAABB);

__global__ void generate_training_data_nrc_with_inference(const size_t nElements,
	uint trainPixelOffset, uint trainPixelStride,
	TrainBuffer<GuidedInputNRC, GuidedOutputNRC>* trainBuffer,
	NRCInputBuffer<GuidedInputNRC>* NRCinputBuffer,
	const GuidedPixelStateBuffer* guidedState,
	const AABB sceneAABB);

__global__ void generate_inference_data(const size_t nElements,
	ScatterRayQueue* scatterRayQueue, float* data, const AABB sceneAABB);

__global__ void generate_inference_data_AR(const size_t nElements,
	ScatterRayQueue* scatterRayQueue, const float* wiBuffer, float* data, 
	float* dataPhi, const AABB sceneAABB);


__global__ void generate_NRCinput_from_training_data(const size_t nElements, 
	uint trainPixelOffset, uint trainPixelStride, 
	NRCInputBuffer<GuidedInputNRC>* NRCinputBuffer,
	GuidedPixelStateBuffer* guidedState, 
	const AABB sceneAABB);

__global__ void compute_AR_pdf_train_combined(const size_t nElements, 
	float* wiBuffer, const precision_t* predThetaPDFBuffer, 
	const precision_t* predPhiPDFBuffer, const GuidedOutput* outputReference);


__global__ void apply_NRCoutput_to_training_data(const size_t nElements, 
	const int* depths, const int* pixelIds, const float* NRCoutput,
	GuidedPixelStateBuffer* guidedState);

__global__ void divide_training_data_by_NRCoutput(const size_t nElements, 
	const int* depths, const int* pixelIds, const float* NRCoutput,
	GuidedPixelStateBuffer* guidedState);

// Compute the CDF from the PDF
KRR_CALLABLE void cdf_from_pdf(const float* pdf, float* cdf, int n) {
	cdf[0] = pdf[0];
	for (int i = 1; i < n; i++)
		cdf[i] = cdf[i - 1] + pdf[i];
}

// Binary search for the index of the first element that is greater than rand_v
KRR_CALLABLE int search_sorted(const float* cdf, float rand_v, int n) {
	int l = 0, r = n - 1;
	while (l < r) {
		int mid = (l + r) / 2;
		if (rand_v < cdf[mid])
			r = mid;
		else
			l = mid + 1;
	}
	return l;
}

// Compute CDF from PDF up until rand_v for efficiency
template <int N>
KRR_CALLABLE int search_sorted_cdf_nearest(const float* pdf, 
	float* cdf_i_minus_one, float* cdf_i, float rand_v) {
	*cdf_i_minus_one = 0.f;
	int i = 0;

	TCNN_PRAGMA_UNROLL
	for (i = 0; i < N; i++) {
		*cdf_i = *cdf_i_minus_one + pdf[i] / N;
		if (rand_v < *cdf_i)
			break;
		*cdf_i_minus_one = *cdf_i;
	}
	return i;
}

template <int N, bool isTheta>
KRR_CALLABLE int search_sorted_cdf_linearinterpolation(const float* pdf, 
	float* cdf_i_minus_one, float* cdf_i, float rand_v) {
	*cdf_i_minus_one = 0.f;
	int i = 0;

	// if constexpr (isTheta)
	// 	*cdf_i = pdf[0] * 0.5f / N;
	// else{
	// 	*cdf_i = (pdf[N-1] + pdf[0]) * 0.5f / (2 * N);
	// }
	*cdf_i = pdf[0] * 0.5f / N;

	if (rand_v < *cdf_i)
		return i;
	
	
	*cdf_i_minus_one = *cdf_i;

	TCNN_PRAGMA_UNROLL
	for (i = 1; i < N; i++) {
		*cdf_i = *cdf_i_minus_one + (pdf[i-1] + pdf[i]) / (2 * N);
		if (rand_v < *cdf_i)
			return i;
		*cdf_i_minus_one = *cdf_i;
	}

	assert(i == N);
	// if constexpr (isTheta)
	// 	*cdf_i = *cdf_i_minus_one + pdf[N-1] * 0.5f / N;
	// else{
	// 	*cdf_i = *cdf_i_minus_one + (pdf[N-1] + pdf[0]) * 0.5f / (2 * N);
	// }
	*cdf_i = *cdf_i_minus_one + pdf[N-1] * 0.5f / N;

	return i;
}


template <typename T, bool isTheta>
KRR_CALLABLE void get_idxoffset_and_multiplier(int* idxoffset, T* multiplier, T* inv_multiplier){
	if constexpr (isTheta) {
		*idxoffset = 0;
		*multiplier = (T)M_PI;
		*inv_multiplier = (T)M_INV_PI;
	}
	else {
		*idxoffset = 1;
		*multiplier = (T)M_2PI;
		*inv_multiplier = (T)M_INV_2PI;
	}
}


template <bool isTheta, uint32_t dim_output, uint32_t padded_dim_output>
inline KRR_DEVICE void sample_dir_guided_inputs_F(const size_t nElements,
	const ScatterRayQueue* scatterRayQueue,
	const PixelStateBuffer* pixelState,
	const float* predPDFBuffer, const AuxInfo* auxData, float* wiBuffer) {
	const size_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= nElements) return;

	const AuxInfo* auxData_i = auxData + i;
	// Returns if the current sample is not guided
	if (i >= scatterRayQueue->size() || !(auxData_i->guided_pass)) return;
	
	const ScatterRayWorkItem w = scatterRayQueue->operator[](i);
	Sampler sampler = &pixelState->sampler[w.pixelId];
		
	const float* pdf_i = predPDFBuffer + i * padded_dim_output;

	int idxoffset = 0;
	if constexpr (!isTheta){
		idxoffset = 1;
	}

	float cdf_i_minus_one, cdf_i;	
	const float rand_v = sampler.get1D(); // Sample a random number between 0 and 1

	int wioffset = 0;
	float* wi_i = wiBuffer + i * (N_WI_PER_PIXEL * N_DIM_WI) + wioffset + idxoffset;

	const float one_over_dim_output = 1.f / dim_output;

#if (AR_LINEAR_INTERPOLATION == 1)
	int v_i_idx = search_sorted_cdf_linearinterpolation<dim_output, isTheta>(pdf_i, &cdf_i_minus_one, &cdf_i, rand_v);
	int v_i_idx_minus_one = v_i_idx - 1;
	float v_i_minus_one = v_i_idx_minus_one + 0.5f;
	float edge_case_multiplier = 1.f;
	float v_i_minus_one_offset = 0.f;


	if (v_i_idx_minus_one < 0){
		// if constexpr (isTheta)
		// 	v_i_idx_minus_one = v_i_idx;
		// else
		// 	v_i_idx_minus_one = dim_output - 1;
		v_i_idx_minus_one = v_i_idx;

		v_i_minus_one_offset = 0.5f;
		edge_case_multiplier = 0.5f;
	}
	else if (v_i_idx > dim_output - 1){
		// if constexpr (isTheta)
		// 	v_i_idx = v_i_idx_minus_one;
		// else
		// 	v_i_idx = 0;
		v_i_idx = v_i_idx_minus_one;
		edge_case_multiplier = 0.5f;
	}

	const float m = (pdf_i[v_i_idx] - pdf_i[v_i_idx_minus_one]) * dim_output;
	const float B = pdf_i[v_i_idx_minus_one] * (1 + v_i_minus_one) - pdf_i[v_i_idx] * v_i_minus_one;

	v_i_minus_one += v_i_minus_one_offset;

	const float A = m * 0.5f;
	// When A is 0, we have to linearly interpolate;
	if (abs(A) < M_EPSILON){
		const float v_diff = rand_v - cdf_i_minus_one;
		const float v_cdf_diff = cdf_i - cdf_i_minus_one;
		const float weight = v_cdf_diff > M_EPSILON ? v_diff / v_cdf_diff: sampler.get1D();
		const float sampled_dir_nearest = (v_i_minus_one + weight * edge_case_multiplier) * one_over_dim_output;

		*wi_i = min(sampled_dir_nearest, 1.f - M_EPSILON);
		return;
	}


	const float C = -(m * 0.5f) * (v_i_minus_one * v_i_minus_one * one_over_dim_output * one_over_dim_output) - (B * v_i_minus_one * one_over_dim_output) + cdf_i_minus_one - rand_v;
	const float delta = B * B - 4 * A * C;
	float sampled_dir;
	if (delta < M_EPSILON){
		sampled_dir = (v_i_minus_one + sampler.get1D() * edge_case_multiplier) * one_over_dim_output;
	}
	else{
		const float sqrt_delta = sqrtf(delta);
		const float sol_0 = (-B + sqrt_delta) / (2 * A);
		const float sol_1 = (-B - sqrt_delta) / (2 * A);
		const float sol = A >= 0 ? max(sol_0, sol_1) : min(sol_0, sol_1);
		sampled_dir = sol;
	}

	// if (sampled_dir < 0.f || sampled_dir > 1.f) {
	// 	printf("problematic v_i_idx: %d, v_i_idx_minus_one: %d, v_i_minus_one: %f, rand_v: %f, A: %f, B: %f, C: %f, delta: %f, sampled_dir: %f\n", v_i_idx, v_i_idx_minus_one, v_i_minus_one, rand_v, A, B, C, delta, sampled_dir);
	// }
#else
	const int v_i = search_sorted_cdf_nearest<dim_output>(pdf_i, &cdf_i_minus_one, &cdf_i, rand_v);
	const float v_diff = rand_v - cdf_i_minus_one;
	const float v_cdf_diff = cdf_i - cdf_i_minus_one;
	const float weight = v_cdf_diff > M_EPSILON ? v_diff / v_cdf_diff: sampler.get1D();
	const float sampled_dir = (v_i + weight) / dim_output;
#endif
	// Writing sampled direction to wiBuffer
	// if constexpr (isTheta)
	// 	*wi_i = min(sampled_dir, 1.f);
	// else
	// 	*wi_i = fmod(sampled_dir, 1.f);
	*wi_i = min(sampled_dir, 1.f - M_EPSILON);
}


template <bool isTheta, uint32_t dim_output, uint32_t padded_dim_output>
__global__ void sample_dir_guided_inputs(const size_t nElements,
	const ScatterRayQueue* scatterRayQueue,
	const PixelStateBuffer* pixelState,
	const float* predPDFBuffer, const AuxInfo* auxData, float* wiBuffer){
	sample_dir_guided_inputs_F<isTheta, dim_output, padded_dim_output>(nElements, scatterRayQueue, pixelState, predPDFBuffer, auxData, wiBuffer);
}

template <bool isTheta, uint32_t dim_output, uint32_t padded_dim_output>
__global__ void sample_dir_guided_inputs_mouseover_sampling(const size_t nElements,
	PCGSampler* samplers,
	const float* predPDFBuffer, float* wiBuffer) {
	const size_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= nElements) return;
	
	Sampler sampler = &samplers[i % MOUSEOVER_SAMPLING_N_SAMPLERS];
	size_t pdfIdxOffset = 0;
	size_t idxoffset = 0;
	if constexpr (!isTheta){
		pdfIdxOffset = i;
		idxoffset = 1;
	}
	const float* pdf_i = predPDFBuffer + pdfIdxOffset * padded_dim_output;
	
	float cdf_i_minus_one, cdf_i;
	const float rand_v = sampler.get1D(); // Sample a random number between 0 and 1
	float* wi_i = wiBuffer + i * 2 + idxoffset;

	const float one_over_dim_output = 1.f / dim_output;

#if (AR_LINEAR_INTERPOLATION == 1)
	int v_i_idx = search_sorted_cdf_linearinterpolation<dim_output, isTheta>(pdf_i, &cdf_i_minus_one, &cdf_i, rand_v);
	int v_i_idx_minus_one = v_i_idx - 1;
	float v_i_minus_one = v_i_idx_minus_one + 0.5f;
	float edge_case_multiplier = 1.f;
	float v_i_minus_one_offset = 0.f;


	if (v_i_idx_minus_one < 0){
		// if constexpr (isTheta)
		// 	v_i_idx_minus_one = v_i_idx;
		// else
		// 	v_i_idx_minus_one = dim_output - 1;
		v_i_idx_minus_one = v_i_idx;

		v_i_minus_one_offset = 0.5f;
		edge_case_multiplier = 0.5f;
	}
	else if (v_i_idx > dim_output - 1){
		// if constexpr (isTheta)
		// 	v_i_idx = v_i_idx_minus_one;
		// else
		// 	v_i_idx = 0;
		v_i_idx = v_i_idx_minus_one;

		edge_case_multiplier = 0.5f;
	}

	const float m = (pdf_i[v_i_idx] - pdf_i[v_i_idx_minus_one]) * dim_output;
	const float b = pdf_i[v_i_idx_minus_one] * (1 + v_i_minus_one) - pdf_i[v_i_idx] * v_i_minus_one;

	v_i_minus_one += v_i_minus_one_offset;

	const float A = m * 0.5f;
	// When A is 0, we have to linearly interpolate;
	if (abs(A) < M_EPSILON){
		const float v_diff = rand_v - cdf_i_minus_one;
		const float v_cdf_diff = cdf_i - cdf_i_minus_one;
		const float weight = v_cdf_diff > M_EPSILON ? v_diff / v_cdf_diff: sampler.get1D();
		const float sampled_dir_nearest = (v_i_minus_one + weight * edge_case_multiplier) * one_over_dim_output;
		*wi_i = min(sampled_dir_nearest, 1.f - M_EPSILON);
		return;
	}

	const float B = b;
	const float C = -(m * 0.5f) * (v_i_minus_one * v_i_minus_one * one_over_dim_output * one_over_dim_output) - (B * v_i_minus_one * one_over_dim_output) + cdf_i_minus_one - rand_v;
	const float delta = B * B - 4 * A * C;
	float sampled_dir;
	if (delta < M_EPSILON){
		sampled_dir = (v_i_minus_one + sampler.get1D() * edge_case_multiplier) * one_over_dim_output;
	}
	else{
		const float sqrt_delta = sqrtf(max(delta, 0.f));
		const float sol_0 = (-B + sqrt_delta) / (2 * A);
		const float sol_1 = (-B - sqrt_delta) / (2 * A);
		const float sol = A >= 0 ? max(sol_0, sol_1) : min(sol_0, sol_1);
		sampled_dir = sol;
	}

	// if (sampled_dir < 0.f || sampled_dir >= 1.f) {
	// 	printf("problematic v_i_idx: %d, v_i_idx_minus_one: %d, v_i_minus_one: %f, rand_v: %f, A: %f, B: %f, C: %f, delta: %f, sampled_dir: %f\n", v_i_idx, v_i_idx_minus_one, v_i_minus_one, rand_v, A, B, C, delta, sampled_dir);
	// }
#else
	const int v_i = search_sorted_cdf_nearest<dim_output>(pdf_i, &cdf_i_minus_one, &cdf_i, rand_v);
	const float v_diff = rand_v - cdf_i_minus_one;
	const float v_cdf_diff = cdf_i - cdf_i_minus_one;
	const float weight = v_cdf_diff > M_EPSILON ? v_diff / v_cdf_diff: sampler.get1D();
	const float sampled_dir = (v_i + weight) * one_over_dim_output;
#endif

	// Writing sampled direction to wiBuffer
	// if constexpr (isTheta)
		// *wi_i = min(sampled_dir, 1.f);
	// else
	// 	*wi_i = fmod(sampled_dir, 1.f);
	*wi_i = min(sampled_dir, 1.f - M_EPSILON);
}


template <typename T, uint16_t n_decimals>
KRR_CALLABLE float round(const T val) {
	const float n_decimals_f = powf(10, n_decimals);
	const float n_decimals_f_inv = 1.0f / n_decimals_f;
	return roundf((float)val * n_decimals_f) * n_decimals_f_inv;
}

template <typename T>
KRR_CALLABLE int get_nearest_idx(T idx, int N){
	const int idx_floor = min(int(idx), N - 1);
	const int idx_ceil = min(idx_floor + 1, N - 1);
	// const T dir_weight = round<T, 2>(idx - idx_floor);
	const T dir_weight = idx - idx_floor;
	const int idx_nearest = dir_weight <= 0.5f ? idx_floor : idx_ceil;
	return idx_nearest;
}

template <typename T>
KRR_CALLABLE int safe_floor(T val, int N){
	return min((int)floor(val), N - 1);
}

template <typename T, bool isTheta>
KRR_CALLABLE float sample_pdf(const T* pdf_samples, float idx, int N){
#if (AR_LINEAR_INTERPOLATION == 1)
	idx -= 0.5f;
	int idx_floor = safe_floor(idx, N);
	int idx_ceil = idx_floor + 1;
	float dir_weight = idx - idx_floor;

	// For theta, we do nearest neighbor at the edges
	if (idx_floor < 0){
		idx_floor = N - 1;
		// if constexpr (isTheta)
			dir_weight = 1.0f;
	}
	if (idx_ceil > N - 1){
		idx_ceil = 0;
		// if constexpr (isTheta)
			dir_weight = 0.0f;
	}

	const float pdf0 = (float)pdf_samples[idx_floor];
	const float pdf1 = (float)pdf_samples[idx_ceil];
	return (1.0f - dir_weight) * pdf0 + dir_weight * pdf1;
#else
	// Doing neirest neighbor
	return (float)pdf_samples[safe_floor(idx, N)];
#endif
}

template <typename T, bool isTheta, bool isNEE>
__global__ void compute_AR_pdf(const size_t nElements, T* wiBuffer, const T* predPDFBuffer) {
	const size_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= nElements) return;

	int idxoffset;
	T multiplier, inv_multiplier;
	get_idxoffset_and_multiplier<T, isTheta>(&idxoffset, &multiplier, &inv_multiplier);

	const T* NEE_wi_i = wiBuffer + i * (N_WI_PER_PIXEL * N_DIM_WI) + 0 + idxoffset;
	T* NEE_savePDF_i = wiBuffer + i * (N_WI_PER_PIXEL * N_DIM_WI) + 2 + idxoffset; // +2 to skip theta, phi. idxoffset to skip theta in case it's phi

	int wioffset = 0;

	T* savePDF_i = NEE_savePDF_i + wioffset;
	const T* wi_i = NEE_wi_i + wioffset;


	if constexpr (isTheta){
		const T* pdf_i = predPDFBuffer + i * N_DIM_PADDED_OUTPUT;
		// For theta, we compute PDF for both NEE and BSDF/MIS using the same predPDFBuffer

		if constexpr (isNEE) {
			// const T NEE_dir_pdf = sample_pdf(pdf_i, (*NEE_wi_i) * inv_multiplier * (float)N_DIM_OUTPUT, N_DIM_OUTPUT);
			const T NEE_dir_pdf = sample_pdf<T, isTheta>(pdf_i, (*NEE_wi_i) * (float)N_DIM_OUTPUT, N_DIM_OUTPUT);
			*NEE_savePDF_i = NEE_dir_pdf;
		}else{
			const T dir_pdf = sample_pdf<T, isTheta>(pdf_i, (*wi_i) * (float)N_DIM_OUTPUT, N_DIM_OUTPUT);
			*savePDF_i = dir_pdf;
		}
	}
	else{ // For phi, we need to use a different predPDFBuffer for NEE, so we cannot compute the pdf for both in the same call
		const T* pdf_i = predPDFBuffer + i * N_DIM_PADDED_OUTPUT_PHI;
		if constexpr (isNEE) {
			// const T NEE_dir_pdf = sample_pdf(pdf_i, (*NEE_wi_i) * inv_multiplier * (float)N_DIM_OUTPUT, N_DIM_OUTPUT);
			const T NEE_dir_pdf = sample_pdf<T, isTheta>(pdf_i, (*NEE_wi_i) * (float)N_DIM_OUTPUT_PHI, N_DIM_OUTPUT_PHI);
			*NEE_savePDF_i = NEE_dir_pdf;
		}
		else{
			// const T dir_pdf = sample_pdf(pdf_i, (*wi_i) * inv_multiplier * (float)N_DIM_OUTPUT, N_DIM_OUTPUT);
			const T dir_pdf = sample_pdf<T, isTheta>(pdf_i, (*wi_i) * (float)N_DIM_OUTPUT_PHI, N_DIM_OUTPUT_PHI);
			*savePDF_i = dir_pdf;
		}
	}
}

template <typename T, uint32_t dim_output, uint32_t padded_dim_output, uint32_t dim_output_phi, uint32_t padded_dim_output_phi, bool isNEE>
inline KRR_DEVICE void compute_AR_pdf_combined_F(const size_t nElements, T* wiBuffer, const T* predPDFThetaBuffer, const T* predPDFPhiBuffer) {
	const size_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= nElements) return;

	const T* pdfTheta_i = predPDFThetaBuffer + i * padded_dim_output;
	const T* pdfPhi_i = predPDFPhiBuffer + i * padded_dim_output_phi;

	unsigned int NEEoffset = 0;

	const T* wi_Theta_i = wiBuffer + i * (N_WI_PER_PIXEL * N_DIM_WI) + NEEoffset;
	const T* wi_Phi_i = wiBuffer + i * (N_WI_PER_PIXEL * N_DIM_WI) + NEEoffset + 1;
	T* savePDF_Theta_i = wiBuffer + i * (N_WI_PER_PIXEL * N_DIM_WI) + NEEoffset + 2; // +2 to skip theta, phi. idxoffset to skip theta in case it's phi
	T* savePDF_Phi_i = wiBuffer + i * (N_WI_PER_PIXEL * N_DIM_WI) + NEEoffset + 3; // +2 to skip theta, phi. idxoffset to skip theta in case it's phi

	// Theta
	const T theta_pdf = sample_pdf<T, true>(pdfTheta_i, (*wi_Theta_i) * (float)dim_output, dim_output);
	*savePDF_Theta_i *= theta_pdf; // Multiplies with initial value (1.0 or -1.0) to retain sign of one-sided BSDFs

	// Phi
	const T phi_pdf = sample_pdf<T, false>(pdfPhi_i, (*wi_Phi_i) * (float)dim_output_phi, dim_output_phi);
	*savePDF_Phi_i *= phi_pdf; // Multiplies with initial value (1.0 or -1.0) to retain sign of one-sided BSDFs
}

template <typename T, uint32_t padded_dim_output, uint32_t dim_output_phi, uint32_t padded_dim_output_phi, bool isNEE>
__global__ void compute_AR_pdf_combined(const size_t nElements, T* wiBuffer, const T* predPDFThetaBuffer, const T* predPDFPhiBuffer) {
	compute_AR_pdf_combined_F<T, padded_dim_output, dim_output_phi, padded_dim_output_phi, isNEE>(nElements, wiBuffer, predPDFThetaBuffer, predPDFPhiBuffer);
}


template <bool isTheta>
__global__ void compute_AR_pdf_train(const size_t nElements, float* wiBuffer, const precision_t* predPDFBuffer, const GuidedOutput* outputReference) {
	const size_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= nElements) return;

	int idxoffset;
	float multiplier, inv_multiplier;
	get_idxoffset_and_multiplier<float, isTheta>(&idxoffset, &multiplier, &inv_multiplier);

	// Save wi from outputRef to wiBuffer
	GuidedOutput output_i = outputReference[i];
	float* wi_i = wiBuffer + i * N_DIM_WI + 0 + idxoffset;
	uint32_t dim_output, dim_padded_output;
	if constexpr (isTheta) {
		// *wi_i = clamp((float)output_i.dir[0], 0.0f, (float)1.f - M_EPSILON);
		*wi_i = output_i.dir[0];
		dim_output = N_DIM_OUTPUT;
		dim_padded_output = N_DIM_PADDED_OUTPUT;
	}
	else {
		// *wi_i = clamp((float)output_i.dir[1], 0.0f, (float)1.f - M_EPSILON);
		*wi_i = output_i.dir[1];
		dim_output = N_DIM_OUTPUT_PHI;
		dim_padded_output = N_DIM_PADDED_OUTPUT_PHI;
	}

	// Compute pdf for wi from predPDFBuffer
	const precision_t* pdf_i = predPDFBuffer + i * dim_padded_output;
	float* savePDF_i = wiBuffer + i * N_DIM_WI + 2 + idxoffset;

	const float dir_pdf = sample_pdf<precision_t, isTheta>(pdf_i, (*wi_i) * dim_output, dim_output);
	*savePDF_i = dir_pdf;
}

template <typename T, bool isNEE>
inline KRR_DEVICE void add_theta_to_input_F(const size_t nElements, const T* wiBuffer, T* inputBuffer){
	const size_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= nElements) return;

	int offset = 0;

	const T* wi_i = wiBuffer + i * (N_WI_PER_PIXEL * N_DIM_WI) + offset;
	T* input_i = inputBuffer + i * N_DIM_INPUT_PHI + (N_DIM_INPUT_PHI - 1);

	#if (AR_LINEAR_INTERPOLATION == 1)
		*input_i = *wi_i;
		const int theta_floor_idx = (float) safe_floor(*wi_i * N_DIM_OUTPUT, N_DIM_OUTPUT);
		const float weight = *wi_i * N_DIM_OUTPUT - theta_floor_idx;
		const float theta_linear = (theta_floor_idx + weight) / N_DIM_OUTPUT;
		*input_i = theta_linear; // Adds theta as last input to phi network
	#else
		*input_i = (T) ((float) safe_floor(*wi_i * N_DIM_OUTPUT, N_DIM_OUTPUT) / N_DIM_OUTPUT);
	#endif
}

template <typename T, bool isNEE>
__global__ void add_theta_to_input(const size_t nElements, const T* wiBuffer, T* inputBuffer){
	add_theta_to_input_F<T, isNEE>(nElements, wiBuffer, inputBuffer);
}

template <typename T>
__global__ void preparePhiInputs_mouseover_sampling(const size_t nElements, const T* wiBuffer, const T* inputBuffer, T* inputPhiBuffer){
	const size_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= nElements) return;
	
	const T* input_i = inputBuffer + 0 * N_DIM_INPUT; // Input to Theta Network (DIM = N_DIM_INPUT) is always the first index
	T* inputPhi_i = inputPhiBuffer + i * N_DIM_INPUT_PHI; // Input to Phi Network (DIM = N_DIM_PHI_INPUT)
	
	TCNN_PRAGMA_UNROLL
	for (int j = 0; j < N_DIM_INPUT; j++) {
		inputPhi_i[j] = input_i[j];
	}

	// Adding Theta to inputPhi
	const T* wi_i = wiBuffer + i * 2;
	#if (AR_LINEAR_INTERPOLATION == 1)
		const int theta_floor_idx = (float) safe_floor(*wi_i * N_DIM_OUTPUT, N_DIM_OUTPUT);
		const float weight = *wi_i * N_DIM_OUTPUT - theta_floor_idx;
		const float theta_linear = (theta_floor_idx + weight) / N_DIM_OUTPUT;
		inputPhi_i[N_DIM_INPUT] = theta_linear;
	#else
		inputPhi_i[N_DIM_INPUT] = (T) ((float) safe_floor(*wi_i * N_DIM_OUTPUT, N_DIM_OUTPUT) / N_DIM_OUTPUT);
	#endif
}

template <uint32_t N, typename T>
__host__ __device__ inline void softmax2(T vals[N]) {
	float max = 0.0f;
	TCNN_PRAGMA_UNROLL
		for (uint32_t i = 0; i < N; ++i) {
			max = fmaxf(max, (float)vals[i]);
		}
	
	float sum_exp_diff = 0.0f;
	TCNN_PRAGMA_UNROLL
		for (uint32_t i = 0; i < N; ++i) {
			sum_exp_diff += expf((float)vals[i] - max);
		}

	TCNN_PRAGMA_UNROLL
		for  (uint32_t i = 0; i < N; ++i) {
			vals[i] = (T) (expf((float)vals[i] - max - logf(sum_exp_diff)));
		}
}

template <typename T, uint32_t dim_output, uint32_t padded_dim_output>
__global__ void computeSoftmax(const size_t nElements, const T* data, T* softmax) {
	const size_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= nElements) return;
	softmax2<dim_output, T>(softmax + i * padded_dim_output);
}

template <typename T, uint32_t dim_output, uint32_t padded_dim_output, uint32_t factor>
inline KRR_DEVICE void softmaxMultFactor_F(const size_t nElements, T* softmax) {
	const size_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= nElements) return;
	T* softmax_i = softmax + i * padded_dim_output;
	softmax2<dim_output, T>(softmax_i);
	TCNN_PRAGMA_UNROLL
		for (uint32_t j = 0; j < dim_output; ++j) {
			softmax_i[j] *= (T) factor;
		}
}

// Macro to calm intelisense down
#ifdef __CUDACC__
    #define LBOUNDS(x) __launch_bounds__(x)
#else
    #define LBOUNDS(x)
#endif

template <typename T, uint32_t dim_output, uint32_t padded_dim_output, uint32_t factor>
__global__ void LBOUNDS(1024) softmaxMultFactor(const size_t nElements, T* softmax) {
	softmaxMultFactor_F<T, dim_output, padded_dim_output, factor>(nElements, softmax);
}

template <typename T, uint32_t dim_output, uint32_t padded_dim_output, uint32_t factor>
__global__ void thetaPostInference(const size_t nElements, T* networkOutputs, T* inputPhiBuffer, 
	const ScatterRayQueue* scatterRayQueue,
	const PixelStateBuffer* pixelState,
	const AuxInfo* auxData, float* wiBuffer) {
	const size_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= nElements) return;

	// SoftmaxMultFactor
	softmaxMultFactor_F<T, dim_output, padded_dim_output, factor>(nElements, networkOutputs);

	// Theta Sampling
	sample_dir_guided_inputs_F<true, dim_output, padded_dim_output>(nElements, scatterRayQueue, pixelState, networkOutputs, auxData, wiBuffer);

	// Add theta to inputPhi
	add_theta_to_input_F<T, false>(nElements, wiBuffer, inputPhiBuffer);
}

template <typename T, uint32_t dim_output, uint32_t padded_dim_output, uint32_t dim_output_phi, uint32_t padded_dim_output_phi, uint32_t factor>
__global__ void phiPostInference(const size_t nElements, T* networkOutputs, T* networkOutputsPhi, 
	const ScatterRayQueue* scatterRayQueue,
	const PixelStateBuffer* pixelState,
	const AuxInfo* auxData, float* wiBuffer) {
	const size_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= nElements) return;

	// SoftmaxMultFactor
	softmaxMultFactor_F<T, dim_output_phi, padded_dim_output_phi, factor>(nElements, networkOutputsPhi);

	// Phi Sampling
	sample_dir_guided_inputs_F<false, dim_output_phi, padded_dim_output_phi>(nElements, scatterRayQueue, pixelState, networkOutputsPhi, auxData, wiBuffer);

	// Compute PDF on MIS-BSDF Inputs
	compute_AR_pdf_combined_F<T, dim_output, padded_dim_output, dim_output_phi, padded_dim_output_phi, false>(nElements, wiBuffer, networkOutputs, networkOutputsPhi);
}

template <typename LearnedDist>
__global__ void combine_AR_pdf_and_dl_doutput(const size_t nElements,
	GuidedOutput* outputReference,
	precision_t* pdfs_theta, precision_t* pdfs_phi,
	precision_t* dL_doutput_theta, precision_t* dL_doutput_phi, 
	float* likelihood, float loss_scale){
	
	const size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (tid >= nElements) return;

	// Combining pdfs
	GuidedOutput output_i = outputReference[tid];
	precision_t* pdf_data_theta = pdfs_theta + tid * N_DIM_PADDED_OUTPUT;
	precision_t* pdf_data_phi = pdfs_phi + tid * N_DIM_PADDED_OUTPUT_PHI;

	float guidingData[4] = {output_i.dir[0], output_i.dir[1], 0.0f, 0.0f};
	guidingData[2] = sample_pdf<precision_t, true>(pdf_data_theta, output_i.dir[0] * N_DIM_OUTPUT, N_DIM_OUTPUT);
	guidingData[3] = sample_pdf<precision_t, false>(pdf_data_phi, output_i.dir[1] * N_DIM_OUTPUT_PHI, N_DIM_OUTPUT_PHI);

	// Computing gradients
	precision_t* gradient_data_theta = dL_doutput_theta + tid * N_DIM_PADDED_OUTPUT;
	precision_t* gradient_data_phi = dL_doutput_phi + tid * N_DIM_PADDED_OUTPUT_PHI;
	constexpr uint output_padding = N_DIM_PADDED_OUTPUT - N_DIM_OUTPUT;
	auto dist = LearnedDist::getDistribution(guidingData, 0, output_padding);

	float Li		= outputReference[tid].L.mean();
	float wiPdf		= outputReference[tid].wiPdf + M_EPSILON;
	float guidedPdf  = dist.pdf() * dist.pdfTransformInv() + M_EPSILON;
	if (isnan(guidedPdf)) printf("Found NaN in guidePdf!\n");
	
	loss_scale /= nElements;
	float prefix	= -Li / wiPdf / guidedPdf * loss_scale;
	likelihood[tid] = -Li / wiPdf * logf(guidedPdf);
	dist.compute_gradients(prefix, pdf_data_theta, pdf_data_phi, gradient_data_theta, gradient_data_phi);
}

template <std::size_t N>
KRR_DEVICE float3 calcLerp(const float x, const float3 (&data)[N]){
    const float xClamped = fminf(fmaxf(x, 0.0f), 1.0f);
    const float x0 = xClamped * (N - 1);
    const int i0 = static_cast<int>(x0);
    const int i1 = i0 + 1;
    const float t = x0 - i0;
	const Vector3f interp = Vector3f(data[i0]) * (1 - t) + Vector3f(data[i1]) * t;
    return make_float3(interp[0], interp[1], interp[2]);
}

template <typename FloatType>
KRR_DEVICE float3 luminanceToViridis(const FloatType lum) {
    constexpr float3 data[] =
        {
            { 0.267004, 0.004874, 0.329415 },
            { 0.268510, 0.009605, 0.335427 },
            { 0.269944, 0.014625, 0.341379 },
            { 0.271305, 0.019942, 0.347269 },
            { 0.272594, 0.025563, 0.353093 },
            { 0.273809, 0.031497, 0.358853 },
            { 0.274952, 0.037752, 0.364543 },
            { 0.276022, 0.044167, 0.370164 },
            { 0.277018, 0.050344, 0.375715 },
            { 0.277941, 0.056324, 0.381191 },
            { 0.278791, 0.062145, 0.386592 },
            { 0.279566, 0.067836, 0.391917 },
            { 0.280267, 0.073417, 0.397163 },
            { 0.280894, 0.078907, 0.402329 },
            { 0.281446, 0.084320, 0.407414 },
            { 0.281924, 0.089666, 0.412415 },
            { 0.282327, 0.094955, 0.417331 },
            { 0.282656, 0.100196, 0.422160 },
            { 0.282910, 0.105393, 0.426902 },
            { 0.283091, 0.110553, 0.431554 },
            { 0.283197, 0.115680, 0.436115 },
            { 0.283229, 0.120777, 0.440584 },
            { 0.283187, 0.125848, 0.444960 },
            { 0.283072, 0.130895, 0.449241 },
            { 0.282884, 0.135920, 0.453427 },
            { 0.282623, 0.140926, 0.457517 },
            { 0.282290, 0.145912, 0.461510 },
            { 0.281887, 0.150881, 0.465405 },
            { 0.281412, 0.155834, 0.469201 },
            { 0.280868, 0.160771, 0.472899 },
            { 0.280255, 0.165693, 0.476498 },
            { 0.279574, 0.170599, 0.479997 },
            { 0.278826, 0.175490, 0.483397 },
            { 0.278012, 0.180367, 0.486697 },
            { 0.277134, 0.185228, 0.489898 },
            { 0.276194, 0.190074, 0.493001 },
            { 0.275191, 0.194905, 0.496005 },
            { 0.274128, 0.199721, 0.498911 },
            { 0.273006, 0.204520, 0.501721 },
            { 0.271828, 0.209303, 0.504434 },
            { 0.270595, 0.214069, 0.507052 },
            { 0.269308, 0.218818, 0.509577 },
            { 0.267968, 0.223549, 0.512008 },
            { 0.266580, 0.228262, 0.514349 },
            { 0.265145, 0.232956, 0.516599 },
            { 0.263663, 0.237631, 0.518762 },
            { 0.262138, 0.242286, 0.520837 },
            { 0.260571, 0.246922, 0.522828 },
            { 0.258965, 0.251537, 0.524736 },
            { 0.257322, 0.256130, 0.526563 },
            { 0.255645, 0.260703, 0.528312 },
            { 0.253935, 0.265254, 0.529983 },
            { 0.252194, 0.269783, 0.531579 },
            { 0.250425, 0.274290, 0.533103 },
            { 0.248629, 0.278775, 0.534556 },
            { 0.246811, 0.283237, 0.535941 },
            { 0.244972, 0.287675, 0.537260 },
            { 0.243113, 0.292092, 0.538516 },
            { 0.241237, 0.296485, 0.539709 },
            { 0.239346, 0.300855, 0.540844 },
            { 0.237441, 0.305202, 0.541921 },
            { 0.235526, 0.309527, 0.542944 },
            { 0.233603, 0.313828, 0.543914 },
            { 0.231674, 0.318106, 0.544834 },
            { 0.229739, 0.322361, 0.545706 },
            { 0.227802, 0.326594, 0.546532 },
            { 0.225863, 0.330805, 0.547314 },
            { 0.223925, 0.334994, 0.548053 },
            { 0.221989, 0.339161, 0.548752 },
            { 0.220057, 0.343307, 0.549413 },
            { 0.218130, 0.347432, 0.550038 },
            { 0.216210, 0.351535, 0.550627 },
            { 0.214298, 0.355619, 0.551184 },
            { 0.212395, 0.359683, 0.551710 },
            { 0.210503, 0.363727, 0.552206 },
            { 0.208623, 0.367752, 0.552675 },
            { 0.206756, 0.371758, 0.553117 },
            { 0.204903, 0.375746, 0.553533 },
            { 0.203063, 0.379716, 0.553925 },
            { 0.201239, 0.383670, 0.554294 },
            { 0.199430, 0.387607, 0.554642 },
            { 0.197636, 0.391528, 0.554969 },
            { 0.195860, 0.395433, 0.555276 },
            { 0.194100, 0.399323, 0.555565 },
            { 0.192357, 0.403199, 0.555836 },
            { 0.190631, 0.407061, 0.556089 },
            { 0.188923, 0.410910, 0.556326 },
            { 0.187231, 0.414746, 0.556547 },
            { 0.185556, 0.418570, 0.556753 },
            { 0.183898, 0.422383, 0.556944 },
            { 0.182256, 0.426184, 0.557120 },
            { 0.180629, 0.429975, 0.557282 },
            { 0.179019, 0.433756, 0.557430 },
            { 0.177423, 0.437527, 0.557565 },
            { 0.175841, 0.441290, 0.557685 },
            { 0.174274, 0.445044, 0.557792 },
            { 0.172719, 0.448791, 0.557885 },
            { 0.171176, 0.452530, 0.557965 },
            { 0.169646, 0.456262, 0.558030 },
            { 0.168126, 0.459988, 0.558082 },
            { 0.166617, 0.463708, 0.558119 },
            { 0.165117, 0.467423, 0.558141 },
            { 0.163625, 0.471133, 0.558148 },
            { 0.162142, 0.474838, 0.558140 },
            { 0.160665, 0.478540, 0.558115 },
            { 0.159194, 0.482237, 0.558073 },
            { 0.157729, 0.485932, 0.558013 },
            { 0.156270, 0.489624, 0.557936 },
            { 0.154815, 0.493313, 0.557840 },
            { 0.153364, 0.497000, 0.557724 },
            { 0.151918, 0.500685, 0.557587 },
            { 0.150476, 0.504369, 0.557430 },
            { 0.149039, 0.508051, 0.557250 },
            { 0.147607, 0.511733, 0.557049 },
            { 0.146180, 0.515413, 0.556823 },
            { 0.144759, 0.519093, 0.556572 },
            { 0.143343, 0.522773, 0.556295 },
            { 0.141935, 0.526453, 0.555991 },
            { 0.140536, 0.530132, 0.555659 },
            { 0.139147, 0.533812, 0.555298 },
            { 0.137770, 0.537492, 0.554906 },
            { 0.136408, 0.541173, 0.554483 },
            { 0.135066, 0.544853, 0.554029 },
            { 0.133743, 0.548535, 0.553541 },
            { 0.132444, 0.552216, 0.553018 },
            { 0.131172, 0.555899, 0.552459 },
            { 0.129933, 0.559582, 0.551864 },
            { 0.128729, 0.563265, 0.551229 },
            { 0.127568, 0.566949, 0.550556 },
            { 0.126453, 0.570633, 0.549841 },
            { 0.125394, 0.574318, 0.549086 },
            { 0.124395, 0.578002, 0.548287 },
            { 0.123463, 0.581687, 0.547445 },
            { 0.122606, 0.585371, 0.546557 },
            { 0.121831, 0.589055, 0.545623 },
            { 0.121148, 0.592739, 0.544641 },
            { 0.120565, 0.596422, 0.543611 },
            { 0.120092, 0.600104, 0.542530 },
            { 0.119738, 0.603785, 0.541400 },
            { 0.119512, 0.607464, 0.540218 },
            { 0.119423, 0.611141, 0.538982 },
            { 0.119483, 0.614817, 0.537692 },
            { 0.119699, 0.618490, 0.536347 },
            { 0.120081, 0.622161, 0.534946 },
            { 0.120638, 0.625828, 0.533488 },
            { 0.121380, 0.629492, 0.531973 },
            { 0.122312, 0.633153, 0.530398 },
            { 0.123444, 0.636809, 0.528763 },
            { 0.124780, 0.640461, 0.527068 },
            { 0.126326, 0.644107, 0.525311 },
            { 0.128087, 0.647749, 0.523491 },
            { 0.130067, 0.651384, 0.521608 },
            { 0.132268, 0.655014, 0.519661 },
            { 0.134692, 0.658636, 0.517649 },
            { 0.137339, 0.662252, 0.515571 },
            { 0.140210, 0.665859, 0.513427 },
            { 0.143303, 0.669459, 0.511215 },
            { 0.146616, 0.673050, 0.508936 },
            { 0.150148, 0.676631, 0.506589 },
            { 0.153894, 0.680203, 0.504172 },
            { 0.157851, 0.683765, 0.501686 },
            { 0.162016, 0.687316, 0.499129 },
            { 0.166383, 0.690856, 0.496502 },
            { 0.170948, 0.694384, 0.493803 },
            { 0.175707, 0.697900, 0.491033 },
            { 0.180653, 0.701402, 0.488189 },
            { 0.185783, 0.704891, 0.485273 },
            { 0.191090, 0.708366, 0.482284 },
            { 0.196571, 0.711827, 0.479221 },
            { 0.202219, 0.715272, 0.476084 },
            { 0.208030, 0.718701, 0.472873 },
            { 0.214000, 0.722114, 0.469588 },
            { 0.220124, 0.725509, 0.466226 },
            { 0.226397, 0.728888, 0.462789 },
            { 0.232815, 0.732247, 0.459277 },
            { 0.239374, 0.735588, 0.455688 },
            { 0.246070, 0.738910, 0.452024 },
            { 0.252899, 0.742211, 0.448284 },
            { 0.259857, 0.745492, 0.444467 },
            { 0.266941, 0.748751, 0.440573 },
            { 0.274149, 0.751988, 0.436601 },
            { 0.281477, 0.755203, 0.432552 },
            { 0.288921, 0.758394, 0.428426 },
            { 0.296479, 0.761561, 0.424223 },
            { 0.304148, 0.764704, 0.419943 },
            { 0.311925, 0.767822, 0.415586 },
            { 0.319809, 0.770914, 0.411152 },
            { 0.327796, 0.773980, 0.406640 },
            { 0.335885, 0.777018, 0.402049 },
            { 0.344074, 0.780029, 0.397381 },
            { 0.352360, 0.783011, 0.392636 },
            { 0.360741, 0.785964, 0.387814 },
            { 0.369214, 0.788888, 0.382914 },
            { 0.377779, 0.791781, 0.377939 },
            { 0.386433, 0.794644, 0.372886 },
            { 0.395174, 0.797475, 0.367757 },
            { 0.404001, 0.800275, 0.362552 },
            { 0.412913, 0.803041, 0.357269 },
            { 0.421908, 0.805774, 0.351910 },
            { 0.430983, 0.808473, 0.346476 },
            { 0.440137, 0.811138, 0.340967 },
            { 0.449368, 0.813768, 0.335384 },
            { 0.458674, 0.816363, 0.329727 },
            { 0.468053, 0.818921, 0.323998 },
            { 0.477504, 0.821444, 0.318195 },
            { 0.487026, 0.823929, 0.312321 },
            { 0.496615, 0.826376, 0.306377 },
            { 0.506271, 0.828786, 0.300362 },
            { 0.515992, 0.831158, 0.294279 },
            { 0.525776, 0.833491, 0.288127 },
            { 0.535621, 0.835785, 0.281908 },
            { 0.545524, 0.838039, 0.275626 },
            { 0.555484, 0.840254, 0.269281 },
            { 0.565498, 0.842430, 0.262877 },
            { 0.575563, 0.844566, 0.256415 },
            { 0.585678, 0.846661, 0.249897 },
            { 0.595839, 0.848717, 0.243329 },
            { 0.606045, 0.850733, 0.236712 },
            { 0.616293, 0.852709, 0.230052 },
            { 0.626579, 0.854645, 0.223353 },
            { 0.636902, 0.856542, 0.216620 },
            { 0.647257, 0.858400, 0.209861 },
            { 0.657642, 0.860219, 0.203082 },
            { 0.668054, 0.861999, 0.196293 },
            { 0.678489, 0.863742, 0.189503 },
            { 0.688944, 0.865448, 0.182725 },
            { 0.699415, 0.867117, 0.175971 },
            { 0.709898, 0.868751, 0.169257 },
            { 0.720391, 0.870350, 0.162603 },
            { 0.730889, 0.871916, 0.156029 },
            { 0.741388, 0.873449, 0.149561 },
            { 0.751884, 0.874951, 0.143228 },
            { 0.762373, 0.876424, 0.137064 },
            { 0.772852, 0.877868, 0.131109 },
            { 0.783315, 0.879285, 0.125405 },
            { 0.793760, 0.880678, 0.120005 },
            { 0.804182, 0.882046, 0.114965 },
            { 0.814576, 0.883393, 0.110347 },
            { 0.824940, 0.884720, 0.106217 },
            { 0.835270, 0.886029, 0.102646 },
            { 0.845561, 0.887322, 0.099702 },
            { 0.855810, 0.888601, 0.097452 },
            { 0.866013, 0.889868, 0.095953 },
            { 0.876168, 0.891125, 0.095250 },
            { 0.886271, 0.892374, 0.095374 },
            { 0.896320, 0.893616, 0.096335 },
            { 0.906311, 0.894855, 0.098125 },
            { 0.916242, 0.896091, 0.100717 },
            { 0.926106, 0.897330, 0.104071 },
            { 0.935904, 0.898570, 0.108131 },
            { 0.945636, 0.899815, 0.112838 },
            { 0.955300, 0.901065, 0.118128 },
            { 0.964894, 0.902323, 0.123941 },
            { 0.974417, 0.903590, 0.130215 },
            { 0.983868, 0.904867, 0.136897 },
            { 0.993248, 0.906157, 0.143936 }
        };
    return calcLerp((float)lum, data);
}

// The SH encoding requires normalized unit vector transformed into unit cube via v' = (v + 1) / 2
KRR_CALLABLE Vector3f warp_direction_for_sh(const Vector3f& dir) {
	return (dir + Vector3f::Ones()) * 0.5f;
}

KRR_CALLABLE float warp_roughness_for_ob(const float roughness) {
	return 1 - expf(-roughness);
}


KRR_NAMESPACE_END