#pragma once
#include "common.h"

template <uint32_t dim, uint32_t multiple> struct padded_dim_compiletime {
    static const uint32_t value = (dim + multiple - 1) / multiple * multiple;
};

#define GUIDED_AUXILIARY_INPUT		1
#define GUIDED_PRODUCT_SAMPLING		1


// Possible types of NRC optimization:
// 0: no NRC optimization
// 1: NRC 1-step optimization
// 2: NRC N-step optimization
// 3: NRC inverse 0-step optimization
// 4: 1 + 3
// 5: 2 + 3
#define OPTIMIZE_ON_NRC 4
#define AR_LINEAR_INTERPOLATION 1

#define RENDER_NOISY_ONLY 0

constexpr unsigned int MAX_TRAIN_DEPTH = 5;			// the MC estimate collected before that depth is only used for training
constexpr unsigned int NETWORK_ALIGNMENT = 16;		// both the cutlass and fully-fused network needs 16byte-aligned.
constexpr unsigned int N_DIM_SPATIAL_INPUT = 3;		// how many dims do spatial data (i.e., positions, and optionally directions)
constexpr unsigned int N_DIM_DIRECTIONAL_INPUT = 
	GUIDED_PRODUCT_SAMPLING ? 3 : 0;				// whether enable product learning MLP(x, w_o) -> f_r(w_o, w_i) * L (w_i)

#if GUIDED_AUXILIARY_INPUT
	constexpr unsigned int N_DIM_AUXILIARY_INPUT = 3;
#else
	constexpr unsigned int N_DIM_AUXILIARY_INPUT = 0; // how many dims do auxiliary data (e.g., normals, roughness) have?
#endif

constexpr unsigned int N_WI_PER_PIXEL = 1; // MIS or BSDF sampling

constexpr unsigned int N_DIM_INPUT = N_DIM_DIRECTIONAL_INPUT + 
	N_DIM_SPATIAL_INPUT + N_DIM_AUXILIARY_INPUT;

constexpr unsigned int N_DIM_OUTPUT = 16;
constexpr unsigned int N_DIM_PADDED_OUTPUT = padded_dim_compiletime<N_DIM_OUTPUT, 16>::value;		// network output size with next_multiple of 16!

constexpr unsigned int N_DIM_WI = 4; // (theta, phi, pdf(theta), pdf(phi))
constexpr float TRAIN_LOSS_SCALE = 1024.f;

constexpr unsigned int N_DIM_INPUT_PHI = 1 + N_DIM_INPUT;
constexpr unsigned int N_DIM_OUTPUT_PHI = N_DIM_OUTPUT * 2;
constexpr unsigned int N_DIM_PADDED_OUTPUT_PHI = padded_dim_compiletime<N_DIM_OUTPUT_PHI, 16>::value; // network output size with next_multiple of 16!

constexpr unsigned int MAX_RESOLUTION = 1280 * 720; // max size of the rendering frame 
constexpr int TRAIN_BUFFER_SIZE = 
	MAX_TRAIN_DEPTH * MAX_RESOLUTION;				// [resolution-affected]
constexpr size_t TRAIN_BATCH_SIZE = 65'536 * 8;		
constexpr size_t MIN_TRAIN_BATCH_SIZE = 65'536;		// the minimum batch size we can tolerate (to avoid unstable training)
constexpr int MAX_INFERENCE_NUM = MAX_RESOLUTION;	// [resolution-affected]

constexpr unsigned int LOSS_GRAPH_SIZE = 256;
constexpr unsigned int MOUSEOVER_SAMPLING_N_SAMPLERS = 20'000'000;

constexpr unsigned int N_DIM_INPUT_NRC = 15;
constexpr unsigned int N_DIM_OUTPUT_NRC = 3;
constexpr unsigned int N_DIM_PADDED_OUTPUT_NRC = 16;
constexpr float TRAIN_LOSS_SCALE_NRC = 1024.f;
constexpr unsigned int NRC_ITER_COUNT = 4;
constexpr unsigned int TRAIN_BATCH_SIZE_NRC = 1'048'576;

#if (OPTIMIZE_ON_NRC > 0)
	constexpr int N_TRAIN_ITER_NRC = -1;
	constexpr int START_TRAIN_ITER_NRC = 0;
#endif