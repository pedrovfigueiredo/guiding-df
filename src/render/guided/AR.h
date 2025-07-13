#pragma once
#include <cuda.h>
#include <cuda_runtime.h>
#include <assert.h>

#include "device/cuda.h"
#include "util/math_utils.h"
#include "common.h"

#include "parameters.h"

#include "tiny-cuda-nn/common.h"

using precision_t = tcnn::network_precision_t;

KRR_NAMESPACE_BEGIN

class AutoRegressive {
public:
    AutoRegressive() = default;
    ~AutoRegressive() = default;

    template <typename T>
	// maps network's raw output to valid parameters
	KRR_CALLABLE AutoRegressive (T* data) {
		mTheta = (float) data[0];
        mPhi = (float) data[1];

        mIsTwoSided = data[2] > 0.f && data[3] > 0.f;
        mPdfTheta = fabs((float) data[2]);
        mPdfPhi = fabs((float) data[3]);

        assert(mIsTwoSided);
        mPdfTransform = utils::squareToUniformSpherePdf();
	}

    KRR_CALLABLE bool isTwoSided() const { return mIsTwoSided; }
    KRR_CALLABLE float pdfTransform() const { return mPdfTransform; }
    KRR_CALLABLE float pdfTransformInv() const { return 1.f / mPdfTransform; } // mPdfTransform is always non-zero

    KRR_CALLABLE float pdf() const {  
        return mPdfTheta * mPdfPhi * mPdfTransform;
    }

    KRR_CALLABLE Vector3f sample() const{
        const Vector2f wi_square = { mTheta, mPhi };
        return utils::squareToUniformSphere(wi_square);
    }

    template <uint32_t dim_output, bool isTheta>
    static KRR_CALLABLE void compute_gradients_specialized_linear_interpolation(const float wi_square, const float PDFOther, 
        const float prefix, const precision_t* pdfs, precision_t* output_grad){ 

        const float inv_res = 1.0f / (dim_output);
        const float inv_res_grad = dim_output;
        const float multiplier = prefix * PDFOther * inv_res_grad;
        
        const float idx = wi_square * (dim_output) - 0.5f;
        int idx_floor = min((int)floor(idx), (int)(dim_output) - 1); // safe_floor
        int idx_ceil = idx_floor + 1;
        float dir_weight = idx - idx_floor;

        // For theta, we do nearest neighbor at the edges
        if (idx_floor < 0){
            idx_floor = dim_output - 1;
            // if constexpr (isTheta)
                dir_weight = 1.0f;
        }
        if (idx_ceil > dim_output - 1){
            idx_ceil = 0;
            // if constexpr (isTheta)
                dir_weight = 0.0f;
        }  

        const float pdfIdx0 = (float) pdfs[idx_floor] * inv_res;
        const float pdfIdx1 = (float) pdfs[idx_ceil] * inv_res;
        TCNN_PRAGMA_UNROLL
        for (int i = 0; i < dim_output; i++) {
            const float pdfI = (float) pdfs[i] * inv_res;
            const float Jiidx0 = idx_floor == i ? pdfIdx0 * (1.0f - pdfIdx0) : -pdfIdx0 * pdfI;
            const float Jiidx1 = idx_ceil == i ? pdfIdx1 * (1.0f - pdfIdx1) : -pdfIdx1 * pdfI;
            const float Jiidx = (1.0f - dir_weight) * Jiidx0 + dir_weight * Jiidx1;
            output_grad[i] = (precision_t) (multiplier * Jiidx);
        }
    }

    template <uint32_t dim_output>
    static KRR_CALLABLE void compute_gradients_specialized_nearest(const float wi_square, const float PDFOther, 
        const float prefix, const precision_t* pdfs, precision_t* output_grad){ 
        
        const float idx = wi_square * (dim_output);
        const int idx_floor = min((int)idx, (int)(dim_output) - 1); // safe_floor
        const float inv_res = 1.0f / (dim_output);
        const float inv_res_grad = dim_output;
        const float multiplier = prefix * PDFOther * inv_res_grad;

        const float pdfIdx = (float) pdfs[idx_floor] * inv_res;
        TCNN_PRAGMA_UNROLL
        for (int i = 0; i < dim_output; i++) {
            const float pdfI = (float) pdfs[i] * inv_res;
            const float Jiidx = idx_floor == i ? pdfIdx * (1.0f - pdfIdx) : -pdfIdx * pdfI;
            output_grad[i] = (precision_t) (multiplier * Jiidx);
        }
    }
    
    KRR_CALLABLE void compute_gradients(const float prefix, const precision_t* pdfs_theta, const precision_t* pdfs_phi, 
        precision_t* output_grad_theta, precision_t* output_grad_phi) const{
        const Vector2f wi_square = { mTheta, mPhi };
    #if (AR_LINEAR_INTERPOLATION == 1)
        AutoRegressive::compute_gradients_specialized_linear_interpolation<N_DIM_OUTPUT, true>(wi_square[0], mPdfPhi, prefix, pdfs_theta, output_grad_theta);
        AutoRegressive::compute_gradients_specialized_linear_interpolation<N_DIM_OUTPUT_PHI, false>(wi_square[1], mPdfTheta, prefix, pdfs_phi, output_grad_phi);
    #else
        AutoRegressive::compute_gradients_specialized_nearest<N_DIM_OUTPUT>(wi_square[0], mPdfPhi, prefix, pdfs_theta, output_grad_theta);
        AutoRegressive::compute_gradients_specialized_nearest<N_DIM_OUTPUT_PHI>(wi_square[1], mPdfTheta, prefix, pdfs_phi, output_grad_phi);
    #endif
    }

    float mTheta, mPhi, mPdfTheta, mPdfPhi, mPdfTransform;
    bool mIsTwoSided;
};

KRR_NAMESPACE_END