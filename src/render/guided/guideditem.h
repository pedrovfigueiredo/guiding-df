#pragma once
#include "common.h"
#include "raytracing.h"
#include "render/shared.h"
// inherits all items' definition from the wavefront pathtracer
#include "render/wavefront/workitem.h"
#include "render/guided/parameters.h"

KRR_NAMESPACE_BEGIN

using namespace shader;

/* Remember to copy these definitions to guideditem.soa whenever changing them. */

struct BsdfEvalWorkItem {
	uint itemId;		// the index of work item in scatter ray queue
};

struct GuidedInferenceWorkItem {
	uint itemId; // the index of work item in scatter ray queue
};

struct RadianceRecordItem {
	Color L;			// local radiance 
	Color thp;			// throughput at current radiance, used to calculate local radiance
	Color Le;			// Le emitted from light source
	Vector3f pos;		// normalized to [-1, 1]^3?
	Vector2f dir;		// [theta, phi] wi, where theta in [0, pi] and phi in [0, 2pi]
	float wiPdf;		// the combined pdf of the sampled direction
	float bsdfPdf;		// [optional] the pdf of the sampled bsdf, for learning selection
	float misWeight;	// [optional] for MIS-aware distribution learning?
	bool delta;			// is this scatter event is sampled from a delta lobe?
	bool miss;			// is this scatter event added for NRC alignment since it's a miss?
	bool isTwoSided;	// is the surface two-sided?
	
	Color bsdfVal;		// [optional] for learning 5-D product distribution L_i * f_r
	Vector2f wo;		// [optional] for learning 5-D product distribution 
	Vector2f normal;	// [auxiliary] surface (shading) normal where theta in [0, pi] and phi in [0, 2pi]
	float roughness;	// [auxiliary] roughness of the surface

	// Optionally add diffuse and specular as input
	Vector3f diffuse;   // [auxiliary] diffuse color of the surface
	Vector3f specular;  // [auxiliary] specular color of the surface

	KRR_CALLABLE void record(const Color& r) { L += r; }
};

struct GuidedPixelState {
	RadianceRecordItem records[MAX_TRAIN_DEPTH];
	uint curDepth{};
};

struct GuidedInput {
	Vector3f pos;		/* normalized pos to [0, 1]^3 */
#if GUIDED_PRODUCT_SAMPLING
	Vector3f dir;		/* normalized dir with 1-norm */
#endif
#if GUIDED_AUXILIARY_INPUT
	float auxiliary[N_DIM_AUXILIARY_INPUT];
#endif
};

struct GuidedInputPhi {
	Vector3f pos;		/* normalized pos to [0, 1]^3 */
#if GUIDED_PRODUCT_SAMPLING
	Vector3f dir;		/* normalized dir with 1-norm */
#endif
#if GUIDED_AUXILIARY_INPUT
	float auxiliary[N_DIM_AUXILIARY_INPUT];
#endif
	float theta;
};

struct GuidedInputwNRC {
	GuidedInput input;
	GuidedInput inputNRC;	
};

// info that used to get the gradients of network output (dL_dy)
struct GuidedOutput {
	Vector2f dir;
	Color L;
	float wiPdf;
#if GUIDED_PRODUCT_SAMPLING
	Color bsdfVal;
#endif
};

struct GuidedInputNRC {
	Vector3f pos;		/* normalized pos to [0, 1]^3 */
	Vector3f dir;		/* normalized dir with 1-norm */
	float auxiliary[9];
};

struct GuidedOutputNRC{
	Color L;
};

struct AuxInfo {
	// NEE
	Ray shadowRay;
	Color Li;
	float lightPdf;

	// BSDF
	BSDFSample bsdfSample;

	bool RR_pass;
	bool guided_pass;
};

#pragma warning (push, 0)
#pragma warning (disable: ALL_CODE_ANALYSIS_WARNINGS)
#include "render/guided/guideditem_soa.h"
#pragma warning (pop)

KRR_NAMESPACE_END