#pragma once
#include <atomic>

#include "common.h"
#include "logger.h"
#include "util/math_utils.h"
#include "device/cuda.h"

#include "render/profiler/profiler.h"
#include "render/wavefront/workqueue.h"
#include "render/guided/guideditem.h"

KRR_NAMESPACE_BEGIN

struct GuidedPixelStateBuffer : public SOA<GuidedPixelState> {
public:
	GuidedPixelStateBuffer() = default;
	GuidedPixelStateBuffer(int n, Allocator alloc) : SOA<GuidedPixelState>(n, alloc) {}

	KRR_CALLABLE void reset(int pixelId) {
		// reset a guided state (call when begining a new frame...
		curDepth[pixelId] = 0;
	}

	/* Records raw (unnormalized) vertex data along the path of the current pixel */
	KRR_CALLABLE void incrementDepth(int pixelId,
		const Ray& ray,				// current scattered ray
		const Color& thp,			// current throughput
		const float pdf,			// effective pdf of the scatter direction
		const float bsdfPdf = 0,	// bsdf pdf for the scatter direction
		const float misWeight = 0,	// effectiveGuidedPdf / combinedWiPdf
		const Color& L = {},		// current radiance 
		bool delta = false,			// is this scatter event sampled from a delta lobe?
		/* below optional / auxiliary data */
		const Color& bsdfVal = {},	// bsdf value for modeling 5-D product distribution
		const ShadingData& sd = {}	// may obtain other auxiliary data from this
	) {
		int depth = curDepth[pixelId];
		if (depth >= MAX_TRAIN_DEPTH) return;
		records[depth].L[pixelId]		  = L;
		records[depth].thp[pixelId]		  = thp;
		records[depth].pos[pixelId]		  = ray.origin;
		records[depth].dir[pixelId]		  = utils::cartesianToSpherical(ray.dir);
		records[depth].wiPdf[pixelId]	  = pdf;
		records[depth].bsdfPdf[pixelId]   = bsdfPdf; 
		records[depth].misWeight[pixelId] = misWeight;
		records[depth].delta[pixelId]	  = delta;
		records[depth].miss[pixelId]	  = false;
		curDepth[pixelId]				  = depth + 1;
		records[depth].normal[pixelId]	  = utils::cartesianToSpherical(sd.frame.N);
		records[depth].roughness[pixelId] = sd.roughness;
		records[depth].diffuse[pixelId]	  = sd.diffuse;
		records[depth].specular[pixelId]  = sd.specular;
		records[depth].bsdfVal[pixelId] = bsdfVal;
		records[depth].wo[pixelId] = cartesianToSpherical(sd.wo);
		records[depth].Le[pixelId] = Color(0.f);
		records[depth].isTwoSided[pixelId] = sd.isTwoSided();
	}

	KRR_CALLABLE void incrementDepthLastBounceNRC(int pixelId,
		bool miss = false,
		const ShadingData& sd = {}	// may obtain other auxiliary data from this
	) {
		int depth = curDepth[pixelId];
		if (depth >= MAX_TRAIN_DEPTH) return;
		records[depth].pos[pixelId]		  = sd.pos;
		curDepth[pixelId]				  = depth + 1;
		records[depth].normal[pixelId]	  = utils::cartesianToSpherical(sd.frame.N);
		records[depth].roughness[pixelId] = sd.roughness;
		records[depth].diffuse[pixelId]	  = sd.diffuse;
		records[depth].specular[pixelId]  = sd.specular;
		records[depth].wo[pixelId] = cartesianToSpherical(sd.wo);
		records[depth].miss[pixelId]	  = miss;

		records[depth].Le[pixelId] = Color(0.f);
		records[depth].delta[pixelId]	  = !(sd.getBsdfType() & BSDF_SMOOTH);
		records[depth].L[pixelId]		  = Color(0.f);
		records[depth].thp[pixelId]		  = Color(0.f);
	}

	/* Two types of radiance contribution call this routine: 
		Emissive intersection and Next event estimation. */
	KRR_CALLABLE void recordRadiance(int pixelId,
		const Color& L) {
		int depth = min(curDepth[pixelId], (uint)MAX_TRAIN_DEPTH);
		for (int i = 0; i < depth; i++) {
			// local radiance should be obtained via L / thp.
			const Color& prev = records[i].L[pixelId];
			records[i].L[pixelId] = prev + L;
		}
	}

	KRR_CALLABLE void overwriteRadiance(int pixelId, int depth, const Color& L) {
		if (depth >= MAX_TRAIN_DEPTH || depth < 0) return;
		records[depth].L[pixelId] = L;
	}

	KRR_CALLABLE void overwriteLe(int pixelId, int depth, const Color& Le) {
		if (depth >= MAX_TRAIN_DEPTH || depth < 0) return;
		records[depth].Le[pixelId] = Le;
	}

	KRR_CALLABLE void overwriteLe(int pixelId, const Color& Le) {
		int depth = curDepth[pixelId] - 1;
		overwriteLe(pixelId, depth, Le);
	}

	KRR_CALLABLE bool isValidRecord(int pixelId, int depth) const {
		return depth >= 0 && depth < MAX_TRAIN_DEPTH && depth < curDepth[pixelId];
	}

	// Divide the radiance by given L
	KRR_CALLABLE void divideRadiance(int pixelId, int depth, const Color& L) {
		if (!isValidRecord(pixelId, depth)) return;
		for (int ch = 0; ch < Color::dim; ch++) {
			if (L[ch] > M_EPSILON)
				records[depth].L[pixelId][ch] = records[depth].L[pixelId][ch] / L[ch];
		}
	}

	// Replaces indirect radiance to pixel.
	KRR_CALLABLE void replaceIndirectRadiance(int pixelId, int depth, const Color& L) {
		if (!isValidRecord(pixelId, depth)) return;
		const Color NRCPlusDirectCont = records[depth].thp[pixelId] * (L + records[depth].Le[pixelId]);
		records[depth].L[pixelId] = NRCPlusDirectCont;
	}

	// Replaces indirect radiance to pixel.
	KRR_CALLABLE void replaceIndirectRadianceSkipDelta(int pixelId, int depth, const Color& L) {
		const int initialDepth = depth;
		while(depth >= 0 && records[depth].delta[pixelId]) depth--;
		if (!isValidRecord(pixelId, depth)) return;
		const Color NRCPlusDirectCont = records[initialDepth].thp[pixelId] * (L + records[initialDepth].Le[pixelId]);
		records[depth].L[pixelId] = NRCPlusDirectCont;
	}

	// Propagates indirect radiance along ray path
	KRR_CALLABLE void propagateIndirectRadiance(int pixelId, int depth, const Color& L) {
		if (!isValidRecord(pixelId, depth)) return;
		
		const Color indirectRadiance = records[depth].thp[pixelId] * L;
		for (int i = 0; i <= depth; i++)
			records[i].L[pixelId] += indirectRadiance;
	}

	KRR_CALLABLE uint getCurDepth(int pixelId) const { return curDepth[pixelId]; }
};

class BsdfEvalQueue : public WorkQueue<BsdfEvalWorkItem> {
public:
	using WorkQueue::WorkQueue;
	using WorkQueue::push;

	KRR_CALLABLE int push(uint index) {
		return push(BsdfEvalWorkItem{ index });
	}
};

class GuidedInferenceQueue : public WorkQueue<GuidedInferenceWorkItem> {
public:
	using WorkQueue::push;
	using WorkQueue::WorkQueue;

	KRR_CALLABLE int push(uint index) { return push(GuidedInferenceWorkItem{ index }); }
};


KRR_NAMESPACE_END