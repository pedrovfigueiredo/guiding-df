#pragma once
#include "common.h"

#include "scene.h"
#include "render/wavefront/workqueue.h"
#include "guided.h"
#include "device/context.h"
#include "device/optix.h"

KRR_NAMESPACE_BEGIN

class GuidedPathTracer;

class OptiXGuidedBackend : public OptiXBackend {
public:
	OptiXGuidedBackend() = default;
	OptiXGuidedBackend(Scene& scene);
	void setScene(Scene& scene);

	void traceClosest(int numRays,
		RayQueue* currentRayQueue,
		MissRayQueue* missRayQueue,
		HitLightRayQueue* hitLightRayQueue,
		ScatterRayQueue* scatterRayQueue,
		RayQueue* nextRayQueue);

	void traceShadow(int numRays,
		ShadowRayQueue* shadowRayQueue,
		PixelStateBuffer* pixelState,
		GuidedPixelStateBuffer* guidedState,
		const TrainState& trainState);
	
	void traceOneHit(int numRays,
		RayQueue* currentRayQueue,
		ScatterRayQueue* scatterRayQueue);

protected:
	OptixProgramGroup createRaygenPG(const char* entrypoint) const;
	OptixProgramGroup createMissPG(const char* entrypoint) const;
	OptixProgramGroup createIntersectionPG(const char* closest, const char* any,
		const char* intersect) const;

private:
	friend class GuidedPathTracer;
	
	OptixModule optixModule;
	OptixPipeline optixPipeline;
	OptixDeviceContext optixContext;
	CUstream cudaStream;

	OptixShaderBindingTable closestSBT{};
	OptixShaderBindingTable shadowSBT{};
	OptixShaderBindingTable oneHitSBT{};
	OptixTraversableHandle optixTraversable{};

	Scene::SceneData sceneData{};

	inter::vector<RaygenRecord> raygenClosestRecords;
	inter::vector<HitgroupRecord> hitgroupClosestRecords;
	inter::vector<MissRecord> missClosestRecords;

	inter::vector<RaygenRecord> raygenShadowRecords;
	inter::vector<HitgroupRecord> hitgroupShadowRecords;
	inter::vector<MissRecord> missShadowRecords;

	inter::vector<RaygenRecord> raygenOneHitRecords;
	inter::vector<HitgroupRecord> hitgroupOneHitRecords;
	inter::vector<MissRecord> missOneHitRecords;
};

KRR_NAMESPACE_END