#include "backend.h"
#include "device/cuda.h"
#include "render/profiler/profiler.h"
#include "render/shared.h"

KRR_NAMESPACE_BEGIN

#define OPTIX_LOG_SIZE 4096
extern "C" char GUIDED_PTX[];
LaunchParamsGuided *launchParams{};

OptiXGuidedBackend::OptiXGuidedBackend(Scene &scene) { setScene(scene); }

void OptiXGuidedBackend::setScene(Scene &scene) {
	char log[OPTIX_LOG_SIZE];
	size_t logSize = sizeof(log);

	// use global context and stream for now
	optixContext = gpContext->optixContext;
	cudaStream	 = gpContext->cudaStream;

	// creating optix module from ptx
	optixModule = createOptiXModule(optixContext, GUIDED_PTX);
	// creating program groups
	OptixProgramGroup raygenClosestPG = createRaygenPG("__raygen__Closest");
	OptixProgramGroup raygenShadowPG  = createRaygenPG("__raygen__Shadow");
	OptixProgramGroup raygenOneHitPG  = createRaygenPG("__raygen__OneHit");

	OptixProgramGroup missClosestPG	  = createMissPG("__miss__Closest");
	OptixProgramGroup missShadowPG	  = createMissPG("__miss__Shadow");
	OptixProgramGroup missOneHitPG	  = createMissPG("__miss__OneHit");

	OptixProgramGroup hitClosestPG =
		createIntersectionPG("__closesthit__Closest", "__anyhit__Closest", nullptr);
	OptixProgramGroup hitShadowPG = createIntersectionPG(nullptr, "__anyhit__Shadow", nullptr);
	OptixProgramGroup hitOneHitPG =
		createIntersectionPG("__closesthit__OneHit", "__anyhit__OneHit", nullptr);

	std::vector<OptixProgramGroup> allPGs = { raygenClosestPG, 	raygenShadowPG, raygenOneHitPG,
											  missClosestPG,   	missShadowPG,  	missOneHitPG,	
											  hitClosestPG,   	hitShadowPG, 	hitOneHitPG };

	// creating optix pipeline from all program groups
	OptixPipelineCompileOptions pipelineCompileOptions = getPipelineCompileOptions();
	OptixPipelineLinkOptions pipelineLinkOptions	   = {};
	pipelineLinkOptions.maxTraceDepth				   = 5;
#ifdef KRR_DEBUG_BUILD
	pipelineLinkOptions.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_FULL;
#endif
	OPTIX_CHECK_WITH_LOG(optixPipelineCreate(optixContext, &pipelineCompileOptions,
											 &pipelineLinkOptions, allPGs.data(), allPGs.size(),
											 log, &logSize, &optixPipeline),
						 log);
	logDebug(log);

	OPTIX_CHECK(optixPipelineSetStackSize(/* [in] The pipeline to configure the stack size for */
										  optixPipeline, 4 * 1024, 4 * 1024, 4 * 1024, 1));
	logDebug(log);

	// creating shader binding table...
	RaygenRecord raygenRecord	  = {};
	MissRecord missRecord		  = {};
	HitgroupRecord hitgroupRecord = {};

	OPTIX_CHECK(optixSbtRecordPackHeader(raygenClosestPG, &raygenRecord));
	raygenClosestRecords.push_back(raygenRecord);
	OPTIX_CHECK(optixSbtRecordPackHeader(raygenShadowPG, &raygenRecord));
	raygenShadowRecords.push_back(raygenRecord);
	OPTIX_CHECK(optixSbtRecordPackHeader(raygenOneHitPG, &raygenRecord));
	raygenOneHitRecords.push_back(raygenRecord);

	OPTIX_CHECK(optixSbtRecordPackHeader(missClosestPG, &missRecord));
	missClosestRecords.push_back(missRecord);
	OPTIX_CHECK(optixSbtRecordPackHeader(missShadowPG, &missRecord));
	missShadowRecords.push_back(missRecord);
	OPTIX_CHECK(optixSbtRecordPackHeader(missOneHitPG, &missRecord));
	missOneHitRecords.push_back(missRecord);

	for (MeshData &meshData : *scene.mData.meshes) {
		hitgroupRecord.data = { &meshData };
		OPTIX_CHECK(optixSbtRecordPackHeader(hitClosestPG, &hitgroupRecord));
		hitgroupClosestRecords.push_back(hitgroupRecord);
		OPTIX_CHECK(optixSbtRecordPackHeader(hitShadowPG, &hitgroupRecord));
		hitgroupShadowRecords.push_back(hitgroupRecord);
		OPTIX_CHECK(optixSbtRecordPackHeader(hitOneHitPG, &hitgroupRecord));
		hitgroupOneHitRecords.push_back(hitgroupRecord);
	}

	closestSBT.raygenRecord				   = (CUdeviceptr) raygenClosestRecords.data();
	closestSBT.missRecordBase			   = (CUdeviceptr) missClosestRecords.data();
	closestSBT.missRecordCount			   = 1;
	closestSBT.missRecordStrideInBytes	   = sizeof(MissRecord);
	closestSBT.hitgroupRecordBase		   = (CUdeviceptr) hitgroupClosestRecords.data();
	closestSBT.hitgroupRecordCount		   = hitgroupClosestRecords.size();
	closestSBT.hitgroupRecordStrideInBytes = sizeof(HitgroupRecord);

	shadowSBT.raygenRecord				  = (CUdeviceptr) raygenShadowRecords.data();
	shadowSBT.missRecordBase			  = (CUdeviceptr) missShadowRecords.data();
	shadowSBT.missRecordCount			  = 1;
	shadowSBT.missRecordStrideInBytes	  = sizeof(MissRecord);
	shadowSBT.hitgroupRecordBase		  = (CUdeviceptr) hitgroupShadowRecords.data();
	shadowSBT.hitgroupRecordCount		  = hitgroupShadowRecords.size();
	shadowSBT.hitgroupRecordStrideInBytes = sizeof(HitgroupRecord);

	oneHitSBT.raygenRecord				  = (CUdeviceptr) raygenOneHitRecords.data();
	oneHitSBT.missRecordBase			  = (CUdeviceptr) missOneHitRecords.data();
	oneHitSBT.missRecordCount			  = 1;
	oneHitSBT.missRecordStrideInBytes	  = sizeof(MissRecord);
	oneHitSBT.hitgroupRecordBase		  = (CUdeviceptr) hitgroupOneHitRecords.data();
	oneHitSBT.hitgroupRecordCount		  = hitgroupOneHitRecords.size();
	oneHitSBT.hitgroupRecordStrideInBytes = sizeof(HitgroupRecord);

	optixTraversable = buildAccelStructure(optixContext, cudaStream, scene);

	if (!launchParams)
		launchParams = gpContext->alloc->new_object<LaunchParamsGuided>();
	launchParams->sceneData	  = scene.mData;
	launchParams->traversable = optixTraversable;
	sceneData				  = scene.mData;
}

void OptiXGuidedBackend::traceClosest(int numRays, RayQueue *currentRayQueue,
									  MissRayQueue *missRayQueue,
									  HitLightRayQueue *hitLightRayQueue,
									  ScatterRayQueue *scatterRayQueue, 
									  RayQueue *nextRayQueue) {
	if (optixTraversable) {
		PROFILE("Trace intersect rays");
		static LaunchParamsGuided params = {};
		params.traversable				 = optixTraversable;
		params.sceneData				 = sceneData;
		params.currentRayQueue			 = currentRayQueue;
		params.missRayQueue				 = missRayQueue;
		params.hitLightRayQueue			 = hitLightRayQueue;
		params.scatterRayQueue			 = scatterRayQueue;
		params.nextRayQueue				 = nextRayQueue;
		cudaMemcpy(launchParams, &params, sizeof(LaunchParamsGuided), cudaMemcpyHostToDevice);
		OPTIX_CHECK(optixLaunch(optixPipeline, cudaStream, (CUdeviceptr) launchParams,
								sizeof(LaunchParamsGuided), &closestSBT, numRays, 1, 1));
#ifdef KRR_DEBUG_BUILD
		CUDA_SYNC_CHECK();
#endif
	}
}

void OptiXGuidedBackend::traceShadow(int numRays, ShadowRayQueue *shadowRayQueue,
									 PixelStateBuffer *pixelState,
									 GuidedPixelStateBuffer *guidedState,
									 const TrainState &trainState) {
	if (optixTraversable) {
		PROFILE("Trace shadow rays");
		static LaunchParamsGuided params = {};
		params.traversable				 = optixTraversable;
		params.sceneData				 = sceneData;
		params.shadowRayQueue			 = shadowRayQueue;
		params.pixelState				 = pixelState;
		params.guidedState				 = guidedState;
		params.trainState				 = trainState;
		cudaMemcpy(launchParams, &params, sizeof(LaunchParamsGuided), cudaMemcpyHostToDevice);
		OPTIX_CHECK(optixLaunch(optixPipeline, cudaStream, (CUdeviceptr) launchParams,
								sizeof(LaunchParamsGuided), &shadowSBT, numRays, 1, 1));
#ifdef KRR_DEBUG_BUILD
		CUDA_SYNC_CHECK();
#endif
	}
}

void OptiXGuidedBackend::traceOneHit(int numRays, RayQueue *currentRayQueue,
									 ScatterRayQueue *scatterRayQueue) {
	if (optixTraversable) {
		PROFILE("Trace one hit rays");
		static LaunchParamsGuided params = {};
		params.traversable				 = optixTraversable;
		params.sceneData				 = sceneData;
		params.currentRayQueue			 = currentRayQueue;
		params.scatterRayQueue			 = scatterRayQueue;
		cudaMemcpy(launchParams, &params, sizeof(LaunchParamsGuided), cudaMemcpyHostToDevice);
		OPTIX_CHECK(optixLaunch(optixPipeline, cudaStream, (CUdeviceptr) launchParams,
								sizeof(LaunchParamsGuided), &oneHitSBT, numRays, 1, 1));
	}
}

OptixProgramGroup OptiXGuidedBackend::createRaygenPG(const char *entrypoint) const {
	return OptiXBackend::createRaygenPG(optixContext, optixModule, entrypoint);
}

OptixProgramGroup OptiXGuidedBackend::createMissPG(const char *entrypoint) const {
	return OptiXBackend::createMissPG(optixContext, optixModule, entrypoint);
}

OptixProgramGroup OptiXGuidedBackend::createIntersectionPG(const char *closest, const char *any,
														   const char *intersect) const {
	return OptiXBackend::createIntersectionPG(optixContext, optixModule, closest, any, intersect);
}

KRR_NAMESPACE_END