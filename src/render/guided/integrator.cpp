#include <cuda.h>
#include <thrust/reduce.h>
#include <thrust/execution_policy.h>
#include <cuda_runtime.h>
#include <cuda/atomic>

#include <tiny-cuda-nn/common.h>
#include <tiny-cuda-nn/network.h>
#include <tiny-cuda-nn/encoding.h>
#include <tiny-cuda-nn/gpu_matrix.h>
#include <tiny-cuda-nn/gpu_memory.h>
#include <tiny-cuda-nn/optimizer.h>
#include <tiny-cuda-nn/trainer.h>
#include <tiny-cuda-nn/network_with_input_encoding.h>

#include "train.h"
#include "integrator.h"
#include "render/profiler/profiler.h"
#include "util/ema.h"
#include "util/film.h"

KRR_NAMESPACE_BEGIN
using namespace tcnn;

template <typename T>
using GPUMatrix = tcnn::GPUMatrix<T, tcnn::MatrixLayout::ColumnMajor>;

namespace {
	// lossgraph and training info logging / plotting 
	std::vector<float> lossGraph(LOSS_GRAPH_SIZE, 0);
	size_t numLossSamples{ 0 };
	size_t numTrainingSamples{ 0 };
	Ema curLossScalar{Ema::Type::Time, 50};

#if (OPTIMIZE_ON_NRC > 0)
	std::vector<float> lossGraphNRC(LOSS_GRAPH_SIZE, 0);
	size_t numLossSamplesNRC{ 0 };
	size_t numTrainingSamplesNRC{ 0 };
	Ema curLossScalarNRC{Ema::Type::Time, 50};
#endif

	// for saving sampled wis
	GPUMemory<AuxInfo> neeInfoBuffer;
}

GuidedPathTracer::GuidedPathTracer(Scene& scene){
	initialize();
	setScene(std::shared_ptr<Scene>(&scene));
}

template <typename... Args> 
KRR_DEVICE_FUNCTION void GuidedPathTracer::debugPrint(uint pixelId, const char *fmt, Args &&...args) {
	if (pixelId == debugPixel)
		printf(fmt, std::forward<Args>(args)...);
}


#if (RENDER_NOISY_ONLY == 0)
	glu::Texture2D GuidedPathTracer::mMouseOverImageTexture;
	cudau::Array GuidedPathTracer::mMouseOverImageArray;
	cudau::InteropSurfaceObjectHolder<1>  GuidedPathTracer::mMouseOverImageTextureBufferSurfaceHolder;
	GPUMemory<float> GuidedPathTracer::mGTMouseOverImage;
	GPUMemory<float> GuidedPathTracer::mMouseOverImage;
#endif


void GuidedPathTracer::initialize(){
	Allocator& alloc = *gpContext->alloc;
	maxQueueSize = mFrameSize[0] * mFrameSize[1];
	CUDA_SYNC_CHECK();	// necessary, preventing kernel accessing memories tobe free'ed...
	for (int i = 0; i < 2; i++) {
		if (rayQueue[i]) rayQueue[i]->resize(maxQueueSize, alloc);
		else rayQueue[i] = alloc.new_object<RayQueue>(maxQueueSize, alloc);
	}
	if (missRayQueue)  missRayQueue->resize(maxQueueSize, alloc);
	else missRayQueue = alloc.new_object<MissRayQueue>(maxQueueSize, alloc);
	if (hitLightRayQueue)  hitLightRayQueue->resize(maxQueueSize, alloc);
	else hitLightRayQueue = alloc.new_object<HitLightRayQueue>(maxQueueSize, alloc);
	if (shadowRayQueue) shadowRayQueue->resize(maxQueueSize, alloc);
	else shadowRayQueue = alloc.new_object<ShadowRayQueue>(maxQueueSize, alloc);
	if (scatterRayQueue) scatterRayQueue->resize(maxQueueSize, alloc);
	else scatterRayQueue = alloc.new_object<ScatterRayQueue>(maxQueueSize, alloc);
#if (RENDER_NOISY_ONLY == 0)
	if (bsdfEvalQueue) bsdfEvalQueue->resize(maxQueueSize, alloc);
	else bsdfEvalQueue = alloc.new_object<BsdfEvalQueue>(maxQueueSize, alloc);
#endif

	if (pixelState) pixelState->resize(maxQueueSize, alloc);
	else pixelState = alloc.new_object<PixelStateBuffer>(maxQueueSize, alloc);
	if (guidedState) guidedState->resize(maxQueueSize, alloc);
	else guidedState = alloc.new_object<GuidedPixelStateBuffer>(maxQueueSize, alloc);
	mDist.initialize();
	#if (OPTIMIZE_ON_NRC > 0)
		mNRC.initialize();
	#endif
	cudaDeviceSynchronize();

	if (!camera) camera = alloc.new_object<Camera>();
	if (!backend) backend = new OptiXGuidedBackend();

	if (!profilerIntegrator) profilerIntegrator = Profiler::instancePtr();
	profilerIntegrator->setEnabled(true);
	profilerIntegrator->startCapture();

#if (RENDER_NOISY_ONLY == 0)
	if (!mMouseOverInfoIn) alloc.new_object<MouseOverInfoInput>();
	if (!mMouseOverInfoOut) alloc.new_object<MouseOverInfoOutput>();
	cudaMalloc(&mMouseOverInfoIn, sizeof(MouseOverInfoInput));
	cudaMalloc(&mMouseOverInfoOut, sizeof(MouseOverInfoOutput));

	if (mMouseOverImageTexture.isInitialized()){
		mMouseOverImageTexture.finalize();
		mMouseOverImageArray.finalize();
		mMouseOverImageTextureBufferSurfaceHolder.finalize();
	}

	mMouseOverImageTexture.initialize(GL_RGBA32F, mMouseOverImageResolution[0], mMouseOverImageResolution[1], 1);
    
    mMouseOverImageArray.initializeFromGLTexture2D(
        gpContext->cudaContext, mMouseOverImageTexture.getHandle(),
        cudau::ArraySurface::Enable, cudau::ArrayTextureGather::Disable);
    mMouseOverImageTextureBufferSurfaceHolder.initialize({ &mMouseOverImageArray });

	mGTMouseOverImage = GPUMemory<float>(mFrameSize[0] * mFrameSize[1] * 4);
#endif

	// @addition Benchmark and Offline mode
	if (renderedImage) renderedImage->resize(mFrameSize);
	else renderedImage = alloc.new_object<Film>(mFrameSize);
	renderedImage->reset();
	CUDA_SYNC_CHECK();
	task.reset();
	ui::StyleColorsDark();

	#if (OPTIMIZE_ON_NRC > 0)
		json config_nrc = File::loadJSON("common/configs/nn/base_hashgrid_nrc_df.json");
		mNRC.resetNetwork(config_nrc["nn"]);
	#endif

	srand((unsigned int)time(NULL));
	randomLaunchTimeSeed = (int) ((float) rand() / RAND_MAX * 10000.f) + 1;
}

void GuidedPathTracer::handleEmissiveHit(){
	PROFILE("Process intersected rays");
	ForAllQueued(hitLightRayQueue, maxQueueSize,
		KRR_DEVICE_LAMBDA(const HitLightWorkItem & w){
		Color Le = w.light.L(w.p, w.n, w.uv, w.wo);
		float misWeight = 1.f;
		Color contrib = Le * w.thp * misWeight;
		pixelState->addRadiance(w.pixelId, contrib);
		if (mDist.mGuiding.isTrainingPixel(w.pixelId)){
			guidedState->recordRadiance(w.pixelId, contrib);
			const Color& LNothp = Le * misWeight;
			guidedState->overwriteLe(w.pixelId, LNothp);
		}
	});
}

void GuidedPathTracer::handleMiss(){
	PROFILE("Process escaped rays");
	Scene::SceneData& sceneData = mpScene->mData;
	ForAllQueued(missRayQueue, maxQueueSize,
		KRR_DEVICE_LAMBDA(const MissRayWorkItem& w) {
		Color L = {};
		Interaction intr(w.ray.origin);
		for (const InfiniteLight& light : *sceneData.infiniteLights) {
			float misWeight = 1;
			L += light.Li(w.ray.dir) * misWeight;
		}
		Color contrib = L * w.thp;
		if (w.thp.isNaN().any()) {
			printf("handleMiss: NaN detected in w.thp: (%f,%f,%f).\nTerminating...\n", w.thp[0],w.thp[1], w.thp[2]);
			assert(false);
		}
		pixelState->addRadiance(w.pixelId, contrib);
		if (mDist.mGuiding.isTrainingPixel(w.pixelId)){
			guidedState->recordRadiance(w.pixelId, contrib);
			guidedState->overwriteLe(w.pixelId, L);
			 
			#if (OPTIMIZE_ON_NRC > 0) 
			/* 
				Adding dummy record for NRC alignment. 
				Will not be used as input. 
			*/ 
				if (w.depth <= mDist.mGuiding.maxTrainDepth) 
					guidedState->incrementDepthLastBounceNRC(w.pixelId, true, {}); 
			#endif 
		}
	});
}

void GuidedPathTracer::handleNRCLastBounce(){ 
	PROFILE("NRC last bounce"); 
	ForAllQueued(scatterRayQueue, maxQueueSize, 
		KRR_DEVICE_LAMBDA(const ScatterRayWorkItem & w) { 
		const ShadingData& sd = w.sd; 
		if (mDist.mGuiding.isTrainingPixel(w.pixelId)) 
			guidedState->incrementDepthLastBounceNRC(w.pixelId, false, sd); 
	}); 
}

void GuidedPathTracer::handleIntersections(){
	PROFILE("Process intersections");
	const float* guidingDataPtr = mDist.guidingDataPtr();
	const AuxInfo* auxInfoPtr = neeInfoBuffer.data();
	ForAllQueued(scatterRayQueue, maxQueueSize,
		KRR_DEVICE_LAMBDA(ScatterRayWorkItem & w) {

		int tid = blockIdx.x * blockDim.x + threadIdx.x; // I is the index of currentWorkItem!
		const AuxInfo* auxInfoPtr_i = auxInfoPtr + tid;
		if (auxInfoPtr_i->RR_pass) return;
		w.thp /= probRR;

		const ShadingData& sd = w.sd;
		// IF the scatter event contains only DELTA lobe(s): do not use guiding distribution;
		const BSDFType bsdfType = sd.getBsdfType();
		float bsdfSamplingFraction = mDist.mGuiding.bsdfSamplingFraction;
		
		Vector3f wi, wiLocal, wiWorld;
		float bsdfPdf, guidedPdf, scatterPdf;
		bool isDelta = false;
		Color bsdfVal = {};
		RayWorkItem r = {};
		
		/* Deciding whether to adopt the guiding distribution. */
		if (auxInfoPtr_i->guided_pass) // Guided Sampling
		{
			Sampler sampler = &pixelState->sampler[w.pixelId];
			auto dist = LearnedDistribution::getDistribution<const float, false, false>(guidingDataPtr, tid);
			wi = dist.sample();
			const Vector3f woLocal = sd.frame.toLocal(sd.wo);
		
		
			wiWorld = wi;
			wiLocal = sd.frame.toLocal(wi);

			guidedPdf	 = dist.pdf();
			bsdfPdf = BxDF::pdf(sd, woLocal, wiLocal, bsdfType);
			scatterPdf = (1 - bsdfSamplingFraction) * guidedPdf + bsdfSamplingFraction * bsdfPdf;

			bsdfVal = BxDF::f(sd, woLocal, wiLocal, bsdfType);
			r.bsdfType = BSDF_GLOSSY | (SameHemisphere(wiLocal, woLocal) ? BSDF_REFLECTION : BSDF_TRANSMISSION );
		}
		else // BSDF Sampling
		{
			BSDFSample sample = auxInfoPtr_i->bsdfSample;
			wiWorld = sd.frame.toWorld(sample.wi);
			wiLocal = sample.wi;

		
			wi = wiWorld;
			bsdfPdf	 = sample.pdf;
			scatterPdf = bsdfPdf;

			// if the sampled BSDF is a Dirac delta function, the guiding pdf need not to be evaluated.
			if (mDist.mGuiding.isEnableGuiding(w.depth) && bsdfSamplingFraction < 1 
				&& (bsdfType & BSDF_SMOOTH)) /* TODO: Check why this cannot be "sample.isNonSpecular()" */ {
				auto dist = LearnedDistribution::getDistribution<const float, false, false>(guidingDataPtr, tid);
				guidedPdf  = dist.pdf();
				scatterPdf = bsdfPdf * bsdfSamplingFraction + (1 - bsdfSamplingFraction) * guidedPdf;
			}
			
			bsdfVal = sample.f;
			r.bsdfType = sample.flags;
			isDelta = sample.isSpecular();
		}


		if ((scatterPdf && any(bsdfVal))) {
			Vector3f p = offsetRayOrigin(sd.pos, sd.frame.N, wiWorld);
			r.pdf = scatterPdf;
			r.ray = { p, wiWorld };
			r.ctx = { sd.pos, sd.frame.N };
			r.pixelId = w.pixelId;
			r.depth = w.depth + 1;
			r.thp = w.thp * bsdfVal * fabs(wiLocal[2]) / (r.pdf + M_EPSILON);
			nextRayQueue(w.depth)->push(r);
			/* enter next guided step */
			if ((mDist.mGuiding.isEnableTraining(w.depth) || mDist.mGuiding.isOneStep(w.depth)) && mDist.mGuiding.isTrainingPixel(w.pixelId)){
				guidedState->incrementDepth(w.pixelId, 
					Ray{sd.pos, wi}, r.thp, scatterPdf, bsdfPdf,
					mDist.mGuiding.misAware ? (1 - bsdfSamplingFraction) * guidedPdf / scatterPdf : 1, 
					0 /*init radiance*/, isDelta /*delta*/,
					mDist.mGuiding.productWithCosine ? bsdfVal * fabs(wiLocal[2]) : bsdfVal, 
					sd);
				// validSamplesQueue->push({w.pixelId});
			}
		}
	#if (OPTIMIZE_ON_NRC > 0)
		else{
			// Registering invalid samples for NRC alignment, so we can query NRC at the invalid point for indirect light at the last valid bounce.
			if ((mDist.mGuiding.isEnableTraining(w.depth) || mDist.mGuiding.isOneStep(w.depth)) && mDist.mGuiding.isTrainingPixel(w.pixelId)){
				guidedState->incrementDepthLastBounceNRC(w.pixelId, false, sd);
			}
		}
	#endif
	});
}

void GuidedPathTracer::generateCameraRays(int sampleId, bool fixedSample){
	PROFILE("Generate camera rays");
	RayQueue* cameraRayQueue = currentRayQueue(0);
	ParallelFor(maxQueueSize, KRR_DEVICE_LAMBDA(int pixelId){
		Sampler sampler = &pixelState->sampler[pixelId];
		Vector2i pixelCoord = { pixelId % mFrameSize[0], pixelId / mFrameSize[0] };
		Ray cameraRay = camera->getRay(pixelCoord, mFrameSize, sampler, fixedSample);
		cameraRayQueue->pushCameraRay(cameraRay, pixelId);
	});
}

void GuidedPathTracer::resize(const Vector2i& size){
	if (size[0] * size[1] > MAX_RESOLUTION) 
		Log(Fatal, "Currently maximum number of pixels is limited to %d", MAX_RESOLUTION);
	RenderPass::resize(size);
	initialize();		// need to resize the queues
}

void GuidedPathTracer::setScene(Scene::SharedPtr scene){
	scene->mTwoSidedLights = false; // disable two-sided lights
	scene->toDevice();
	mpScene = scene;
	lightSampler = scene->getSceneData().lightSampler;
	initialize();
	backend->setScene(*scene);
}

void GuidedPathTracer::beginFrame(CUDABuffer& frame){
	if (!mpScene || !maxQueueSize) return;
	PROFILE("Begin frame");
	cudaMemcpy(camera, &mpScene->getCamera(), sizeof(Camera), cudaMemcpyHostToDevice);
	Color4f *frameBuffer = (Color4f *) frame.data();
	ParallelFor(maxQueueSize, KRR_DEVICE_LAMBDA(int pixelId){	
		// reset per-pixel radiance & sample state
		Vector2i pixelCoord = { pixelId % mFrameSize[0], pixelId / mFrameSize[0] };
		pixelState->L[pixelId] = 0;
		pixelState->sampler[pixelId].setPixelSample(pixelCoord, frameId * samplesPerPixel * randomLaunchTimeSeed);
		pixelState->sampler[pixelId].advance(256 * pixelId);
		guidedState->reset(pixelId);
		frameBuffer[pixelId] = Color4f(0);
	});
	// GPUCall(KRR_DEVICE_LAMBDA() { trainBuffer->clear(); });
	#if (OPTIMIZE_ON_NRC > 0)
		mNRC.beginFrame();
	#endif
	mDist.beginFrame();
	//@modified OFFLINE MODE
#if (RENDER_NOISY_ONLY == 0)
	if (mMouseOverImage.size() < mMouseOverImageResolution[0] * mMouseOverImageResolution[1])
		mMouseOverImage.resize(mMouseOverImageResolution[0] * mMouseOverImageResolution[1]);
#endif
}


void GuidedPathTracer::sampleWi(){
	PROFILE("Sampling pre-Inference Step");
	float* wiPtr = LearnedDistribution::inferenceWiPtr();
	AuxInfo* auxInfoPtr = neeInfoBuffer.data();
	ForAllQueued(scatterRayQueue, maxQueueSize,
		KRR_DEVICE_LAMBDA(const ScatterRayWorkItem & w) {
		Sampler sampler = &pixelState->sampler[w.pixelId];
		const int tid = blockIdx.x * blockDim.x + threadIdx.x; // I is the index of currentWorkItem!
		AuxInfo* auxInfoPtr_i = auxInfoPtr + tid;
		float probRR_rand = sampler.get1D();
		const bool RR_pass = probRR_rand >= probRR;
		auxInfoPtr_i->RR_pass = RR_pass;
		if (RR_pass) return;	

		const float bsdfSamplingFraction = mDist.mGuiding.bsdfSamplingFraction;
		const ShadingData& sd = w.sd;
		// IF the scatter event contains only DELTA lobe(s): do not use guiding distribution;
		const BSDFType bsdfType = sd.getBsdfType();
		float* wiPtr_i = wiPtr + tid * N_DIM_WI * N_WI_PER_PIXEL; // Stores (theta,phi) for each wi on NPM and (theta, phi, pdf(theta), pdf(phi)) for each wi on AR

		int wioffset = 0; // 0 if NEE is not enabled in compilation. N_DIM_WI otherwise.

		// decide whether to adopt the guiding distribution vs bsdf
		const bool guided_pass = (mDist.mGuiding.isEnableGuiding(w.depth) && (bsdfType & BSDF_SMOOTH) &&
			(bsdfSamplingFraction == 0 || sampler.get1D() >= bsdfSamplingFraction));
		auxInfoPtr_i->guided_pass = guided_pass;

		// We assume two-sided BSDFs (PDFs are stored as non-negative values)
		wiPtr_i[wioffset + 2] = 1.f;
		wiPtr_i[wioffset + 3] = 1.f;

		if (!guided_pass) {
			Vector3f woLocal = sd.frame.toLocal(sd.wo);
			BSDFSample sample = BxDF::sample(sd, woLocal, sampler, (int)sd.bsdfType);
			auxInfoPtr_i->bsdfSample = sample;
			if (sample.pdf && any(sample.f)) {
				Vector3f wiLocal = sample.wi;
				Vector3f wiWorld = sd.frame.toWorld(sample.wi);
				Vector2f wiSq = uniformSphereToSquare(wiWorld);
				
				// record the sampled wi
				wiPtr_i[wioffset] = wiSq[0];
				wiPtr_i[wioffset + 1] = wiSq[1];
			}
		}
	});
}

#if (RENDER_NOISY_ONLY == 0)
void GuidedPathTracer::renderGT(int samplesPerPixel, int maxDepth, bool saveRawPDF, bool productBSDF, bool productCosine, Vector2i frameSize){
	PROFILE("Render GT");
	Log(Info, "Starting GT rendering with %d spp and %d max depth", samplesPerPixel, maxDepth);
	int	maxQueueSizeLocal = frameSize[0] * frameSize[1];
	if (!maxQueueSizeLocal || maxQueueSizeLocal > maxQueueSize) return;

	ParallelFor(maxQueueSizeLocal, KRR_DEVICE_LAMBDA(int pixelId){
		pixelState->L[pixelId] = 0;
	});

	for (int sampleId = 0; sampleId < samplesPerPixel; sampleId++) {
		{
			GPUCall(KRR_DEVICE_LAMBDA() { currentRayQueue(0)->reset(); });
			PROFILE("Generate camera rays");
			RayQueue* cameraRayQueue = currentRayQueue(0);
			ParallelFor(maxQueueSizeLocal, KRR_DEVICE_LAMBDA(int pixelId){
				Sampler sampler = &pixelState->sampler[pixelId];
				Vector2i pixelCoord = { pixelId % frameSize[0], pixelId / frameSize[0] };
				
				Vector2f pixelCoordNormalized = Vector2f(pixelCoord) / Vector2f(frameSize); // [0, 1]
				Vector3f wi = normalize(squareToUniformSphere(pixelCoordNormalized));
				Vector3f p = offsetRayOrigin(mMouseOverInfoOut->raw_position, mMouseOverInfoOut->raw_normal, wi);
				Ray cameraRay = {p, wi};

				Color startingThp = Color::Ones();

				// Incorporate BSDF and/or cos into thp in first RayWorkItem
				if (productBSDF || productCosine){
					ShadingData sd = mMouseOverInfoOut->sd;
					Vector3f woLocal = sd.frame.toLocal(sd.wo);
					Vector3f wiLocal = sd.frame.toLocal(wi);
					Color bsdfVal = BxDF::f(sd, woLocal, wiLocal, (int)sd.bsdfType);
					float cosVal = fabs(wiLocal[2]);
					if (productBSDF)
						startingThp *= bsdfVal;
					if (productCosine)
						startingThp *= cosVal;
				}

				cameraRayQueue->pushCameraRay(cameraRay, pixelId, startingThp);
			});
		}

		// [STEP#2] do radiance estimation recursively
		for (int depth = 0; true; depth++) {
			GPUCall(KRR_DEVICE_LAMBDA() {
				nextRayQueue(depth)->reset();
				hitLightRayQueue->reset();
				missRayQueue->reset();
				shadowRayQueue->reset();
				scatterRayQueue->reset();
				bsdfEvalQueue->reset();
			});
			// [STEP#2.1] find closest intersections, filling in scatterRayQueue and hitLightQueue
			backend->traceClosest(
				maxQueueSizeLocal,	// cuda::automic can not be accessed directly in host code
				currentRayQueue(depth),
				missRayQueue,
				hitLightRayQueue,
				scatterRayQueue,
				nextRayQueue(depth));

			// [STEP#2.2] handle hit and missed rays, contribute to pixels
			// handleEmissiveHit();
			{
				PROFILE("Process intersected rays");
				ForAllQueued(hitLightRayQueue, maxQueueSizeLocal,
					KRR_DEVICE_LAMBDA(const HitLightWorkItem & w){
					Color Le = w.light.L(w.p, w.n, w.uv, w.wo);
					float misWeight = 1.f;
					Color contrib = Le * w.thp * misWeight;
					if (w.thp.isNaN().any()) {
						printf("handleEmissiveHit: NaN detected in w.thp: (%f,%f,%f).\nTerminating...\n", w.thp[0],w.thp[1], w.thp[2]);
						assert(false);
					}
					pixelState->addRadiance(w.pixelId, contrib);
				});
			}
			if (depth || !transparentBackground)  handleMiss();
			// [Sidestory] break on maximum bounce, but after handling emissive intersections.
			if (depth == maxDepth) break;

			{
				PROFILE("NEE sampling");
				AuxInfo* auxInfoPtr = neeInfoBuffer.data();
				ForAllQueued(scatterRayQueue, maxQueueSizeLocal,
					KRR_DEVICE_LAMBDA(const ScatterRayWorkItem & w) {
					Sampler sampler = &pixelState->sampler[w.pixelId];
					const int tid = blockIdx.x * blockDim.x + threadIdx.x; // I is the index of currentWorkItem!
					AuxInfo* auxInfoPtr_i = auxInfoPtr + tid;
					float probRR_rand = sampler.get1D();
					const bool RR_pass = probRR_rand >= probRR;
					auxInfoPtr_i->RR_pass = RR_pass;
					if (RR_pass) return;

					// decide whether to adopt the guiding distribution vs bsdf
					auxInfoPtr_i->guided_pass = false;
				});
			}
			{
				PROFILE("BSDF sampling");
				AuxInfo* auxInfoPtr = neeInfoBuffer.data();
				ForAllQueued(scatterRayQueue, maxQueueSizeLocal,
					KRR_DEVICE_LAMBDA(const ScatterRayWorkItem & w) {
					const int tid = blockIdx.x * blockDim.x + threadIdx.x; // I is the index of currentWorkItem!
					AuxInfo* auxInfoPtr_i = auxInfoPtr + tid;

					if (auxInfoPtr_i->RR_pass) return;

					Sampler sampler = &pixelState->sampler[w.pixelId];
					const ShadingData& sd = w.sd;
					const BSDFType bsdfType = sd.getBsdfType();
					Vector3f woLocal = sd.frame.toLocal(sd.wo);

					/* sample BSDF */
					BSDFSample sample = BxDF::sample(sd, woLocal, sampler, (int)sd.bsdfType);
					auxInfoPtr_i->bsdfSample = sample;

					bsdfEvalQueue->push(tid);
				});
			}

			{
				PROFILE("BSDF handling");
				const AuxInfo* auxInfoPtr = neeInfoBuffer.data();
				ForAllQueued(bsdfEvalQueue, maxQueueSizeLocal,
					KRR_DEVICE_LAMBDA(const BsdfEvalWorkItem & id) {
					const ScatterRayWorkItem w = scatterRayQueue->operator[](id.itemId);
					const ShadingData& sd = w.sd;
					const BSDFType bsdfType = sd.getBsdfType();
					Vector3f woLocal = sd.frame.toLocal(sd.wo);

					/* sample BSDF */
					const AuxInfo* auxInfoPtr_i = auxInfoPtr + id.itemId;
					BSDFSample sample = auxInfoPtr_i->bsdfSample;

					if (sample.pdf && any(sample.f)) {
						Vector3f wiWorld = sd.frame.toWorld(sample.wi);
						RayWorkItem r = {};
						Vector3f p = offsetRayOrigin(sd.pos, sd.frame.N, wiWorld);
						float bsdfPdf	 = sample.pdf, guidedPdf{};
						float scatterPdf = bsdfPdf;

						r.pdf = scatterPdf;
						r.ray = { p, wiWorld };
						r.ctx = { sd.pos, sd.frame.N };
						r.pixelId = w.pixelId;
						r.depth = w.depth + 1;
						r.thp = w.thp * sample.f * fabs(sample.wi[2]) / (r.pdf + M_EPSILON);
						if (r.thp.isNaN().any()) {
							printf("handleBsdfSampling: NaN detected in r.thp: (%f,%f,%f). r.pdf: %f; w.thp: (%f,%f,%f) \nTerminating...\n", r.thp[0],r.thp[1], r.thp[2], r.pdf, w.thp[0],w.thp[1], w.thp[2]);
							assert(false);
						}
						r.bsdfType = sample.flags;
						nextRayQueue(w.depth)->push(r);
					}
				});
			}
		}
	}

	Color4f *outputFrameBuffer = (Color4f *) mGTMouseOverImage.data();
	ParallelFor(maxQueueSizeLocal, KRR_DEVICE_LAMBDA(int pixelId){
		Color L = pixelState->L[pixelId] / float(samplesPerPixel);
		if (!saveRawPDF){
			const float3 colorViridis = luminanceToViridis(
				logf(1 + L.mean())
			);
			L = Color(colorViridis.x, colorViridis.y, colorViridis.z);
		}
		outputFrameBuffer[pixelId] = Color4f(L, 1.0f);
	});
}
#endif

void GuidedPathTracer::render(CUDABuffer& frame){
	if (!mpScene || !maxQueueSize) return;
	PROFILE("Guided Path Tracer");

	for (int sampleId = 0; sampleId < samplesPerPixel; sampleId++) {
		// [STEP#1] generate camera / primary rays
		GPUCall(KRR_DEVICE_LAMBDA() { currentRayQueue(0)->reset(); });
		generateCameraRays(sampleId);
		// [STEP#2] do radiance estimation recursively
		for (int depth = 0; true; depth++) {
			GPUCall(KRR_DEVICE_LAMBDA() {
				nextRayQueue(depth)->reset();
				hitLightRayQueue->reset();
				missRayQueue->reset();
				shadowRayQueue->reset();
				scatterRayQueue->reset();
			});
			// [STEP#2.1] find closest intersections, filling in scatterRayQueue and hitLightQueue
			backend->traceClosest(
				maxQueueSize,	// cuda::automic can not be accessed directly in host code
				currentRayQueue(depth),
				missRayQueue,
				hitLightRayQueue,
				scatterRayQueue,
				nextRayQueue(depth));

			// [STEP#2.2] handle hit and missed rays, contribute to pixels
			handleEmissiveHit();
			if (depth || !transparentBackground)  handleMiss();

			#if (OPTIMIZE_ON_NRC > 0) 
				/*
				If we are using NRC as reference, we must register the bounce after the 
				last training bounce for it to be an input to NRC.
				*/
				if (depth == mDist.mGuiding.maxTrainDepth){
					handleNRCLastBounce();
				}
			#endif 

			// [Sidestory] break on maximum bounce, but after handling emissive intersections.
			if (depth == maxDepth)break;

			// Samples wi to evaluate BSDF and NEE wis on guiding network and decide which paths need to be sampled from network.
			sampleWi();

			// [STEP#2.3] handle intersections and shadow rays
			// generate dist for every intersections, since both nee and guided sampling needs it
			if (mDist.mGuiding.isEnableGuiding(depth)){ 
				inferenceStep();
			}
			// tracing shadow rays should be before increment depth (the NEE contribution not included in current vertex)
			handleIntersections();	
		}
	}

	// write results of the current frame...
	Color4f *frameBuffer = (Color4f *) frame.data();
	if (visMode == VisualizationMode::Noisy){
		ParallelFor(maxQueueSize, KRR_DEVICE_LAMBDA(int pixelId){
			Color L = pixelState->L[pixelId] / float(samplesPerPixel);
			if (enableClamp) L = clamp(L, 0.f, clampMax);
			renderedImage->put(Color4f(L, 1), pixelId);
			if (renderMode == RenderMode::Interactive)
				frameBuffer[pixelId] = Color4f(L, 1);
			else if (renderMode == RenderMode::Offline)
				frameBuffer[pixelId] = renderedImage->getPixel(pixelId);
		});
	}

#if (RENDER_NOISY_ONLY == 0)
	// For all other visualization modes, we spawn camera rays again
	GPUCall(KRR_DEVICE_LAMBDA() { currentRayQueue(0)->reset(); scatterRayQueue->reset();});
	generateCameraRays(0, true);
	backend->traceOneHit(
		maxQueueSize,
		currentRayQueue(0),
		scatterRayQueue);
	
	// Retrieve gBuffer data from the first hit and write the results to the frame buffer
	const AABB sceneAABB = mpScene->getAABB();
	ForAllQueued(scatterRayQueue, maxQueueSize,
		KRR_DEVICE_LAMBDA(ScatterRayWorkItem & w) {	
		const ShadingData& sd = w.sd;
		const uint pixelId = w.pixelId;
		const int tid = blockIdx.x * blockDim.x + threadIdx.x;
		Color L;
		
		switch (visMode) {
			case VisualizationMode::Diffuse:
				frameBuffer[pixelId] = Color4f(sd.diffuse, 1);
				break;
			case VisualizationMode::Specular:
				frameBuffer[pixelId] = Color4f(sd.specular, 1);
				break;
			case VisualizationMode::Normal:
				frameBuffer[pixelId] = Color4f(Color(sd.frame.N * 0.5f + Vector3f(0.5f)), 1);
				break;
			default:
				break;
		}

		// Collect mouseover info
		const Vector2i pixelCoord = { w.pixelId % mFrameSize[0], mFrameSize[1] - w.pixelId / mFrameSize[0] }; // Y is flipped since pixelId is counted from the top-left corner. Flipping aligns it with the bottom-left corner.
		if (!(mMouseOverInfoIn->lock) && (pixelCoord[0] == mMouseOverInfoIn->mouseCoords[0] && pixelCoord[1] == mMouseOverInfoIn->mouseCoords[1])) {
			mMouseOverInfoOut->position = normalizeSpatialCoord(sd.pos, sceneAABB);
			mMouseOverInfoOut->raw_position = sd.pos;
			mMouseOverInfoOut->normal = utils::cartesianToSphericalNormalized(sd.frame.N);
			mMouseOverInfoOut->raw_normal = sd.frame.N;
			mMouseOverInfoOut->diffuse = sd.diffuse;
			mMouseOverInfoOut->specular = sd.specular;
			mMouseOverInfoOut->uv = sd.uv;
			mMouseOverInfoOut->roughness = sd.roughness;
			mMouseOverInfoOut->wo = sd.wo.normalized();
			mMouseOverInfoOut->isEmissive = sd.light != nullptr;
			mMouseOverInfoOut->sd = sd;
		}
	});

	mouseOverInference();
#endif
}

#if (RENDER_NOISY_ONLY == 0)
void GuidedPathTracer::mouseOverInference(){
	PROFILE("Mouseover inference");


#if (OPTIMIZE_ON_NRC > 0)
	if (showNRCMouseOver){
		mNRC.mouseOverInference(
			mMouseOverInfoOut,
			mMouseOverImage.data(),
			mMouseOverImageResolution,
			currentRayQueue(0),
			scatterRayQueue,
			pixelState,
			backend,
			mpScene->getAABB()
		);
	}else{
#endif
		// Write wi for each pixel in mouseOverImage
		{
			float* wiPtr = LearnedDistribution::inferenceWiPtr();
			ParallelFor(mMouseOverImageResolution[0] * mMouseOverImageResolution[1], KRR_DEVICE_LAMBDA(int pixelId){
				const int pixelCoord[2] = { pixelId % mMouseOverImageResolution[0], pixelId / mMouseOverImageResolution[0] };
				float* wiPtr_i = wiPtr + pixelId * (N_WI_PER_PIXEL * N_DIM_WI);
				wiPtr_i[0] = (float)pixelCoord[0] / mMouseOverImageResolution[0];
				wiPtr_i[1] = ((float)pixelCoord[1] / mMouseOverImageResolution[1]);

				wiPtr_i[0] += 0.5f / mMouseOverImageResolution[0];
				wiPtr_i[1] += 0.5f / mMouseOverImageResolution[1];
			});
			cudaDeviceSynchronize();
		}

		// Inference
		{
			mDist.mouseOverInference(mMouseOverInfoOut, mMouseOverImage.data(), mMouseOverImageResolution);
		}
#if (OPTIMIZE_ON_NRC > 0)
	}
#endif

	// Normalize values in mouseOverImage
	{
		const float* image_ptr = mMouseOverImage.data();
		mMouseOverImageTextureBufferSurfaceHolder.beginCUDAAccess(0);
		const CUsurfObject surfaceObj = mMouseOverImageTextureBufferSurfaceHolder.getNext();
		ParallelFor(mMouseOverImageResolution[0] * mMouseOverImageResolution[1], KRR_DEVICE_LAMBDA(int pixelId){
			const int x = pixelId % mMouseOverImageResolution[0];
			const int y = pixelId / mMouseOverImageResolution[0];
			float val = image_ptr[pixelId];
			if (mMouseOverApplyLog){
				val = logf(1 + val);
			}
			
			float3 color;
			switch (mMouseOverVisualizationMode) {
				case 0: // MouseOverVisualizationMode::LuminanceBW
					color = make_float3(val, val, val);
					break;
				case 1: // MouseOverVisualizationMode::LuminanceViridis
					color = luminanceToViridis(val);
					break;
				default:
					break;
			}

			union Alias {
				float4 targetType;
				uint4 uiValue;
			} u;
			u.targetType = make_float4(color.x, color.y, color.z, 1.0f);

			surf2Dwrite(u.uiValue, surfaceObj, x * sizeof(float4), y, cudaBoundaryModeTrap);

		});
		mMouseOverImageTextureBufferSurfaceHolder.endCUDAAccess(0, false);
	}
}

void GuidedPathTracer::saveMouseOverSampling(const string& filename, uint numSamples, bool saveRaw){
	static GPUMemory<PCGSampler> mouseOverSamplers = GPUMemory<PCGSampler>(MOUSEOVER_SAMPLING_N_SAMPLERS);
	static bool samplersInitialized = false;
	GPUMemory<float> normalizedDirections = GPUMemory<float>(numSamples*2);
	GPUMemory<uint> samplingPixelCount = GPUMemory<uint>(mMouseOverImageResolution[0] * mMouseOverImageResolution[1]);
	GPUMemory<float> mouseOverImageRGBA = GPUMemory<float>(mMouseOverImageResolution[0] * mMouseOverImageResolution[1] * 4);


	normalizedDirections.memset(0, normalizedDirections.size(), 0);
	samplingPixelCount.memset(0, samplingPixelCount.size(), 0);
	mouseOverImageRGBA.memset(0, mouseOverImageRGBA.size(), 0);
	PCGSampler* mouseOverSamplersPtr = mouseOverSamplers.data();
	if (!samplersInitialized){
		ParallelFor(MOUSEOVER_SAMPLING_N_SAMPLERS, KRR_DEVICE_LAMBDA(int idx){
			mouseOverSamplersPtr[idx].setSeed(idx * 256);
		});
		cudaDeviceSynchronize();
		samplersInitialized = true;
	}else{
		ParallelFor(MOUSEOVER_SAMPLING_N_SAMPLERS, KRR_DEVICE_LAMBDA(int idx){
			mouseOverSamplersPtr[idx].advance(idx * 256);
		});
		cudaDeviceSynchronize();
	}

	mDist.mouseOverSampling(mMouseOverInfoOut, normalizedDirections.data(), numSamples, mouseOverSamplersPtr);
	CUDA_SYNC_CHECK();

	{
		const float* normalizedDirectionsPtr = normalizedDirections.data();
		uint* samplingImagePtr = samplingPixelCount.data();
		ParallelFor(numSamples, KRR_DEVICE_LAMBDA(int idx){
			if (idx >= numSamples) return;
			const float* normalizedDirection = normalizedDirectionsPtr + idx * 2;
			const int x = (int)(normalizedDirection[0] * mMouseOverImageResolution[0]);
			const int y = (int)(normalizedDirection[1] * mMouseOverImageResolution[1]);
			if (x >= mMouseOverImageResolution[0] || y >= mMouseOverImageResolution[1] || x < 0 || y < 0){
				// printf("Idx: %d. Invalid pixel coordinates: %d %d\n", idx, x, y);
				return;
			}
			const int pixelId = y * mMouseOverImageResolution[0] + x;
			atomicAdd(samplingImagePtr + pixelId, 1);
		});
		CUDA_SYNC_CHECK();
	}

	{
		// Find max value (in parallel)
		// const uint maxNSamplesPerBin = (uint) thrust::reduce(thrust::device, samplingPixelCount.data(), samplingPixelCount.data() + samplingPixelCount.size(), 0, thrust::maximum<uint>());
		cudaDeviceSynchronize();
		uint* samplingImagePtr = samplingPixelCount.data();
		float* mouseOverImageRGBAPtr = mouseOverImageRGBA.data();
		const int saveRawPDFMouseOverInt = saveRaw ? 1 : 0;
		ParallelFor(mMouseOverImageResolution[0] * mMouseOverImageResolution[1], KRR_DEVICE_LAMBDA(int pixelId){
			if (pixelId >= mMouseOverImageResolution[0] * mMouseOverImageResolution[1]) return;
			const int x = pixelId % mMouseOverImageResolution[0];
			const int y = pixelId / mMouseOverImageResolution[0];
			const float pdf_transform = squareToUniformSpherePdf();
			float val = (float) samplingImagePtr[pixelId] * (mMouseOverImageResolution[0] * mMouseOverImageResolution[1]) / (double) numSamples * pdf_transform;
			
			float3 color;
			if (saveRawPDFMouseOverInt == 0){
				if (mMouseOverApplyLog){
					val = logf(1 + val);
				}
				switch (mMouseOverVisualizationMode) {
					case 0: // MouseOverVisualizationMode::LuminanceBW
						color = make_float3(val, val, val);
						break;
					case 1: // MouseOverVisualizationMode::LuminanceViridis
						color = luminanceToViridis(val);
						break;
					default:
						break;
				}
			}

			else{
				color = make_float3(val, val, val);
			}
			
			mouseOverImageRGBAPtr[pixelId * 4] = color.x;
			mouseOverImageRGBAPtr[pixelId * 4 + 1] = color.y;
			mouseOverImageRGBAPtr[pixelId * 4 + 2] = color.z;
			mouseOverImageRGBAPtr[pixelId * 4 + 3] = 1.0f;
		});
		CUDA_SYNC_CHECK();
		saveImage(filename, mouseOverImageRGBAPtr, mMouseOverImageResolution[0], mMouseOverImageResolution[1]);
	}
}
#endif

void GuidedPathTracer::endFrame(CUDABuffer& frame){
	if (mDist.mGuiding.isEnableTraining() && !mDist.mGuiding.isTrainingFinished && !trainDebug) trainStep();
	if (trainDebug && mDist.mGuiding.trainState.oneStep) { trainStep(); mDist.mGuiding.trainState.oneStep = false; }
	frameId++;
	// OFFLINE MODE
	task.tickFrame();

	mDist.endFrame();
	
	Budget budget = task.getBudget();
	if ( mDist.mGuiding.autoTrain && !mDist.mGuiding.isTrainingFinished && /* determine whether training phase is over */
		((budget.type == BudgetType::Spp && task.getProgress() >= trainingBudgetSpp) ||
		(budget.type == BudgetType::Time && task.getProgress() >= trainingBudgetTime))) {
		mDist.mGuiding.isTrainingFinished = true;
		cudaDeviceSynchronize();
		if (discardTraining)
			renderedImage->reset();
		CUDA_SYNC_CHECK();
	}
	if (task.isFinished())
		gpContext->requestExit();
}

void GuidedPathTracer::finalize() {
	// save results and quit
	cudaDeviceSynchronize();

	auto capture = profilerIntegrator->endCapture();
	Profiler::saveToCSV(capture, "profiler.csv");

#if (RENDER_NOISY_ONLY == 0)
	cudaFree(mMouseOverInfoIn);
	cudaFree(mMouseOverInfoOut);
#endif

	string output_name = gpContext->getGlobalConfig().contains("name") ? 
		gpContext->getGlobalConfig()["name"] : "result";
	fs::path save_path = File::outputDir() / (output_name + ".exr");
	renderedImage->save(save_path);
	Log(Info, "Total SPP: %zd (%zd), elapsed time: %.1f", task.getCurrentSpp(), frameId, task.getElapsedTime());
	Log(Success, "Task finished, saving results to %s", save_path.string().c_str());
	CUDA_SYNC_CHECK();
}

void GuidedPathTracer::renderUI(){
	ui::Text("Render parameters");
	ui::InputInt("Samples per pixel", &samplesPerPixel);
	ui::InputInt("Max bounces", &maxDepth, 1);
	ui::SliderFloat("Russian roulette", &probRR, 0, 1);

	// Output Buffer Visualizations
#if (RENDER_NOISY_ONLY == 0)
	static const char* vis_modes[] = { "Noisy", "Diffuse", "Specular", "Normal"};
	ui::Combo("Visualization Mode", (int*)&visMode, vis_modes, (int)VisualizationMode::NumTypes);
#else
	static const char* vis_modes[] = { "Noisy"};
	ui::Combo("Visualization Mode [RENDER_NOISY_ONLY]", (int*)&visMode, vis_modes, 1);
#endif

	if (mDist.hasInitializedNetwork()) {	
		ui::Text("Guidance");
		mDist.renderUI();
	}

	// Mouseover visualization
#if (RENDER_NOISY_ONLY == 0)
	// Show mouseover info
	static bool showMouseoverInfo = false;
	ui::Checkbox("Mouseover info", &showMouseoverInfo);
	if (showMouseoverInfo) {
		ui::Begin("Mouseover info", &showMouseoverInfo);
		auto mouseOverInfoInHost = MouseOverInfoInput();
		auto mouseOverInfoOutHost = MouseOverInfoOutput();
		
		cudaMemcpy(&mouseOverInfoInHost, mMouseOverInfoIn, sizeof(MouseOverInfoInput), cudaMemcpyDeviceToHost);
		cudaMemcpy(&mouseOverInfoOutHost, mMouseOverInfoOut, sizeof(MouseOverInfoOutput), cudaMemcpyDeviceToHost);

		ui::Text("Cursor Info: (%d, %d)", mouseOverInfoInHost.mouseCoords[0], mouseOverInfoInHost.mouseCoords[1]);
		ui::Text("Position: (%f, %f, %f)", mouseOverInfoOutHost.position[0], mouseOverInfoOutHost.position[1], mouseOverInfoOutHost.position[2]);
		ui::Text("Wo: (%f, %f, %f)", mouseOverInfoOutHost.wo[0], mouseOverInfoOutHost.wo[1], mouseOverInfoOutHost.wo[2]);
		ui::Text("Normal: (%f, %f)", mouseOverInfoOutHost.normal[0], mouseOverInfoOutHost.normal[1]);

		ui::Text("Diffuse: (%f, %f, %f)", mouseOverInfoOutHost.diffuse[0], mouseOverInfoOutHost.diffuse[1], mouseOverInfoOutHost.diffuse[2]);
		ui::SameLine();
		auto drawList = ui::GetWindowDrawList();
		const int sz = ui::GetTextLineHeight();
		
		auto diffuseColor = ImColor(mouseOverInfoOutHost.diffuse[0], mouseOverInfoOutHost.diffuse[1], mouseOverInfoOutHost.diffuse[2], 1.0f);
		ImVec2 p = ImGui::GetCursorScreenPos();
		drawList->AddRectFilled(p, ImVec2(p.x + sz, p.y + sz), diffuseColor);
		ui::Dummy(ImVec2(sz, sz));
		
		ui::Text("Specular: (%f, %f, %f)", mouseOverInfoOutHost.specular[0], mouseOverInfoOutHost.specular[1], mouseOverInfoOutHost.specular[2]);
		ui::SameLine();
		auto specularColor = ImColor(mouseOverInfoOutHost.specular[0], mouseOverInfoOutHost.specular[1], mouseOverInfoOutHost.specular[2], 1.0f);
		p = ImGui::GetCursorScreenPos();
		drawList->AddRectFilled(p, ImVec2(p.x + sz, p.y + sz), specularColor);
		ui::Dummy(ImVec2(sz, sz));

		ui::Text("UV: (%f, %f)", mouseOverInfoOutHost.uv[0], mouseOverInfoOutHost.uv[1]);
		ui::Text("Roughness: %f", mouseOverInfoOutHost.roughness);
		ui::Text("Is emissive: %s", mouseOverInfoOutHost.isEmissive ? "true" : "false");
		ui::Text("Is Two Sided: %s", mouseOverInfoOutHost.sd.isTwoSided() ? "true" : "false");
		ui::Separator();

		const std::string lock_unlock_message = mouseOverInfoInHost.lock ? "[SPACE] Unlock mouseover" : "[SPACE] Lock mouseover";
		if (ui::Button(lock_unlock_message.c_str())) {
			invertMouseOverLock();
		}

		ui::Text("PDF visualization");
		ui::InputInt2("Output Resolution", (int*)&mMouseOverImageResolution);
		ui::Text("Visualization Mode");
		ui::RadioButton("Luminance - BW", &mMouseOverVisualizationMode, (int)MouseOverVisualizationMode::LuminanceBW);
		ui::RadioButton("Luminance - Viridis", &mMouseOverVisualizationMode, (int)MouseOverVisualizationMode::LuminanceViridis);

		#if (OPTIMIZE_ON_NRC > 0)
			ui::Checkbox("Show NRC Visualization", &showNRCMouseOver);
		#else
			ui::Text("set \"OPTIMIZE_ON_NRC\" to a value greater than 0 enable NRC Visualization");
		#endif
		
		const ImVec2 avail_size = ui::GetContentRegionAvail();
		ui::Spacing();
		ui::Image((void*)(intptr_t)mMouseOverImageTexture.getHandle(), ImVec2(avail_size.x, avail_size.y - 30));

		static char mouseOverOutputPath[256] = "./mouseOver.exr";
		const float* mouseOverImagePtr = mMouseOverImage.data();
		static bool saveRawPDFMouseOver = false;
		ui::Text("Save MouseOver Visualization");
		ui::InputText("MouseOver output path", mouseOverOutputPath, 256);
		ui::Checkbox("Save Raw PDF Mouseover", &saveRawPDFMouseOver);
		ui::Checkbox("Apply 1+Log Transform to PDF Mouseover", &mMouseOverApplyLog);
		if (ui::Button("Save MouseOver Image")) {

			GPUMemory<float> mouseOverImageRGBA = GPUMemory<float>(mMouseOverImageResolution[0] * mMouseOverImageResolution[1] * 4);
			float* mouseOverImageRGBAPtr = mouseOverImageRGBA.data();
			const int saveRawPDFMouseOverInt = saveRawPDFMouseOver ? 1 : 0;
			ParallelFor(mMouseOverImageResolution[0] * mMouseOverImageResolution[1], KRR_DEVICE_LAMBDA(int pixelId){
				const int x = pixelId % mMouseOverImageResolution[0];
				const int y = pixelId / mMouseOverImageResolution[0];
				float val = mouseOverImagePtr[pixelId];
				float3 color;
				if (saveRawPDFMouseOverInt == 0){
					if (mMouseOverApplyLog){
						val = logf(1 + val);
					}
					switch (mMouseOverVisualizationMode) {
						case 0: // MouseOverVisualizationMode::LuminanceBW
							color = make_float3(val, val, val);
							break;
						case 1: // MouseOverVisualizationMode::LuminanceViridis
							color = luminanceToViridis(val);
							break;
						default:
							break;
					}
				}
				else{
					color = make_float3(val, val, val);
				}
				
				mouseOverImageRGBAPtr[pixelId * 4] = color.x;
				mouseOverImageRGBAPtr[pixelId * 4 + 1] = color.y;
				mouseOverImageRGBAPtr[pixelId * 4 + 2] = color.z;
				mouseOverImageRGBAPtr[pixelId * 4 + 3] = 1.0f;
			});
			cudaDeviceSynchronize();
			saveImage({mouseOverOutputPath}, mouseOverImageRGBAPtr, mMouseOverImageResolution[0], mMouseOverImageResolution[1]);
		}

		static char gt_PDF_path[256] = "./gt_PDF.exr";
		static int gt_PDF_spp = samplesPerPixel;
		static int gt_PDF_maxDepth = maxDepth;
		static bool saveRawPDF = false;
		static bool saveIndividualSamples = false;
		static bool productBSDF = false, productCos = false;
		static Vector2i frameSizeLocal = mFrameSize;
		ui::Separator();
		ui::Text("Compute GT PDF");
		ui::InputText("GT PDF path", gt_PDF_path, 256);
		ui::Checkbox("Save Raw PDF GT", &saveRawPDF);
		ui::SameLine();
		ui::Checkbox("Save Individual Samples GT", &saveIndividualSamples);
		ui::Checkbox("Product BSDF", &productBSDF);
		ui::SameLine();
		ui::Checkbox("Product cos", &productCos);
		ui::InputInt2("GT PDF resolution", (int*)&frameSizeLocal);
		ui::InputInt("GT PDF spp", &gt_PDF_spp);
		ui::SameLine();
		ui::InputInt("GT PDF maxDepth", &gt_PDF_maxDepth);
		
		if (ui::Button("Compute PDF GT")) {
			if (!saveIndividualSamples){
				renderGT(gt_PDF_spp, gt_PDF_maxDepth, saveRawPDF, productBSDF, productCos, frameSizeLocal);
				saveImage({gt_PDF_path}, mGTMouseOverImage.data(), frameSizeLocal[0], frameSizeLocal[1]);
			}
			else{
				// Save individual samples
				fs::path base_path = fs::path(gt_PDF_path).parent_path();
				fs::path filename = fs::path(gt_PDF_path).filename();
				fs::path individual_samples_path = base_path / "individual_samples";
				fs::create_directories(individual_samples_path);
				for (int i = 0; i < gt_PDF_spp; i++){
					renderGT(1, gt_PDF_maxDepth, saveRawPDF, productBSDF, productCos, frameSizeLocal);
					const string gt_pdf_i = (individual_samples_path / (filename.stem().string() + "_" + std::to_string(i) + ".exr")).string();
					saveImage({gt_pdf_i}, mGTMouseOverImage.data(), frameSizeLocal[0], frameSizeLocal[1]);
				}
			}
		}
		
		static char samplingOutputPath[256] = "./sampling.exr";
		static int numSamples = 1024;
		static bool saveRawSampling = false;
		ui::Separator();
		ui::Text("Sample from MouseOver Position");
		ui::InputText("Sampling representation path", samplingOutputPath, 256);
		ui::Checkbox("Save Raw Samples", &saveRawSampling);
		ui::InputInt("Number of samples", &numSamples);
		if (ui::Button("Sample")) {
			saveMouseOverSampling(samplingOutputPath, numSamples, saveRawSampling);
		}
		ui::End();
	}

#else
	ui::Text("To enable Mouseover Visualization, unset RENDER_NOISY_ONLY");
#endif

	ui::Text("Current step: %d; %d samples; loss: %f", numLossSamples, numTrainingSamples, curLossScalar.emaVal());
	ui::PlotLines("Loss graph", lossGraph.data(), min(numLossSamples, lossGraph.size()),
		numLossSamples < LOSS_GRAPH_SIZE ? 0 : numLossSamples % LOSS_GRAPH_SIZE, 0, FLT_MAX, FLT_MAX, ImVec2(0, 50));

	
#if (OPTIMIZE_ON_NRC > 0)
	ui::Text("NRC Loss stats");
	ui::Text("Current step: %d; %d samples; loss: %f", numLossSamplesNRC, numTrainingSamplesNRC, curLossScalarNRC.emaVal());
	ui::PlotLines("Loss graph", lossGraphNRC.data(), min(numLossSamplesNRC, lossGraphNRC.size()),
		numLossSamplesNRC < LOSS_GRAPH_SIZE ? 0 : numLossSamplesNRC % LOSS_GRAPH_SIZE, 0, FLT_MAX, FLT_MAX, ImVec2(0, 50));
#endif
	
	if (ui::CollapsingHeader("Save/Load")){
		static char networkSavePath[256] = "./";
		ui::Text("Save network weights");
		ui::InputText("Save network weights base path", networkSavePath, 256);
		if (ui::Button("Save")){
			mDist.saveNetwork(std::string(networkSavePath));
		}

		static char networkLoadPath[256] = "./";
		ui::Text("Load network weights");
		ui::InputText("Load network weights base path", networkLoadPath, 256);
		if (ui::Button("Load")){
			mDist.loadNetwork(std::string(networkLoadPath));
		}
	}
	
	if (ui::CollapsingHeader("Task progress")) {
		task.renderUI();
	}
	ui::Checkbox("Train debug", &trainDebug);
	if (ui::Button("Reset parameters")) {
		resetTraining();
	}
	ui::Checkbox("Clamping pixel value", &enableClamp);
	if (enableClamp) {
		ui::SameLine();
		ui::DragFloat("Max:", &clampMax, 1, 1, 1e5f, "%.1f");
	}
}

void GuidedPathTracer::imageFlipY(float* imageDevice, int width, int height, int channels){
	ParallelFor(width * height / 2, KRR_DEVICE_LAMBDA(int pixelId){
		const int x = pixelId % width;
		const int y = pixelId / width;
		const int y_flipped = height - y - 1;

		Vector4f temp;
		TCNN_PRAGMA_UNROLL
		for (int c = 0; c < channels; c++) {
			temp[c] = imageDevice[(y * width + x) * channels + c];
		}
		
		TCNN_PRAGMA_UNROLL
		for (int c = 0; c < channels; c++) {
			imageDevice[(y * width + x) * channels + c] = imageDevice[(y_flipped * width + x) * channels + c];
			imageDevice[(y_flipped * width + x) * channels + c] = temp[c];
		}
	});
}

void GuidedPathTracer::saveImage(const string& filename, float* imagePtr, int width, int height, bool flipY){
	if (flipY){
		imageFlipY(imagePtr, width, height, 4);
		cudaDeviceSynchronize();
	}

	Image image({width, height}, Image::Format::RGBAfloat);
	cudaMemcpy(image.data(), imagePtr, width * height * 4 * sizeof(float), cudaMemcpyDeviceToHost);
	fs::path filepath = File::resolve(filename);
	if (!fs::exists(filepath.parent_path()))
		fs::create_directories(filepath.parent_path());
	image.saveImage(filepath);
	logSuccess("Image saved to " + filepath.string());
}

void GuidedPathTracer::resetNetwork(json config){
	neeInfoBuffer = GPUMemory<AuxInfo>(MAX_INFERENCE_NUM);
	mDist.resetNetwork(config);
}

void GuidedPathTracer::resetTraining() {
	mDist.resetTraining();
	task.reset();
	numLossSamples = 0;
}

/*
* Do inference for all intersections that needs scattering events.
* [1] generate inference inputs and let network predict raw outputs;
* [2] importance sample on the output spherical basis;
* [3] push rayqueue, update throughput, increment guided depth, etc.
*/
void GuidedPathTracer::inferenceStep(){
	PROFILE("Inference");
	mDist.inferenceStep(scatterRayQueue, pixelState, neeInfoBuffer.data(), mpScene->getAABB());
}

void GuidedPathTracer::trainStep(){
	// If optimizing on NRC, we must record NRC's output to L in guidedState before performing the training step
#if (OPTIMIZE_ON_NRC > 0)

	bool skipInferenceDataGeneration = false;
    if (N_TRAIN_ITER_NRC == -1 || (int) numLossSamples < N_TRAIN_ITER_NRC){
		float normalized_lossNRC = mNRC.trainStep(guidedState, maxQueueSize, mpScene->getAABB(), numTrainingSamplesNRC);
		curLossScalarNRC.update(normalized_lossNRC);
		lossGraphNRC[numLossSamplesNRC++ % LOSS_GRAPH_SIZE] = curLossScalarNRC.emaVal();
		skipInferenceDataGeneration = true;
	}

	if (numLossSamples >= START_TRAIN_ITER_NRC){ 
		mNRC.inferenceStepFromTrainingData( 
			guidedState, maxQueueSize, mpScene->getAABB(), skipInferenceDataGeneration
		); 
	}else{ 
		// If we are not using NRC as inputs to our method, we must decrement the depth of the guidedState to remove the last invalid bounce 
		// Only applies if we have not already decremented the depth in trainStep
		if (!skipInferenceDataGeneration){
			ParallelFor(maxQueueSize, KRR_DEVICE_LAMBDA(int tid){ 
				int pixelId = mNRC.trainPixelOffset + tid * mNRC.trainPixelStride; 
				int depth = guidedState->curDepth[pixelId]; 
				guidedState->curDepth[pixelId]--;
			}); 
			cudaDeviceSynchronize(); 
		}
		mNRC.clearInferenceBuffer();
	} 
#endif

	float normalized_loss = mDist.trainStep(guidedState, maxQueueSize, mpScene->getAABB(), numTrainingSamples);
	curLossScalar.update(normalized_loss);
	lossGraph[numLossSamples++ % LOSS_GRAPH_SIZE] = curLossScalar.emaVal();
}

KRR_REGISTER_PASS_DEF(GuidedPathTracer);
KRR_NAMESPACE_END