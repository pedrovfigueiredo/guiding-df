#pragma once
#include "json.hpp"
#include <tiny-cuda-nn/gpu_matrix.h>
#include <tiny-cuda-nn/gpu_memory.h>
#include <tiny-cuda-nn/common.h>

#include "kiraray.h"
#include "window.h"
#include "scene.h"
#include "camera.h"
#include "file.h"
#include "renderpass.h"

#include "device/buffer.h"
#include "device/context.h"
#include "device/cuda.h"
#include "backend.h"
#include "train.h"
#include "render/guided/workqueue.h"
#include "util/task.h"

#include "render/common/gl_util.h"
#include "render/common/cuda_util.h"

#if (OPTIMIZE_ON_NRC > 0)
	#include "nrc.h"
#endif

#include "learnedDistribution.h"

namespace tcnn {
	template <typename T> class Loss;
	template <typename T> class Optimizer;
	template <typename T> class Encoding;
	template <typename T> class GPUMemory;
	template <typename T> class GPUMatrixDynamic;
	template <typename T, typename PARAMS_T> class Network;
	template <typename T> class NetworkWithInputEncoding;
	template <typename T, typename PARAMS_T, typename COMPUTE_T> class Trainer;
	template <uint32_t N_DIMS, uint32_t RANK, typename T> class TrainableBuffer;
}

KRR_NAMESPACE_BEGIN

using nlohmann::json;
using precision_t = tcnn::network_precision_t;
using namespace tcnn;
using learned_dist_type = LearnedDistribution;

enum class MouseOverVisualizationMode {
	LuminanceBW = 0,
	LuminanceViridis
};

class Film;

class GuidedPathTracer : public RenderPass {
public:
	using SharedPtr = std::shared_ptr<GuidedPathTracer>;
	KRR_REGISTER_PASS_DEC(GuidedPathTracer);
	enum class RenderMode {Interactive, Offline};
	enum class VisualizationMode {Noisy, Diffuse, Specular, Normal, NumTypes};

	GuidedPathTracer() = default;
	GuidedPathTracer(Scene& scene);
	~GuidedPathTracer() = default;

	void resize(const Vector2i& size) override;
	void setScene(Scene::SharedPtr scene) override;
	void beginFrame(CUDABuffer& frame) override;
	void endFrame(CUDABuffer& frame) override;
	void render(CUDABuffer& frame) override;
	void renderUI() override;
	void finalize() override;

	void initialize();

	string getName() const override { return "GuidedPathTracer"; }

	template <typename F>
	void ParallelFor(int nElements, F&& func, CUstream stream = 0) {
		DCHECK_GT(nElements, 0);
		GPUParallelFor(nElements, func, stream);
	}

	void imageFlipY(float* imageDevice, int width, int height, int channels);
	void saveImage(const string& filename, float* imagePtr, int width, int height, bool flipY = true);
	void handleIntersections();
	void handleEmissiveHit();
	void handleMiss();
	void handleNRCLastBounce();

	void generateCameraRays(int sampleId, bool fixedSample = false);

	KRR_CALLABLE RayQueue* currentRayQueue(int depth) { return rayQueue[depth & 1]; }
	KRR_CALLABLE RayQueue* nextRayQueue(int depth) { return rayQueue[(depth & 1) ^ 1]; }

	template <typename... Args>
	KRR_DEVICE_FUNCTION void debugPrint(uint pixelId, const char *fmt, Args &&...args);

	// guided path routines
	void trainStep();
	void resetTraining();
	void inferenceStep();

	// Sample wi routine
	void sampleWi();

	OptiXGuidedBackend* backend;
	Camera* camera{ };
	LightSampler lightSampler;

#if (RENDER_NOISY_ONLY == 0)
	void mouseOverInference();
	void saveMouseOverSampling(const string& filename, uint numSamples, bool saveRaw);
	void renderGT(int samplesPerPixel, int maxDepth, bool saveRawPDF = false, bool productBSDF = false, bool productCosine = false, Vector2i frameSize = {128,128});

	bool onMouseEvent(const io::MouseEvent& mouseEvent) override {
		if (mouseEvent.type != io::MouseEvent::Type::Move) return false;
		// Only assigns if mMouseOverInfoIn is allocated
		auto attr = cudaPointerAttributes();
		auto return_code = cudaPointerGetAttributes(&attr, mMouseOverInfoIn);
		if (return_code != cudaSuccess || attr.type != cudaMemoryTypeDevice)
			return false;

		const Vector2i screenPos = {(int)mouseEvent.screenPos[0], (int)mouseEvent.screenPos[1]};
		cudaMemcpy(&(mMouseOverInfoIn->mouseCoords), &screenPos, sizeof(Vector2i), cudaMemcpyHostToDevice);
		return false;
	}

	void invertMouseOverLock() {
		bool old_lock;
		cudaMemcpy(&old_lock, &(mMouseOverInfoIn->lock), sizeof(bool), cudaMemcpyDeviceToHost);
		bool new_lock = !old_lock;
		cudaMemcpy(&(mMouseOverInfoIn->lock), &new_lock, sizeof(bool), cudaMemcpyHostToDevice);
	}

	bool onKeyEvent(const io::KeyboardEvent &keyEvent) override {
		if (keyEvent.type == io::KeyboardEvent::Type::KeyPressed && keyEvent.key == io::KeyboardEvent::Key::Space)
			invertMouseOverLock();
		return false;
	}

	MouseOverInfoInput* mMouseOverInfoIn { };
	MouseOverInfoOutput* mMouseOverInfoOut { };

	static GPUMemory<float> mGTMouseOverImage;
	static glu::Texture2D mMouseOverImageTexture;
	static cudau::Array mMouseOverImageArray;
	static cudau::InteropSurfaceObjectHolder<1>  mMouseOverImageTextureBufferSurfaceHolder;
	static GPUMemory<float> mMouseOverImage;
	int mMouseOverVisualizationMode{ (int) MouseOverVisualizationMode::LuminanceViridis };
	bool mMouseOverApplyLog{ false };
	Vector2i mMouseOverImageResolution{ 128, 128 };
#endif

	// work queues
	RayQueue* rayQueue[2]{ };				// switching bewteen current and next queue
	MissRayQueue* missRayQueue{ };
	HitLightRayQueue* hitLightRayQueue{ };
	ShadowRayQueue* shadowRayQueue{ };	
	ScatterRayQueue* scatterRayQueue{ };	// bsdf evaluation (plus shadow ray generation)

#if (RENDER_NOISY_ONLY == 0)
	BsdfEvalQueue* bsdfEvalQueue{ }; 
#endif

	// SamplesCountQueue *validSamplesQueue{};
	// SamplesCountQueue *invalidSamplesQueue{};

	PixelStateBuffer* pixelState;
	GuidedPixelStateBuffer* guidedState;
	
	// global properties
	bool transparentBackground{ };
	bool debugOutput{ };
	uint debugPixel{ };
	int maxQueueSize;
	int frameId{ 0 };
	std::shared_ptr<Profiler> profilerIntegrator{ };

	// path tracing parameters
	int samplesPerPixel{ 1 };
	int maxDepth{ 6 };
	float probRR{ 1 };
	bool enableClamp{ false };
	float clampMax{ 1e4f };

	void resetNetwork(json config);
	bool trainDebug{};

	int randomLaunchTimeSeed;


	// Visualization at the end of Render()
	VisualizationMode visMode{ VisualizationMode::Noisy };

	// offline mode 
	RenderMode renderMode{ RenderMode::Interactive };
	RenderTask task{};
	float trainingBudgetTime{ 0.3 };
	float trainingBudgetSpp{ 0.25 };
	bool discardTraining{ false };		// whether samples in training process contribute to final rendering
	Film *renderedImage{ nullptr };

	learned_dist_type mDist;

#if (OPTIMIZE_ON_NRC > 0)
	NRC mNRC;
	bool showNRCMouseOver = false; 
#endif

	friend void to_json(json &j, const GuidedPathTracer &p) {
		j = json{
			{ "mode", p.renderMode},
			{ "max_depth", p.maxDepth },
			{ "rr", p.probRR },			
			{ "enable_clamp", p.enableClamp },
			{ "clamp_max", p.clampMax },
			{ "bsdf_fraction", p.mDist.mGuiding.bsdfSamplingFraction },
			{ "cosine_aware",  p.mDist.mGuiding.cosineAware},
			{ "mis_aware", p.mDist.mGuiding.misAware },
			{ "product_cosine", p.mDist.mGuiding.productWithCosine },
			{ "max_train_depth", p.mDist.mGuiding.maxTrainDepth },
			{ "max_guided_depth", p.mDist.mGuiding.maxGuidedDepth },
			{ "auto_train", p.mDist.mGuiding.autoTrain },
			{ "train_budget_spp", p.trainingBudgetSpp},
			{ "train_budget_time", p.trainingBudgetTime},
			{ "batch_per_frame", p.mDist.mGuiding.batchPerFrame },
			{ "batch_size", p.mDist.mGuiding.batchSize },
			{ "budget", p.task }
		};
	}

	friend void from_json(const json &j, GuidedPathTracer &p) {
		p.renderMode = j.value("mode", GuidedPathTracer::RenderMode::Interactive);
		p.maxDepth	 = j.value("max_depth", 6);
		p.probRR	 = j.value("rr", 0.8);
		p.enableClamp					= j.value("enable_clamp", false);
		p.clampMax						= j.value("clamp_max", 1e4f);
		p.mDist.mGuiding.bsdfSamplingFraction = j.value("bsdf_fraction", 0.5);
		p.mDist.mGuiding.cosineAware			= j.value("cosine_aware", true);
		p.mDist.mGuiding.misAware				= j.value("mis_aware", false);
		p.mDist.mGuiding.productWithCosine	= j.value("product_cosine", false);
		p.mDist.mGuiding.maxTrainDepth		= j.value("max_train_depth", 3);
		p.mDist.mGuiding.maxGuidedDepth		= j.value("max_guided_depth", 10);
		p.mDist.mGuiding.autoTrain			= j.value("auto_train", false);
		p.discardTraining				= j.value("discard_training", false);
		p.trainingBudgetSpp				= j.value("training_budget_spp", 0.25f);
		p.trainingBudgetTime			= j.value("training_budget_time", 0.3f);
		p.mDist.mGuiding.batchPerFrame		= j.value("batch_per_frame", 5);
		p.mDist.mGuiding.batchSize			= j.value("batch_size", TRAIN_BATCH_SIZE);
		p.task							= j.value("budget", RenderTask{});

		CHECK_LOG(p.mDist.mGuiding.maxTrainDepth <= MAX_TRAIN_DEPTH, 
				"Max train depth %d exceeds limit %d!", p.mDist.mGuiding.maxTrainDepth,
				  MAX_TRAIN_DEPTH);

		if (j.contains("config")) {
			string config_path = j.at("config");
			std::ifstream f(config_path);
			if (f.fail())
				logFatal("Open network config file failed!");
			json config = json::parse(f, nullptr, true, true);
			p.resetNetwork(config["nn"]);
		} else {
			Log(Warning, "Network config do not specified!"
				"Assuming guiding is disabled.");
		}
			
	}
};

KRR_ENUM_DEFINE(GuidedPathTracer::RenderMode, {
	{GuidedPathTracer::RenderMode::Interactive, "interactive"},
	{GuidedPathTracer::RenderMode::Offline, "offline"},
})

KRR_NAMESPACE_END