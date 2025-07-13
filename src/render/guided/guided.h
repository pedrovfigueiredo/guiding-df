#pragma once

#include "sampler.h"
#include "scene.h"
#include "render/lightsampler.h"
#include "render/bsdf.h"
#include "device/optix.h"
#include "render/wavefront/workqueue.h"
#include "render/guided/workqueue.h"

KRR_NAMESPACE_BEGIN

class PixelStateBuffer;
class GuidedPixelStateBuffer;

using namespace shader;

class TrainState {
public:
	KRR_CALLABLE bool isEnableGuiding() const { return enableGuiding; }

	KRR_CALLABLE bool isEnableTraining() const { return enableTraining; }

	KRR_CALLABLE bool isOneStep() const { return oneStep; }

	KRR_CALLABLE bool isTrainingPixel(uint pixelId) const {
		return (enableTraining || oneStep) && (pixelId - trainPixelOffset) % trainPixelStride == 0;
	}

	KRR_HOST void renderUI() {
		ImGui::Checkbox("Enable Guiding", &enableGuiding);
		ImGui::Checkbox("Enable Training", &enableTraining);
		ImGui::InputInt("Train Pixel Stride", (int*)&trainPixelStride);
		if (ui::Button("Train one step")) oneStep = true;
		ImGui::Checkbox("Accumulate Training Data", &accumulateTrainingData);
	}

	bool enableTraining{ false };
	bool enableGuiding{ false };
	uint trainPixelOffset{ 0 };
	uint trainPixelStride{ 1 };
	bool oneStep{ false };
	bool accumulateTrainingData{ true };
};

typedef struct {
	// Input from host
	Vector2i mouseCoords;
	bool lock;
} MouseOverInfoInput;

typedef struct {
	// Output from optix
	Vector3f position;
	Vector2f normal;
	Vector3f diffuse;
	Vector3f specular;
	Vector2f uv;
	float roughness;
	Vector3f wo;
	bool isEmissive;
	Vector3f raw_position;
	Vector3f raw_normal;
	ShadingData sd;
} MouseOverInfoOutput;

typedef struct {
	RayQueue* currentRayQueue;
	RayQueue* nextRayQueue;
	ShadowRayQueue* shadowRayQueue;
	MissRayQueue* missRayQueue;
	HitLightRayQueue* hitLightRayQueue;
	ScatterRayQueue* scatterRayQueue;
	
	PixelStateBuffer* pixelState;
	GuidedPixelStateBuffer* guidedState;
	TrainState trainState;
	Scene::SceneData sceneData;
	OptixTraversableHandle traversable;
} LaunchParamsGuided;

KRR_NAMESPACE_END