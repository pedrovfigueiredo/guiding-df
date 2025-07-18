#pragma once

#include "sampler.h"
#include "scene.h"
#include "render/lightsampler.h"
#include "render/bsdf.h"

#include "pathstate.h"

KRR_NAMESPACE_BEGIN

using namespace shader;

struct LaunchParamsBDPT {
	uint frameID{ 0 };
	Vector2i fbSize = Vector2i::Zero();
	bool debugOutput	= false;
	Vector2i debugPixel = { 114, 514 };

	int maxDepth		 = 10;
	float probRR		 = 0.2;
	float clampThreshold = 1e4f; 

	Camera camera;
	LightSampler lightSampler;
	Scene::SceneData sceneData;

	Color4f* colorBuffer{ nullptr };
	BDPTPathStateBuffer *pathState{ nullptr };
	OptixTraversableHandle traversable{ 0 };
};

template <typename Type> 
class ScopedAssignment {
public:
	// ScopedAssignment (RAII styled class, taken from pbrt) 
	ScopedAssignment(Type *target = nullptr, Type value = Type{}) : target(target) {
		if (target) {
			backup	= *target;
			*target = value;
		}
	}
	~ScopedAssignment() {
		if (target) *target = backup;
	}
	ScopedAssignment(const ScopedAssignment &)			  = delete;
	ScopedAssignment &operator=(const ScopedAssignment &) = delete;

	ScopedAssignment &operator=(ScopedAssignment &&other) {
		target		 = other.target;
		backup		 = other.backup;
		other.target = nullptr;
		return *this;
	}

private:
	Type *target, backup;
};

KRR_NAMESPACE_END