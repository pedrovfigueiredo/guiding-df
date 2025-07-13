#pragma once

#include "texture.h"
#include "host/timer.h"
#include "renderpass.h"
#include "window.h"
#include "logger.h"

KRR_NAMESPACE_BEGIN

enum class ImageFormat { HDR, SDR, Count };
enum class IntervalType { Frame, Time, Count };

KRR_ENUM_DEFINE(ImageFormat, {
	{ImageFormat::HDR, "hdr"},
	{ImageFormat::SDR, "sdr"}
});

KRR_ENUM_DEFINE(IntervalType, {
    {IntervalType::Frame, "frame"},
    {IntervalType::Time, "time"}
});

class ImageSavePass : public RenderPass {
public:
    using RenderPass::RenderPass;
    using SharedPtr = std::shared_ptr<ImageSavePass>;
    KRR_REGISTER_PASS_DEC(ImageSavePass);

    ImageSavePass() {mStartTime	   = CpuTimer::getCurrentTimePoint(); mLastSaveTime = mStartTime;};
    ~ImageSavePass() = default;
    void beginFrame(CUDABuffer& frame) override;
    void render(CUDABuffer& frame) override;
    void endFrame(CUDABuffer& frame) override;
    void renderUI() override;
    void resize(const Vector2i& size) override;
    void finalize() override;

    string getName() const override { return "ImageSavePass"; }
protected:
    typedef struct {
		size_t timestep;
		double timepoint;
		Image frame;
	} FrameInfo;

    void reset();
    void getOutputFilename(fs::path& filepath, size_t frame_number, double elapsed_time);
    void saveImage(const fs::path& filepath, Image& image);

    bool mNeedsSave{}, mContinuousSave{}, mSaveAtFinalize{}, mLogResults{};
    size_t mFrameNumber{0}, mSaveInterval{ 1 };
    std::vector<FrameInfo> mFrameInfos;
    ImageFormat mFormat{ ImageFormat::HDR };
    IntervalType mIntervalType{ IntervalType::Frame };
    CpuTimer::TimePoint mLastSaveTime;
    CpuTimer::TimePoint mStartTime;

    friend void to_json(json &j, const ImageSavePass &p) { 
		j = json{ 
			{ "continuous", p.mContinuousSave },
			{ "interval", p.mSaveInterval },
            { "interval_type", p.mIntervalType},
            { "log", p.mLogResults },
            { "save_at_finalize", p.mSaveAtFinalize },
            { "format", p.mFormat}
		};
	}

	friend void from_json(const json &j, ImageSavePass &p) {
		p.mContinuousSave = j.value("continuous", false);
		p.mSaveInterval	  = j.value("interval", 1);
        p.mLogResults = j.value("log", false);
        p.mSaveAtFinalize = j.value("save_at_finalize", true);
        p.mFormat = j.value("format", ImageFormat::HDR);
        p.mIntervalType = j.value("interval_type", IntervalType::Frame);
	}
};

KRR_NAMESPACE_END