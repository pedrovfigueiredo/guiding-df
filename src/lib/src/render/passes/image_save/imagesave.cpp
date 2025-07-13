#include "imagesave.h"
#include "render/profiler/profiler.h"

KRR_NAMESPACE_BEGIN

void ImageSavePass::beginFrame(CUDABuffer &frame) {
	mFrameNumber++;

	switch(mIntervalType){
		case IntervalType::Frame:
			mNeedsSave |= mContinuousSave && (mFrameNumber % mSaveInterval == 0);
			break;
		case IntervalType::Time:
			mNeedsSave |= mContinuousSave && (CpuTimer::calcElapsedTime(mLastSaveTime) * 1e-3 > mSaveInterval);
			break;
		default:
			break;
	};
}

void ImageSavePass::render(CUDABuffer &frame) {
	if (mNeedsSave) {
		PROFILE("Image saving");
        Image image({mFrameSize[0], mFrameSize[1]}, Image::Format::RGBAfloat);
        cudaMemcpy(image.data(), (float*) frame.data(), mFrameSize[0] * mFrameSize[1] * 4 * sizeof(float), cudaMemcpyDeviceToHost);
        const double elapsed_time = CpuTimer::calcElapsedTime(mStartTime) * 1e-3;

        if (mSaveAtFinalize) {
            if (mLogResults)
			    Log(Info, "Saving frame #%zd to memory.", mFrameNumber);
            mFrameInfos.push_back({mFrameNumber, elapsed_time, image});
        } else {
            fs::path save_path;
            getOutputFilename(save_path, mFrameNumber, elapsed_time);
            if (mLogResults)
			    Log(Info, "Saving frame #%zd (%f) to path: %s", mFrameNumber, elapsed_time, save_path.string().c_str());
            saveImage(save_path, image);
        }
		mLastSaveTime = CpuTimer::getCurrentTimePoint();
		mNeedsSave = false;
	}
}

void ImageSavePass::getOutputFilename(fs::path& filepath, size_t frame_number, double elapsed_time){
    std::stringstream ss;
    ss.precision(3);
    ss << frame_number << "_" << elapsed_time << (mFormat == ImageFormat::HDR ? ".exr" : ".png");
    filepath = File::outputDir() / "images" / ss.str();
}

void ImageSavePass::saveImage(const fs::path& filepath, Image& image){
	if (!fs::exists(filepath.parent_path()))
		fs::create_directories(filepath.parent_path());
	image.saveImage(filepath);
}

void ImageSavePass::endFrame(CUDABuffer &frame) {}

void ImageSavePass::resize(const Vector2i &size) { RenderPass::resize(size); }

void ImageSavePass::finalize() {
	if (mSaveAtFinalize) {
        Log(Info, "Saving frames at finalize...");
		for (FrameInfo &f : mFrameInfos) {
            fs::path save_path;
			getOutputFilename(save_path, f.timestep, f.timepoint);
            saveImage(save_path, f.frame);
		}
        Log(Info, "Done.");
	}
}

void ImageSavePass::renderUI() { 
	ui::Checkbox("Enabled", &mEnable);
	if (mEnable) {
		
		ui::Checkbox("Continuous save", &mContinuousSave);
        if (mContinuousSave)
            ui::InputScalar("Save every", ImGuiDataType_::ImGuiDataType_U64,
                            &mSaveInterval);
			if (ui::Button("Save")) mNeedsSave = 1;

		ui::Checkbox("Log results", &mLogResults);
		ui::Checkbox("Save at Finalize", &mSaveAtFinalize);

        if (ui::Button("Reset")) reset();
	}
}

void ImageSavePass::reset() {
	mNeedsSave = false;
	mStartTime = CpuTimer::getCurrentTimePoint();
}


KRR_REGISTER_PASS_DEF(ImageSavePass);
KRR_NAMESPACE_END