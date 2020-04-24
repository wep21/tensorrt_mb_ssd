#include <iostream>
#include <fstream>
#include <memory>
#include <cassert>
#include <chrono>

// tensorrt
#include <NvInfer.h>
#include <NvOnnxParser.h>

// opencv
#include <opencv/cv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>

#include <cuda_utils.h>

struct deleter {
    template<typename T>
    void operator()(T* obj) const {
  	if( obj ) {
  	    obj->destroy();
  	}
    }
};

class Logger : public nvinfer1::ILogger {
    public:
        Logger(bool verbose)
        : _verbose(verbose) {
        }

        void log(Severity severity, const char *msg) override {
            if (_verbose || (severity != Severity::kINFO) && (severity != Severity::kVERBOSE))
                std::cout << msg << std::endl;
        }
    private:
        bool _verbose{false};
};

int main(int argc, char* argv[]){
	assert(argc == 4);
    Logger logger(false);
	const int num_classes = 21;
	const int num_detection = 3000;
	const float score_threshold = 0.3;
	const float iou_threshold = 0.45;
	std::vector<std::string> label{"BG", "aeroplane", "bicycle",  "bird",
	                               "boat", "bottle", "bus", "car", "cat",
								   "chair", "cow", "diningtable", "dog",
								   "horse", "motorbike", "person", "pottedplant",
								   "sheep", "sofa", "train", "tvmonitor"};
	
	std::chrono::high_resolution_clock::time_point  start, end;
	const int input_w = 300;
	const int input_h = 300;
	
	auto builder = std::unique_ptr<nvinfer1::IBuilder, deleter>(nvinfer1::createInferBuilder(logger));
	if (!builder)
    {
        return false;
    }
	const auto explicitBatch = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);     
    auto network = std::unique_ptr<nvinfer1::INetworkDefinition, deleter>(builder->createNetworkV2(explicitBatch));
    if (!network)
    {
        return false;
    }
	auto config = std::unique_ptr<nvinfer1::IBuilderConfig, deleter>(builder->createBuilderConfig());
    if (!config)
    {
        return false;
    }
	auto parser = std::unique_ptr<nvonnxparser::IParser, deleter>(nvonnxparser::createParser(*network, logger));
    if (!parser)
    {
        return false;
    }
	if (!parser->parseFromFile(argv[1], static_cast<int>(nvinfer1::ILogger::Severity::kERROR)))
	{
    	return false;
    }
	int max_batch_size = 1;
	builder->setMaxBatchSize(max_batch_size);
    config->setMaxWorkspaceSize(16 << 20);
	// config->setFlag(nvinfer1::BuilderFlag::kFP16);
	std::unique_ptr<nvinfer1::ICudaEngine, deleter> engine;
	std::ifstream file(argv[2], std::ios::in | std::ios::binary);
    if (!file.is_open())
	{
		std::cout << "Create engine..." << std::endl;
		engine = std::unique_ptr<nvinfer1::ICudaEngine, deleter>(builder->buildEngineWithConfig(*network, *config));
		if (!engine)
		{
			std::cout << "Fail to create engine" << std::endl;
			return false;
		}
		std::cout << "Successfully create engine" << std::endl;
		auto selialized = std::unique_ptr<nvinfer1::IHostMemory, deleter>(engine->serialize());
		std::cout << "Save engine: " << argv[2] << std::endl;
		std::ofstream file;
		file.open(argv[2], std::ios::binary | std::ios::out);
		if (!file.is_open()) return false;
		file.write((const char *)selialized->data(), selialized->size());
		file.close();
	}
	else
	{
		file.seekg (0, file.end);
		size_t size = file.tellg();
		file.seekg (0, file.beg);

		char *buffer = new char[size];
		file.read(buffer, size);
		file.close();
        auto runtime = std::unique_ptr<nvinfer1::IRuntime, deleter>(nvinfer1::createInferRuntime(logger));
		engine = std::unique_ptr<nvinfer1::ICudaEngine, deleter>(runtime->deserializeCudaEngine(buffer, size, nullptr));

		delete[] buffer;
	}

	start = std::chrono::high_resolution_clock::now();
	std::cout << "Preparing data..." << std::endl;
	auto orig_image = cv::imread(argv[3], cv::IMREAD_COLOR);
	auto image = orig_image.clone();
	cv::resize(image, image, cv::Size(input_w, input_h));
	cv::Mat pixels;
	image.convertTo(pixels, CV_32FC3, 1.0 / 255, 0);
	int channels = 3;
	std::vector<float> img;
	std::vector<float> data (channels * input_w * input_h);
	if (pixels.isContinuous())
		img.assign((float*)pixels.datastart, (float*)pixels.dataend);
	else {
		std::cerr << "Error reading image " << argv[3] << std::endl;
		return false;
	}

	std::vector<float> mean {0.485, 0.456, 0.406};
	std::vector<float> std {0.229, 0.224, 0.225};

	for (int c = 0; c < channels; c++) {
		for (int j = 0, hw = input_w * input_h; j < hw; j++) {
			data[c * hw + j] = (img[channels * j + 2 - c] - mean[c]) / std[c];
		}
	}

	// inference
	auto context = std::unique_ptr<nvinfer1::IExecutionContext, deleter>(engine->createExecutionContext());
	cudaStream_t stream = nullptr;
	cudaStreamCreate(&stream);
	auto data_d = cuda::make_unique<float[]>(channels * input_h * input_w);
	cudaMemcpy(data_d.get(), data.data(), data.size() * sizeof(float), cudaMemcpyHostToDevice);
	auto boxes_d = cuda::make_unique<float[]>(num_detection * 4);
	auto scores_d = cuda::make_unique<float[]>(num_detection * num_classes);
	std::vector<void *> buffers = { data_d.get(), scores_d.get(), boxes_d.get() };
	context->enqueueV2(buffers.data(), stream, nullptr);
	auto boxes = std::make_unique<float[]>(num_detection * 4);
	auto scores = std::make_unique<float[]>(num_detection * num_classes);
	cudaMemcpyAsync(boxes.get(), boxes_d.get(), sizeof(float) * num_detection * 4, cudaMemcpyDeviceToHost, stream);
	cudaMemcpyAsync(scores.get(), scores_d.get(), sizeof(float) * num_detection * num_classes, cudaMemcpyDeviceToHost, stream);
	cudaStreamSynchronize(stream);
	auto out_image = orig_image.clone();
	for (int i = 0; i < num_classes; i++)
	{
		std::vector<float> probs;
		std::vector<cv::Rect2d> subset_boxes;
		std::vector<int> indices;
		for (int j = 0; j < num_detection; j++)
		{
			probs.push_back(scores[i + j * num_classes]);
			subset_boxes.push_back(cv::Rect2d(cv::Point2d(boxes[j * 4], boxes[j * 4 + 1]), 
				                              cv::Point2d(boxes[j * 4 + 2], boxes[j * 4 + 3])));
			if (probs.size() == 0)
			{
				continue;
			}
		}
		cv::dnn::NMSBoxes(subset_boxes, probs, score_threshold, iou_threshold, indices);
		for (const auto& index: indices)
		{
			if (i != 0)
			{
				cv::Point2f tl(subset_boxes[index].tl().x * orig_image.cols, subset_boxes[index].tl().y * orig_image.rows);
				cv::Point2f br(subset_boxes[index].br().x * orig_image.cols, subset_boxes[index].br().y * orig_image.rows);
				cv::rectangle(out_image, tl, br, cv::Scalar(255, 255 / num_classes * i, 0), 3);
				cv::putText(out_image, label[i], cvPoint(tl.x, tl.y -10), CV_FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(255, 255 / num_classes * i, 0), 3);
			}
		}
	}
	end = std::chrono::high_resolution_clock::now();
	double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count();
	std::cout << "exec time: " << elapsed << std::endl;
	cv::imwrite("output.jpg", out_image);
	return 0;
}