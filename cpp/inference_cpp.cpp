#include "argsParser.h"
#include "buffers.h"
#include "common.h"
#include "logger.h"
#include "parserOnnxConfig.h"

#include "NvInfer.h"
#include <cuda_runtime_api.h>

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>

using samplesCommon::SampleUniquePtr;

class FaceMaskTRT
{
public:
    FaceMaskTRT(const samplesCommon::OnnxSampleParams& params)
        : mParams(params)
        , mEngine(nullptr)
    {
    }

    bool build();
    bool infer();

private:
    samplesCommon::OnnxSampleParams mParams; 

    nvinfer1::Dims mInputDims;  
    nvinfer1::Dims mOutputDims;
    int mNumber{0};        
    std::shared_ptr<nvinfer1::ICudaEngine> mEngine; 

    bool constructNetwork(SampleUniquePtr<nvinfer1::IBuilder>& builder,
        SampleUniquePtr<nvinfer1::INetworkDefinition>& network, SampleUniquePtr<nvinfer1::IBuilderConfig>& config,
        SampleUniquePtr<nvonnxparser::IParser>& parser);

    bool processInput(const samplesCommon::BufferManager& buffers);
    bool verifyOutput(const samplesCommon::BufferManager& buffers);
};

// Returns empty string iff can't read the file
string readBuffer(string const& path)
{
    string buffer;
    ifstream stream(path.c_str(), ios::binary);

    if (stream)
    {
        stream >> noskipws;
        copy(istream_iterator<char>(stream), istream_iterator<char>(), back_inserter(buffer));
    }

    return buffer;
}

ICudaEngine* getCudaEngine(string const& TrtModelPath)
{
    string enginePath{TrtModelPath};
    ICudaEngine* engine{nullptr};

    string buffer = readBuffer(enginePath);
    if (buffer.size())
    {
        // try to deserialize engine
        unique_ptr<IRuntime, Destroy> runtime{createInferRuntime(gLogger)};
        engine = runtime->deserializeCudaEngine(buffer.data(), buffer.size(), nullptr);
    }

    if (!engine)
    {
        // Fallback to creating engine from scratch
        engine = createCudaEngine(onnxModelPath, batchSize);

        if (engine)
        {
            unique_ptr<IHostMemory, Destroy> engine_plan{engine->serialize()};
            // try to save engine for future uses
            writeBuffer(engine_plan->data(), engine_plan->size(), enginePath);
        }
    }
    return engine;
}

bool FaceMaskTRT::infer()
{
    // Create RAII buffer manager object
    samplesCommon::BufferManager buffers(mEngine);

    auto context = SampleUniquePtr<nvinfer1::IExecutionContext>(mEngine->createExecutionContext());
    if (!context)
    {
        return false;
    }

    ASSERT(mParams.inputTensorNames.size() == 1);
    if (!processInput(buffers))
    {
        return false;
    }

    // Memcpy from host input buffers to device input buffers
    buffers.copyInputToDevice();

    bool status = context->executeV2(buffers.getDeviceBindings().data());
    if (!status)
    {
        return false;
    }

    // Memcpy from device output buffers to host output buffers
    buffers.copyOutputToHost();

    // Verify results
    if (!verifyOutput(buffers))
    {
        return false;
    }

    return true;
}

samplesCommon::OnnxSampleParams initializeSampleParams(const samplesCommon::Args& args)
{
    samplesCommon::OnnxSampleParams params;
    if (args.dataDirs.empty()) 
    {
        params.dataDirs.push_back("data/mnist/");
        params.dataDirs.push_back("data/samples/mnist/");
    }
    else 
    {
        params.dataDirs = args.dataDirs;
    }
    params.onnxFileName = "mnist.onnx";
    params.inputTensorNames.push_back("Input3");
    params.outputTensorNames.push_back("Plus214_Output_0");
    params.dlaCore = args.useDLACore;
    params.int8 = args.runInInt8;
    params.fp16 = args.runInFp16;

    return params;
}

void printHelpInfo()
{
    std::cout
        << "Usage: ./sample_onnx_mnist [-h or --help] [-d or --datadir=<path to data directory>] [--useDLACore=<int>]"
        << std::endl;
    std::cout << "--help          Display help information" << std::endl;
    std::cout << "--datadir       Specify path to a data directory, overriding the default. This option can be used "
                 "multiple times to add multiple directories. If no data directories are given, the default is to use "
                 "(data/samples/mnist/, data/mnist/)"
              << std::endl;
    std::cout << "--useDLACore=N  Specify a DLA engine for layers that support DLA. Value can range from 0 to n-1, "
                 "where n is the number of DLA engines on the platform."
              << std::endl;
    std::cout << "--int8          Run in Int8 mode." << std::endl;
    std::cout << "--fp16          Run in FP16 mode." << std::endl;
}

int main(int argc, char** argv)
{
    cv::VideoCapture capture();
    cv::Mat frame;
    while (1) {
        if (!capture.isOpened()) {
            break; //do some logging here or something else - webcam not available
        }

        //Create image frames from capture
        capture >> frame;

        if (!frame.empty()) {
            //do something with your image (e.g. provide it)
            lastImage = frame.clone();
        }
    }

    samplesCommon::Args args;
    bool argsOK = samplesCommon::parseArgs(args, argc, argv);
    if (!argsOK)
    {
        sample::gLogError << "Invalid arguments" << std::endl;
        printHelpInfo();
        return EXIT_FAILURE;
    }
    if (args.help)
    {
        printHelpInfo();
        return EXIT_SUCCESS;
    }

    auto sampleTest = sample::gLogger.defineTest(gSampleName, argc, argv);

    sample::gLogger.reportTestStart(sampleTest);

    FaceMaskTRT sample(initializeSampleParams(args));

    sample::gLogInfo << "Building and running a GPU inference engine for Onnx" << std::endl;

    if (!sample.build())
    {
        return sample::gLogger.reportFail(sampleTest);
    }
    if (!sample.infer())
    {
        return sample::gLogger.reportFail(sampleTest);
    }

    return sample::gLogger.reportPass(sampleTest);
}

}