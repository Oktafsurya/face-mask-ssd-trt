#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <iostream>
#include <fstream>  
#include <string>
#include <memory>

using namespace nvinfer1;
using namespace std;
using namespace nvonnxparser;

// class to log errors, warnings, and other information during the build and inference phases
class Logger : public nvinfer1::ILogger
{
public:
    void log(Severity severity, const char* msg) override {
        // remove this 'if' if you need more logged info
        if ((severity == Severity::kERROR) || (severity == Severity::kINTERNAL_ERROR)) {
            std::cout << msg << "\n";
        }
    }
} gLogger;

struct TRTDestroy
{
    template <class T>
    void operator()(T* obj) const
    {
        if (obj)
        {
            obj->destroy();
        }
    }
};

template <class T>
using TRTUniquePtr = std::unique_ptr<T, TRTDestroy>;

string getBasename(string const& path)
{
#ifdef _WIN32
    constexpr char SEPARATOR = '\\';
#else
    constexpr char SEPARATOR = '/';
#endif
    int baseId = path.rfind(SEPARATOR) + 1;
    return path.substr(baseId, path.rfind('.') - baseId);
}

void writeBuffer(void* buffer, size_t size, string const& path)
{
    ofstream stream(path.c_str(), ios::binary);

    if (stream)
        stream.write(static_cast<char*>(buffer), size);
        if (!stream.fail()){
            cout << "Engine successfully saved to:" << path << endl;
        }
}

void createCudaEngine(string const& onnxModelPath, int batchSize)
{ 
    ICudaEngine* engine{nullptr};
    const auto explicitBatch = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH); 
    TRTUniquePtr<nvinfer1::IBuilder> builder{nvinfer1::createInferBuilder(gLogger)};
    TRTUniquePtr<nvinfer1::INetworkDefinition> network{builder->createNetworkV2(explicitBatch)};
    TRTUniquePtr<nvonnxparser::IParser> parser{nvonnxparser::createParser(*network, gLogger)};
    TRTUniquePtr<nvinfer1::IBuilderConfig> config{builder->createBuilderConfig()};

    if (parser->parseFromFile(onnxModelPath.c_str(), static_cast<int>(ILogger::Severity::kINFO)))
    {
        string enginePath{getBasename(onnxModelPath) + "_batch" + to_string(batchSize) + ".engine"};
        constexpr size_t MAX_WORKSPACE_SIZE = 1ULL << 30; // 1 GB
        config->setMaxWorkspaceSize(MAX_WORKSPACE_SIZE);
        if (builder->platformHasFastFp16())
        {
            config->setFlag(nvinfer1::BuilderFlag::kFP16);
        }
        builder->setMaxBatchSize(batchSize);
    
        auto profile = builder->createOptimizationProfile();
        profile->setDimensions(network->getInput(0)->getName(), OptProfileSelector::kMIN, Dims4{1, 3, 300 , 300});
        profile->setDimensions(network->getInput(0)->getName(), OptProfileSelector::kOPT, Dims4{1, 3, 300 , 300});
        profile->setDimensions(network->getInput(0)->getName(), OptProfileSelector::kMAX, Dims4{1, 3, 300 , 300});    
        config->addOptimizationProfile(profile);

        engine = builder->buildEngineWithConfig(*network, *config);

        if (engine){
            TRTUniquePtr<IHostMemory> engine_plan{engine->serialize()};
            // Try to save engine for future uses.
            writeBuffer(engine_plan->data(), engine_plan->size(), enginePath);
        }    
    }
    else{
        cout << "ERROR: could not parse input engine." << endl;
    }
}

int main(int argc, char** argv){
    // command to run: ./onnx2trt_converter onnxpath
    string onnxPath = argv[1];
    int batchSize = 1;
    ICudaEngine* cuda_engine{nullptr};
    createCudaEngine(onnxPath, batchSize);
}
