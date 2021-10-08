#include <NvInfer.h>
#include <NvOnnxParser.h>

using namespace nvinfer1;
using namespace std;
using namespace nvonnxparser;

ICudaEngine* createCudaEngine(string const& onnxModelPath, int batchSize)
{ 
    const auto explicitBatch = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH); 
    unique_ptr<nvinfer1::IBuilder, Destroy<nvinfer1::IBuilder>> builder{nvinfer1::createInferBuilder(gLogger)};
    unique_ptr<nvinfer1::INetworkDefinition, Destroy<nvinfer1::INetworkDefinition>> network{builder->createNetworkV2(explicitBatch)};
    unique_ptr<nvonnxparser::IParser, Destroy<nvonnxparser::IParser>> parser{nvonnxparser::createParser(*network, gLogger)};
    unique_ptr<nvinfer1::IBuilderConfig,Destroy<nvinfer1::IBuilderConfig>> config{builder->createBuilderConfig()};

    if (!parser->parseFromFile(onnxModelPath.c_str(), static_cast<int>(ILogger::Severity::kINFO)))
    {
        cout << "ERROR: could not parse input engine." << endl;
        return nullptr;
    }

    config->setMaxWorkspaceSize(MAX_WORKSPACE_SIZE);
    builder->setFp16Mode(builder->platformHasFastFp16());
    builder->setMaxBatchSize(batchSize);
    
    auto profile = builder->createOptimizationProfile();
    profile->setDimensions(network->getInput(0)->getName(), OptProfileSelector::kMIN, Dims4{1, 3, 256 , 256});
    profile->setDimensions(network->getInput(0)->getName(), OptProfileSelector::kOPT, Dims4{1, 3, 256 , 256});
    profile->setDimensions(network->getInput(0)->getName(), OptProfileSelector::kMAX, Dims4{32, 3, 256 , 256});    
    config->addOptimizationProfile(profile);

    return builder->buildEngineWithConfig(*network, *config);   
}

bool saveEngine(const ICudaEngine& engine, const std::string& fileName, std::ostream& err)
{
    std::ofstream engineFile(fileName, std::ios::binary);
    if (!engineFile)
    {
        err << "Cannot open engine file: " << fileName << std::endl;
        return false;
    }

    TrtUniquePtr<IHostMemory> serializedEngine{engine.serialize()};
    if (serializedEngine == nullptr)
    {
        err << "Engine serialization failed" << std::endl;
        return false;
    }

    engineFile.write(static_cast<char*>(serializedEngine->data()), serializedEngine->size());
    return !engineFile.fail();
}

std::vector< std::string > getClassNames(const std::string& imagenet_classes)
{
    std::ifstream classes_file(imagenet_classes);
    std::vector< std::string > classes;
    if (!classes_file.good())
    {
        std::cerr << "ERROR: can't read file with classes names.n";
        return classes;
    }
    std::string class_name;
    while (std::getline(classes_file, class_name))
    {
        classes.push_back(class_name);
    }
    return classes;
}

int main(int argc, char** argv){
    // ./onnx2trt_converter onnxpath filename
    string onnxPath = argv[1]
    string trtFilename = argv[2]
    ICudaEngine* cuda_engine{nullptr};
    int batchSize = 1;

    cuda_engine = createCudaEngine(onnxPath, batchSize)
    saveEngine(cuda_engine, trtFilename)
}
