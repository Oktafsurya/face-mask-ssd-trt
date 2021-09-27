//----------------------------------------------------------------------------
//----------------------------------------------------------------------------
std::vector<std::string> load_labels(const std::string& fileName)
{
    std::ifstream ins(fileName);
    if (!ins.is_open())
    {
        std::cerr << "Couldn't open " << fileName << std::endl;
        abort();
    }

    std::vector<std::string> labels;
    std::string line;

    while (getline(ins, line))
        labels.push_back(line);

    ins.close();

    return labels;
}

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------
torch::Tensor read_image(const std::string& imageName)
{
    cv::Mat img = cv::imread(imageName);
    img = crop_center(img);
    cv::resize(img, img, cv::Size(224,224));

    cv::imshow("image", img);

    if (img.channels()==1)
        cv::cvtColor(img, img, cv::COLOR_GRAY2RGB);
    else
        cv::cvtColor(img, img, cv::COLOR_BGR2RGB);

    img.convertTo( img, CV_32FC3, 1/255.0 );

    torch::Tensor img_tensor = torch::from_blob(img.data, {img.rows, img.cols, 3}, c10::kFloat);
    img_tensor = img_tensor.permute({2, 0, 1});
    img_tensor.unsqueeze_(0);

    img_tensor = torch::data::transforms::Normalize<>(norm_mean, norm_std)(img_tensor);

    return img_tensor.clone();
}


//----------------------------------------------------------------------------
//----------------------------------------------------------------------------
cv::Mat crop_center(const cv::Mat &img)
{
    const int rows = img.rows;
    const int cols = img.cols;

    const int cropSize = std::min(rows,cols);
    const int offsetW = (cols - cropSize) / 2;
    const int offsetH = (rows - cropSize) / 2;
    const cv::Rect roi(offsetW, offsetH, cropSize, cropSize);

    return img(roi);
}