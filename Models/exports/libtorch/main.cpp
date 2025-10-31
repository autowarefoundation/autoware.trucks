/*
**
FILE:   main.cpp
DESC:   libTorch implementation of exported SceneSeg Network
**
*/

#include <assert.h>
#include <iostream>

#include <torch/script.h>
#include <torch/torch.h>
#include <torch/cuda.h>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp>

using namespace cv; 
using namespace std; 

/*
**
FUNC:   main()
DESC:   example program entry point
**
*/
int main(int argc, char **argv) 
{
    // Get exported script module
    if (argc != 3) {
        std::cerr << "usage: deploy_libtorch <path-to-exported-script-module>\n";
        return -1;
    }

    // Create a module object to hold the network
    torch::jit::script::Module module_;
    
    // Try to load the script module
    try {
        // Deserialize the ScriptModule from a file using torch::jit::load().
        module_ = torch::jit::load(argv[1], torch::kCUDA);
    }
    catch (const c10::Error& e) {
        std::cerr << "ERROR: Failed to load the model." << std::endl;
        return -1;
    }

    //
    std::cout << "INFO: Model <" << argv[1] << "> loaded successfully." << std::endl;
    //

    // Load an input image
    cv::Mat frame = cv::imread(argv[2], cv::IMREAD_COLOR); 
    cv::Mat preImg;
    cvtColor(frame, preImg, cv::COLOR_BGR2RGB);

    // Resize
    cv::Mat img;
    resize(preImg, img, Size(640, 320));    // Fixed ratio from original implementation

    // Convert input cv image to Tensor
    at::Tensor tensor_image = torch::from_blob(img.data, { img.rows, img.cols, 3 }, at::kByte);

    // Convert to float and scale it 
    tensor_image = tensor_image.toType(c10::kFloat).div(255);

    // Transpose the image
    tensor_image = tensor_image.permute({ (2),(0),(1) });

    // Create a batch dimension for input
    tensor_image.unsqueeze_(0);

    // Normalise the input values
    std::vector<double> norm_mean = {0.485, 0.456, 0.406};
    std::vector<double> norm_std = {0.229, 0.224, 0.225};
    tensor_image = torch::data::transforms::Normalize<>(norm_mean, norm_std)(tensor_image);

    // Create to "input values"
    auto input_to_net = std::vector<torch::jit::IValue>{tensor_image.to(at::kCUDA)};

    // Push the image through the network forward compute path
    at::Tensor prediction = module_.forward(input_to_net).toTensor();

    //
    std::cout << "INFO: Forward compute path executed succssfully." << std::endl;
    //

    // Post-process the output tensor
    prediction = prediction.squeeze(0).to(at::kCPU).detach();

    // Transpose back
    c10::IntArrayRef dims = {1, 2, 0};
    prediction = prediction.permute(dims);

    // Take max values across 2 dimensions
    auto final_output = std::get<1>(max(prediction,2,false));

    // Create an image to hold the segmentation mask
    cv::Mat vis_predict_object(img.rows, img.cols, CV_8UC3, Scalar(0, 0, 0));

    // Process the tensor output and crate output segmentation mask
    for (int x=0;x<img.rows;x++) // rows
    {
        for (int y=0;y<img.cols;y++) // cols
        {
            int value = final_output[x][y].item<int>();
            
            if (value==0) 
            {
                // Background colour
                vis_predict_object.at<cv::Vec3b>(x,y)=cv::Vec3b(255,93,61);
            }
            if (value==1)
            {
                // Foreground colour
                vis_predict_object.at<cv::Vec3b>(x,y)=cv::Vec3b(145,28,255);
            }
            if (value==2)
            {
                // Background colour
                vis_predict_object.at<cv::Vec3b>(x,y)=cv::Vec3b(255,93,61);
            }
        }
    }

    //
    std::cout << "INFO: Segmentation mask created succssfully." << std::endl;
    //

    // Write out the segmentation mask to file
    cv::imwrite("output_seg_mask.jpg", vis_predict_object);

    // Transparency factor
    auto alpha = 0.5;

    // Create an image to hold the segmentation mask
    cv::Mat final_output_image(frame.rows, frame.cols, CV_8UC3, Scalar(0, 0, 0));

    // Resize
    cv::Mat out_img;
    resize(vis_predict_object, out_img, Size(frame.cols, frame.rows));    // Fixed ratio from original implementation

    // Alpha Blend of Segmentation mask onto original input image
    addWeighted( out_img, alpha, frame, 1-alpha, 0.0, final_output_image);

    //
    std::cout << "INFO: Output imag created succssfully." << std::endl;
    //

    // Write out the segmentation mask to file
    cv::imwrite("output_image.jpg", final_output_image);

    // Done
    return 0;
}
/*
**
End of File
**
*/