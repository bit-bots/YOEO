#include "openvino/openvino.hpp"
#include <opencv2/opencv.hpp>
#include <iterator>
#include <memory>
#include <sstream>
#include <string>
#include <vector>
// #include <opencv2/opencv.hpp>
#include <ngraph/type/element_type.hpp>
// #include "format_reader_ptr.h"
std::shared_ptr getData(Mat& img) {
    int width = img.cols;
    int height = img.rows;

    std::shared_ptr<unsigned char> _data;
    size_t size = width * height * img.channels();

    _data.reset(new unsigned char[size], std::default_delete<unsigned char[]>());
    Mat resized(cv::Size(width, height), img.type(), _data.get());

    cv::resize(img, resized, cv::Size(width, height));
    return _data;
}

int main()
{
    auto xml = "/home/ss21mipt/Documents/starkit/DIPLOMA/YOEO/config/IR&onnx_for_416_Petr_1/yoeo.xml";
    ov::Core core;

    // PROVERKA INPUT and OUTPUT
    // std::shared_ptr<ov::Model> model = core.read_model(xml);
    // std::cout << model->input(0).get_partial_shape() << std::endl;
    // std::cout << model->output(0).get_partial_shape() << std::endl;
    // std::cout << model->output(1).get_partial_shape() << std::endl;

    std::shared_ptr<ov::Model> net = core.read_model(xml);    // net = ie.ReadNetwork(model_path);
    ov::preprocess::PrePostProcessor ppp(net);        
    ov::preprocess::InputInfo& input = ppp.input(0);            // inputs = net.getInputsInfo(); 
    auto input_shape = net->input(0).get_partial_shape();
    std::cout << net->input(0).get_partial_shape() << std::endl;
    // auto output_1 = ppp.output();       // outputs = net.getOutputsInfo();
    // ov::preprocess::InputInfo& output_2 = ppp.output();
    
    input.tensor().set_element_type(ov::element::f32);   // NAPISAL PODRYGOMY (NE NHWC) почему здесь другая нумерация, как это может быть правильно написанным вариантом?
    input.model().set_layout("NCHW");                                               // input_data->setLayout(Layout::NCHW);       
    input.model().set_layout("NHWC");
    // input.tensor().set_shape({1, 416, 416, 3});
    // Model expects shape {1, 3, 480, 640}
    // input.preprocess().convert_layout({0, 3, 1, 2});
    input.tensor().set_element_type(ov::element::f32);
    // input.tensor().set_color_format(ov::preprocess::ColorFormat::NV12_TWO_PLANES);  // add NV12 to BGR conversion
    std::cout << "before converting color" << std::endl;
    // input.preprocess().convert_color(ov::preprocess::ColorFormat::RGB);             // input_data->getPreProcess().setColorFormat(ColorFormat::RGB);
    
    // output_1.tensor().set_element_type(ov::element::f32);                   // output_data->setPrecision(Precision::FP32);
    // output_2.tensor().set_element_type(ov::element::f32);                   // output_data->setPrecision(Precision::FP32);
    
    net = ppp.build();
    
    // INPUT = [1,3,H,W]    OUTPUTS = OUTPUT[0], OUTPUT[1] = [1, ( 3*(H/32)*(W/32) + 3*(H/16)*(W/16) ), (5 + numof_classes)], [1, H, W]

    // for (auto item : inputs)
    // {
    //     inputsName = item.first;
    //     auto input_data = item.second;
    //     input_data->setPrecision(Precision::FP32);
    //     input_data->setLayout(Layout::NCHW);
    //     input_data->getPreProcess().setColorFormat(ColorFormat::RGB);
    //     std::cout << "input name = " << m_inputName << std::endl;
    // }
 
 
    // for (auto item : outputs)
    // {
    //     auto output_data = item.second;
    //     output_data->setPrecision(Precision::FP32);
    //     outputsName = item.first;
    //     std::cout << "Loading model to the device " << device << std::endl;
    // }
    // std::cout << "Loading model to the device " << device << std::endl;

    ov::CompiledModel compiled_model = core.compile_model(net, "CPU");    // auto executable_network = ie.LoadNetwork(net, device);
    ov::InferRequest infer_request = compiled_model.create_infer_request();    // infer_request = executable_network.CreateInferRequest();
 
    ov::Tensor m_inputData = infer_request.get_input_tensor(0);    // m_inputData = infer_request.GetBlob(inputsName);
    
    std::cout << "Shape of input tensor: " << m_inputData.get_shape() << std::endl; 
    std::cout << "Type of input tensor: " << m_inputData.get_element_type() << std::endl;    
    auto m_numChannels = m_inputData.get_shape()[1];
    auto m_inputW = m_inputData.get_shape()[3];
    auto m_inputH = m_inputData.get_shape()[2];
    auto m_imageSize = m_inputH * m_inputW;
    auto data1 = m_inputData.data();
    std::cout << "m_numChannels: " << m_numChannels <<  std::endl <<"m_inputW: " << m_inputW <<  std::endl << "m_inputH: " << m_inputH <<  std::endl<< "m_imageSize: " << m_imageSize << std::endl;
    
    ov::element::Type input_tensor_type = ov::element::f32;


    cv::VideoCapture cap(0);
    cv::Mat image;
    cv::Mat segments;

    std::cout << "doshlo" << std::endl;

    while (cap.isOpened()){
        cap >> image;
        std::cout << "doshlo" << std::endl;
        if (image.empty() || !image.data) {
            return false;
        }
        std::cout << "SIZES"  << image.rows << image.cols <<  image.type() << std::endl;
        // std::shared_ptr input_data = getData(data_img);
        // m_inputData = ov::Tensor(input_tensor_type, input_shape, input_data.get());   
        // FILLING THE DATA1
        // for (size_t row = 0; row < m_inputH; row++) {
        //     for (size_t col = 0; col < m_inputW; col++) {
        //         for (size_t ch = 0; ch < m_numChannels; ch++) {
        //             data1[m_imageSize * ch + row * m_inputW + col] = float(image.at<cv::Vec3b>(row, col)[ch]);
        //         }
        //     }
        // }


        cv::imshow("webcam", image);
        // cv::imshow("segmentation", segments);
        std::cout << "doshlo" << std::endl;
        if(cv::waitKey(30)>=0)
                break;
        std::cout << "Allo blyat " << std::endl;
    }
    // cv::destroyWindow("segmentation");
    cv::destroyWindow("webcam");

}