#include "openvino/openvino.hpp"
#include <opencv2/opencv.hpp>
#include <iterator>
#include <memory>
#include <sstream>
#include <string>
#include <vector>
// #include <opencv2/opencv.hpp>
#include <ngraph/type/element_type.hpp>
#include "include/format_reader_ptr.h"


cv::Mat tensorToMat(const ov::Tensor &tensor)
{
    auto sizes = tensor.get_shape();
    int elem[5] = {5,6,7,8,9};
    int flag = 0;
    int skok = 0;
    std::cout << "in tensorToMat check sizes" << sizes[0] << " " << sizes[1] << " " << sizes[2] << std::endl;
    cv::Mat result = cv::Mat{sizes[1], sizes[2], CV_32FC(sizes[0]), tensor.data<float_t>()};
    for (size_t row = 0; row < sizes[1]; row++) {
        for (size_t col = 0; col < sizes[2]; col++) {
            for (size_t ch = 0; ch < sizes[0]; ch++) {
                for (int k = 0; k<5; k++){
                    if (result.at<cv::Vec3b>(row, col)[ch] == elem[k]){
                        flag++;
                    }
                }
                if (flag == 0){
                    elem[0+skok] = result.at<cv::Vec3b>(row, col)[ch];
                    std::cout << result.at<cv::Vec3b>(row, col)[ch] << std::endl;
                    skok++;
                }
                flag = 0;
                result.at<cv::Vec3b>(row, col)[ch] = result.at<cv::Vec3b>(row, col)[ch] * 100;
            }
        }
    }
    std::cout << "UNIQUE ARE: " << elem[0] << " " << elem[1] << " " << elem[2] << " " << elem[3] << " " << elem[4] << std::endl;
    return result;
}

// void colorizeSegmentation(const Mat &score, Mat &segm)
// {
//     const int rows = score.size[2];
//     const int cols = score.size[3];
//     const int chns = score.size[1];
//     if (colors.empty())
//     {
//         // Generate colors.
//         colors.push_back(Vec3b());
//         for (int i = 1; i < chns; ++i)
//         {
//             Vec3b color;
//             for (int j = 0; j < 3; ++j)
//                 color[j] = (colors[i - 1][j] + rand() % 256) / 2;
//             colors.push_back(color);
//         }
//     }
//     else if (chns != (int)colors.size())
//     {
//         CV_Error(Error::StsError, format("Number of output classes does not match "
//                                          "number of colors (%d != %d)", chns, colors.size()));
//     }
//     Mat maxCl = Mat::zeros(rows, cols, CV_8UC1);
//     Mat maxVal(rows, cols, CV_32FC1, score.data);
//     for (int ch = 1; ch < chns; ch++)
//     {
//         for (int row = 0; row < rows; row++)
//         {
//             const float *ptrScore = score.ptr<float>(0, ch, row);
//             uint8_t *ptrMaxCl = maxCl.ptr<uint8_t>(row);
//             float *ptrMaxVal = maxVal.ptr<float>(row);
//             for (int col = 0; col < cols; col++)
//             {
//                 if (ptrScore[col] > ptrMaxVal[col])
//                 {
//                     ptrMaxVal[col] = ptrScore[col];
//                     ptrMaxCl[col] = (uchar)ch;
//                 }
//             }
//         }
//     }
//     segm.create(rows, cols, CV_8UC3);
//     for (int row = 0; row < rows; row++)
//     {
//         const uchar *ptrMaxCl = maxCl.ptr<uchar>(row);
//         Vec3b *ptrSegm = segm.ptr<Vec3b>(row);
//         for (int col = 0; col < cols; col++)
//         {
//             ptrSegm[col] = colors[ptrMaxCl[col]];
//         }
//     }
// }

int main()
{
    auto xml = "/home/ss21mipt/Documents/starkit/DIPLOMA/YOEO/config/IR&onnx_for_416_Petr_1/yoeo.xml";
    // auto xml = "/home/ss21mipt/Documents/starkit/DIPLOMA/to_rhoban/weights/Feds_yolov8_2_openvino/best.xml";
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
    auto output1_shape = net->output(0).get_partial_shape();
    std::cout << net->input(0).get_partial_shape() << output1_shape << std::endl;
    // auto output_1 = ppp.output(0);       // outputs = net.getOutputsInfo();
    // auto output_2 = ppp.output(1);       // outputs = net.getOutputsInfo();
    
    // auto output_1 = ppp.output(0).postprocess().convert_element_type(ov::element::i32);
    // auto output_2 = ppp.output(1).postprocess().convert_element_type(ov::element::i32);
    
    // std::cout << output_1 << output_2 << std::endl;

    input.tensor().set_element_type(ov::element::f32);   // NAPISAL PODRYGOMY (NE NHWC) почему здесь другая нумерация, как это может быть правильно написанным вариантом?
    input.model().set_layout("NCHW");                                               // input_data->setLayout(Layout::NCHW);       
    input.model().set_layout("NHWC");
    // input.tensor().set_shape({1, 416, 416, 3});
    // Model expects shape {1, 3, 480, 640}
    // input.preprocess().convert_layout({0, 3, 1, 2});
    // input.tensor().set_element_type(ov::element::f32);
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
    auto data1 = m_inputData.data<float_t>();
    std::cout << "m_numChannels: " << m_numChannels <<  std::endl <<"m_inputW: " << m_inputW <<  std::endl << "m_inputH: " << m_inputH <<  std::endl<< "m_imageSize: " << m_imageSize << std::endl;
    
    // ov::element::Type input_tensor_type = ov::element::f32;


    cv::VideoCapture cap(0);
    cv::Mat image;
    cv::Mat segments;
    cv::Mat mask;
    cv::Mat test_mask;
    std::cout << "doshlo" << std::endl;

    while (cap.isOpened()){
        cap >> image;
        std::cout << "doshlo" << std::endl;
        if (image.empty() || !image.data) {
            return false;
        }
        cv::Size scale(416, 416);  
        cv::resize(image, image, scale);    

        // ПЕРЕБИВАНИЕ КАРТИНКИ В ВЕКТОР

        
        std::cout << "SIZES of Mat: "  << image.size[0] << " " << image.size[1] << " " << image.channels()<<  std::endl;

        // auto data1 = input_tensor1.data<int32_t>();
        // std::shared_ptr input_data = getData(data_img);
        // m_inputData = ov::Tensor(input_tensor_type, input_shape, input_data.get());   
        // FILLING THE DATA1
        
        for (size_t row = 0; row < m_inputH; row++) {
            for (size_t col = 0; col < m_inputW; col++) {
                for (size_t ch = 0; ch < m_numChannels; ch++) {
        // #ifdef NCS2
                    data1[m_imageSize * ch + row * m_inputW + col] = float(image.at<cv::Vec3b>(row, col)[ch]);
                    // std::cout << "XYETA " << float(image.at<cv::Vec3b>(row, col)[ch]) << data1[m_imageSize * ch + row * m_inputW + col] << std::endl ;
        // #else
                    // data[m_imageSize * ch + row * m_inputW + col] = float(blob_image.at<cv::Vec3b>(row, col)[ch] / 255.0);
        // #endif // NCS2
                }
            }
        }

        // data1 = &array_mat[0];

        infer_request.infer();
        std::cout << "HAHA" << std::endl;
        ov::Tensor output_tensor1 = infer_request.get_output_tensor(0);

        ov::Tensor output_tensor2 = infer_request.get_output_tensor(1);
        
        // IR v10 works with converted precisions (i64 -> i32)
        auto out_data1 = output_tensor1.data<float_t>();
        auto out_data2 = output_tensor2.data<float_t>();
        auto mask_shape = output_tensor2.get_shape();
        
        std::cout << "SEGMENTATION MASK" << mask_shape << std::endl;
        std::cout << "SEGMENTATION MASK" << *out_data2 << std::endl;
        test_mask = cv::Mat::ones(mask_shape[1], mask_shape[2], CV_8UC1);
        // TENSOR TO MAT
        std::cout << test_mask.channels() << test_mask.size() << std::endl;
        for(size_t i=0; i<(m_inputH); i++){
            for(size_t j=0; j<m_inputW; j++){
                // mask.at<float>(i, j) = float(data1[i * m_inputW + j]);
                // std::cout << test_mask.at(i) << std::endl;
                test_mask.at<uchar>(i*m_inputH+j) = (char)(out_data2[i*m_inputH+j]*100);
                std::cout << " ZDOROVA" << std::endl;
             
                    
                // std::cout << data1[i+j] << " " << mask.at<float>(i, j) << std::endl;
                // std::cout << data1[i+j] << std::endl;
                // std::cout << out_data1[i+j] << std::endl;
            } 
        }

        // auto segments = tensorToMat(output_tensor2);
        cv::imshow("webcam", image);
        cv::imshow("segmentation", test_mask);
        // std::cout << "doshlo" << mask << std::endl;
        if(cv::waitKey(30)>=0)
                break;
        std::cout << "Allo blyat " << std::endl;
        // cv::imwrite("alpha.png", mask);
        // break;
    }
    cv::destroyWindow("segmentation");
    cv::destroyWindow("webcam");

}