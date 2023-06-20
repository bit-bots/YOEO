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

#include <chrono>
#include <algorithm>
#define CLOCK std::chrono::steady_clock
#define CLOCK_CAST std::chrono::duration_cast<std::chrono::microseconds>

int main()
{
    auto xml;

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
    // std::cout << net->input(0).get_partial_shape() << output1_shape << std::endl;
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
    // std::cout << "before converting color" << std::endl;
    // input.preprocess().convert_color(ov::preprocess::ColorFormat::RGB);             // input_data->getPreProcess().setColorFormat(ColorFormat::RGB);
    
    // output_1.tensor().set_element_type(ov::element::f32);                   // output_data->setPrecision(Precision::FP32);
    // output_2.tensor().set_element_type(ov::element::f32);                   // output_data->setPrecision(Precision::FP32);
    
    net = ppp.build();
    
    // INPUT = [1,3,H,W]    OUTPUTS = OUTPUT[0], OUTPUT[1] = [1, ( 3*(H/32)*(W/32) + 3*(H/16)*(W/16) ), (5 + numof_classes)], [1, H, W]

    ov::CompiledModel compiled_model = core.compile_model(net, "CPU");    // auto executable_network = ie.LoadNetwork(net, device);
    ov::InferRequest infer_request = compiled_model.create_infer_request();    // infer_request = executable_network.CreateInferRequest();
 
    ov::Tensor m_inputData = infer_request.get_input_tensor(0);    // m_inputData = infer_request.GetBlob(inputsName);
    
    // std::cout << "Shape of input tensor: " << m_inputData.get_shape() << std::endl; 
    // std::cout << "Type of input tensor: " << m_inputData.get_element_type() << std::endl;    
    auto m_numChannels = m_inputData.get_shape()[1];
    auto m_inputW = m_inputData.get_shape()[3];
    auto m_inputH = m_inputData.get_shape()[2];
    auto m_imageSize = m_inputH * m_inputW;
    auto data1 = m_inputData.data<float_t>();
    // std::cout << "m_numChannels: " << m_numChannels <<  std::endl <<"m_inputW: " << m_inputW <<  std::endl << "m_inputH: " << m_inputH <<  std::endl<< "m_imageSize: " << m_imageSize << std::endl;
    
    // ov::element::Type input_tensor_type = ov::element::f32;


    // cv::VideoCapture cap(0);
    cv::VideoCapture cap("/home/ss21mipt/DIPLOMA/test_data/sahr3/video.avi");
    cv::Mat image;
    cv::Mat segments;
    cv::Mat mask;
    cv::Mat test_mask;
    int index = 0;
    std::chrono::steady_clock::time_point begin;
    std::chrono::steady_clock::time_point end;
    std::chrono::steady_clock::time_point begin_global;
    std::chrono::steady_clock::time_point end_global;
    auto time_preproc= 0;
    auto t_preproc = 0;
    auto t_preproc_max = 0;
    auto t_preproc_min = 100;
    auto time_infer = 0;
    auto t_infer = 0;
    auto t_infer_max = 0;
    auto t_infer_min = 100;
    auto time_postproc = 0;
    auto t_postproc = 0;
    auto t_postproc_max = 0;
    auto t_postproc_min = 100;
    // std::cout << "doshlo" << std::endl;
    begin_global = CLOCK::now();
    // while (cap.isOpened()){
    while (index < 500){
        begin = CLOCK::now();
        cap >> image;
        // image = cv::imread("/home/ss21mipt/DIPLOMA/IoU_tool/ground_t/frame111.jpg");
        // std::cout << "doshlo" << std::endl;
        if (image.empty() || !image.data) {
            return false;
        }
        cv::Size scale(416, 416);  
        cv::resize(image, image, scale);    

        // ПЕРЕБИВАНИЕ КАРТИНКИ В ВЕКТОР

        
        // std::cout << "SIZES of Mat: "  << image.size[0] << " " << image.size[1] << " " << image.channels()<<  std::endl;

        // auto data1 = input_tensor1.data<int32_t>();
        // std::shared_ptr input_data = getData(data_img);
        // m_inputData = ov::Tensor(input_tensor_type, input_shape, input_data.get());   
        // FILLING THE DATA1
        for (size_t row = 0; row < m_inputH; row++) {
            for (size_t col = 0; col < m_inputW; col++) {
                for (size_t ch = 0; ch < m_numChannels; ch++) {
        // #ifdef NCS2
                    data1[m_imageSize * ch + row * m_inputW + col] = float(image.at<cv::Vec3b>(row, col)[ch]);
                    
        // #else
                    // data[m_imageSize * ch + row * m_inputW + col] = float(blob_image.at<cv::Vec3b>(row, col)[ch] / 255.0);
        // #endif // NCS2
                }
            }
        }
        end = CLOCK::now();
        time_preproc = time_preproc + (CLOCK_CAST(end - begin).count() / 1000.0);
        t_preproc = (CLOCK_CAST(end - begin).count() / 1000.0);
        if (t_preproc > t_preproc_max) {
            t_preproc_max = t_preproc;
        }
        if (t_preproc < t_preproc_min) {
            t_preproc_min = t_preproc;
        }
        std::cout << "FPS #1 PREPROCESSING: " <<  (CLOCK_CAST(end - begin).count() / 1000.0) << std::endl;

        // data1 = &array_mat[0];
        begin = CLOCK::now();
        infer_request.infer();
        std::cout << "HAHA" << std::endl;
        end = CLOCK::now();
        time_infer = time_infer + (CLOCK_CAST(end - begin).count() / 1000.0);
        t_infer = (CLOCK_CAST(end - begin).count() / 1000.0);
        if (t_infer > t_infer_max) {
            t_infer_max = t_infer;
        }
        if (t_infer < t_infer_min) {
            t_infer_min = t_infer;
        }
        std::cout << "FPS #2 INFERENCE: " <<  (CLOCK_CAST(end - begin).count() / 1000.0) << std::endl;

        begin = CLOCK::now();

        ov::Tensor output_tensor1 = infer_request.get_output_tensor(0);
        ov::Tensor output_tensor2 = infer_request.get_output_tensor(1);
        
        // IR v10 works with converted precisions (i64 -> i32)
        auto out_data1 = output_tensor1.data<float_t>();
        auto out_data2 = output_tensor2.data<float_t>();
        auto mask_shape = output_tensor2.get_shape();
        
        // std::cout << "SEGMENTATION MASK" << mask_shape << std::endl;
        // std::cout << "SEGMENTATION MASK" << *out_data2 << std::endl;
        test_mask = cv::Mat::ones(mask_shape[1], mask_shape[2], CV_8UC1);
        // TENSOR TO MAT
        // std::cout << test_mask.channels() << test_mask.size() << std::endl;
        for(size_t i=0; i<(m_inputH); i++){
            for(size_t j=0; j<m_inputW; j++){
                // mask.at<float>(i, j) = float(data1[i * m_inputW + j]);
                // std::cout << test_mask.at(i) << std::endl;
                if (out_data2[i*m_inputH+j] == 1) {
                    out_data2[i*m_inputH+j] = 0;    // зануляем фильтр зеленого
                }
                test_mask.at<uchar>(i*m_inputH+j) = (char)(out_data2[i*m_inputH+j]*100);
            } 
        }
        // end = CLOCK::now();
        // std::cout << "FPS #2 INTO FINAL MASK " <<  (CLOCK_CAST(end - begin).count() / 1000.0) << std::endl;
        
        // begin = CLOCK::now();
        // auto segments = tensorToMat(output_tensor2);
        // cv::imshow("webcam", image);
        cv::resize(test_mask, test_mask, cv::Size(720,540));
        // cv::imshow("YOEO_segmentation", test_mask);
        // std::cout << "doshlo" << mask << std::endl;
        // if(cv::waitKey(30)>=0)
        //         break;
        // std::cout << "Allo blyat " << std::endl;
        // cv::imwrite("alpha.png", mask);
        // break;
        index++;
        end = CLOCK::now();
        time_postproc = time_postproc + (CLOCK_CAST(end - begin).count() / 1000.0);
        t_postproc = (CLOCK_CAST(end - begin).count() / 1000.0);
        if (t_postproc > t_postproc_max) {
            t_postproc_max = t_postproc;
        }
        if (t_postproc < t_postproc_min) {
            t_postproc_min = t_postproc;
        }
        std::cout << "FPS #3 POSTPROCESSING " <<  (CLOCK_CAST(end - begin).count() / 1000.0) << std::endl;
        std::cout << "index = " << index << std::endl;
    }
    // cv::imwrite("/home/ss21mipt/DIPLOMA/IoU_tool/YOEO_segmentation.jpg", test_mask);
    end_global = CLOCK::now();
    std::cout << "AVERAGE FPS " <<  (index * 1.0 / (CLOCK_CAST(end_global - begin_global).count() / 1000000.0)) << std::endl;
    std::cout << "AVERAGE PREPROCESSING FPS: " << index*1.0/(time_preproc / 1000.0) << std::endl;
    std::cout << "AVERAGE INFERENCE FPS: " << index*1.0/(time_infer / 1000.0) << std::endl;
    std::cout << "AVERAGE POSTPROCESSING FPS: " << index*1.0/(time_postproc / 1000.0) << std::endl;

    std::cout << "AVERAGE PREPROCESSING time: " << ((time_preproc) / (index*1.0)) << " in range [" << t_preproc_min << ", " << t_preproc_max << "]" << std::endl;
    std::cout << "AVERAGE INFERENCE time: " << ((time_infer) / (index*1.0)) << " in range [" << t_infer_min << ", " << t_infer_max << "]" << std::endl;
    std::cout << "AVERAGE POSTPROCESSING time: " << ((time_postproc) / (index*1.0)) << " in range [" << t_postproc_min << ", " << t_postproc_max << "]" << std::endl;
    cv::destroyWindow("YOEO_segmentation");
    // cv::destroyWindow("webcam");

}