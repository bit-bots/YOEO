#include "openvino/openvino.hpp"
#include <iterator>
#include <memory>
#include <sstream>
#include <string>
#include <vector>
// #include <opencv2/opencv.hpp>
#include <ngraph/type/element_type.hpp>
#include "format_reader_ptr.h"

int main()
{
    auto xml = "/home/ss21mipt/Documents/starkit/DIPLOMA/YOEO/config/IR&onnx_for_416_Petr_1/yoeo.xml";
    ov::Core core;

    // PROVERKA INPUT and OUTPUT
    std::shared_ptr<ov::Model> model = core.read_model(xml);
    std::cout << model->input(0).get_partial_shape() << std::endl;
    std::cout << model->output(0).get_partial_shape() << std::endl;
    std::cout << model->output(1).get_partial_shape() << std::endl;
    
    ov::CompiledModel compiled_model = core.compile_model("model.xml", "AUTO");
    ov::InferRequest infer_request = compiled_model.create_infer_request();
    auto segment = model->output(1);

    // ov::CompiledModel compiled_model = core.compile_model(xml, "AUTO");
    // ov::InferRequest infer_request = compiled_model.create_infer_request();
    
    // Get input port for model with one input
    // auto input_port = compiled_model.input();
    
    // Create tensor from external memory
    // ov::Tensor input_tensor(input_port.get_element_type(), input_port.get_shape(), memory_ptr);
    
    // Set input tensor for model with one input
    // infer_request.set_input_tensor(input_tensor);
    // infer_request.start_async();
    // infer_request.wait();   
    
    // Get output tensor by tensor name
    // auto output = infer_request.get_tensor("tensor_name");
    // const float \*output_buffer = output.data<const float>();
    // output_buffer[] - accessing output tensor data
}