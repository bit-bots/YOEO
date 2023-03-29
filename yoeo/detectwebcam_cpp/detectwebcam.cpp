/*
def _draw_and_save_output_image(image, detections, seg, img_size, output_path, classes):
    """Draws detections in output image and stores this.

    :param image_path: Path to input image
    :type image_path: str
    :param detections: List of detections on image
    :type detections: [Tensor]
    :param seg: Segmentation image
    :type seg: Tensor
    :param img_size: Size of each image dimension for yolo
    :type img_size: int
    :param output_path: Path of output directory
    :type output_path: str
    :param classes: List of class names
    :type classes: [str]
    """
    # Create plot
    img = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    plt.figure()
    fig, ax = plt.subplots(1)
    # Get segmentation
    seg = seg.cpu().detach().numpy().astype(np.uint8)
    # seg = seg.astype(np.uint8)
    # Draw all of it
    seg = seg[0]
    print(f"ETO EST SEG {seg}")
    # The amount of padding that was added
    print("GOVNINA")
    print(img_size / max(img.shape[:2]))
    print(max(img.shape[0] - img.shape[1], 0))
    print(img.shape[0], img.shape[1])
    print("end of GOVNINA")
    pad_x = max(img.shape[0] - img.shape[1], 0) * (img_size / max(img.shape[:2])) // 2
    # pad_x = 21.0
    pad_y = max(img.shape[1] - img.shape[0], 0) * (img_size / max(img.shape[:2])) // 2
    print(f"CHEKAI PADI {pad_x, pad_y}")

    seg_map = seg[
                int(pad_y) : int(img_size - pad_y),
                int(pad_x) : int(img_size - pad_x),
                ] * 255

    print(f"MILLIARDNAYA {img, pad_y, pad_x}")
    ax.imshow(
        SegmentationMapsOnImage(
            seg[
                int(pad_y) : int(img_size - pad_y),
                int(pad_x) : int(img_size - pad_x),
                ], shape=img.shape).draw_on_image(img)[0])
    print("JJEEPPAAA")
    # Rescale boxes to original image

    detections = rescale_boxes(detections, img_size, img.shape[:2])
    unique_labels = detections[:, -1].cpu().unique()
    n_cls_preds = len(unique_labels)
    # Bounding-box colors
    cmap = plt.get_cmap("tab20b")
    colors = [cmap(i) for i in np.linspace(0, 1, len(classes))]
    for x1, y1, x2, y2, conf, cls_pred in detections:

        print(f"\t+ Label: {classes[int(cls_pred)]} | Confidence: {conf.item():0.4f}")

        box_w = x2 - x1
        box_h = y2 - y1

        # Create a Rectangle patch
        bbox = patches.Rectangle((x1, y1), box_w, box_h, linewidth=2, edgecolor=colors[int(cls_pred)], facecolor="none")
        # Add the bbox to the plot
        ax.add_patch(bbox)
        # Add label
        """
        plt.text(
            x1,
            y1,
            s=classes[int(cls_pred)],
            color="white",
            verticalalignment="top",
            bbox={"color": colors[int(cls_pred)], "pad": 0})
        """

    # Save generated image with detections
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.gca().xaxis.set_major_locator(NullLocator())
    plt.gca().yaxis.set_major_locator(NullLocator())
    # filename = os.path.basename(image_path).split(".")[0]
    # output_path_1 = os.path.join(output_path, f"{filename}.png")
    # redraw the canvas
    fig.canvas.draw()
    # convert canvas to image
    img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8,
                                    sep='')
    print(f"summing up PICTURE 0 : {img.shape}")                                
    img  = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    # img is rgb, convert to opencv's default bgr
    img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
    print(f"summing up PICTURE 1 : {img.shape}")
    # cv2.imwrite(output_path_1, img)
    cv2.imshow('inference', img)
    # cv2.waitKey(1)



model = load_model(model_path, weights_path)
print("NY PRIVET 7")

cam = cv2.VideoCapture(0)
key = cv2.waitKey(1)

while key != 27:
    print("START OF INFERENCE OF IMAGE")
    t = time.time()
    _, image = cam.read()

    fps = int(cam.get(cv2.CAP_PROP_FPS))
    fps_text = "FPS: {:.2f}".format(fps)
    cv2.putText(image, fps_text, (5, 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 1)
    
    cv2.imshow('raw', image)

    model.eval()  # Set model to evaluation mode

    # Configure input
    input_img = transforms.Compose([
        DEFAULT_TRANSFORMS,
        Resize(img_size)])((
            image,
            np.empty((1, 5)),
            np.empty((img_size, img_size), dtype=np.uint8)))[0].unsqueeze(0)

    print(f"raw image shape: {image.shape}")
    print(f"torch image shape: {input_img.shape}")
    

    if torch.cuda.is_available():
        input_img = input_img.to("cuda")
        print(f"torch image shape: {input_img.shape}")
    # Get detections
    with torch.no_grad():

        detections, segmentations = model(input_img)
        detections = non_max_suppression(detections, conf_thres, nms_thres)
        detections = rescale_boxes(detections[0], img_size, image.shape[0:2])
        segmentations = rescale_segmentation(segmentations, image.shape[0:2])
        print(f"detections shape: {detections.shape}")
        print(f"detections shape: {detections.shape}")

    _draw_and_save_output_image(image, detections, segmentations, img_size, output_path, classes)

    if cv2.waitKey(1) == 27:
        cam.release()
        cv2.destroyAllWindows()
        break
*/

#include <torch/script.h>
// #include <opencv2/opencv.hpp>
// #include <ATen/ATen.h>
#include <iostream>
#include <opencv2/opencv.hpp>
// #include <opencv2/highgui.hpp>

auto ToInput(torch::Tensor tensor_image)
{
    // Create a vector of inputs.
    return std::vector<torch::jit::IValue>{tensor_image};
}

auto ToTensor(cv::Mat img, bool show_output = false, bool unsqueeze = false, int unsqueeze_dim = 0)
{
    std::cout << "image shape: " << img.size() << std::endl;
    torch::Tensor tensor_image = torch::from_blob(img.data, {1, img.rows, img.cols, 3 }, torch::kByte);

    if (unsqueeze)
    {
        tensor_image.unsqueeze_(unsqueeze_dim);
        std::cout << "tensors new shape: " << tensor_image.sizes() << std::endl;
    }

    if (show_output)
    {
        std::cout << tensor_image.slice(2, 0, 1) << std::endl;
    }
    std::cout << "tenor shape: " << tensor_image.sizes() << std::endl;
    return tensor_image;
}

torch::Tensor CVtoTensor(cv::Mat img,int unsqueeze_dim = 0) {
    cv::Size scale(640, 640);
    cv::resize(img, img, scale);
    std::cout << "== simply resize: " << img.size() << " ==" << std::endl;
    cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
    auto img_tensor = torch::from_blob(img.data, {1, img.rows, img.cols, 3 }, torch::kByte);
    img_tensor = img_tensor.permute({ 0, 3, 1, 2 }).toType(torch::kFloat);
    std::cout << "tensor shape: " << img_tensor.sizes() << std::endl;
    // std::cout << "tensors new shape: " << img_tensor.sizes() << std::endl;
    return img_tensor;
}

int main() {

    int kIMAGE_SIZE = 416;
    std::cout << "OpenCV version : " << CV_VERSION << std::endl;
    std::cout << "Major version : " << CV_MAJOR_VERSION << std::endl;
    std::cout << "Minor version : " << CV_MINOR_VERSION << std::endl;
    std::cout << "Subminor version : " << CV_SUBMINOR_VERSION << std::endl;

    torch::jit::script::Module module;
    std::cout << "BEFORE LOADING MODEL" << std::endl;
    // module.load_jit("/home/ss21mipt/Documents/starkit/DIPLOMA/YOEO/weights/yoeo.pt");
    module = torch::jit::load("/home/ss21mipt/Documents/starkit/DIPLOMA/YOEO/weights/ochko.pt");
    // module = torch::jit::load("/home/ss21mipt/Documents/starkit/DIPLOMA/YOEO/weights/expected_flow.pt");
    std::cout << "AFTER LOADING MODEL" << std::endl;
    module.to(torch::kCPU);
    // module.eval();
    
    cv::VideoCapture cap(0);
    cv::Mat image;
    cv::Mat segments;
    torch::Tensor img;
    int dets = 5;
    int segs = 4;
    while (cap.isOpened()){
        cap >> image;
        if (image.empty() || !image.data) {
            return false;
        }
        img = CVtoTensor(image);
        std::cout << "dim 0: " << img.sizes()[0] << std::endl;
        std::cout << "dim 1: " << img.sizes()[1] << std::endl;
        std::cout << "dim 2: " << img.sizes()[2] << std::endl;
        std::cout << "dim 3: " << img.sizes()[3] << std::endl;
        
        // input_tensor.sub_(0.5).div_(0.5);    
        std::vector<torch::jit::IValue> input;
        // // inputs.emplace_back(img);
        input.push_back(img);
        // inputs = inputs.to(torch::kCPU);
        std::cout << "BEFORE OUTPUT worked" << std::endl;
        // std::cout << typeid().name() << std::
        auto output = module.forward(input).toTensorList();
        // std::cout << "MID OUTPUT worked" << std::endl;
        // auto out = output.toTensorList();
        std::cout << "AFTER OUTPUT worked" << std::endl;

        // inputs.push_back(torch::ones({1, 3, 416, 416}));
        // std::cout << "ToInput worked" << std::endl;
        // torch::IValue out = module.forward({input_tensor});
        // std::cout << "ZDOROVO" << std::endl;
        for (int i=0; i<output.size(); i++)
        {
            // std::cout << "try: " << i << output.size() << std::endl;
            torch::Tensor ten = output[i];
            auto x = ten.sizes();
            // std::cout << "SUCCEED" <<std::endl;
            // std::cout << "try: " << ten.sizes().size() << std::endl;
            std::cout << "Shape of every elem: " << ten.sizes() << std::endl;
            if (ten.sizes().size() == dets)
            {
                std::cout << "first: " << ten.sizes() << std::endl;
            }
            if (ten.sizes().size() == segs)
            {
                std::cout << "second: " << ten.sizes() << std::endl;
                ten = ten.squeeze().detach().to(torch::kInt); 
                ten = ten.mul(255).clamp(0,255);
                // std::cout << "second: " << ten << std::endl;
                std::cout << "tyt X " << std::endl;
                cv::Mat resultImg(416, 416,CV_8UC3,(void*)ten.data_ptr());
                cv::cvtColor(resultImg, resultImg, cv::COLOR_BGR2RGB);
                std::cout << "tyt Y " << std::endl;
                segments = resultImg;
                std::cout << "tyt Z " << std::endl;
            }

        }
        // return 1;
        cv::imshow("webcam", image);
        cv::imshow("segmentation", segments);
        if(cv::waitKey(30)>=0)
                break;
        std::cout << "Allo blyat " << std::endl;
    }
    cv::destroyWindow("segmentation");
            // std::cout << "try: " << output[0].slice() << std::endl;
        // auto output = out.toTensor();
        // std::cout << "output initialized" << std::endl;
        
        // auto results = out.sort(-1, true);
        // auto softmaxs = std::get<0>(results)[0].softmax(0);
        // auto indexs = std::get<1>(results)[0];

        //sizes() gives shape. 
        // std::cout << output.sizes() << std::endl;
        //std::cout << "output: " << output[0] << std::endl;
        //std::cout << output.slice(/*dim=*/1, /*start=*/0, /*end=*/5) << '\n';
        
        // output = torch::sigmoid(output);
        
        // auto out_tensor = output.squeeze(0).detach().permute({ 1, 2, 0 });
        
        //auto out_tensor = output.squeeze().detach();
        // std::cout << "out_tensor (after squeeze & detach): " << out_tensor.sizes() << std::endl;
        // out_tensor = out_tensor.mul(255).clamp(0, 255).to(torch::kU8);
        // out_tensor = out_tensor.to(torch::kCPU);
        // cv::Mat resultImg(448, 448, CV_8UC3);
        // std::memcpy((void*)resultImg.data, out_tensor.data_ptr(), sizeof(torch::kU8) * out_tensor.numel());
        // cv::resize(resultImg, resultImg, cv::Size(1280, 720), 0, 0, cv::INTER_AREA);

        // cv::imshow("segmentation", resultImg);
        // cv::waitKey(0);
        // cv::destroyWindow("segmentation");  
    return 0;
}