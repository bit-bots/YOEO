// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#ifdef USE_OPENCV
#    include <fstream>
#    include <iostream>

// clang-format off
#    include <opencv2/opencv.hpp>

#    include "samples/slog.hpp"
#    include "opencv_wrapper.h"
// clang-format on

using namespace std;
using namespace FormatReader;

OCVReader::OCVReader(const string& filename) {
    img = cv::imread(filename);
    _size = 0;

    if (img.empty()) {
        return;
    }

    _size = img.size().width * img.size().height * img.channels();
    _width = img.size().width;
    _height = img.size().height;
}

std::shared_ptr<unsigned char> OCVReader::getData(size_t width = 0, size_t height = 0) {
    if (width == 0)
        width = img.cols;

    if (height == 0)
        height = img.rows;

    size_t size = width * height * img.channels();
    _data.reset(new unsigned char[size], std::default_delete<unsigned char[]>());

    cv::Mat resized(cv::Size(width, height), img.type(), _data.get());

    if (width != img.cols || height != img.rows) {
        slog::warn << "Image is resized from (" << img.cols << ", " << img.rows << ") to (" << width << ", " << height
                   << ")" << slog::endl;
    }
    // cv::resize() just copy data to output image if sizes are the same
    cv::resize(img, resized, cv::Size(width, height));

    return _data;
}
#endif
