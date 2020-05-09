/**
MIT License

Copyright (c) 2018 NVIDIA CORPORATION. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*
*/
#include "ds_image.h"
#include "trt_utils.h"
#include "yolov3.h"
#include "yolo_config_parser.h"

#include <fstream>
#include <string>
#include <sys/time.h>


using namespace std;
using namespace cv;

int main(int argc, char** argv)
{
    NetworkInfo yoloInfo = getYoloNetworkInfo();
    InferParams yoloInferParams = getYoloInferParams();
    uint64_t seed = getSeed();
    std::string networkType = getNetworkType();
    std::string precision = getPrecision();
    bool decode = getDecode();

    uint batchSize = getBatchSize();


    srand(unsigned(seed));

    std::unique_ptr<Yolo> inferNet{nullptr};

    inferNet = std::unique_ptr<Yolo>{new YoloV3(batchSize, yoloInfo, yoloInferParams)};


    Mat frame;
    VideoCapture cap;
    string videofile;
    if(argc>1)
    {
        videofile = argv[1];
        if(argc>2)
        {
        }
    }
    else
    {
        videofile = "/work/workspace/track_video/video/3.mp4";
    }
    cap.open(videofile);
       if (!cap.isOpened())
           return -1;

    std::vector<DsImage> dsImages;
    const int barWidth = 70;
    double inferElapsed = 0;

    std::ofstream fout;
    bool written = false;

    // Batched inference loop
    for (uint loopIdx = 0; ; loopIdx += batchSize)
    {
        cap >> frame;
        // Load a new batch
        dsImages.clear();

        dsImages.emplace_back(frame.data, frame.rows, frame.cols, inferNet->getInputH(), inferNet->getInputW());


        cv::Mat trtInput = blobFromDsImages(dsImages, inferNet->getInputH(), inferNet->getInputW());
        struct timeval inferStart, inferEnd;
        gettimeofday(&inferStart, NULL);
        inferNet->doInference(trtInput.data, dsImages.size());
        gettimeofday(&inferEnd, NULL);
        inferElapsed += ((inferEnd.tv_sec - inferStart.tv_sec)
                         + (inferEnd.tv_usec - inferStart.tv_usec) / 1000000.0)
            * 1000;

        if (decode)
        {
            for (uint imageIdx = 0; imageIdx < dsImages.size(); ++imageIdx)
            {
                auto curImage = dsImages.at(imageIdx);
                auto binfo = inferNet->decodeDetections(imageIdx, curImage.getImageHeight(),
                                                        curImage.getImageWidth());
                auto remaining
                    = nmsAllClasses(inferNet->getNMSThresh(), binfo, inferNet->getNumClasses());
                for (auto b : remaining)
                {
                    if (inferNet->isPrintPredictions())
                    {
                        printPredictions(b, inferNet->getClassName(b.label));
                    }
                    curImage.addBBox(b, inferNet->getClassName(b.label));
                }
                curImage.showImage();
            }
        }
    }


    return 0;
}
