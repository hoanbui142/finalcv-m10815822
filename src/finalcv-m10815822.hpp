#pragma once 

#include <map>
#include <string>
#include <vector>

#include "opencv2/opencv.hpp"

struct xyzPoint
{
    int x;
    int y;
    int z;
};

class Reconstruct3D
{
public:
    std::string CalibrationDataDir;
    std::vector<cv::Mat> LeftIntrinsic;
    std::vector<cv::Mat> RightIntrinsic;
    std::vector<cv::Mat> LeftExtrinsic;
    std::vector<cv::Mat> RightExtrinsic;
    std::vector<cv::Mat> FMatrix;

    std::vector<cv::Point2i> list_l;
    std::vector<cv::Point2i> list_r;

    void readParams();
    void printinfo();
    void checkpoint();
    void exportXYZ();
    void colorizing();
    void exportXYZRGB();   

};