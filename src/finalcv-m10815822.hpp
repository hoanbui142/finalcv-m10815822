#pragma once 

#include <map>
#include <string>
#include <vector>

#include "opencv2/opencv.hpp"

class Reconstruct3D
{
public:
    std::string CalibrationDataDir;
    std::vector<cv::Mat> LeftIntrinsic;
    std::vector<cv::Mat> RightIntrinsic;
    std::vector<cv::Mat> LeftExtrinsic;
    std::vector<cv::Mat> RightExtrinsic;
    std::vector<cv::Mat> FMatrix;

    std::vector<cv::Point2f> list_2d_color;
    std::vector<cv::Point3f> list_3d_color;
    std::vector<cv::Point3f> list_3d;

    void readParams();
    void printinfo();
    void checkpoint();
    void exportXYZ();
    void colorizing();
private:
    bool exist(std::vector<cv::Point3f> a, size_t n, cv::Point3f p) 
    { 
        // Count the same coordinates to pass
        for (size_t i = 0; i < a.size(); i++)
        {
            if (p == a[i])
            {
                return true;
                break;
            }
        }
        return false;
    } 
};