#include <iostream>
#include <fstream>
#include <sstream>

#include "finalcv-m10815822.hpp"

void Reconstruct3D::readParams()
{
    //Read file CalibarationData.txt

    std::ifstream paramFile("CalibrationData.txt");
    std::string line;
    
    while (std::getline(paramFile, line))
    {
        if (line.find("#Left Camera Intrinsic parameter") != std::string::npos)
        {
            double LeftIntrinsicValue[9];
            for (int i = 0; i < 3; i++)
            {
                std::getline(paramFile, line);
                std::istringstream iss(line);
                iss >> LeftIntrinsicValue[i * 3 + 0] >> LeftIntrinsicValue[i * 3 + 1] >> LeftIntrinsicValue[i * 3 + 2];
            }
            LeftIntrinsic.push_back(cv::Mat(3, 3, CV_64F, LeftIntrinsicValue).clone());
        }
        else if (line.find("#Left Camera Extrinsic parameter") != std::string::npos)
        {
            double LeftExtrinsicValue[12];
            for (int i = 0; i < 3; i++)
            {
                std::getline(paramFile, line);
                std::istringstream iss(line);
                iss >> LeftExtrinsicValue[i * 4 + 0] >> LeftExtrinsicValue[i * 4 + 1] >> LeftExtrinsicValue[i * 4 + 2] >> LeftExtrinsicValue[i * 4 + 3];
            }
            LeftExtrinsic.push_back(cv::Mat(3, 4, CV_64F, LeftExtrinsicValue).clone());
        }
        else if (line.find("#Right Camera Intrinsic parameter") != std::string::npos)
        {
            double RightIntrinsicValue[9];
            for (int i = 0; i < 3; i++)
            {
                std::getline(paramFile, line);
                std::istringstream iss(line);
                iss >> RightIntrinsicValue[i * 3 + 0] >> RightIntrinsicValue[i * 3 + 1] >> RightIntrinsicValue[i * 3 + 2];
            }
            RightIntrinsic.push_back(cv::Mat(3, 3, CV_64F, RightIntrinsicValue).clone());
        }
        else if (line.find("#Right Camera Extrinsic parameter") != std::string::npos)
        {
            double RightExtrinsicValue[12];
            for (int i = 0; i < 3; i++)
            {
                std::getline(paramFile, line);
                std::istringstream iss(line);
                iss >> RightExtrinsicValue[i * 4 + 0] >> RightExtrinsicValue[i * 4 + 1] >> RightExtrinsicValue[i * 4 + 2] >> RightExtrinsicValue[i * 4 + 3];
            }
            RightExtrinsic.push_back(cv::Mat(3, 4, CV_64F, RightExtrinsicValue).clone());
        }
        else if (line.find("#FMatrix") != std::string::npos)
        {
            double FundamentalMatrix[9];
            for (int i = 0; i < 3; i++)
            {
                std::getline(paramFile, line);
                std::istringstream iss(line);
                iss >> FundamentalMatrix[i * 3 + 0] >> FundamentalMatrix[i * 3 + 1] >> FundamentalMatrix[i * 3 + 2];
            }
            FMatrix.push_back(cv::Mat(3,3, CV_64F, FundamentalMatrix).clone());
        }
        else
        {
            continue;
        }
    }
}

void Reconstruct3D::printinfo()
{
    // Information of CalibrationData
    std::cout << "Left Intrinsic matrix :" << std::endl
              << LeftIntrinsic[0] << std::endl;
    std::cout << std::endl;
    std::cout << "Left Extrinsic matrix :" << std::endl
              << LeftExtrinsic[0] << std::endl;
    std::cout << std::endl;
    std::cout << "Right Intrinsic matrix :" << std::endl
              << RightIntrinsic[0] << std::endl;
    std::cout << std::endl;
    std::cout << "Right Extrinsic matrix :" << std::endl
              << RightExtrinsic[0] << std::endl;
    std::cout << std::endl;
    std::cout << "Fundamental matrix :" << std::endl
              << FMatrix[0] << std::endl;
    std::cout << std::endl;
}

void Reconstruct3D::checkpoint()
{
    //Prepare K1 matrix
    cv::Mat K1 = LeftIntrinsic[0].clone();
    //std::cout << K1 << std::endl;

    //Prepare R|t1 matrix
    cv::Mat Rt1 = LeftExtrinsic[0].clone();
    //std::cout << Rt1 << std::endl;

    //Calculate P1
    cv::Mat P1 = K1 * Rt1;
    std::cout << "P1 :" << std::endl
              << P1 << std::endl;
    std::cout << std::endl;

    //Calculate P2
    cv::Mat K2 = RightIntrinsic[0].clone();
    cv::Mat Rt2 = RightExtrinsic[0].clone();
    cv::Mat P2 = K2 * Rt2;

    std::cout << "P2 :" << std::endl
              << P2 << std::endl;
    std::cout << std::endl;


    ///////////////////////////////////////////////
    /// Read images to get mask
    ///////////////////////////////////////////////

    cv::Mat left_img = cv::imread("L000.JPG");
    cv::Mat right_img = cv::imread("R000.JPG");
    double thresh = 50;   //threshold value with 50 is fixed threholding value.
    double maxValue = 255; //maximum value that can be assigned out to the output of threholding
    cv::Mat mask1;
    cv::threshold(left_img, mask1, thresh, maxValue, cv::THRESH_BINARY);
    cv::imwrite("Mask1.jpg", mask1);
    cv::Mat mask2;
    cv::threshold(right_img, mask2, thresh, maxValue, cv::THRESH_BINARY);
    cv::imwrite("Mask2.jpg", mask2);
    cv::Mat test = mask2.clone();
    //cv::imshow("left_img",mask1);
    //cv::imshow("right_img",mask2);
    //cv::imshow("left_img",left_img);
    //cv::imshow("right_img",right_img);

    
    // Scan 2D point of Left Images
    for (int x = 0; x < mask1.cols; x++)
    {
        for (int y = 0; y < mask1.rows; y++)
        {
            cv::Vec3b tmp = mask1.at<cv::Vec3b>(y, x);
            if (tmp.val[0] > 0 || tmp.val[1] > 0 || tmp.val[2] > 0)
            {   
                list_l.push_back(cv::Point2i(x,y));
                std::cout << "x: " << x << " y: " << y << std::endl;
            }            
        }
    }
    

    //std::vector<cv::Vec3d> l;  
    cv::Mat F = FMatrix[0].clone();
    //list_l.push_back(cv::Point2f(174,623));
    //cv::computeCorrespondEpilines(list_l, 1, F, l);
    //cv::Mat x1t = x1.t();
    
    double datapoint1[] = {170, 625, 1};
    cv::Mat point1 = cv::Mat(1,3, CV_64F,datapoint1);
    cv::Mat point1t = point1.t();
    cv::Mat l = F * point1t;
    std::cout << l << std::endl;
    //std::cout << "[" << l[0] << ", " << l[1] << ", " << l[2] << "]\n" << std::endl;
    double a = l.at<double>(0,0);
    double b = l.at<double>(1,0);
    double c = l.at<double>(2,0);
    std::cout << a << ", "<< b << ", "<< c << std::endl;
    
    double x0, y0, x01, y01;
    x0 = 0;
    y0 = (-c-a*x0)/b;
    x01 = mask2.cols;
    y01 = (-c-a*x01)/b;
    //std::cout<<"error: "<< a * point1.at<int>(0,0) + b * point1.at<int>(0,1) + c <<std::endl;
	cv::line(test, cv::Point2d(x0,y0), cv::Point2d(x01,y01), cv::Scalar(0,0,255), 1);
    cv::imwrite("rightImageEpipolarLine.jpg",test);
    //cv::imshow("right_img",test);

    for (int x = 0; x < mask2.cols; x++)
    {
        for (int y = 0; y < mask2.rows; y++)
        {
            cv::Vec3b tmp = mask2.at<cv::Vec3b>(y, x);
            //std::cout << "color:" << tmp[0] << std::endl;
            //cv::Vec3b tmp1 = left_img.at<cv::Vec3b>()

            if (std::abs(a*x+b*y+c) < 0.0001)
            {
                if (tmp.val[0] > 0 || tmp.val[1] > 0 || tmp.val[2] > 0)
                {   
                    std::cout << "x: " << x << " y: " << y << std::endl;
                    double datapoint2[] = {double(x), double(y), 1};
                    cv::Mat point2 = cv::Mat(1,3, CV_64F,datapoint2);
                    cv::Mat p1t = P1.row(0);
                    cv::Mat p2t = P1.row(1);
                    cv::Mat p3t = P1.row(2);
                    cv::Mat pp1t = P2.row(0);
                    cv::Mat pp2t = P2.row(1);
                    cv::Mat pp3t = P2.row(2);

                    cv::Mat A1 = point1.col(0) * p3t - p1t;
                    cv::Mat A2 = point1.col(1) * p3t - p2t;
                    cv::Mat A3 = point2.col(0) * pp3t - pp1t;
                    cv::Mat A4 = point2.col(1) * pp3t - pp2t;
                    std::cout << "A1 = " << A1 << std::endl;
                    std::cout << "A2 = " << A2 << std::endl;
                    std::cout << "A3 = " << A3 << std::endl;
                    std::cout << "A4 = " << A4 << std::endl;

                    cv::Mat A = A1;
                    cv::vconcat(A, A2, A);
                    cv::vconcat(A, A3, A);
                    cv::vconcat(A, A4, A);
                    
                    std::cout << "A = " << A << std::endl;
                    cv::Mat S, U, V;
                    cv::SVD::compute(A, S, U, V, cv::SVD::FULL_UV);
                    cv::Mat Vt = V.t();
                    std::cout << "V" << std::endl << Vt << std::endl << std::endl;
                    cv::Mat V_normalize = Vt/Vt.row(3).col(3);
                    std::cout << "V_normalize" << std::endl << V_normalize << std::endl << std::endl;
                    double X3d = V_normalize.at<double>(0,3);
                    double Y3d = V_normalize.at<double>(1,3);
                    double Z3d = V_normalize.at<double>(2,3);
                }    
            }
        }
    }
    cv::waitKey();
}

void Reconstruct3D::exportXYZ()
{
    std::string xyzFileURL("test.xyz");
    std::ofstream xyzFileStream;
    xyzFileStream.open(xyzFileURL);
    //xyzFileStream << int(X3d) << " " << int(Y3d) << " " << int(Z3d) << std::endl;
    xyzFileStream.close();
}