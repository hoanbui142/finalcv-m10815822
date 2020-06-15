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
    //Calculate P1
    cv::Mat K1 = LeftIntrinsic[0].clone();
    cv::Mat Rt1 = LeftExtrinsic[0].clone();
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
    cv::Mat F = FMatrix[0].clone();

    ///////////////////////////////////////////////
    /// Read images to get mask
    ///////////////////////////////////////////////
    std::stringstream ssl;
    std::stringstream ssr;
    for (size_t i = 0;i < 293; i++)
    {
        ssl.str(""); // Clear the string stream
        ssl << "L" << std::setfill('0') << std::setw(3) << i << ".jpg";
        std::cout << ssl.str() << std::endl;
        cv::Mat left_img = cv::imread("L\\" + ssl.str());

        ssr.str("");
        ssr << "R" << std::setfill('0') << std::setw(3) << i << ".jpg";
        cv::Mat right_img = cv::imread("R\\" + ssr.str());
        std::cout << ssr.str() << std::endl;

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
                    //std::cout << "x: " << x << " y: " << y << std::endl;
                }            
            }
        }

        
        for (std::size_t i = 0; i < list_l.size(); i++)
        {
            double datapoint1[] = {double(list_l[i].x),double(list_l[i].y), 1};
            cv::Mat point1 = cv::Mat(1,3, CV_64F,datapoint1);
            cv::Mat point1t = point1.t();
            cv::Mat l = F * point1t;
            //std::cout << l << std::endl;
            //std::cout << "[" << l[0] << ", " << l[1] << ", " << l[2] << "]\n" << std::endl;
            double a = l.at<double>(0,0);
            double b = l.at<double>(1,0);
            double c = l.at<double>(2,0);
            //std::cout << a << ", "<< b << ", "<< c << std::endl;
            
            /*double x0, y0, x01, y01;
            x0 = 0;
            y0 = (-c-a*x0)/b;
            x01 = mask2.cols;
            y01 = (-c-a*x01)/b;
            //std::cout<<"error: "<< a * point1.at<int>(0,0) + b * point1.at<int>(0,1) + c <<std::endl;
            cv::line(test, cv::Point2d(x0,y0), cv::Point2d(x01,y01), cv::Scalar(0,0,255), 1);
            cv::imwrite("rightImageEpipolarLine.jpg",test);
            //cv::imshow("right_img",test);*/

            for (int x = 0; x < mask2.cols; x++)
            {
                for (int y = 0; y < mask2.rows; y++)
                {
                    cv::Vec3b tmp = mask2.at<cv::Vec3b>(y, x);
                    if (std::abs(a*x+b*y+c) < 0.0001)
                    {
                        if (tmp.val[0] > 0 || tmp.val[1] > 0 || tmp.val[2] > 0)
                        {   
                            //std::cout << "x: " << x << " y: " << y << std::endl;
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
                            //std::cout << "A1 = " << A1 << std::endl;
                            //std::cout << "A2 = " << A2 << std::endl;
                            //std::cout << "A3 = " << A3 << std::endl;
                            //std::cout << "A4 = " << A4 << std::endl;

                            cv::Mat A = A1;
                            cv::vconcat(A, A2, A);
                            cv::vconcat(A, A3, A);
                            cv::vconcat(A, A4, A);
                            
                            //std::cout << "A = " << A << std::endl;
                            cv::Mat S, U, V;
                            cv::SVD::compute(A, S, U, V, cv::SVD::FULL_UV);
                            cv::Mat Vt = V.t();
                            //std::cout << "V" << std::endl << Vt << std::endl << std::endl;
                            cv::Mat V_normalize = Vt/Vt.row(3).col(3);
                            //std::cout << "V_normalize" << std::endl << V_normalize << std::endl << std::endl;
                            double X3d = V_normalize.at<double>(0,3);
                            double Y3d = V_normalize.at<double>(1,3);
                            double Z3d = V_normalize.at<double>(2,3);
                            list_3d.push_back(cv::Point3i(int(X3d),int(Y3d),int(Z3d)));                            
                            //std::cout << int(X3d) << " " << int(Y3d) << " " << int(Z3d) << std::endl;
                        }    
                    }
                }
            }
        }
        list_l.clear();
    }
    
}

void Reconstruct3D::exportXYZ()
{
    std::string xyzFileURL("reconstruct3D.xyz");
    std::ofstream xyzFileStream;
    xyzFileStream.open(xyzFileURL);
    for(std::size_t i = 0; i < list_3d.size(); i++)
    {
        xyzFileStream << list_3d[i].x << " " << list_3d[i].y << " " << list_3d[i].z << std::endl;
    }
    xyzFileStream.close();
}

void Reconstruct3D::colorizing()
{
    list_3d_color.push_back(cv::Point3d(9,-55,178));
    list_3d_color.push_back(cv::Point3d(2,-36,174));
    list_3d_color.push_back(cv::Point3d(17,-36,177));
    list_3d_color.push_back(cv::Point3d(9,-4,163));
    list_3d_color.push_back(cv::Point3d(-12,-7,158));
    list_3d_color.push_back(cv::Point3d(10,56,171));
    
    list_2d_color.push_back(cv::Point2d(2074,1112));
    list_2d_color.push_back(cv::Point2d(1967,1455));
    list_2d_color.push_back(cv::Point2d(2191,1452));
    list_2d_color.push_back(cv::Point2d(2044,1539));
    list_2d_color.push_back(cv::Point2d(1626,1976));
    list_2d_color.push_back(cv::Point2d(2039,3455));

    

    cv::Mat texture = cv::imread("Texture.JPG");

    double dataX1[] = {0,0,75,1};
    cv::Mat X1 = cv::Mat(1,4,CV_64F,dataX1);
    double dataX2[] = {0,0,25,1};
    cv::Mat X2 = cv::Mat(1,4,CV_64F,dataX2);
    double dataX3[] = {100,0,25,1};
    cv::Mat X3 = cv::Mat(1,4,CV_64F,dataX3);
    double dataX4[] = {120,90,15,1};
    cv::Mat X4 = cv::Mat(1,4,CV_64F,dataX4);
    double dataX5[] = {90,50,60,1};
    cv::Mat X5 = cv::Mat(1,4,CV_64F,dataX5);
    double dataX6[] = {0,100,25,1};
    cv::Mat X6 = cv::Mat(1,4,CV_64F,dataX6);
    double dataX7[] = {60,40,20,1};
    cv::Mat X7 = cv::Mat(1,4,CV_64F,dataX7);

    double datax1[] = {83,146,1};
    cv::Mat x1 = cv::Mat(1,3,CV_64F,datax1);
    double datax2[] = {103,259,1};
    cv::Mat x2 = cv::Mat(1,3,CV_64F,datax2);
    double datax3[] = {346,315,1};
    cv::Mat x3 = cv::Mat(1,3,CV_64F,datax3);
    double datax4[] = {454,218,1};
    cv::Mat x4 = cv::Mat(1,3,CV_64F,datax4);
    double datax5[] = {365,161,1};
    cv::Mat x5 = cv::Mat(1,3,CV_64F,datax5);
    double datax6[] = {218,144,1};
    cv::Mat x6 = cv::Mat(1,3,CV_64F,datax6);
    double datax7[] = {286,244,1};
    cv::Mat x7 = cv::Mat(1,3,CV_64F,datax7);

    cv::Mat zeros = cv::Mat::zeros(1,4,CV_64F);

    cv::Mat A1 = X1;
    cv::hconcat(A1,zeros,A1);
    cv::hconcat(A1,(-x1.col(0)*X1),A1);
    std::cout << "A1 =" << A1 << std::endl;
    cv::Mat A2 = zeros;
    cv::hconcat(A2,X1,A2);
    cv::hconcat(A2,(-x1.col(1)*X1),A2);
    std::cout << "A2 =" << A2 << std::endl;

    cv::Mat A3 = X2;
    cv::hconcat(A3,zeros,A3);
    cv::hconcat(A3,(-x2.col(0)*X2),A3);
    std::cout << "A3 =" << A3 << std::endl;
    cv::Mat A4 = zeros;
    cv::hconcat(A4,X2,A4);
    cv::hconcat(A4,(-x2.col(1)*X2),A4);
    std::cout << "A4 =" << A4 << std::endl;

    cv::Mat A5 = X3;
    cv::hconcat(A5,zeros,A5);
    cv::hconcat(A5,(-x3.col(0)*X3),A5);
    std::cout << "A5 =" << A5 << std::endl;
    cv::Mat A6 = zeros;
    cv::hconcat(A6,X3,A6);
    cv::hconcat(A6,(-x3.col(1)*X3),A6);
    std::cout << "A6 =" << A6 << std::endl;

    cv::Mat A7 = X4;
    cv::hconcat(A7,zeros,A7);
    cv::hconcat(A7,(-x4.col(0)*X4),A7);
    std::cout << "A7 =" << A7 << std::endl;
    cv::Mat A8 = zeros;
    cv::hconcat(A8,X4,A8);
    cv::hconcat(A8,(-x4.col(1)*X4),A8);
    std::cout << "A8 =" << A8 << std::endl;

    cv::Mat A9 = X5;
    cv::hconcat(A9,zeros,A9);
    cv::hconcat(A9,(-x5.col(0)*X5),A9);
    std::cout << "A9 =" << A9 << std::endl;
    cv::Mat A10 = zeros;
    cv::hconcat(A10,X5,A10);
    cv::hconcat(A10,(-x5.col(1)*X5),A10);
    std::cout << "A10 =" << A10 << std::endl;

    cv::Mat A11 = X6;
    cv::hconcat(A11,zeros,A11);
    cv::hconcat(A11,(-x6.col(0)*X6),A11);
    std::cout << "A11 =" << A11 << std::endl;
    cv::Mat A12 = zeros;
    cv::hconcat(A12,X6,A12);
    cv::hconcat(A12,(-x6.col(1)*X6),A12);
    std::cout << "A12 =" << A12 << std::endl;

    cv::Mat A13 = X7;
    cv::hconcat(A13,zeros,A13);
    cv::hconcat(A13,(-x7.col(0)*X7),A13);
    std::cout << "A13 =" << A13 << std::endl;
    cv::Mat A14 = zeros;
    cv::hconcat(A14,X7,A14);
    cv::hconcat(A14,(-x7.col(1)*X7),A14);
    std::cout << "A14 =" << A14 << std::endl;

    cv::Mat A = A1;
    cv::vconcat(A,A2,A);
    cv::vconcat(A,A3,A);
    cv::vconcat(A,A4,A);
    cv::vconcat(A,A5,A);
    cv::vconcat(A,A6,A);
    cv::vconcat(A,A7,A);
    cv::vconcat(A,A8,A);
    cv::vconcat(A,A9,A);
    cv::vconcat(A,A10,A);
    cv::vconcat(A,A11,A);
    cv::vconcat(A,A12,A);
    cv::vconcat(A,A13,A);
    cv::vconcat(A,A14,A);

    cv::Mat S, U, V;
    cv::SVD::compute(A, S, U, V, cv::SVD::FULL_UV);
    cv::Mat Vt = V.t();
    std::cout << "V" << std::endl << Vt << std::endl << std::endl;
    //cv::Mat V_normalize = Vt/Vt.row(3).col(3);
    //std::cout << "V_normalize" << std::endl << V_normalize << std::endl << std::endl;
    double p1 = Vt.at<double>(0,11);
    double p2 = Vt.at<double>(1,11);
    double p3 = Vt.at<double>(2,11);
    double p4 = Vt.at<double>(3,11);
    double p5 = Vt.at<double>(4,11);
    double p6 = Vt.at<double>(5,11);
    double p7 = Vt.at<double>(6,11);
    double p8 = Vt.at<double>(7,11);
    double p9 = Vt.at<double>(8,11);
    double p10 = Vt.at<double>(9,11);
    double p11 = Vt.at<double>(10,11);
    double p12 = Vt.at<double>(11,11);

    double dataP[] = {p1,p2,p3,p4,
                    p5,p6,p7,p8,
                    p9,p10,p11,p12};
    cv::Mat P = cv::Mat(3,4, CV_64F, dataP);
    cv::Mat P_normalize = P/P.row(2).col(3);
    std::cout << "P= " << std::endl << P_normalize << std::endl << std::endl;
    
    
}
