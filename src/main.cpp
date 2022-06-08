
#include <iostream>
#include <opencv2/opencv.hpp>

#include <pcl/common/common.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/filters/radius_outlier_removal.h>
#include "levmar.h"
#include <pcl/io/ply_io.h>

#include "AutoOiling.h"
#include "CameraController.h"


using namespace std;
using namespace cv;

//Optimal Related
int optm_samples = 0;
int Pos_num = 5;  // Multi_Pos
vector<Mat> Calib_World(Pos_num);

CameraController camera;


bool isCapturingFinish = false;
void callback_single();

void callback_single()
{
    //std::cout << "callback() : I'm called!" << std::endl;
    isCapturingFinish = true;
}


void ResizeShow_two(Mat img1, Mat img2, string WindowName, string message)
{
    Mat dispImg;
    int w, w1, h, h1;
    h = 600;
    h1 = img1.rows;
    w1 = img1.cols;
    w = (w1 * h) / h1;

    dispImg.create(Size(100 + 2 * w, 100 + h), CV_8UC1);

    Mat imgROI = dispImg(Rect(20, 80, w, h));
    resize(img1, imgROI, Size(w, h), 0, 0, CV_INTER_LINEAR);

    imgROI = dispImg(Rect(40 + w, 80, w, h));
    resize(img2, imgROI, Size(w, h), 0, 0, CV_INTER_LINEAR);

    putText(dispImg, message, Point(30, 30), FONT_HERSHEY_SIMPLEX, 1, Scalar(255, 23, 0), 4, 8);

    imshow(WindowName, dispImg);
}



void ResizeShow(cv::Mat img1, std::string WindowName)
{
    namedWindow(WindowName, 0);//´´½¨´°¿Ú
    imshow(WindowName, img1);//ÔÚ´´½¨µÄ´°¿ÚÖÐÏÔÊ¾Í¼Æ¬

}


void viewerOneOff(pcl::visualization::PCLVisualizer& viewer)
{
    viewer.setBackgroundColor(0.5, 0.7, 0.9);
    std::cout << "i only run once" << std::endl;
}


// Levmar Related Function

struct xtradata {
    char msg[128];

};

void expfunc_refsurf_test(double* p, double* x, int m, int n, void* data)

{
    struct xtradata* dat;

    dat = (struct xtradata*)data;

    //p[0]->k  p[1]->a0 
    //p[2],p[3],p[4] -> r1,r2,r3
    //p[5],p[6] -> t1,t2,   (t3=0)
    //p[7],p[8] -> gama,d
    double error_max = 10;
    double single_error = 0;
    int Camera_height = Calib_World[0].rows;
    int Camera_width = Calib_World[0].cols;
    int i = 0;
    for (int h = 0; h < Camera_height; h++)
    {
        for (int w = 0; w < Camera_width; w++)
        {
            for (int Pos_index = 0; Pos_index < Pos_num; Pos_index++)
            {
                float* data_calib = Calib_World[Pos_index].ptr<float>(h);
                if (data_calib[4 * w + 2] > 0 && i < n)
                {
                    float sin_2a = sin(2 * (p[0] * data_calib[4 * w + 3] + p[1]));
                    float cos_2a = cos(2 * (p[0] * data_calib[4 * w + 3] + p[1]));
                    float tan_gama = tan(p[7]);
                    float theta = sqrt(p[2] * p[2] + p[3] * p[3] + p[4] * p[4]);
                    float cos_theta = cos(theta);
                    float sin_theta = sin(theta);

                    float a_11 = cos_theta + p[2] * p[2] * (1 - cos_theta) / theta / theta;
                    float a_12 = p[2] * p[3] * (1 - cos_theta) / theta / theta - p[4] * sin_theta / theta;
                    float a_13 = p[2] * p[4] * (1 - cos_theta) / theta / theta + p[3] * sin_theta / theta;
                    float a_14 = p[5];
                    float a1 = sin_2a * (a_11 * data_calib[4 * w + 0] + a_12 * data_calib[4 * w + 1] + a_13 * data_calib[4 * w + 2] + a_14);

                    float a_21 = p[2] * p[3] * (1 - cos_theta) / theta / theta + p[4] * sin_theta / theta;
                    float a_22 = cos_theta + p[3] * p[3] * (1 - cos_theta) / theta / theta;
                    float a_23 = p[3] * p[4] * (1 - cos_theta) / theta / theta - p[2] * sin_theta / theta;
                    float a_24 = p[6];
                    float a2 = cos_2a * (a_21 * data_calib[4 * w + 0] + a_22 * data_calib[4 * w + 1] + a_23 * data_calib[4 * w + 2] + a_24);

                    float a_31 = p[2] * p[4] * (1 - cos_theta) / theta / theta - p[3] * sin_theta / theta;
                    float a_32 = p[3] * p[4] * (1 - cos_theta) / theta / theta + p[2] * sin_theta / theta;
                    float a_33 = cos_theta + p[4] * p[4] * (1 - cos_theta) / theta / theta;
                    float a_34 = 0;
                    float a3 = tan_gama * (a_31 * data_calib[4 * w + 0] + a_32 * data_calib[4 * w + 1] + a_33 * data_calib[4 * w + 2] + a_34);

                    x[i] = a1 + a2 + a3 - p[8];

                    single_error = x[i] / sqrt((sin_2a * a_11 + cos_2a * a_21 + tan_gama * a_31) * (sin_2a * a_11 + cos_2a * a_21 + tan_gama * a_31) + (sin_2a * a_12 + cos_2a * a_22 + tan_gama * a_32) * (sin_2a * a_12 + cos_2a * a_22 + tan_gama * a_32) + (sin_2a * a_13 + cos_2a * a_23 + tan_gama * a_33) * (sin_2a * a_13 + cos_2a * a_23 + tan_gama * a_33));
                    x[i] = single_error;
                    error_max += abs(single_error);
                    i++;
                }
            }
        }
    }
    error_max = error_max / i;
    cout << "Average Error(mm): " << error_max << endl;
}

float Calculate_error(double* p, float x, float y, float z, float index_line)
{
    float sin_2a = sin(2 * (p[0] * index_line + p[1]));
    float cos_2a = cos(2 * (p[0] * index_line + p[1]));
    float tan_gama = tan(p[7]);
    float theta = sqrt(p[2] * p[2] + p[3] * p[3] + p[4] * p[4]);
    float cos_theta = cos(theta);
    float sin_theta = sin(theta);

    float a_11 = cos_theta + p[2] * p[2] * (1 - cos_theta) / theta / theta;
    float a_12 = p[2] * p[3] * (1 - cos_theta) / theta / theta - p[4] * sin_theta / theta;
    float a_13 = p[2] * p[4] * (1 - cos_theta) / theta / theta + p[3] * sin_theta / theta;
    float a_14 = p[5];
    float a1 = sin_2a * (a_11 * x + a_12 * y + a_13 * z + a_14);

    float a_21 = p[2] * p[3] * (1 - cos_theta) / theta / theta + p[4] * sin_theta / theta;
    float a_22 = cos_theta + p[3] * p[3] * (1 - cos_theta) / theta / theta;
    float a_23 = p[3] * p[4] * (1 - cos_theta) / theta / theta - p[2] * sin_theta / theta;
    float a_24 = p[6];
    float a2 = cos_2a * (a_21 * x + a_22 * y + a_23 * z + a_24);

    float a_31 = p[2] * p[4] * (1 - cos_theta) / theta / theta - p[3] * sin_theta / theta;
    float a_32 = p[3] * p[4] * (1 - cos_theta) / theta / theta + p[2] * sin_theta / theta;
    float a_33 = cos_theta + p[4] * p[4] * (1 - cos_theta) / theta / theta;
    float a_34 = 0;
    float a3 = tan_gama * (a_31 * x + a_32 * y + a_33 * z + a_34);

    float  error = a1 + a2 + a3 - p[8];

    return  error / sqrt((sin_2a * a_11 + cos_2a * a_21 + tan_gama * a_31) * (sin_2a * a_11 + cos_2a * a_21 + tan_gama * a_31) + (sin_2a * a_12 + cos_2a * a_22 + tan_gama * a_32) * (sin_2a * a_12 + cos_2a * a_22 + tan_gama * a_32) + (sin_2a * a_13 + cos_2a * a_23 + tan_gama * a_33) * (sin_2a * a_13 + cos_2a * a_23 + tan_gama * a_33));
}



// int main()     int main_MultiPos()   Version1   Binocular-vision
int main_MultiPos()
{
    AutoOiling autoOiling;

    Mat leftImage, rightImage, L_see, R_see, leftImage_src, rightImage_src;
    string filename = "Calib\\";

    int Camera_width = 2048;
    int Camera_height = 1536;

    leftImage_src = Mat(Camera_height, Camera_width, CV_8UC1, cv::Scalar::all(0));
    rightImage_src = Mat(Camera_height, Camera_width, CV_8UC1, cv::Scalar::all(0));

    Mat img_center_left(Camera_height, Camera_width, CV_8UC1, cv::Scalar::all(0));
    Mat img_center_right(Camera_height, Camera_width, CV_8UC1, cv::Scalar::all(0));

    Mat img_center_left_see(Camera_height, Camera_width, CV_8UC1, cv::Scalar::all(0));
    Mat img_center_right_see(Camera_height, Camera_width, CV_8UC1, cv::Scalar::all(0));

    Mat img_center_left_save(Camera_height, Camera_width, CV_32FC3, cv::Scalar::all(0));

    Mat Laser_left, Laser_widL;
    Mat Laser_right, Laser_widR;
    Mat laser_world;
    Mat laser_world_sum = Mat(Camera_height, Camera_width, CV_32FC3, cv::Scalar::all(0));

    autoOiling.setResolution(Camera_width, Camera_height);

    pcl::visualization::CloudViewer viewer("Simple Cloud Viewer");
    viewer.runOnVisualizationThreadOnce(viewerOneOff);
    double start, end;

    int filter_core = 5; //  Laser width pixel / sqrt(3)

    int gray_temp = 80;// Laser intensity threshold

    autoOiling.CAMERAPARAM = "C:\\Phd_Research\\2021\\0326\\result.yml"; // Camera calibration result

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
    Mat laser_world_sum_PM = Mat(Camera_height, Camera_width, CV_32FC3, cv::Scalar::all(0));

    int start_num = 1;
    int end_num = 90;
    vector<Mat> laser_world_index(Pos_num);
    if (autoOiling.loadCameraParameters())
    {
        int index = 0;
        for (int Pos_index = 0; Pos_index < Pos_num; Pos_index++)
        {
            cout << "Processing Pos" + std::to_string(Pos_index + 1) + " !" << endl;
            laser_world_index[Pos_index] = Mat(Camera_height, Camera_width, CV_32FC4, cv::Scalar::all(0));
            for (int i = start_num; i < end_num; i += 1)
            {
                leftImage = imread("C:\\Phd_Research\\2021\\0326\\" + filename + std::to_string(Pos_index + 1) + "\\left\\" + std::to_string(i) + ".bmp", 0);
                rightImage = imread("C:\\Phd_Research\\2021\\0326\\" + filename + std::to_string(Pos_index + 1) + "\\right\\" + std::to_string(i) + ".bmp", 0);
                leftImage_src = imread("C:\\Phd_Research\\2021\\0326\\" + filename + std::to_string(Pos_index + 1) + "\\left\\" + std::to_string(0) + ".bmp", 0);
                leftImage_src = imread("C:\\Phd_Research\\2021\\0326\\" + filename + std::to_string(Pos_index + 1) + "\\right\\" + std::to_string(0) + ".bmp", 0);

                leftImage = leftImage - leftImage_src;
                rightImage = rightImage - rightImage_src;

                autoOiling.rectifyImages(leftImage, rightImage);
                leftImage = autoOiling.leftRectifiedImage.clone();
                rightImage = autoOiling.rightRectifiedImage.clone();

                leftImage.convertTo(leftImage, CV_32FC1);
                rightImage.convertTo(rightImage, CV_32FC1);

                img_center_left = Mat(Camera_height, Camera_width, CV_8UC1, cv::Scalar::all(0));
                img_center_right = Mat(Camera_height, Camera_width, CV_8UC1, cv::Scalar::all(0));
                Mat temp = leftImage.clone();
                GaussianBlur(leftImage, leftImage, Size(0, 0), filter_core, filter_core);
                autoOiling.Detect_line_center(leftImage, img_center_left, gray_temp);
                GaussianBlur(rightImage, rightImage, Size(0, 0), filter_core, filter_core);
                autoOiling.Detect_line_center(rightImage, img_center_right, gray_temp);

                for (int h = 0; h < Camera_height; h++)
                {
                    uchar* data_mask = img_center_left.ptr<uchar>(h);
                    float* data_save = img_center_left_save.ptr<float>(h);
                    for (int w = 0; w < Camera_width; w++)
                    {
                        if (data_mask[w] > 0)
                        {
                            data_save[3 * w] = 10;
                            data_save[3 * w + 1] = i;
                            data_save[3 * w + 2] += 1;
                        }
                    }
                }


                autoOiling.Acquire_obauque(leftImage, img_center_left, Laser_left, Laser_widL);
                autoOiling.Acquire_obauque(rightImage, img_center_right, Laser_right, Laser_widR);

                laser_world = Mat(Camera_height, Camera_width, CV_32FC3, cv::Scalar::all(0));
                autoOiling.reconstruct_frontback_laser(Laser_left, Laser_widL, Laser_right, Laser_widR, laser_world);
                autoOiling.Add_laser_world(laser_world_sum, laser_world);

                img_center_left_see = img_center_left_see + img_center_left;
                img_center_right_see = img_center_right_see + img_center_right;

                ResizeShow_two(img_center_left_see, img_center_right_see, "Line_dectect", "Line");
                waitKey(10);
                cout << i << endl;
                index += 1;

            }


            autoOiling.PM_filter(laser_world_sum, laser_world_sum_PM, 5, 10);
            autoOiling.convert2Pointcloud_new(laser_world_sum_PM, cloud, 0);
            viewer.showCloud(cloud);
            for (int h = 0; h < Camera_height; h++)
            {
                float* data = laser_world_sum_PM.ptr<float>(h);
                float* data_save = img_center_left_save.ptr<float>(h);
                float* data_world_index = laser_world_index[Pos_index].ptr<float>(h);
                for (int w = 0; w < Camera_width; w++)
                {
                    if (data_save[3 * w] > 0 && data_save[3 * w + 2] < 2 && data[3 * w + 2] > 0)
                    {
                        data_world_index[4 * w + 0] = data[3 * w + 0];
                        data_world_index[4 * w + 1] = data[3 * w + 1];
                        data_world_index[4 * w + 2] = data[3 * w + 2];
                        data_world_index[4 * w + 3] = data_save[3 * w + 1];
                    }

                }
            }

            //Initial Parameters
            img_center_left_see = Mat(Camera_height, Camera_width, CV_8UC1, cv::Scalar::all(0));
            img_center_right_see = Mat(Camera_height, Camera_width, CV_8UC1, cv::Scalar::all(0));
            laser_world_sum_PM = Mat(Camera_height, Camera_width, CV_32FC3, cv::Scalar::all(0));
            laser_world_sum = Mat(Camera_height, Camera_width, CV_32FC3, cv::Scalar::all(0));
            img_center_left_save = Mat(Camera_height, Camera_width, CV_32FC3, cv::Scalar::all(0));

        }


    }
    else
    {
        std::cout << "load parameter failed" << std::endl;
    }


    //Save Data ->  Multi_Plane PointCloud
    FileStorage fs("Calib_Multi.xml", FileStorage::WRITE);
    for (int Pos_index = 0; Pos_index < Pos_num; Pos_index++)
    {
        fs << "Calib" + to_string(Pos_index) << laser_world_index[Pos_index];
    }
    fs.release();
    cout << "Save Success!" << endl;


    system("pause");
    return 0;

}



//int main()     int main_Optimal_MultiPos()
int main_Optimal_MultiPos()
{
    pcl::visualization::CloudViewer viewer("Simple Cloud Viewer");
    viewer.runOnVisualizationThreadOnce(viewerOneOff);
    vector<Mat> laser_world_index(Pos_num);
    FileStorage fs("Calib_Multi.xml", FileStorage::READ);
    for (int Pos_index = 0; Pos_index < Pos_num; Pos_index++)
    {
        fs["Calib" + to_string(Pos_index)] >> laser_world_index[Pos_index];
    }
    fs.release();
    cout << "Load Success!" << endl;

    AutoOiling Optimal;

    int Camera_width = laser_world_index[0].cols;
    int Camera_height = laser_world_index[0].rows;

    float plane[4] = { 0 };
    std::vector <float> X_vector, Y_vector, Z_vector;
    CvMat* points_mat;

    // Fit Ax+By+Cz=D;
    // Construct Plane
    int start_num = 1;
    int end_num = 90;
    Mat plane_all(end_num, 1, CV_32FC4, cv::Scalar::all(0));

    for (int i = start_num; i < end_num; i++)
    {
        plane[0] = 0;
        plane[1] = 0;
        plane[2] = 0;
        plane[3] = 0;
        float* data_plane = plane_all.ptr<float>(i);
        for (int Pos_index = 0; Pos_index < Pos_num; Pos_index++)
        {
            for (int h = 0; h < Camera_height; h++)
            {
                float* data = laser_world_index[Pos_index].ptr<float>(h);
                for (int w = 0; w < Camera_width; w++)
                {
                    if (abs(data[4 * w + 3] - i) < 0.1)
                    {
                        X_vector.push_back(data[4 * w + 0]);
                        Y_vector.push_back(data[4 * w + 1]);
                        Z_vector.push_back(data[4 * w + 2]);
                    }

                }
            }
        }
        points_mat = cvCreateMat(X_vector.size(), 3, CV_32FC1);
        if (X_vector.size() < 500)
        {
            X_vector.clear();
            Y_vector.clear();
            Z_vector.clear();
            //cout << i << endl;
            //cout << "No Fit!"<< endl;
            continue;
        }
        for (int m = 0; m < X_vector.size(); ++m)
        {
            points_mat->data.fl[m * 3 + 0] = X_vector[m];
            points_mat->data.fl[m * 3 + 1] = Y_vector[m];
            points_mat->data.fl[m * 3 + 2] = Z_vector[m];

        }
        Optimal.cvFitPlane(points_mat, plane);
        data_plane[0] = plane[0];
        data_plane[1] = plane[1];
        data_plane[2] = plane[2];
        data_plane[3] = plane[3];
        cout << i << endl;
        cout << X_vector.size() << endl;
        X_vector.clear();
        Y_vector.clear();
        Z_vector.clear();
    }

    //Intersect to Z axis
    Mat z_direc_all(end_num, 1, CV_32FC3, cv::Scalar::all(0));
    int step = (end_num - start_num) / 2;
    int z_samples = 0;
    Point3f z_direc(0, 0, 0), z_point(0, 0, 0), galvo_orgin, camera_center(0, 0, 0);
    for (int i = start_num; i < end_num - step; i++)
    {
        float* data_plane1 = plane_all.ptr<float>(i);
        float* data_plane2 = plane_all.ptr<float>(i + step);
        float* data_z_direc = z_direc_all.ptr<float>(i);

        data_z_direc[0] = data_plane1[1] * data_plane2[2] - data_plane1[2] * data_plane2[1];
        data_z_direc[1] = data_plane1[2] * data_plane2[0] - data_plane1[0] * data_plane2[2];
        data_z_direc[2] = data_plane1[0] * data_plane2[1] - data_plane1[1] * data_plane2[0];
    }

    for (int i = start_num; i < end_num - step; i++)
    {
        float* data_z_direc = z_direc_all.ptr<float>(i);
        z_direc.x += data_z_direc[0];
        z_direc.y += data_z_direc[1];
        z_direc.z += data_z_direc[2];
        z_samples += 1;
    }
    z_direc.x = z_direc.x / z_samples;
    z_direc.y = z_direc.y / z_samples;
    z_direc.z = z_direc.z / z_samples;
    z_direc = z_direc / norm(z_direc);



    Mat Eq_left(end_num, 3, CV_32FC1, cv::Scalar::all(0));
    Mat Eq_right(end_num, 1, CV_32FC1, cv::Scalar::all(0));
    Mat result(1, 3, CV_32FC1, cv::Scalar::all(0));
    for (int i = start_num; i < end_num; i++)
    {
        float* data_plane = plane_all.ptr<float>(i);
        float* data_temp = Eq_left.ptr<float>(i);
        float* data_right = Eq_right.ptr<float>(i);
        data_temp[0] = data_plane[0];
        data_temp[1] = data_plane[1];
        data_temp[2] = data_plane[2];
        data_right[0] = -data_plane[3];
    }


    solve(Eq_left, Eq_right, result, CV_SVD);
    float* data_z_point = result.ptr<float>(0);
    z_point.x = data_z_point[0];
    z_point.y = data_z_point[1];
    z_point.z = data_z_point[2];
    galvo_orgin = Optimal.getFootPoint(camera_center, z_point, z_point + z_direc);

    vector<Point3f> camera_points, galvo_points;
    Point3f x_direc = galvo_orgin - camera_center;
    x_direc = x_direc / norm(x_direc);

    camera_points.push_back(camera_center);
    galvo_points.push_back(galvo_orgin);

    camera_points.push_back(camera_center + Point3f(1, 0, 0));
    galvo_points.push_back(galvo_orgin + Point3f(x_direc.x, x_direc.y, x_direc.z));
    camera_points.push_back(camera_center + Point3f(2, 0, 0));
    galvo_points.push_back(galvo_orgin + Point3f(2 * x_direc.x, 2 * x_direc.y, 2 * x_direc.z));

    camera_points.push_back(camera_center + Point3f(0, 0, 1));
    galvo_points.push_back(galvo_orgin + Point3f(z_direc.x, z_direc.y, z_direc.z));

    camera_points.push_back(camera_center + Point3f(0, 0, 2));
    galvo_points.push_back(galvo_orgin + Point3f(2 * z_direc.x, 2 * z_direc.y, 2 * z_direc.z));
    Mat RT_coor(4, 4, CV_64FC1);

    RT_coor = Optimal.Get3DR_TransMatrix(camera_points, galvo_points).inv();
    //RT_coor = Optimal.Get3DR_TransMatrix(camera_points, galvo_points);
    Mat r_vec;
    //RT_coor = Optimal.Get3DR_TransMatrix(galvo_points, camera_points);
    for (int r = 0; r < RT_coor.rows; r++)
    {
        for (int c = 0; c < RT_coor.cols; c++)
        {
            printf("%f, ", RT_coor.at<double>(r, c));
        }
        printf("\n");
    }
    Mat R_init(3, 3, CV_64FC1), T_init(3, 1, CV_64FC1);
    for (int i = 0; i < 3; i++)
    {
        for (int j = 0; j < 3; j++)
        {
            R_init.at<double>(i, j) = RT_coor.at<double>(i, j);
        }
    }
    for (int i = 0; i < 3; i++)
    {
        T_init.at<double>(i, 0) = RT_coor.at<double>(i, 3);
    }



    Rodrigues(R_init, r_vec);
    cout << r_vec << endl;



    // Levmar  Optimal
    const int  m = 9; // optm_samples measurements, m parameters
   // double p[m], x[optm_samples], opts[LM_OPTS_SZ], info[LM_INFO_SZ];
    double p[m], opts[LM_OPTS_SZ], info[LM_INFO_SZ];



    vector<Mat> test_see(Pos_num);
    for (int Pos_index = 0; Pos_index < Pos_num; Pos_index++)
    {
        test_see[Pos_index] = Mat(Camera_height, Camera_width, CV_32FC3, cv::Scalar::all(0));
        Calib_World[Pos_index] = Mat(Camera_height, Camera_width, CV_32FC4, cv::Scalar::all(0));
    }



    int index_num = 0;
    for (int h = 0; h < Camera_height; h++)
    {
        for (int w = 0; w < Camera_width; w++)
        {
            for (int Pos_index = 0; Pos_index < Pos_num; Pos_index++)
            {
                float* data_world_index = laser_world_index[Pos_index].ptr<float>(h);
                float* data_test = test_see[Pos_index].ptr<float>(h);
                float* data_calib = Calib_World[Pos_index].ptr<float>(h);
                if ((data_world_index[4 * w + 2] > 0)) //&& (index_num < optm_samples))
                {
                    data_test[3 * w + 0] = data_world_index[4 * w + 0];
                    data_test[3 * w + 1] = data_world_index[4 * w + 1];
                    data_test[3 * w + 2] = data_world_index[4 * w + 2];

                    data_calib[4 * w + 0] = data_world_index[4 * w + 0];
                    data_calib[4 * w + 1] = data_world_index[4 * w + 1];
                    data_calib[4 * w + 2] = data_world_index[4 * w + 2];
                    data_calib[4 * w + 3] = data_world_index[4 * w + 3];
                    index_num++;
                    optm_samples++;
                }
            }
        }
    }

    double* x = new double[optm_samples];
    // Data Input
    for (int i = 0; i < optm_samples; i++)
    {
        x[i] = 0;
    }

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
    for (int Pos_index = 0; Pos_index < Pos_num; Pos_index++)
    {
        Optimal.convert2Pointcloud_new(test_see[Pos_index], cloud, 0);
        viewer.showCloud(cloud);
    }



    int ret;

    struct xtradata data;


    /*initial parameters estimate: */
    double M_PI_design = 3.141592653;
    //p[0]->k  p[1]->a0 
    //p[2],p[3],p[4] -> r1,r2,r3
    //p[5],p[6] -> t1,t2,   (t3=0)
    //p[7],p[8] -> gama,d  
    p[0] = -0.00277; p[1] = 0.7199;
    p[2] = r_vec.at<double>(0, 0); p[3] = r_vec.at<double>(1, 0);
    p[4] = r_vec.at<double>(2, 0); p[5] = T_init.at<double>(0, 0); p[6] = T_init.at<double>(1, 0); p[7] = 0.0;
    p[8] = 0.0;

    /*optimization control parameters; passing to levmar NULL instead of opts revertsto defaults */

    opts[0] = LM_INIT_MU; opts[1] = 1E-15; opts[2] = 1E-15; opts[3] = 1E-20;

    opts[4] = LM_DIFF_DELTA; // relevant only if the finite difference Jacobian version isused

    printf("Before parameters: %.7g %.7g %.7g %.7g\n", p[0], p[1], p[2], p[3]);
    printf("Before parameters: %.7g %.7g %.7g %.7g\n", p[4], p[5], p[6], p[7]);
    printf("Before parameters: %.7g \n", p[8]);




    strcpy_s(data.msg, "Hello there!");


    ret = dlevmar_dif(expfunc_refsurf_test, p, x, m, optm_samples, 5000, opts, info, NULL, NULL, (void*)&data); //withoutJacobian

    printf("Levenberg-Marquardtreturned in %g iter, reason %g\n", info[5], info[6]);
    printf("Bestfit parameters: %.7g %.7g %.7g %.7g\n", p[0], p[1], p[2], p[3]);
    printf("Bestfit parameters: %.7g %.7g %.7g %.7g\n", p[4], p[5], p[6], p[7]);
    printf("Before parameters: %.7g \n", p[8]);

    //Error Estimate
    vector<Mat> Error_calib(Pos_num);
    for (int Pos_index = 0; Pos_index < Pos_num; Pos_index++)
    {
        Error_calib[Pos_index] = Mat(Camera_height, Camera_width, CV_32FC1, cv::Scalar::all(0));

    }
    for (int h = 0; h < Camera_height; h++)
    {
        for (int w = 0; w < Camera_width; w++)
        {
            for (int Pos_index = 0; Pos_index < Pos_num; Pos_index++)
            {
                float* data_world_index = laser_world_index[Pos_index].ptr<float>(h);
                float* data_error = Error_calib[Pos_index].ptr<float>(h);
                if ((data_world_index[4 * w + 2] > 0))
                {
                    data_error[w] = abs(Calculate_error(p, data_world_index[4 * w + 0], data_world_index[4 * w + 1], data_world_index[4 * w + 2], data_world_index[4 * w + 3]));
                }


            }

        }
    }

    FileStorage fs1("Calib_Ref.xml", FileStorage::WRITE);
    Mat Ref_p(9, 1, CV_32FC1, Scalar::all(0));
    for (int i = 0; i < 9; i++)
    {
        Ref_p.at<float>(i, 0) = p[i];
    }
    fs1 << "Ref_p" << Ref_p;
    fs1.release();



    system("pause");
    return 0;

}


// int main()     int main_TransRecon()   
int main_TransRecon()
{

    AutoOiling autoOiling;
    Mat laser_world_indexF, laser_world_indexB;

    Mat leftImage, rightImage, L_see, R_see, leftImage_src, rightImage_src;

    int Camera_width = 2048;
    int Camera_height = 1536;

    leftImage_src = Mat(Camera_height, Camera_width, CV_8UC1, cv::Scalar::all(0));
    rightImage_src = Mat(Camera_height, Camera_width, CV_8UC1, cv::Scalar::all(0));
    Mat left_high = leftImage_src.clone();
    Mat right_high = rightImage_src.clone();


    Mat img_center_left(Camera_height, Camera_width, CV_8UC1, cv::Scalar::all(0));
    Mat img_center_right(Camera_height, Camera_width, CV_8UC1, cv::Scalar::all(0));

    Mat img_center_left_see(Camera_height, Camera_width, CV_8UC1, cv::Scalar::all(0));
    Mat img_center_right_see(Camera_height, Camera_width, CV_8UC1, cv::Scalar::all(0));

    Mat img_center_left_save(Camera_height, Camera_width, CV_32FC3, cv::Scalar::all(0));

    Mat Laser_left, Laser_widL;
    Mat Laser_right, Laser_widR;
    Mat laser_world, laser_worldR;
    Mat laser_world_sumF = Mat(Camera_height, Camera_width, CV_32FC3, cv::Scalar::all(0));
    Mat laser_world_sumFR = Mat(Camera_height, Camera_width, CV_32FC3, cv::Scalar::all(0));
    Mat laser_world_sumB = Mat(Camera_height, Camera_width, CV_32FC3, cv::Scalar::all(0));
    Mat laser_world_sumBR = Mat(Camera_height, Camera_width, CV_32FC3, cv::Scalar::all(0));

    Mat laser_world_sum_PMF = Mat(Camera_height, Camera_width, CV_32FC3, cv::Scalar::all(0));
    Mat laser_world_sum_PMFR(Camera_height, Camera_width, CV_32FC3, cv::Scalar::all(0));
    Mat laser_world_sum_PMB = Mat(Camera_height, Camera_width, CV_32FC3, cv::Scalar::all(0));
    Mat laser_world_sum_PMBR(Camera_height, Camera_width, CV_32FC3, cv::Scalar::all(0));



    // Laser Discriminate Parameters
    vector<Mat> laser_pre(2), laser_preR(2);
    Mat laser_indentify, laser_indentifyR;


    autoOiling.setResolution(Camera_width, Camera_height);

    //pcl::visualization::CloudViewer viewer("Simple Cloud Viewer");
    //viewer.runOnVisualizationThreadOnce(viewerOneOff);
    boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer(new pcl::visualization::PCLVisualizer("3D Viewer"));   //设置视窗的背景色，可以任意设置RGB的颜色，这里是设置为黑色
    viewer->setBackgroundColor(0.5, 0.7, 0.9);
    pcl::visualization::Camera Camera_pos;
    viewer->initCameraParameters();
    Camera_pos.pos[0] = -245.900901;
    Camera_pos.pos[1] = 16.432728;
    Camera_pos.pos[2] = 224.284610;
    Camera_pos.view[0] = 0.071219;
    Camera_pos.view[1] = -0.995488;
    Camera_pos.view[2] = 0.062706;
    Camera_pos.focal[0] = -255.726440;
    Camera_pos.focal[1] = 24.640686;
    Camera_pos.focal[2] = 365.749386;
    viewer->loadCameraParameters("a.cam");


    double start, end;
    int start_num = 91;
    int end_num = 180;

    int filter_core = 5; // < w/sqrt(3)
    int gray_temp = 20;// +20;
    int thr_horiz = 2;
    float thr_dual = 5; //unit : mm
    string filename = "Calib\\Ele";
    string data_str = "0517";
    autoOiling.CAMERAPARAM = "C:\\Phd_Research\\2021\\" + data_str + "\\result.yml";

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_Filter(new pcl::PointCloud<pcl::PointXYZRGB>);

    //Test
    Mat laser_world_fuse = Mat(Camera_height, Camera_width, CV_32FC3, cv::Scalar::all(0));
    Mat laser_world_left = Mat(Camera_height, Camera_width, CV_32FC3, cv::Scalar::all(0));
    Mat laser_world_right = Mat(Camera_height, Camera_width, CV_32FC3, cv::Scalar::all(0));
    Mat laser_world_fuse_PM = Mat(Camera_height, Camera_width, CV_32FC3, cv::Scalar::all(0));

    Mat laser_L2R = Mat(Camera_height, Camera_width, CV_32FC1, cv::Scalar::all(0));
    Mat laser_L2R_table = Mat(Camera_height, Camera_width, CV_32FC2, cv::Scalar::all(0));


    Mat Match_test1, Match_test2, laser_world_filter, laser_world_filterR;

    Mat laser_world_all = Mat(Camera_height, Camera_width, CV_32FC4, cv::Scalar::all(0));

    double start1, end1, during1 = 0, start2, end2, during2 = 0;

    leftImage_src = imread("C:\\Phd_Research\\2021\\" + data_str + "\\" + filename + "\\left\\0.bmp", 0);
    rightImage_src = imread("C:\\Phd_Research\\2021\\" + data_str + "\\" + filename + "\\right\\0.bmp", 0);
    if (autoOiling.loadCameraParameters())
    {
        start2 = clock();
        int points_sum = 0, points_single = 0;
        for (int i = start_num; i < end_num; i += 1)
        {
            start1 = clock();
            leftImage = imread("C:\\Phd_Research\\2021\\" + data_str + "\\" + filename + "\\left\\" + std::to_string(i) + ".bmp", 0);
            rightImage = imread("C:\\Phd_Research\\2021\\" + data_str + "\\" + filename + "\\right\\" + std::to_string(i) + ".bmp", 0);
            end1 = clock();

            during1 += (end1 - start1);


            cout << "Image Reading: " << during1 << "  ms!" << endl;
            leftImage = leftImage - leftImage_src;
            rightImage = rightImage - rightImage_src;

            autoOiling.rectifyImages(leftImage, rightImage);
            leftImage = autoOiling.leftRectifiedImage.clone();
            rightImage = autoOiling.rightRectifiedImage.clone();

            leftImage.convertTo(leftImage, CV_32FC1);
            rightImage.convertTo(rightImage, CV_32FC1);


            //*********************************************** Left Camera Process *********************************************************************//
            laser_world = Mat(Camera_height, Camera_width, CV_32FC3, cv::Scalar::all(0));
            autoOiling.intersect_Ref_nolimit(leftImage, laser_world, float(i), 0, gray_temp);
            //laser_world_filter = laser_world.clone();
            autoOiling.Distinguish_Confident3D(laser_world, laser_world, laser_world_filter, rightImage, thr_horiz, gray_temp, 0);
            //Mat laser_world_filter_PM;
            //autoOiling.PM_filter(laser_world_filter, laser_world_filter_PM, 5 ,10);
            if (i == 92)
            {
                cout << "Test" << endl;
            }
            for (int h = 0; h < Camera_height; h++)
            {
                float* data_3D_filter = laser_world_filter.ptr<float>(h);
                float WZ_possible = 1300;
                int w_possible = 0;
                for (int w = 0; w < Camera_width; w++)
                {
                    if (data_3D_filter[3 * w + 2] < WZ_possible && data_3D_filter[3 * w + 2] > 100)
                    {
                        WZ_possible = data_3D_filter[3 * w + 2];
                        data_3D_filter[3 * w_possible + 0] = 0;
                        data_3D_filter[3 * w_possible + 1] = 0;
                        data_3D_filter[3 * w_possible + 2] = 0;
                        w_possible = w;
                    }
                    else
                    {
                        data_3D_filter[3 * w + 0] = 0;
                        data_3D_filter[3 * w + 1] = 0;
                        data_3D_filter[3 * w + 2] = 0;
                    }
                }
            }



            //points_single = autoOiling.convert2Pointcloud_new(laser_world_filter, cloud,2);
            // Fuse Function
            for (int h = 0; h < Camera_height; h++)
            {
                float* data_3D_filter = laser_world_filter.ptr<float>(h);
                float* data_3D_all = laser_world_left.ptr<float>(h);
                float* data_3D_fuse = laser_world_fuse.ptr<float>(h);
                for (int w = 0; w < Camera_width; w++)
                {
                    if (data_3D_filter[3 * w + 2] > 100 && data_3D_filter[3 * w + 2] < 1400 && data_3D_all[3 * w + 2] == 0)
                    {
                        data_3D_all[3 * w + 0] = data_3D_filter[3 * w + 0];
                        data_3D_all[3 * w + 1] = data_3D_filter[3 * w + 1];
                        data_3D_all[3 * w + 2] = data_3D_filter[3 * w + 2];

                        data_3D_fuse[3 * w + 0] = data_3D_filter[3 * w + 0];
                        data_3D_fuse[3 * w + 1] = data_3D_filter[3 * w + 1];
                        data_3D_fuse[3 * w + 2] = data_3D_filter[3 * w + 2];
                    }
                }
            }
            //points_sum += points_single;


            //*********************************************** Right Camera Process *********************************************************************//
            laser_worldR = Mat(Camera_height, Camera_width, CV_32FC3, cv::Scalar::all(0));
            autoOiling.intersect_Ref_nolimit(rightImage, laser_worldR, float(i), 1, gray_temp);
            autoOiling.Distinguish_Confident3D(laser_worldR, laser_world_filter, laser_world_filterR, leftImage, thr_horiz, gray_temp, 1);
            //Mat laser_world_filter_PMR;
            //autoOiling.PM_filter(laser_world_filterR, laser_world_filter_PMR, 3, 10);

            for (int h = 0; h < Camera_height; h++)
            {
                float* data_3D_filter = laser_world_filterR.ptr<float>(h);
                float WZ_possible = 1300;
                int w_possible = 0;
                for (int w = 0; w < Camera_width; w++)
                {
                    if (data_3D_filter[3 * w + 2] < WZ_possible && data_3D_filter[3 * w + 2] > 100)
                    {
                        WZ_possible = data_3D_filter[3 * w + 2];
                        data_3D_filter[3 * w_possible + 0] = 0;
                        data_3D_filter[3 * w_possible + 1] = 0;
                        data_3D_filter[3 * w_possible + 2] = 0;
                        w_possible = w;
                    }
                    else
                    {
                        data_3D_filter[3 * w + 0] = 0;
                        data_3D_filter[3 * w + 1] = 0;
                        data_3D_filter[3 * w + 2] = 0;
                    }
                }
            }

            points_single = autoOiling.convert2Pointcloud_new(laser_world_filter, cloud, 2);
            points_sum += points_single;
            points_single = autoOiling.convert2Pointcloud_new(laser_world_filterR, cloud, 2);
            points_sum += points_single;




            // Fuse Function
            Mat R2L_compensate = cv::Mat(Camera_height, Camera_width, CV_32FC2, cv::Scalar::all(0));
            autoOiling.Points_3D_Camera(laser_world_filterR, 0, R2L_compensate);
            for (int h = 0; h < Camera_height; h++)
            {
                float* data_3D_filter = laser_world_filterR.ptr<float>(h);
                float* data_R2L_compensate = R2L_compensate.ptr<float>(h);
                float* data_3D_allR = laser_world_right.ptr<float>(h);
                float* data_3D_fuse = laser_world_fuse.ptr<float>(h);

                for (int w = 0; w < Camera_width; w++)
                {
                    int x = round(data_R2L_compensate[2 * w]);
                    int y = round(data_R2L_compensate[2 * w + 1]);
                    if (data_3D_filter[3 * w + 2] > 100 && data_3D_filter[3 * w + 2] < 1400 && data_3D_fuse[3 * x + 2] == 0)
                    {
                        data_3D_allR[3 * x + 0] = data_3D_filter[3 * w + 0];
                        data_3D_allR[3 * x + 1] = data_3D_filter[3 * w + 1];
                        data_3D_allR[3 * x + 2] = data_3D_filter[3 * w + 2];
                        data_3D_fuse[3 * x + 0] = data_3D_filter[3 * w + 0];
                        data_3D_fuse[3 * x + 1] = data_3D_filter[3 * w + 1];
                        data_3D_fuse[3 * x + 2] = data_3D_filter[3 * w + 2];
                    }
                }
            }
            //*********************************************** Display Point Cloud *********************************************************************//
            if (i > 1)
            {
                viewer->removePointCloud(to_string(i - 1));
            }
            viewer->addPointCloud<pcl::PointXYZRGB>(cloud, to_string(i));

            viewer->spinOnce(5);
            viewer->getCameraParameters(Camera_pos);
            printf("%lf,%lf,%lf,", Camera_pos.pos[0], Camera_pos.pos[1], Camera_pos.pos[2]);
            printf("%lf,%lf,%lf,", Camera_pos.view[0], Camera_pos.view[1], Camera_pos.view[2]);
            printf("%lf,%lf,%lf\n", Camera_pos.focal[0], Camera_pos.focal[1], Camera_pos.focal[2]);
            waitKey(10);
            cout << i << endl;

        }
        cout << "Number:" << points_sum << endl;
        end2 = clock();
        cout << "Alogirithm Processing: " << end2 - start2 - during1 << "  ms!" << endl;

        pcl::RadiusOutlierRemoval<pcl::PointXYZRGB> pcFilter;  //创建滤波器对象
        pcFilter.setInputCloud(cloud);             //设置待滤波的点云
        pcFilter.setRadiusSearch(5);               // 设置搜索半径
        pcFilter.setMinNeighborsInRadius(50);      // 设置一个内点最少的邻居数目
        pcFilter.filter(*cloud_Filter);        //滤波结果存储到cloud_filtered

        start2 = clock();
        cout << "PCL Filter: " << start2 - end2 << "  ms!" << endl;
        // Cloud Filter Display
        viewer->removePointCloud(to_string(end_num - 1));
        //autoOiling.PM_filter(laser_world_fuse, laser_world_fuse_PM);
        //autoOiling.convert2Pointcloud_new(laser_world_fuse_PM, cloud_Filter, 2);
        viewer->addPointCloud<pcl::PointXYZRGB>(cloud_Filter, to_string(end_num));




    }

    while (!viewer->wasStopped())
    {
        viewer->spinOnce(5);
        viewer->getCameraParameters(Camera_pos);
        printf("%lf,%lf,%lf,", Camera_pos.pos[0], Camera_pos.pos[1], Camera_pos.pos[2]);
        printf("%lf,%lf,%lf,", Camera_pos.view[0], Camera_pos.view[1], Camera_pos.view[2]);
        printf("%lf,%lf,%lf\n", Camera_pos.focal[0], Camera_pos.focal[1], Camera_pos.focal[2]);
        cout << "PCL Filter: " << start2 - end2 << "  ms!" << endl;
    }
    //pcl::io::savePLYFile("rabbit.ply", *cloud);

    cout << "Finish !" << endl;

    system("pause");

    exit(0);

}


// int main()     int main_CandiateExtract()   
int main()
{
    AutoOiling Trans_Process;


    //**************************************Parameters Setting(Start)******************************************/
    int filter_core = 3;
    int gray_temp = 4;   // lowest intensity
    float Depth_start = 500;  // DOF start
    float Depth_end = 800;  // DOF end
    int   thr_search = 20;
    float thr_joint = 5;    // unit: pixel
    int Camera_width = 2048;
    int Camera_height = 1536;
    int img_start = 1;
    int img_end = 180;
    Trans_Process.setResolution(Camera_width, Camera_height);
    string filename = "Calib\\Cup2";
    string data_str = "0517";
    Trans_Process.CAMERAPARAM = "C:\\Phd_Research\\2021\\" + data_str + "\\result.yml";
    //**************************************Parameters Setting(End)******************************************/


    //**************************************Middle Parameters (Start)******************************************/
    Mat leftImage, rightImage;
    Mat GaussL, GaussR, CentreL, CentreR;;
    Mat laser_world, laser_world_jointL, laser_world_jointR, laser_world_firstL, laser_world_firstR;
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_Filter(new pcl::PointCloud<pcl::PointXYZRGB>);
    //**************************************Middle Parameters (End)******************************************/

    //**************************************PCL Window Settings (Start)******************************************/
    boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer(new pcl::visualization::PCLVisualizer("3D Viewer"));
    viewer->setBackgroundColor(0.5, 0.7, 0.9);
    pcl::visualization::Camera Camera_pos;
    viewer->initCameraParameters();
    Camera_pos.pos[0] = -245.900901;
    Camera_pos.pos[1] = 16.432728;
    Camera_pos.pos[2] = 224.284610;
    Camera_pos.view[0] = 0.071219;
    Camera_pos.view[1] = -0.995488;
    Camera_pos.view[2] = 0.062706;
    Camera_pos.focal[0] = -255.726440;
    Camera_pos.focal[1] = 24.640686;
    Camera_pos.focal[2] = 365.749386;
    viewer->loadCameraParameters("a.cam");
    //**************************************PCL Window Settings (End)******************************************/


    //************************************** Major Function (Start) ***********************************/
    Mat leftImage_src = imread("C:\\Phd_Research\\2021\\" + data_str + "\\" + filename + "\\left\\0.bmp", 0);
    Mat rightImage_src = imread("C:\\Phd_Research\\2021\\" + data_str + "\\" + filename + "\\right\\0.bmp", 0);
    if (Trans_Process.loadCameraParameters())
    {
        for (int img_i = img_start; img_i < img_end; img_i += 1)
        {
            cout << img_i << endl;
            //************************************** Rectfy Images (Start) ***********************************//
            leftImage = imread("C:\\Phd_Research\\2021\\" + data_str + "\\" + filename + "\\left\\" + std::to_string(img_i) + ".bmp", 0);
            rightImage = imread("C:\\Phd_Research\\2021\\" + data_str + "\\" + filename + "\\right\\" + std::to_string(img_i) + ".bmp", 0);
            leftImage = leftImage - leftImage_src;
            rightImage = rightImage - rightImage_src;
            Trans_Process.rectifyImages(leftImage, rightImage);
            leftImage = Trans_Process.leftRectifiedImage.clone();
            rightImage = Trans_Process.rightRectifiedImage.clone();
            leftImage.convertTo(leftImage, CV_32FC1);
            rightImage.convertTo(rightImage, CV_32FC1);
            //************************************** Rectfy Images (End) ***********************************//

            //****************************** Laser Centre Extract(Start)*********************************//
            GaussianBlur(leftImage, GaussL, Size(0, 0), filter_core, filter_core);
            GaussianBlur(rightImage, GaussR, Size(0, 0), filter_core, filter_core);
            Trans_Process.LineCentreExtract(GaussL, CentreL, gray_temp, thr_search);
            Trans_Process.LineCentreExtract(GaussR, CentreR, gray_temp, thr_search);
            //****************************** Laser Centre Extract (End)*********************************//

            //****************************** Left Camera Process(Start)*********************************//
            laser_world = Mat(Camera_height, Camera_width, CV_32FC3, cv::Scalar::all(0));
            Trans_Process.intersect_Ref_DOF(CentreL, laser_world, float(img_i), 0, gray_temp, Depth_start, Depth_end);
            Trans_Process.Joint_DualCamera(laser_world, laser_world_jointL, CentreR, thr_joint, 0);
            Trans_Process.Proxinate_SingleCamera(laser_world_jointL, laser_world_firstL);
            //****************************** Left Camera Process(End)*********************************//

            //****************************** Right Camera Process(Start)*********************************//
            laser_world = Mat(Camera_height, Camera_width, CV_32FC3, cv::Scalar::all(0));
            Trans_Process.intersect_Ref_DOF(CentreR, laser_world, float(img_i), 1, gray_temp, Depth_start, Depth_end);
            Trans_Process.Joint_DualCamera(laser_world, laser_world_jointR, CentreL, thr_joint, 1);
            //****************************** Right Camera Process(End)*********************************//

            //***************************MatToPCL (Start)***************************************//
            Trans_Process.convert2Pointcloud_DOF(laser_world_firstL, cloud, Depth_start, Depth_end);
            if (img_i > 1)
            {
                viewer->removePointCloud(to_string(img_i - 1));
            }
            viewer->addPointCloud<pcl::PointXYZRGB>(cloud, to_string(img_i));
            viewer->spinOnce(5);
            //***************************MatToPCL (End)***************************************//
        }
    }
    //************************************** Major Function (End) ***********************************/

    //************************************** Display PointCloud ***********************************/
    while (!viewer->wasStopped())
    {
        viewer->spinOnce(5);
    }
    cout << "Finish !" << endl;
    system("pause");

    return 0;

}

