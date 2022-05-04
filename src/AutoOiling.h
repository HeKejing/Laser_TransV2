
#include <iostream>
#include <vector>
#include <opencv2\opencv.hpp> 

#include <pcl/common/common.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/visualization/cloud_viewer.h>

//#include "common_type_prv.h"
//#include "LineDetector.h"

namespace cv    // use the data structure of opencv
{
    class Mat;
    template <class T>
    class Point_;
    template <class T>
    class Point3_;
    typedef Point3_<float> Point3f;
    typedef Point_<int> Point2i;
}

class AutoOiling
{
public:
    AutoOiling();
    ~AutoOiling();

    void setResolution(int width, int height);                      // if set. please set before loadCameraParameters()
    int loadCameraParameters();


    //By Kejing
    void rectifyImages(cv::Mat& leftImage, cv::Mat& rightImage);

    void cvFitPlane(const CvMat* points, float* plane);
    cv::Point3f getFootPoint(cv::Point3f point, cv::Point3f line_p1, cv::Point3f line_p2);
    cv::Mat Get3DR_TransMatrix(const std::vector<cv::Point3f>& srcPoints, const std::vector<cv::Point3f>& dstPoints);
    /**
    * @brief        找出每行前后表面反射laser
    * @param[in]    image
    * @param[out]   laser_out   :   找到的激光起始坐标
    * @param[out]   laser_out_width  :   激光宽度
    * @param[out]   laser_see   :   中间参量观测用
    * @param[in]    gray        :   临界灰度
    * @param[in]    error_max   :   允许误差次数
    * @param[in]    flag_pos    :   位置约束，flag = 1 物体左侧  flag = 2 物体右侧
    */
    void find_frontback_laser(cv::Mat& image, cv::Mat& laser_out, cv::Mat& laser_out_width, cv::Mat& laser_see, int gray, int error_max, int flag_pos);

    void reconstruct_frontback_laser(cv::Mat& laser_left, cv::Mat& laser_left_width, cv::Mat& laser_right, cv::Mat& laser_right_width, cv::Mat& laser_world);

    void intersect_Ref(cv::Mat& laser, cv::Mat& laser_width, float line_index, cv::Mat& laser_world, int index);


    void Add_laser_world(cv::Mat& laser_world_sum, cv::Mat& laser_world);
    int PM_filter(cv::Mat& inputMatrix, cv::Mat& outputMatrix, int window_size, int thr_dist);

    /**
    * @brief        计算单个点世界坐标
    */
    void Calculate_world_coord(float x, float y, float dif, float& wx, float& wy, float& wz);
    void Calculate_world_byRef(float x, float y, float line_index, float& wx, float& wy, float& wz, int index);
    /**
    * @brief     转换Mat到点云
    */
    void convert2Pointcloud(cv::Mat laser_world, pcl::PointCloud<pcl::PointXYZRGB>::Ptr& organisedPointCloud);
    int convert2Pointcloud_new(cv::Mat laser_world, pcl::PointCloud<pcl::PointXYZRGB>::Ptr& organisedPointCloud, int flag);
    /**
    * @brief        Hessian matrix -> line center (Steger)
    */
    void Detect_line_center(cv::Mat& image, cv::Mat& laser_center_mask, int threshold);


    /**
    * @brief      2021/02/17 Acquire obaque  reflected laser
    INPUT         image,image_center
    OUTPUT        laser_left laser_width
    */
    void Acquire_obauque(cv::Mat& image, cv::Mat& image_center, cv::Mat& laser_left, cv::Mat& laser_width);

    void calcu_XYZ(float x, float y, float x_p, float y_p, float& X, float& Y, float& Z);

    void intersect_Ref_nolimit(cv::Mat& src, cv::Mat& laser_world, float line_index, int index, int thr_gray);

    void Points_3D_Camera(cv::Mat& Points_surf, int camera_index, cv::Mat& MatchCam);

    void Points_3D_Camera_single(float& X, float& Y, float& Z, int camera_index, float& x_p, float& y_p);

    //flag = 0 (L2R)   flag = 1 (R2L)
    void Distinguish_Confident3D(cv::Mat& laser_world, cv::Mat& laser_common, cv::Mat& laser_world_filter, cv::Mat& image_project, int thr_horiz, int thr_gray, int flag);

    void intersect_Ref_updated(cv::Mat& src, cv::Mat& laser_world, cv::Mat& laser_common, float line_index, int index, int thr_gray, cv::Mat& image_project, int thr_horiz);
    void PM_filter_laser(cv::Mat& inputMatrix, cv::Mat& outputMatrix, int vert_size, float laser_size, float dist_thr);
    cv::Point3f design_Color(float dep_s, float dep_e, float input_z);

    void LineCentreExtract(cv::Mat image_src, cv::Mat& image_out, int gray_thr, int thr_search);
    void intersect_Ref_DOF(cv::Mat& src, cv::Mat& laser_world, float line_index, int index, int thr_gray, float DOF1, float DOF2);
    void Joint_DualCamera(cv::Mat& laser_world, cv::Mat& laser_worldF, cv::Mat& image_project, int thr_horiz, int flag);
    void Proxinate_SingleCamera(cv::Mat& laser_world, cv::Mat& laser_worldF);
    void convert2Pointcloud_DOF(cv::Mat laser_world, pcl::PointCloud<pcl::PointXYZRGB>::Ptr& organisedPointCloud, float DOF1, float DOF2);

    std::string     CAMERAPARAM = "result.yml";
    cv::Mat         cameraMatrixL, cameraDistcoeffL, cameraMatrixR, cameraDistcoeffR;  // camera inner parameters
    cv::Mat         R, T, Rl, Pl, Rr, Pr, Q, E, F;                                     // camera external parameters
    cv::Size        img_size;
    int             resolutionWidth;
    int             resolutionHeight;
    cv::Mat         map11, map12, map21, map22;

    cv::Mat         leftRectifiedImage, rightRectifiedImage;

    cv::Mat         temp_KJ;
    cv::Mat         image2D_;
    cv::Mat         imageFeature_;
    //Reconstruct Parameters
    float fx_d, px_d, py_d, baseline, offset;
    cv::Mat M_trig_L, M_trig_R;
    cv::Mat Ref_p;
    float Ref_para[9];
};

