#include "AutoOiling.h"

//By HeKejing
AutoOiling::AutoOiling()
{
    resolutionWidth = 2048;
    resolutionHeight = 1536;
}

AutoOiling::~AutoOiling()
{

}

void AutoOiling::setResolution(int width, int height)
{
    resolutionWidth = width;
    resolutionHeight = height;
}

int AutoOiling::loadCameraParameters()
{
    img_size = cv::Size(resolutionWidth, resolutionHeight);

    // load config file
    cv::FileStorage fs(CAMERAPARAM, cv::FileStorage::READ);
    if (!fs.isOpened())
    {
        cout << "Failed to open file" << endl;
        return 0;
    }

    // read parameters
    fs["camL_K"] >> cameraMatrixL;
    fs["camL_kc"] >> cameraDistcoeffL;
    fs["camR_K"] >> cameraMatrixR;
    fs["camR_kc"] >> cameraDistcoeffR;
    fs["R"] >> R;
    fs["T"] >> T;
    fs["E"] >> E;
    fs["F"] >> F;
    // fs["Q"] >> Q;

    cout << cameraMatrixL << endl;

    cout << cameraDistcoeffL << endl;

    cout << cameraMatrixR << endl;

    cout << cameraDistcoeffR << endl;


    cout << R << endl;


    cout << T << endl;


    cout << E << endl;


    cout << F << endl;


    //cout << Q << endl;
    // calculate map
    stereoRectify(cameraMatrixL, cameraDistcoeffL, cameraMatrixR, cameraDistcoeffR, img_size, R, T, Rl, Rr, Pl, Pr, Q, 0);

    initUndistortRectifyMap(cameraMatrixL, cameraDistcoeffL, Rl, Pl, img_size, CV_32FC1, map11, map12);
    initUndistortRectifyMap(cameraMatrixR, cameraDistcoeffR, Rr, Pr, img_size, CV_32FC1, map21, map22);




    fx_d = Q.at<double>(2, 3);  // f
    px_d = -Q.at<double>(0, 3); // u0
    py_d = -Q.at<double>(1, 3); // v0
    baseline = 1 / Q.at<double>(3, 2); // -Tx
    offset = -(Q.at<double>(3, 3) / Q.at<double>(3, 2)); //u0-u0'


    M_trig_L = Pl.clone();
    M_trig_R = Pr.clone();

    M_trig_L.convertTo(M_trig_L, CV_32FC1);
    M_trig_R.convertTo(M_trig_R, CV_32FC1);
    cout << M_trig_L << endl;
    cout << M_trig_R << endl;


    cv::FileStorage fs1("Calib_Ref.xml", cv::FileStorage::READ);
    if (!fs1.isOpened())
    {
        cout << "No Ref_P file! Suppose Ref_p = 0." << endl;
        for (int i = 0; i < 9; i++)
        {
            Ref_para[i] = 0.0;
        }
        fs1.release();
    }
    else
    {
        fs1["Ref_p"] >> Ref_p;
        for (int i = 0; i < 9; i++)
        {
            Ref_para[i] = Ref_p.at<float>(i, 0);
        }
        fs1.release();
    }

    return 1;
}

void AutoOiling::calcu_XYZ(float x, float y, float x_p, float y_p, float& X, float& Y, float& Z)
{
    cv::Mat M_XX(4, 3, CV_32FC1), M_YY(4, 1, CV_32FC1), result_end;
    M_XX.at<float>(0, 0) = x * M_trig_L.at<float>(2, 0) - M_trig_L.at<float>(0, 0);
    M_XX.at<float>(0, 1) = x * M_trig_L.at<float>(2, 1) - M_trig_L.at<float>(0, 1);
    M_XX.at<float>(0, 2) = x * M_trig_L.at<float>(2, 2) - M_trig_L.at<float>(0, 2);


    M_XX.at<float>(1, 0) = y * M_trig_L.at<float>(2, 0) - M_trig_L.at<float>(1, 0);
    M_XX.at<float>(1, 1) = y * M_trig_L.at<float>(2, 1) - M_trig_L.at<float>(1, 1);
    M_XX.at<float>(1, 2) = y * M_trig_L.at<float>(2, 2) - M_trig_L.at<float>(1, 2);

    M_XX.at<float>(2, 0) = x_p * M_trig_R.at<float>(2, 0) - M_trig_R.at<float>(0, 0);
    M_XX.at<float>(2, 1) = x_p * M_trig_R.at<float>(2, 1) - M_trig_R.at<float>(0, 1);
    M_XX.at<float>(2, 2) = x_p * M_trig_R.at<float>(2, 2) - M_trig_R.at<float>(0, 2);


    M_XX.at<float>(3, 0) = y_p * M_trig_R.at<float>(2, 0) - M_trig_R.at<float>(1, 0);
    M_XX.at<float>(3, 1) = y_p * M_trig_R.at<float>(2, 1) - M_trig_R.at<float>(1, 1);
    M_XX.at<float>(3, 2) = y_p * M_trig_R.at<float>(2, 2) - M_trig_R.at<float>(1, 2);


    M_YY.at<float>(0, 0) = M_trig_L.at<float>(0, 3) - x * M_trig_L.at<float>(2, 3);
    M_YY.at<float>(1, 0) = M_trig_L.at<float>(1, 3) - y * M_trig_L.at<float>(2, 3);
    M_YY.at<float>(2, 0) = M_trig_R.at<float>(0, 3) - x_p * M_trig_R.at<float>(2, 3);
    M_YY.at<float>(3, 0) = M_trig_R.at<float>(1, 3) - y_p * M_trig_R.at<float>(2, 3);

    result_end = (M_XX.t() * M_XX).inv() * (M_XX.t()) * M_YY;

    X = result_end.at<float>(0, 0);
    Y = result_end.at<float>(1, 0);
    Z = result_end.at<float>(2, 0);
}


void AutoOiling::cvFitPlane(const CvMat* points, float* plane)
{
    // Estimate geometric centroid.
    int nrows = points->rows;
    int ncols = points->cols;
    int type = points->type;
    CvMat* centroid = cvCreateMat(1, ncols, type);
    cvSet(centroid, cvScalar(0));
    for (int c = 0; c < ncols; c++)
    {
        for (int r = 0; r < nrows; r++)
        {
            centroid->data.fl[c] += points->data.fl[ncols * r + c];
        }
        centroid->data.fl[c] /= nrows;
    }
    // Subtract geometric centroid from each point.
    CvMat* points2 = cvCreateMat(nrows, ncols, type);
    for (int r = 0; r < nrows; r++)
        for (int c = 0; c < ncols; c++)
            points2->data.fl[ncols * r + c] = points->data.fl[ncols * r + c] - centroid->data.fl[c];
    // Evaluate SVD of covariance matrix.
    CvMat* A = cvCreateMat(ncols, ncols, type);
    CvMat* W = cvCreateMat(ncols, ncols, type);
    CvMat* V = cvCreateMat(ncols, ncols, type);
    cvGEMM(points2, points, 1, NULL, 0, A, CV_GEMM_A_T);
    cvSVD(A, W, NULL, V, CV_SVD_V_T);
    // Assign plane coefficients by singular vector corresponding to smallest singular value.
    plane[ncols] = 0;
    for (int c = 0; c < ncols; c++) {
        plane[c] = V->data.fl[ncols * (ncols - 1) + c];
        plane[ncols] += plane[c] * centroid->data.fl[c];
    }
    // Release allocated resources.
    cvReleaseMat(&centroid);
    cvReleaseMat(&points2);
    cvReleaseMat(&A);
    cvReleaseMat(&W);
    cvReleaseMat(&V);
}

cv::Point3f AutoOiling::getFootPoint(cv::Point3f point, cv::Point3f line_p1, cv::Point3f line_p2)
{
    float x0 = point.x;
    float y0 = point.y;
    float z0 = point.z;

    float x1 = line_p1.x;
    float y1 = line_p1.y;
    float z1 = line_p1.z;

    float x2 = line_p2.x;
    float y2 = line_p2.y;
    float z2 = line_p2.z;

    float k = -((x1 - x0) * (x2 - x1) + (y1 - y0) * (y2 - y1) + (z1 - z0) * (z2 - z1)) / ((x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1) + (z2 - z1) * (z2 - z1)) * 1.0;

    float xn = k * (x2 - x1) + x1;
    float yn = k * (y2 - y1) + y1;
    float zn = k * (z2 - z1) + z1;

    return cv::Point3f(xn, yn, zn);
}

cv::Mat AutoOiling::Get3DR_TransMatrix(const std::vector<cv::Point3f>& srcPoints, const std::vector<cv::Point3f>& dstPoints)
{
    double srcSumX = 0.0f;
    double srcSumY = 0.0f;
    double srcSumZ = 0.0f;

    double dstSumX = 0.0f;
    double dstSumY = 0.0f;
    double dstSumZ = 0.0f;

    //至少三组点
    if (srcPoints.size() != dstPoints.size() || srcPoints.size() < 3)
    {
        return cv::Mat();
    }

    int pointsNum = srcPoints.size();
    for (int i = 0; i < pointsNum; ++i)
    {
        srcSumX += srcPoints[i].x;
        srcSumY += srcPoints[i].y;
        srcSumZ += srcPoints[i].z;

        dstSumX += dstPoints[i].x;
        dstSumY += dstPoints[i].y;
        dstSumZ += dstPoints[i].z;
    }

    cv::Point3d centerSrc, centerDst;

    centerSrc.x = double(srcSumX / pointsNum);
    centerSrc.y = double(srcSumY / pointsNum);
    centerSrc.z = double(srcSumZ / pointsNum);

    centerDst.x = double(dstSumX / pointsNum);
    centerDst.y = double(dstSumY / pointsNum);
    centerDst.z = double(dstSumZ / pointsNum);

    //Mat::Mat(int rows, int cols, int type)
    cv::Mat srcMat(3, pointsNum, CV_64FC1);
    cv::Mat dstMat(3, pointsNum, CV_64FC1);
    for (int i = 0; i < pointsNum; ++i)//N组点
    {
        //三行
        srcMat.at<double>(0, i) = srcPoints[i].x - centerSrc.x;
        srcMat.at<double>(1, i) = srcPoints[i].y - centerSrc.y;
        srcMat.at<double>(2, i) = srcPoints[i].z - centerSrc.z;

        dstMat.at<double>(0, i) = dstPoints[i].x - centerDst.x;
        dstMat.at<double>(1, i) = dstPoints[i].y - centerDst.y;
        dstMat.at<double>(2, i) = dstPoints[i].z - centerDst.z;

    }

    cv::Mat matS = srcMat * dstMat.t();

    cv::Mat matU, matW, matV;
    cv::SVDecomp(matS, matW, matU, matV);

    cv::Mat matTemp = matU * matV;
    double det = cv::determinant(matTemp);//行列式的值

    double datM[] = { 1, 0, 0, 0, 1, 0, 0, 0, det };
    cv::Mat matM(3, 3, CV_64FC1, datM);

    cv::Mat matR = matV.t() * matM * matU.t();

    double* datR = (double*)(matR.data);
    double delta_X = centerDst.x - (centerSrc.x * datR[0] + centerSrc.y * datR[1] + centerSrc.z * datR[2]);
    double delta_Y = centerDst.y - (centerSrc.x * datR[3] + centerSrc.y * datR[4] + centerSrc.z * datR[5]);
    double delta_Z = centerDst.z - (centerSrc.x * datR[6] + centerSrc.y * datR[7] + centerSrc.z * datR[8]);


    //生成RT齐次矩阵(4*4)
    cv::Mat R_T = (cv::Mat_<double>(4, 4) <<
        matR.at<double>(0, 0), matR.at<double>(0, 1), matR.at<double>(0, 2), delta_X,
        matR.at<double>(1, 0), matR.at<double>(1, 1), matR.at<double>(1, 2), delta_Y,
        matR.at<double>(2, 0), matR.at<double>(2, 1), matR.at<double>(2, 2), delta_Z,
        0, 0, 0, 1
        );

    return R_T;
}




void AutoOiling::rectifyImages(cv::Mat& leftImage, cv::Mat& rightImage)
{
    remap(leftImage, leftRectifiedImage, map11, map12, cv::INTER_LINEAR, cv::BORDER_CONSTANT);
    remap(rightImage, rightRectifiedImage, map21, map22, cv::INTER_LINEAR, cv::BORDER_CONSTANT);
}


void AutoOiling::find_frontback_laser(cv::Mat& image, cv::Mat& laser_out, cv::Mat& laser_out_width, cv::Mat& laser_see, int gray, int error_max, int flag_pos)
{
    int wt = resolutionWidth;
    int ht = resolutionHeight;
    bool flag = false; // whether start
    int error = 0;
    int temp_w = 0;
    int index = 0;
    for (int h = 0; h < ht; h++)
    {
        uchar* data = image.ptr<uchar>(h);
        index = 0;
        for (int w = 1; w < wt; w++)
        {
            if ((data[w] >= gray) && (!flag)) // Not start and find
            {
                if (data[w - 1] <= data[w])
                {
                    flag = true;
                    temp_w = w;
                    laser_see.at<float>(h, temp_w) = 1;
                    laser_see.at<float>(h, 0) += 1; // start pixel tell how many laser find
                    index += 1;
                    laser_out.at<int>(h, 0) = index;
                    if (laser_out.at<int>(h, 0) > 11)
                    {
                        break;
                    }
                    laser_out.at<int>(h, index) = temp_w;
                    error = 0;
                }
            }

            if ((data[w] >= gray) && flag) // start and find
            {
                // laser_see.at<float>(h, temp_w) += 1;
                // laser_out_width.at<int>(h, index) += 1;

                if (data[w - 1] <= data[w])
                {
                    laser_see.at<float>(h, temp_w) += 1;
                    //laser_out.at<int>(h, index) += 1;
                    laser_out_width.at<int>(h, index) += 1;
                }
                else
                {
                    if (error < error_max)
                    {
                        error += 1;
                        laser_see.at<float>(h, temp_w) += 1;
                        //laser_out.at<int>(h, index) += 1;
                        laser_out_width.at<int>(h, index) += 1;
                    }
                    else
                    {
                        flag = false;
                    }
                }
            }

            if ((data[w] < gray) && flag) // start and Not find
            {
                flag = false;
            }

            // Not start and Not find -> No motion

        }


    }
}


void AutoOiling::reconstruct_frontback_laser(cv::Mat& laser_left, cv::Mat& laser_left_width, cv::Mat& laser_right, cv::Mat& laser_right_width, cv::Mat& laser_world)
{
    const double b = -(1 / Q.at<double>(3, 2));   // baseline
    const double f = Q.at<double>(2, 3);          // focus
    float b_mul_f = b * f;
    float d, WX, WY, WZ;
    int left_w;

    int ht = resolutionHeight;

    for (int h = 0; h < ht; h++)
    {
        // Method two
        if (((laser_left.at <float>(h, 0)) == (laser_right.at <float>(h, 0))) && (laser_left.at <float>(h, 0) > 0))
        {
            float width_left = laser_left_width.at<float>(h, 1);
            float width_right = laser_right_width.at<float>(h, 1);
            //width_left = 1;
            for (float i = (-width_left / 2); i < (width_left / 2); i++)
            {
                float* data_left = laser_left.ptr<float>(h);
                float* data_right = laser_right.ptr<float>(h);
                int left_w = round(data_left[1] + i);

                if (left_w > 0 && left_w < resolutionWidth)
                {
                    d = data_left[1] + i - data_right[1] - (i * (width_right / width_left));   // i+1 means sequence start with 1;

                    Calculate_world_coord(left_w, h, d, WX, WY, WZ);
                    laser_world.at<cv::Vec3f>(h, left_w)[0] = WX;
                    laser_world.at<cv::Vec3f>(h, left_w)[1] = WY;
                    laser_world.at<cv::Vec3f>(h, left_w)[2] = WZ;
                }

            }

        }

    }

}

void AutoOiling::intersect_Ref(cv::Mat& laser, cv::Mat& laser_width, float line_index, cv::Mat& laser_world, int index)
{
    int ht = resolutionHeight;
    float  WX, WY, WZ;
    for (int h = 0; h < ht; h++)
    {
        // Method two
        if ((laser.at <float>(h, 0) > 0))
        {
            for (int m = 1; m < laser.at <float>(h, 0) + 1; m++)
            {
                float width_laser = laser_width.at<float>(h, m);
                //width_left = 1;
                for (float i = (-width_laser / 2); i < (width_laser / 2); i++)
                {
                    float* data_laser = laser.ptr<float>(h);
                    int D_x = round(data_laser[m] + i);
                    if (D_x > 0 && D_x < resolutionWidth)
                    {
                        float x_w = data_laser[m] + i;
                        float line_optm = line_index + i * 1.0 / width_laser;
                        Calculate_world_byRef(x_w, h, line_optm, WX, WY, WZ, index);
                        laser_world.at<cv::Vec3f>(h, D_x)[0] = WX;
                        laser_world.at<cv::Vec3f>(h, D_x)[1] = WY;
                        laser_world.at<cv::Vec3f>(h, D_x)[2] = WZ;
                    }
                }

            }
        }

    }
}



void AutoOiling::Add_laser_world(cv::Mat& laser_world_sum, cv::Mat& laser_world)
{
    float depth_s = 0;//270;
    float depth_e = 2000;// 350;
    int wt = laser_world_sum.cols;
    int ht = laser_world_sum.rows;
    for (int h = 0; h < ht; h++)
    {
        float* data_sum = laser_world_sum.ptr<float>(h);
        float* data = laser_world.ptr<float>(h);
        for (int w = 0; w < wt; w++)
        {
            /*if ((data_sum[3 * w + 2] == 0) && (data[3 * w + 2] < 500) && (data[3 * w + 2] > 0))
            {
                data_sum[3 * w] = data[3 * w];
                data_sum[3 * w + 1] = data[3 * w + 1];
                data_sum[3 * w + 2] = data[3 * w + 2];
            }*/
            if ((data_sum[3 * w + 2] == 0) && (data[3 * w + 2] < depth_e) && (data[3 * w + 2] > depth_s))
            {
                data_sum[3 * w] = data[3 * w];
                data_sum[3 * w + 1] = data[3 * w + 1];
                data_sum[3 * w + 2] = data[3 * w + 2];
            }
            else if ((data_sum[3 * w + 2] != 0) && (data[3 * w + 2] < depth_e) && (data[3 * w + 2] > depth_s))
            {
                data_sum[3 * w] = (data_sum[3 * w] + data[3 * w]) / 2.0;
                data_sum[3 * w + 1] = (data_sum[3 * w + 1] + data[3 * w + 1]) / 2.0;
                data_sum[3 * w + 2] = (data_sum[3 * w + 2] + data[3 * w + 2]) / 2.0;
            }

        }
    }

}

void AutoOiling::Calculate_world_coord(float x, float y, float dif, float& wx, float& wy, float& wz)
{
    double X, Y, Z, W;
    X = x - px_d;
    Y = y - py_d;
    Z = fx_d;
    W = (dif - offset) / baseline;
    wx = X / W;
    wy = Y / W;
    wz = Z / W;
}


void AutoOiling::Calculate_world_byRef(float x, float y, float line_index, float& wx, float& wy, float& wz, int index)
{
    float M_XX[3][3], M_YY[3], M_XX_inv[3][3], M_end[3][3], result_end[3] = { 0, 0, 0 };

    float sin_2a = sin(2 * (Ref_para[0] * line_index + Ref_para[1]));
    float cos_2a = cos(2 * (Ref_para[0] * line_index + Ref_para[1]));
    float tan_gama = tan(Ref_para[7]);
    float theta = sqrt(Ref_para[2] * Ref_para[2] + Ref_para[3] * Ref_para[3] + Ref_para[4] * Ref_para[4]);
    float cos_theta = cos(theta);
    float sin_theta = sin(theta);

    float a_11 = cos_theta + Ref_para[2] * Ref_para[2] * (1 - cos_theta) / theta / theta;
    float a_12 = Ref_para[2] * Ref_para[3] * (1 - cos_theta) / theta / theta - Ref_para[4] * sin_theta / theta;
    float a_13 = Ref_para[2] * Ref_para[4] * (1 - cos_theta) / theta / theta + Ref_para[3] * sin_theta / theta;
    float a_14 = Ref_para[5];

    float a_21 = Ref_para[2] * Ref_para[3] * (1 - cos_theta) / theta / theta + Ref_para[4] * sin_theta / theta;
    float a_22 = cos_theta + Ref_para[3] * Ref_para[3] * (1 - cos_theta) / theta / theta;
    float a_23 = Ref_para[3] * Ref_para[4] * (1 - cos_theta) / theta / theta - Ref_para[2] * sin_theta / theta;
    float a_24 = Ref_para[6];

    float a_31 = Ref_para[2] * Ref_para[4] * (1 - cos_theta) / theta / theta - Ref_para[3] * sin_theta / theta;
    float a_32 = Ref_para[3] * Ref_para[4] * (1 - cos_theta) / theta / theta + Ref_para[2] * sin_theta / theta;
    float a_33 = cos_theta + Ref_para[4] * Ref_para[4] * (1 - cos_theta) / theta / theta;
    float a_34 = 0;

    //
    if (index == 0)
    {
        M_XX[0][0] = x * M_trig_L.at<float>(2, 0) - M_trig_L.at<float>(0, 0);
        M_XX[0][1] = x * M_trig_L.at<float>(2, 1) - M_trig_L.at<float>(0, 1);
        M_XX[0][2] = x * M_trig_L.at<float>(2, 2) - M_trig_L.at<float>(0, 2);


        M_XX[1][0] = y * M_trig_L.at<float>(2, 0) - M_trig_L.at<float>(1, 0);
        M_XX[1][1] = y * M_trig_L.at<float>(2, 1) - M_trig_L.at<float>(1, 1);
        M_XX[1][2] = y * M_trig_L.at<float>(2, 2) - M_trig_L.at<float>(1, 2);

        M_XX[2][0] = sin_2a * a_11 + cos_2a * a_21 + tan_gama * a_31;
        M_XX[2][1] = sin_2a * a_12 + cos_2a * a_22 + tan_gama * a_32;
        M_XX[2][2] = sin_2a * a_13 + cos_2a * a_23 + tan_gama * a_33;

        M_YY[0] = M_trig_L.at<float>(0, 3) - x * M_trig_L.at<float>(2, 3);
        M_YY[1] = M_trig_L.at<float>(1, 3) - y * M_trig_L.at<float>(2, 3);
        M_YY[2] = Ref_para[8] - (sin_2a * a_14 + cos_2a * a_24 + tan_gama * a_34);
    }
    else
    {
        M_XX[0][0] = x * M_trig_R.at<float>(2, 0) - M_trig_R.at<float>(0, 0);
        M_XX[0][1] = x * M_trig_R.at<float>(2, 1) - M_trig_R.at<float>(0, 1);
        M_XX[0][2] = x * M_trig_R.at<float>(2, 2) - M_trig_R.at<float>(0, 2);


        M_XX[1][0] = y * M_trig_R.at<float>(2, 0) - M_trig_R.at<float>(1, 0);
        M_XX[1][1] = y * M_trig_R.at<float>(2, 1) - M_trig_R.at<float>(1, 1);
        M_XX[1][2] = y * M_trig_R.at<float>(2, 2) - M_trig_R.at<float>(1, 2);

        M_XX[2][0] = sin_2a * a_11 + cos_2a * a_21 + tan_gama * a_31;
        M_XX[2][1] = sin_2a * a_12 + cos_2a * a_22 + tan_gama * a_32;
        M_XX[2][2] = sin_2a * a_13 + cos_2a * a_23 + tan_gama * a_33;

        M_YY[0] = M_trig_R.at<float>(0, 3) - x * M_trig_R.at<float>(2, 3);
        M_YY[1] = M_trig_R.at<float>(1, 3) - y * M_trig_R.at<float>(2, 3);
        M_YY[2] = Ref_para[8] - (sin_2a * a_14 + cos_2a * a_24 + tan_gama * a_34);
    }
    // M_XX_inv
    float P = M_XX[0][0] * (M_XX[1][1] * M_XX[2][2] - M_XX[1][2] * M_XX[2][1]) - M_XX[1][0] * (M_XX[0][1] * M_XX[2][2] - M_XX[0][2] * M_XX[2][1]) + M_XX[2][0] * (M_XX[0][1] * M_XX[1][2] - M_XX[0][2] * M_XX[1][1]);
    if (abs(P) > 10e-5)
    {
        M_XX_inv[0][0] = (M_XX[1][1] * M_XX[2][2] - M_XX[1][2] * M_XX[2][1]) / P;
        M_XX_inv[1][0] = (M_XX[1][2] * M_XX[2][0] - M_XX[1][0] * M_XX[2][2]) / P;
        M_XX_inv[2][0] = (M_XX[1][0] * M_XX[2][1] - M_XX[1][1] * M_XX[2][0]) / P;
        M_XX_inv[0][1] = (M_XX[0][2] * M_XX[2][1] - M_XX[0][1] * M_XX[2][2]) / P;
        M_XX_inv[1][1] = (M_XX[0][0] * M_XX[2][2] - M_XX[0][2] * M_XX[2][0]) / P;
        M_XX_inv[2][1] = (M_XX[0][1] * M_XX[2][0] - M_XX[0][0] * M_XX[2][1]) / P;
        M_XX_inv[0][2] = (M_XX[0][1] * M_XX[1][2] - M_XX[0][2] * M_XX[1][1]) / P;
        M_XX_inv[1][2] = (M_XX[1][0] * M_XX[0][2] - M_XX[0][0] * M_XX[1][2]) / P;
        M_XX_inv[2][2] = (M_XX[0][0] * M_XX[1][1] - M_XX[1][0] * M_XX[0][1]) / P;
        for (int m = 0; m < 3; m++)
        {
            for (int n = 0; n < 3; n++)
            {
                result_end[m] += M_XX_inv[m][n] * M_YY[n];
            }

        }
        wx = result_end[0];
        wy = result_end[1];
        wz = result_end[2];
    }
    else
    {
        wx = 0;
        wy = 0;
        wz = 0;
    }


}

void AutoOiling::convert2Pointcloud(cv::Mat laser_world, pcl::PointCloud<pcl::PointXYZRGB>::Ptr& organisedPointCloud)
{
    pcl::PointXYZRGB point;
    const int height = laser_world.rows;
    const int width = laser_world.cols;
    organisedPointCloud->height = height;
    organisedPointCloud->width = width;
    organisedPointCloud->resize(height * width);
    for (int h = 0; h < height; h++)
    {
        float* ptr_XYZ = laser_world.ptr<float>(h);
        for (int w = 0; w < width; w++)
        {
            point.x = *(ptr_XYZ + 3 * w);
            point.y = *(ptr_XYZ + 3 * w + 1);
            point.z = *(ptr_XYZ + 3 * w + 2);
            point.b = 250;
            point.g = 250;
            point.r = 250;
            (*organisedPointCloud)(w, h) = point;
        }
    }
}
int AutoOiling::convert2Pointcloud_new(cv::Mat laser_world, pcl::PointCloud<pcl::PointXYZRGB>::Ptr& organisedPointCloud, int flag)
{
    pcl::PointXYZRGB point;
    const int height = laser_world.rows;
    const int width = laser_world.cols;
    int points_num = 0;
    for (int h = 0; h < height; h++)
    {
        float* ptr_XYZ = laser_world.ptr<float>(h);
        for (int w = 0; w < width; w++)
        {
            if ((*(ptr_XYZ + 3 * w + 2) > 100) && (*(ptr_XYZ + 3 * w + 2) < 1400))
            {
                point.x = *(ptr_XYZ + 3 * w);
                point.y = *(ptr_XYZ + 3 * w + 1);
                point.z = *(ptr_XYZ + 3 * w + 2);
                /*point.b = 250;
                point.g = 250;
                point.r = 250;*/
                points_num += 1;
                cv::Point3f color_point = design_Color(600.0, 700.0, point.z);

                if (flag == 0)
                {
                    point.b = 250;
                    point.g = 250;
                    point.r = 250;
                }

                if (flag == 1)
                {
                    point.b = 0;
                    point.g = 250;
                    point.r = 0;
                }

                if (flag == 2)
                {
                    point.b = color_point.x;
                    point.g = color_point.y;
                    point.r = color_point.z;
                }
                organisedPointCloud->push_back(point);
            }
        }
    }
    return points_num;
}

cv::Point3f AutoOiling::design_Color(float dep_s, float dep_e, float input_z)
{
    //JET Output
    cv::Point3f output_color;
    output_color.x = 0;
    output_color.y = 0;
    output_color.z = 0;

    if (input_z < dep_s)
    {
        output_color.x = 128;
    }
    if (input_z > dep_e)
    {
        output_color.z = 252;
    }

    if (input_z >= dep_s && input_z <= dep_e)
    {
        int ratio_z = round((input_z - dep_s) / (dep_e - dep_s) * 255.0);
        if (ratio_z < 32)
        {
            output_color.x = 128 + 4 * ratio_z;
        }
        if (ratio_z == 32)
        {
            output_color.x = 255;
        }
        if (ratio_z >= 33 && ratio_z <= 95)
        {
            output_color.x = 255;
            output_color.y = 4 + 4 * (ratio_z - 33);
        }
        if (ratio_z == 96)
        {
            output_color.x = 254;
            output_color.y = 255;
            output_color.z = 2;
        }
        if (ratio_z >= 97 && ratio_z <= 158)
        {
            output_color.x = 250 - 4 * (ratio_z - 97);
            output_color.y = 255;
            output_color.z = 6 + 4 * (ratio_z - 97);
        }
        if (ratio_z == 159)
        {
            output_color.x = 1;
            output_color.y = 255;
            output_color.z = 254;
        }
        if (ratio_z >= 160 && ratio_z <= 223)
        {
            output_color.y = 252 - 4 * (ratio_z - 160);
            output_color.z = 255;
        }
        if (ratio_z >= 224 && ratio_z <= 255)
        {
            output_color.z = 252 - 4 * (ratio_z - 224);
        }
    }
    return output_color;

}

void AutoOiling::LineCentreExtract(cv::Mat image_src, cv::Mat& image_out, int gray_thr, int thr_search)
{
    image_out = cv::Mat(resolutionHeight, resolutionWidth, CV_32FC1, cv::Scalar::all(0));
    int max_inten;
    int start_w, end_w;
    int index_record = -1;
    float sum_intensity, sum_x;
    float temp;
    for (int h = 0; h < resolutionHeight; h++)
    {
        float* data_src = image_src.ptr<float>(h);
        float* data_out = image_out.ptr<float>(h);
        max_inten = gray_thr;
        for (int w = 0; w < resolutionWidth; w++)
        {
            //******************Find Laser Possible Laser Point(Start)***********************//
            if (data_src[w] > gray_thr)
            {
                max_inten = gray_thr;
                index_record = w;
                //***********Search MaxInten Points ******************//
                start_w = MAX(0, w - thr_search);
                end_w = MIN(resolutionWidth, w + thr_search);
                for (int index_w = start_w; index_w < end_w; index_w++) //Find Max Intensity
                {
                    if (data_src[index_w] >= max_inten)
                    {
                        index_record = index_w;
                        max_inten = data_src[index_w];
                    }
                }

                //******************Find Centre in MaxInten Neighbor******************//
                sum_intensity = 0;
                sum_x = 0;
                start_w = MAX(0, index_record - thr_search);
                end_w = MIN(resolutionWidth, index_record + thr_search);
                for (int index_w = start_w; index_w < end_w; index_w++)
                {
                    if (data_src[index_w] >= (max_inten / 1.5))
                    {
                        sum_intensity += (index_w * data_src[index_w]);
                        sum_x += data_src[index_w];
                    }
                }
                if (sum_intensity > 0)
                {
                    temp = sum_intensity / sum_x;
                    data_out[int(temp)] = temp;
                }

                w += thr_search; // Step Jump
            }
            //******************Find Laser Possible Laser Point (End)***********************//
        }
    }
    // cout << "Test" << endl;
}

void AutoOiling::intersect_Ref_DOF(cv::Mat& src, cv::Mat& laser_world, float line_index, int index, int thr_gray, float DOF1, float DOF2)
{

    int ht = resolutionHeight;
    int wt = resolutionWidth;
    float  WX, WY, WZ;
    int w_possible;
    for (int h = 0; h < ht; h++)
    {
        float* data = src.ptr<float>(h);
        float* data_world = laser_world.ptr<float>(h);
        for (int w = 0; w < wt; w++)
        {
            if (data[w] > 0)
            {
                Calculate_world_byRef(float(data[w]), float(h), line_index, WX, WY, WZ, index);
                if (WZ <= DOF2 && WZ >= DOF1)
                {
                    data_world[3 * w + 0] = WX;
                    data_world[3 * w + 1] = WY;
                    data_world[3 * w + 2] = WZ;
                }
            }
        }
    }
    //cout << "Test" << endl;
}

void AutoOiling::Joint_DualCamera(cv::Mat& laser_world, cv::Mat& laser_worldF, cv::Mat& image_project, int thr_horiz, int flag)
{
    cv::Mat MatchImage;
    laser_worldF = cv::Mat(resolutionHeight, resolutionWidth, CV_32FC3, cv::Scalar::all(0));
    Points_3D_Camera(laser_world, 1 - flag, MatchImage);
    for (int h = 0; h < resolutionHeight; h++)
    {
        float* data_2D = MatchImage.ptr<float>(h);
        float* data_3D_filter = laser_worldF.ptr<float>(h);
        float* data_3D = laser_world.ptr<float>(h);
        float* data_project = image_project.ptr<float>(h);
        for (int w = 0; w < resolutionWidth; w++)
        {
            int x = round(data_2D[2 * w]);
            if (x > 0 && x < resolutionWidth)
            {
                int x_s = std::max(x - thr_horiz, 0);
                int x_e = std::min(x + thr_horiz, resolutionWidth);
                for (int index = x_s; index < x_e; index++)
                {
                    if (data_project[index] > 0)
                    {
                        if (abs(data_2D[2 * w] - data_project[index]) < thr_horiz)
                        {
                            data_3D_filter[3 * w] = data_3D[3 * w];
                            data_3D_filter[3 * w + 1] = data_3D[3 * w + 1];
                            data_3D_filter[3 * w + 2] = data_3D[3 * w + 2];
                            break;
                        }
                    }
                }
            }

        }

    }

    //cout << "Test" << endl;
}

void AutoOiling::Proxinate_SingleCamera(cv::Mat& laser_world, cv::Mat& laser_worldF)
{
    laser_worldF = cv::Mat(resolutionHeight, resolutionWidth, CV_32FC3, cv::Scalar::all(0));
    float min_z = 5000;
    int x_index = -1;
    for (int h = 0; h < resolutionHeight; h++)
    {
        float* data_3D = laser_world.ptr<float>(h);
        float* data_3D_filter = laser_worldF.ptr<float>(h);
        x_index = -1;
        min_z = 5000;
        for (int w = 0; w < resolutionWidth; w++)
        {
            if (data_3D[3 * w + 2] > 0 && data_3D[3 * w + 2] < min_z)
            {
                min_z = data_3D[3 * w + 2];
                x_index = w;
            }
        }
        if (x_index > 0)
        {
            data_3D_filter[3 * x_index] = data_3D[3 * x_index];
            data_3D_filter[3 * x_index + 1] = data_3D[3 * x_index + 1];
            data_3D_filter[3 * x_index + 2] = data_3D[3 * x_index + 2];
        }
    }
}

void AutoOiling::convert2Pointcloud_DOF(cv::Mat laser_world, pcl::PointCloud<pcl::PointXYZRGB>::Ptr& organisedPointCloud, float DOF1, float DOF2)
{
    pcl::PointXYZRGB point;
    const int height = laser_world.rows;
    const int width = laser_world.cols;
    for (int h = 0; h < height; h++)
    {
        float* ptr_XYZ = laser_world.ptr<float>(h);
        for (int w = 0; w < width; w++)
        {
            if ((*(ptr_XYZ + 3 * w + 2) > DOF1) && (*(ptr_XYZ + 3 * w + 2) < DOF2))
            {
                point.x = *(ptr_XYZ + 3 * w);
                point.y = *(ptr_XYZ + 3 * w + 1);
                point.z = *(ptr_XYZ + 3 * w + 2);

                cv::Point3f color_point = design_Color(DOF1, DOF2, point.z);
                point.b = color_point.x;
                point.g = color_point.y;
                point.r = color_point.z;
                organisedPointCloud->push_back(point);
            }
        }
    }

}



int AutoOiling::PM_filter(cv::Mat& inputMatrix, cv::Mat& outputMatrix, int window_size, int thr_dist)
{
    const int height = inputMatrix.rows;
    const int width = inputMatrix.cols;
    int windowSize = window_size;
    outputMatrix = cv::Mat(height, width, CV_32FC3, cv::Scalar::all(0));
    for (int i = 0; i < height; i++) {
        float* ptr_input = inputMatrix.ptr<float>(i);
        float* ptr_output = outputMatrix.ptr<float>(i);
        for (int j = 0; j < width; j++) {
            int num = 0, flag = 0;
            float x = 0, y = 0, z = 0;
            if (*(ptr_input + 3 * j + 2) != 0) {
                for (int k = -windowSize; k < windowSize + 1; k++)
                {
                    if ((i + k) > -1 && (i + k) < height)
                    {
                        float* ptr_input_k = inputMatrix.ptr<float>(i + k);
                        for (int m = -windowSize; m < windowSize + 1; m++)
                        {
                            if ((j + m) > -1 && (j + m) < width && *(ptr_input_k + 3 * (j + m) + 2) != 0)
                            {
                                if (abs(*(ptr_input_k + 3 * (j + m) + 2) - *(ptr_input + 3 * j + 2)) < thr_dist)
                                {
                                    x += *(ptr_input_k + 3 * (j + m));
                                    y += *(ptr_input_k + 3 * (j + m) + 1);
                                    z += *(ptr_input_k + 3 * (j + m) + 2);
                                    num += 1;
                                }
                                else
                                {
                                    flag += 1;
                                }
                            }
                        }
                    }
                }
                if (flag > num || num < windowSize * windowSize)
                {
                    x = 0;
                    y = 0;
                    z = 0;
                }
                if (num != 0)
                {
                    *(ptr_output + 3 * j) = x / num;
                    *(ptr_output + 3 * j + 1) = y / num;
                    *(ptr_output + 3 * j + 2) = z / num;
                }
                else
                {
                    *(ptr_output + 3 * j) = 0;
                    *(ptr_output + 3 * j + 1) = 0;
                    *(ptr_output + 3 * j + 2) = 0;
                }
            }
        }
    }
    return 1;
}

void AutoOiling::Detect_line_center(cv::Mat& img, cv::Mat& laser_center_mask, int threshold)
{
    //Ò»½×Æ«µ¼Êý
    cv::Mat m1, m2;
    m1 = (cv::Mat_<float>(1, 2) << 1, -1);  //xÆ«µ¼
    m2 = (cv::Mat_<float>(2, 1) << 1, -1);  //yÆ«µ¼

    cv::Mat dx, dy;
    filter2D(img, dx, CV_32FC1, m1);
    filter2D(img, dy, CV_32FC1, m2);

    //¶þ½×Æ«µ¼Êý
    cv::Mat m3, m4, m5;
    m3 = (cv::Mat_<float>(1, 3) << 1, -2, 1);   //¶þ½×xÆ«µ¼
    m4 = (cv::Mat_<float>(3, 1) << 1, -2, 1);   //¶þ½×yÆ«µ¼
    m5 = (cv::Mat_<float>(2, 2) << 1, -1, -1, 1);   //¶þ½×xyÆ«µ¼

    cv::Mat dxx, dyy, dxy;
    filter2D(img, dxx, CV_32FC1, m3);
    filter2D(img, dyy, CV_32FC1, m4);
    filter2D(img, dxy, CV_32FC1, m5);

    //hessian¾ØÕó
    double maxD = -1;
    int imgcol = img.cols;
    int imgrow = img.rows;

    for (int i = 0; i < imgcol; i++)
    {
        for (int j = 0; j < imgrow; j++)
        {
            if (img.at<float>(j, i) > threshold)
            {
                cv::Mat hessian(2, 2, CV_32FC1);
                hessian.at<float>(0, 0) = dxx.at<float>(j, i);
                hessian.at<float>(0, 1) = dxy.at<float>(j, i);
                hessian.at<float>(1, 0) = dxy.at<float>(j, i);
                hessian.at<float>(1, 1) = dyy.at<float>(j, i);

                cv::Mat eValue;
                cv::Mat eVectors;
                eigen(hessian, eValue, eVectors);

                double nx, ny;
                double fmaxD = 0;
                if (fabs(eValue.at<float>(0, 0)) >= fabs(eValue.at<float>(1, 0)))  //ÇóÌØÕ÷Öµ×î´óÊ±¶ÔÓ¦µÄÌØÕ÷ÏòÁ¿
                {
                    nx = eVectors.at<float>(0, 0);
                    ny = eVectors.at<float>(0, 1);
                    fmaxD = eValue.at<float>(0, 0);
                }
                else
                {
                    nx = eVectors.at<float>(1, 0);
                    ny = eVectors.at<float>(1, 1);
                    fmaxD = eValue.at<float>(1, 0);
                }

                double t = -(nx * dx.at<float>(j, i) + ny * dy.at<float>(j, i)) / (nx * nx * dxx.at<float>(j, i) + 2 * nx * ny * dxy.at<float>(j, i) + ny * ny * dyy.at<float>(j, i));

                if (fabs(t * nx) <= 0.5 && fabs(t * ny) <= 0.5)
                {
                    laser_center_mask.at<uchar>(j, i) = 255;
                }
            }
        }
    }
}

void AutoOiling::Acquire_obauque(cv::Mat& image, cv::Mat& image_center, cv::Mat& laser_left, cv::Mat& laser_width)
{
    int ht = image.rows;
    int wt = image.cols;
    int width_search = 8;
    int gray = 50;

    laser_left = cv::Mat(ht, 11, CV_32FC1, cv::Scalar::all(0));
    laser_width = cv::Mat(ht, 11, CV_32FC1, cv::Scalar::all(0));

    for (int h = 0; h < ht; h++)
    {
        float* data = image.ptr<float>(h);
        uchar* data_center = image_center.ptr<uchar>(h);
        float* data_laser = laser_left.ptr<float>(h);
        float* data_width = laser_width.ptr<float>(h);
        int index = 1;
        for (int w = 0; w < wt; w++)
        {
            if (index > 10)
                if (index > 10)
                {
                    continue;
                }
            if (data_center[w] > 100)
            {
                // Find One Laser
                data_laser[index] = w;

                // Compute Width
                int start_w = MAX(0, w - width_search);
                int end_w = MIN(wt, w + width_search);
                for (int index_w = start_w; index_w < end_w; index_w++)
                {
                    if (data[index_w] >= gray) // Not start and find
                    {
                        data_width[index] += 1;
                    }
                }

                // Laser_center_norm

                //data_laser[index] = w - data_width[index] / 2.0;

                index += 1;
                data_laser[0] += 1;
            }
        }

    }

}

void AutoOiling::intersect_Ref_nolimit(cv::Mat& src, cv::Mat& laser_world, float line_index, int index, int thr_gray)
{
    int ht = resolutionHeight;
    int wt = resolutionWidth;
    float  WX, WY, WZ;
    int w_possible;
    float  WZ_possible = 1300;
    for (int h = 0; h < ht; h++)
    {
        WZ_possible = 1300;
        float* data = src.ptr<float>(h);
        float* data_world = laser_world.ptr<float>(h);
        for (int w = 0; w < wt; w++)
        {
            if (data[w] > thr_gray)
            {
                Calculate_world_byRef(float(w), float(h), line_index, WX, WY, WZ, index);
                if (WZ < WZ_possible)
                {
                    WZ_possible = WZ;
                    w_possible = w;
                }
                data_world[3 * w + 0] = WX;
                data_world[3 * w + 1] = WY;
                data_world[3 * w + 2] = WZ;
            }
        }
        if (WZ_possible < 1300)
        {
            Calculate_world_byRef(float(w_possible), float(h), line_index, WX, WY, WZ, index);
            data_world[3 * w_possible + 0] = WX;
            data_world[3 * w_possible + 1] = WY;
            data_world[3 * w_possible + 2] = WZ;

            /*float laser_wid = 0;
            for (int m = -10; m < 10; m++)
            {
                if ((w_possible + m) < 0 || (w_possible + m) > wt)
                {
                    continue;
                }
                if (data[w_possible + m] < thr_gray)
                {
                    continue;
                }
                laser_wid++;
            }
            float temp = 0;
            for (int m = -10; m < 10; m++)
            {
                if ((w_possible + m) < 0 || (w_possible + m) > wt)
                {
                    continue;
                }
                if (data[w_possible + m] < thr_gray)
                {
                    continue;
                }
                temp ++;

                Calculate_world_byRef(float(w_possible+m), float(h), line_index - 0.5 + temp / laser_wid, WX, WY, WZ, index);
                data_world[3 * (w_possible + m) + 0] = WX;
                data_world[3 * (w_possible + m) + 1] = WY;
                data_world[3 * (w_possible + m) + 2] = WZ;
            }*/





        }
    }
}

void AutoOiling::Points_3D_Camera(cv::Mat& Points_surf, int camera_index, cv::Mat& MatchCam)
{
    const int height = Points_surf.rows;
    const int width = Points_surf.cols;
    float x_Zc, y_Zc, Zc, x, y;
    MatchCam = cv::Mat(height, width, CV_32FC2, cv::Scalar::all(0));
    for (int h = 0; h < height; h++)
    {
        float* data_3D = Points_surf.ptr<float>(h);
        float* data_2D = MatchCam.ptr<float>(h);
        for (int w = 0; w < width; w++)
        {
            if (data_3D[3 * w + 2] > 0)
            {
                if (camera_index == 0)
                {
                    x_Zc = M_trig_L.at<float>(0, 0) * data_3D[3 * w] + M_trig_L.at<float>(0, 1) * data_3D[3 * w + 1] + M_trig_L.at<float>(0, 2) * data_3D[3 * w + 2] + M_trig_L.at<float>(0, 3);
                    y_Zc = M_trig_L.at<float>(1, 0) * data_3D[3 * w] + M_trig_L.at<float>(1, 1) * data_3D[3 * w + 1] + M_trig_L.at<float>(1, 2) * data_3D[3 * w + 2] + M_trig_L.at<float>(1, 3);
                    Zc = M_trig_L.at<float>(2, 0) * data_3D[3 * w] + M_trig_L.at<float>(2, 1) * data_3D[3 * w + 1] + M_trig_L.at<float>(2, 2) * data_3D[3 * w + 2] + M_trig_L.at<float>(2, 3);

                    x = x_Zc / Zc;
                    y = y_Zc / Zc;

                    data_2D[2 * w] = x;
                    data_2D[2 * w + 1] = y;
                }
                else
                {
                    x_Zc = M_trig_R.at<float>(0, 0) * data_3D[3 * w] + M_trig_R.at<float>(0, 1) * data_3D[3 * w + 1] + M_trig_R.at<float>(0, 2) * data_3D[3 * w + 2] + M_trig_R.at<float>(0, 3);
                    y_Zc = M_trig_R.at<float>(1, 0) * data_3D[3 * w] + M_trig_R.at<float>(1, 1) * data_3D[3 * w + 1] + M_trig_R.at<float>(1, 2) * data_3D[3 * w + 2] + M_trig_R.at<float>(1, 3);
                    Zc = M_trig_R.at<float>(2, 0) * data_3D[3 * w] + M_trig_R.at<float>(2, 1) * data_3D[3 * w + 1] + M_trig_R.at<float>(2, 2) * data_3D[3 * w + 2] + M_trig_R.at<float>(2, 3);

                    x = x_Zc / Zc;
                    y = y_Zc / Zc;

                    data_2D[2 * w] = x;
                    data_2D[2 * w + 1] = y;
                }
            }

        }
    }
}



void AutoOiling::Distinguish_Confident3D(cv::Mat& laser_world, cv::Mat& laser_common, cv::Mat& laser_world_filter, cv::Mat& image_project, int thr_horiz, int thr_gray, int flag)
{
    int Camera_height = laser_world.rows;
    int Camera_width = laser_world.cols;

    if (flag == 0)
    {
        cv::Mat Match_L2R;
        laser_world_filter = cv::Mat(Camera_height, Camera_width, CV_32FC3, cv::Scalar::all(0));
        Points_3D_Camera(laser_world, 1, Match_L2R);
        for (int h = 0; h < Camera_height; h++)
        {
            float* data_2D = Match_L2R.ptr<float>(h);
            float* data_3D_filter = laser_world_filter.ptr<float>(h);
            float* data_3D = laser_world.ptr<float>(h);
            for (int w = 0; w < Camera_width; w++)
            {
                int x = round(data_2D[2 * w]);
                int y = round(data_2D[2 * w + 1]);
                if (x > 0 && y > 0 && x < Camera_width && y < Camera_height)
                {
                    float* data_project = image_project.ptr<float>(y);
                    int x_s = std::max(x - thr_horiz, 0);
                    int x_e = std::min(x + thr_horiz, Camera_width);
                    for (int index = x_s; index < x_e; index++)
                    {
                        if (data_project[index] > thr_gray)
                        {
                            data_3D_filter[3 * w] = data_3D[3 * w];
                            data_3D_filter[3 * w + 1] = data_3D[3 * w + 1];
                            data_3D_filter[3 * w + 2] = data_3D[3 * w + 2];
                            break;
                        }
                    }
                }
            }
        }
    }

    if (flag == 1)
    {
        cv::Mat Match_R2L;
        laser_world_filter = cv::Mat(Camera_height, Camera_width, CV_32FC3, cv::Scalar::all(0));
        Points_3D_Camera(laser_world, 0, Match_R2L);
        for (int h = 0; h < Camera_height; h++)
        {
            float* data_2D = Match_R2L.ptr<float>(h);
            float* data_3D_filter = laser_world_filter.ptr<float>(h);
            float* data_3D_common = laser_common.ptr<float>(h);
            float* data_3D = laser_world.ptr<float>(h);
            bool flag_quit = false;
            for (int w = 0; w < Camera_width; w++)
            {
                int x = round(data_2D[2 * w]);
                int y = round(data_2D[2 * w + 1]);
                if (x > 0 && y > 0 && x < Camera_width && y < Camera_height)
                {
                    //**********************************************************************************// Remove Repeate Pointcloud
                 /*   int w_s = std::max(w - thr_horiz, 0);
                    int w_e = std::min(w + thr_horiz, Camera_width);
                    for (int index_w = w_s; index_w < w_e; index_w++)
                    {
                        if (data_3D_common[3 * index_w + 2] > 0)
                        {
                            flag_quit = true;
                        }
                    }
                    if (flag_quit)
                    {
                        break;
                    }*/
                    //**********************************************************************************//

                    float* data_project = image_project.ptr<float>(y);
                    int x_s = std::max(x - thr_horiz, 0);
                    int x_e = std::min(x + thr_horiz, Camera_width);
                    for (int index = x_s; index < x_e; index++)
                    {
                        if (data_project[index] > thr_gray)
                        {
                            data_3D_filter[3 * w] = data_3D[3 * w];
                            data_3D_filter[3 * w + 1] = data_3D[3 * w + 1];
                            data_3D_filter[3 * w + 2] = data_3D[3 * w + 2];
                            break;
                        }
                    }
                }
            }
        }
    }




}


void AutoOiling::intersect_Ref_updated(cv::Mat& src, cv::Mat& laser_world, cv::Mat& laser_common, float line_index, int index, int thr_gray, cv::Mat& image_project, int thr_horiz)
{
    int ht = resolutionHeight;
    int wt = resolutionWidth;
    float  WX, WY, WZ, x_project, y_project;
    int w_possible;
    float  WZ_possible = 1300;
    for (int h = 0; h < ht; h++)
    {
        WZ_possible = 1300;
        float* data = src.ptr<float>(h);
        float* data_world = laser_world.ptr<float>(h);
        float* data_3D_common = laser_common.ptr<float>(h);
        for (int w = 0; w < wt; w++)
        {
            bool flag_exist = false;
            if (data[w] > thr_gray)
            {
                Calculate_world_byRef(float(w), float(h), line_index, WX, WY, WZ, index);
                if (index == 0)
                {
                    Points_3D_Camera_single(WX, WY, WZ, 1, x_project, y_project);
                    int x = round(x_project);
                    int y = round(y_project);
                    if (x > 0 && y > 0 && x < resolutionWidth && y < resolutionHeight)
                    {
                        float* data_project = image_project.ptr<float>(y);
                        int x_s = std::max(x - thr_horiz, 0);
                        int x_e = std::min(x + thr_horiz, resolutionWidth);
                        for (int index_x = x_s; index_x < x_e; index_x++)
                        {
                            if (data_project[index_x] > thr_gray)
                            {
                                flag_exist = true;
                                break;
                            }
                        }
                    }
                    if (WZ < WZ_possible && flag_exist)
                    {
                        WZ_possible = WZ;
                        w_possible = w;

                    }
                    if (flag_exist)
                    {
                        data_world[3 * w + 0] = WX;
                        data_world[3 * w + 1] = WY;
                        data_world[3 * w + 2] = WZ;
                    }

                }
                if (index == 1)
                {
                    Points_3D_Camera_single(WX, WY, WZ, 0, x_project, y_project);
                    int x = round(x_project);
                    int y = round(y_project);
                    if (x > 0 && y > 0 && x < resolutionWidth && y < resolutionHeight)
                    {
                        float* data_project = image_project.ptr<float>(y);
                        int x_s = std::max(x - thr_horiz, 0);
                        int x_e = std::min(x + thr_horiz, resolutionWidth);
                        for (int index_x = x_s; index_x < x_e; index_x++)
                        {
                            if (data_project[index_x] > thr_gray)
                            {
                                flag_exist = true;
                                break;
                            }
                        }
                    }
                    if (WZ < WZ_possible && flag_exist)
                    {
                        WZ_possible = WZ;
                        w_possible = w;
                    }
                }


            }
        }

        if (WZ_possible < 1300)
        {
            Calculate_world_byRef(float(w_possible), float(h), line_index, WX, WY, WZ, index);
            data_world[3 * w_possible + 0] = WX;
            data_world[3 * w_possible + 1] = WY;
            data_world[3 * w_possible + 2] = WZ;

            if (index == 1) //Remove Repeated Pointcloud
            {
                int w_s = std::max(w_possible - thr_horiz, 0);
                int w_e = std::min(w_possible + thr_horiz, resolutionWidth);
                for (int index_w = w_s; index_w < w_e; index_w++)
                {
                    if (data_3D_common[3 * index_w + 2] > 0)
                    {
                        data_world[3 * w_possible + 0] = 0;
                        data_world[3 * w_possible + 1] = 0;
                        data_world[3 * w_possible + 2] = 0;
                        break;
                    }
                }
            }

        }
    }
}


void AutoOiling::Points_3D_Camera_single(float& X, float& Y, float& Z, int camera_index, float& x_p, float& y_p)
{
    float x_Zc, y_Zc, Zc;
    if (camera_index == 0)
    {
        x_Zc = M_trig_L.at<float>(0, 0) * X + M_trig_L.at<float>(0, 1) * Y + M_trig_L.at<float>(0, 2) * Z + M_trig_L.at<float>(0, 3);
        y_Zc = M_trig_L.at<float>(1, 0) * X + M_trig_L.at<float>(1, 1) * Y + M_trig_L.at<float>(1, 2) * Z + M_trig_L.at<float>(1, 3);
        Zc = M_trig_L.at<float>(2, 0) * X + M_trig_L.at<float>(2, 1) * Y + M_trig_L.at<float>(2, 2) * Z + M_trig_L.at<float>(2, 3);

        x_p = x_Zc / Zc;
        y_p = y_Zc / Zc;

    }
    else
    {
        x_Zc = M_trig_R.at<float>(0, 0) * X + M_trig_R.at<float>(0, 1) * Y + M_trig_R.at<float>(0, 2) * Z + M_trig_R.at<float>(0, 3);
        y_Zc = M_trig_R.at<float>(1, 0) * X + M_trig_R.at<float>(1, 1) * Y + M_trig_R.at<float>(1, 2) * Z + M_trig_R.at<float>(1, 3);
        Zc = M_trig_R.at<float>(2, 0) * X + M_trig_R.at<float>(2, 1) * Y + M_trig_R.at<float>(2, 2) * Z + M_trig_R.at<float>(2, 3);

        x_p = x_Zc / Zc;
        y_p = y_Zc / Zc;
    }
}

void AutoOiling::PM_filter_laser(cv::Mat& inputMatrix, cv::Mat& outputMatrix, int vert_size, float laser_size, float dist_thr)
{
    const int height = inputMatrix.rows;
    const int width = inputMatrix.cols;
    outputMatrix = cv::Mat(height, width, CV_32FC3, cv::Scalar::all(0));

    for (int i = 0; i < height; i++) {
        float* ptr_input = inputMatrix.ptr<float>(i);
        float* ptr_output = outputMatrix.ptr<float>(i);
        for (int j = 0; j < width; j++) {
            int num = 0, flag = 0;
            float x = 0, y = 0, z = 0, num_index = 0;
            if (*(ptr_input + 4 * j + 2) != 0)
            {
                for (int k = -vert_size; k < vert_size + 1; k++)
                {
                    if ((i + k) > -1 && (i + k) < height)
                    {
                        float* ptr_input_k = inputMatrix.ptr<float>(i + k);
                        for (int m = -200; m < 200 + 1; m++)
                        {
                            float dif_index = abs(*(ptr_input_k + 4 * (j + m) + 3) - *(ptr_input + 4 * j + 3));
                            if ((j + m) > -1 && (j + m) < width && *(ptr_input_k + 4 * (j + m) + 2) != 0 && dif_index < laser_size)
                            {
                                if (abs(*(ptr_input_k + 4 * (j + m) + 2) - *(ptr_input + 4 * j + 2)) < dist_thr)
                                {
                                    x += (*(ptr_input_k + 4 * (j + m)) * (laser_size - dif_index) / laser_size);
                                    y += (*(ptr_input_k + 4 * (j + m) + 1) * (laser_size - dif_index) / laser_size);
                                    z += (*(ptr_input_k + 4 * (j + m) + 2) * (laser_size - dif_index) / laser_size);
                                    num += 1;
                                    num_index += ((laser_size - dif_index) / laser_size);

                                    /*  x += *(ptr_input_k + 4 * (j + m)) ;
                                      y += *(ptr_input_k + 4 * (j + m) + 1) ;
                                      z += *(ptr_input_k + 4 * (j + m) + 2);
                                      num += 1;
                                      num_index += 1;*/


                                }
                                else
                                {
                                    flag += 1;
                                }
                            }
                        }
                    }
                }
                if (flag > num)// || num < vert_size * laser_size) 
                {
                    x = 0;
                    y = 0;
                    z = 0;
                }
                if (num != 0)
                {
                    *(ptr_output + 3 * j) = x / num_index;
                    *(ptr_output + 3 * j + 1) = y / num_index;
                    *(ptr_output + 3 * j + 2) = z / num_index;
                }
                else
                {
                    *(ptr_output + 3 * j) = 0;
                    *(ptr_output + 3 * j + 1) = 0;
                    *(ptr_output + 3 * j + 2) = 0;
                }
            }
        }
    }
}

