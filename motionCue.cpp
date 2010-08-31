/*
 *  motionCue.cpp
 *  KaToracker
 *
 *  Created by Tetsuro Kato on 10/08/22.
 *  Copyright 2010 Tetsuro Kato. All rights reserved.
 *
 */

#include "ParticleFilter.h"
#include <opencv.hpp>
#include <algorithm>

motionCue::motionCue()
{
	
}

// デストラクタ
motionCue::~motionCue()
{
	cvReleaseImage (&eig_img);
	cvReleaseImage (&temp_img);
	cvReleaseImage (&prev_pyramid);
	cvReleaseImage (&curr_pyramid);	
}

void motionCue::initialize(Mat& img, const CvRect rect)
{
	prevImg.create(img.size(), CV_8U);
	cvtColor(img, prevImg, CV_BGR2GRAY, 1);
	
	// 怒りのサンプルコードコピペ
	IplImage iplHeader = prevImg;
	IplImage *src_img1 = &iplHeader;

	int corner_count = motionCue::maxCorner; // 最大検出点数

	eig_img = cvCreateImage (cvGetSize (src_img1), IPL_DEPTH_32F, 1);
	temp_img = cvCreateImage (cvGetSize (src_img1), IPL_DEPTH_32F, 1);
	CvPoint2D32f *nextCorners = (CvPoint2D32f *) cvAlloc (corner_count * sizeof (CvPoint2D32f));
	
	prev_pyramid = cvCreateImage (cvSize (src_img1->width + 8, src_img1->height / 3), IPL_DEPTH_8U, 1);
	curr_pyramid = cvCreateImage (cvSize (src_img1->width + 8, src_img1->height / 3), IPL_DEPTH_8U, 1);
	criteria = cvTermCriteria (CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 64, 0.01);
	
	// (2)疎な特徴点を検出
	cvSetImageROI(src_img1, rect);
	cvGoodFeaturesToTrack (src_img1, eig_img, temp_img, nextCorners, &corner_count, 0.001, 5, NULL);
	cvResetImageROI(src_img1);
	
	// vectorに詰める
	for (int i=0; i<corner_count; i++) {
		Point p(nextCorners[i].x+rect.x, nextCorners[i].y+rect.y);
		prevPts.push_back(p);
	}
	
	// debug
/*	Mat debug = img.clone();
	for (int i=0,len=prevPts.size(); i<len; i++) {
		circle(debug, prevPts[i], 3, CV_RGB(255,255,0));
	}
	namedWindow("debug");
	imshow("debug", debug);
	waitKey(10);*/
	
	// free
//	cvFree_(&nextCorners);
}
// 次のステップで必要になる、特徴点を抽出する
void motionCue::update(Mat& img, const CvRect rect)
{
	assert(prevImg.size()==img.size());
	cvtColor(img, prevImg, CV_BGR2GRAY, 1);
	
	IplImage iplHeader = prevImg;
	IplImage *src_img1 = &iplHeader;
	
	int corner_count = motionCue::maxCorner; // 最大検出点数
	CvPoint2D32f *nextCorners = (CvPoint2D32f *) cvAlloc (corner_count * sizeof (CvPoint2D32f));	
	
	// (2)疎な特徴点を検出
	cvGoodFeaturesToTrack (src_img1, eig_img, temp_img, nextCorners, &corner_count, 0.001, 5, NULL);
	
	for (int i=0; i<corner_count; i++) {
		Point p(nextCorners[i].x+rect.x, nextCorners[i].y+rect.y);
		prevPts.push_back(p);
	}
	
	// free
	//	cvFree_(&nextCorners);
	
}
// 全体のモーションフィールドを計算する
void motionCue::calcOpticalFlow(Mat& img)
{
	Mat next;
	next.create(img.size(), CV_8UC1);
	cvtColor(img, next, CV_BGR2GRAY, 1);
	
	IplImage tempIpl = prevImg;
	IplImage *src_img1 = &tempIpl;
	IplImage tempIpl2 = next;
	IplImage *src_img2 = &tempIpl2;
	
	int corner_count = prevPts.size();
	CvPoint2D32f *corners1 = (CvPoint2D32f *) cvAlloc (corner_count * sizeof (CvPoint2D32f));
	for (int i=0; i<prevPts.size(); i++) {
		corners1[i] = prevPts[i];
	}
	CvPoint2D32f *corners2 = (CvPoint2D32f *) cvAlloc (corner_count * sizeof (CvPoint2D32f));
	status = (char*)cvAlloc(corner_count);
	cvCalcOpticalFlowPyrLK (src_img1, src_img2, prev_pyramid, curr_pyramid,
							corners1, corners2, corner_count, cvSize (10, 10), 4, status, NULL, criteria, 0);
	pairList.clear();
	for (int i=0; i<corner_count; i++) {
		if (status[i]) {
			matchingPair p(corners1[i], corners2[i]);
			pairList.push_back(p);
		}
	}
}

// パーティクルの周辺領域を用いて、１つのパーティクルの尤度を評価
double motionCue::evaluateMotion(const int x, const int y, const CvRect neighbor, const Point center)
{
	// すでに計算されている、全体のモーションフィールドから、
	// このパーティクルの近傍領域の部分を抜き出して、モーションを評価する
	// 初期化のとき、１回目のループ、では、計算不可能
	if (pairList.size()==0 || (center.x==0&&center.y==0)) {
		return 0.0;
	}
	vector<double> xList, yList;
	for (int i=0; i<pairList.size(); i++) {
		if (pairList[i].contains(neighbor)) {
			xList.push_back(pairList[i].motion.x);
			yList.push_back(pairList[i].motion.y);
		}
	}
	// 対応点が見つからないとき
	if (xList.size()==0) {
		return 5.0; // やばい、このときどうしよう
	}
	std::sort(xList.begin(), xList.end());
	std::sort(yList.begin(), yList.end());
	double medianX = xList[xList.size()/2];
	double medianY = yList[yList.size()/2];
	
	double estimateX = center.x + medianX;
	double estimateY = center.y + medianY;
	
	double dist = sqrt((estimateX - x)*(estimateX - x) + (estimateY - y) * (estimateY - y));
	
	return dist;
}