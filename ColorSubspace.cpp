/*
 *  ColorSubspace.cpp
 *  KaToracker
 *
 *  Created by Tetsuro Kato on 10/08/15.
 *  Copyright 2010 Tetsuro Kato. All rights reserved.
 *
 */


#include "ParticleFilter.h"

// コンストラクタ
colorSubspace::colorSubspace()
{
	

}

vector<MatND> colorSubspace::getHistograms(Mat& img)
{
	vector<MatND> hists;
	Mat hsv;
	hsv.create(img.size(), img.type());
	cvtColor(img, hsv, CV_BGR2HSV);
	
	MatND temp;
	int hc[] = {0};
	int sc[] = {1};
	int vc[] = {2};
	const int histSize[] = {20};
	float range[] = {0, 180};
	float svrange[] = {0, 256};
	const float *ranges[] = {range};
	const float *svranges[] = {svrange};
	calcHist(&hsv, 1, hc, Mat(), temp, 1, histSize, ranges, true, false);
	normalization(temp);
	hists.push_back(temp);
	
	temp.release();
	calcHist(&hsv, 1, sc, Mat(), temp, 1, histSize, svranges, true, false);
	normalization(temp);
	hists.push_back(temp);

	temp.release();
	calcHist(&hsv, 1, vc, Mat(), temp, 1, histSize, svranges, true, false);
	normalization(temp);
	hists.push_back(temp);
	
	return hists;
}

// ROI部分画像が入力されること
void colorSubspace::update(Mat& img)
{
	if (this->H_History.size() == 10) {
		H_History.erase(H_History.begin());
		S_History.erase(S_History.begin());
		V_History.erase(V_History.begin());
	}
	
	vector<MatND> hists = getHistograms(img);
	
	H_History.push_back(hists[0]);
	S_History.push_back(hists[1]);
	V_History.push_back(hists[2]);
}

double colorSubspace::calcDistance(Mat& img)
{
	assert(H_History.size()==S_History.size() && S_History.size()==V_History.size());
	
	// 入力画像のヒストグラム
	vector<MatND> hists = getHistograms(img);
	
	double dist = MAXFLOAT;

	for (int i=0; i<H_History.size(); i++) {
		double temp = 0.0;
		temp += compareHist(hists[0], H_History[i], CV_COMP_BHATTACHARYYA);
		temp += compareHist(hists[1], S_History[i], CV_COMP_BHATTACHARYYA);
		temp += compareHist(hists[2], V_History[i], CV_COMP_BHATTACHARYYA);
		
		dist = min(dist, temp);
	}
	
	return dist/3;
}

void colorSubspace::normalization(MatND& mat)
{
	double max;
	minMaxLoc(mat, 0, &max);
	for (int i=0; i<mat.size[0]; i++) {
		mat.at<float>(i) /= max;
	}
}