/*
 *  ParticleFilter.h
 *  KaToracker
 *
 *  Created by Tetsuro Kato on 10/08/10.
 *  Copyright 2010 Tetsuro Kato. All rights reserved.
 *
 */

#pragma once

#include <vector>
#include <opencv.hpp>

#include "ParticleFilter.h"

using namespace std;
using namespace cv;

class particle
{
public:
	int x;
	int y;
	double weight;
	
	particle()
	{
		x = y = 0;
		weight = 0.0;
	}
	
	particle(int x, int y, double weight)
	{
		this->x = x;
		this->y = y;
		this->weight = weight;
	}
	
	// コピーコンストラクタ
	particle(const particle& obj)
	{
		this->x = obj.x;
		this->y = obj.y;
		this->weight = obj.weight;
	}
};

class colorSubspace {
private:
	vector<MatND> H_History;
	vector<MatND> S_History;
	vector<MatND> V_History;
	
	vector<MatND> getHistograms(Mat& img);
	void normalization(MatND& mat);
	
public:
	colorSubspace();
	void update(Mat& image);
	double calcDistance(Mat& image);
};

class matchingPair {
public:
	Point prev;
	Point next;
	Point motion;
	matchingPair(Point prev, Point next)
	{
		this->prev = prev;
		this->next = next;
		motion.x = next.x - prev.x;
		motion.y = next.y - prev.y;
	}
	
	bool contains(const CvRect rect)
	{
		return rect.x < next.x && next.x < rect.x+rect.width
		&& rect.y < next.y && next.y < rect.y+rect.height;
	}
};

class motionCue {
private:
	
//	Mat nextImg
	Mat prevImg;
	vector<Point2f> prevPts; // vectorの方が間違いなさそうなので
	vector<matchingPair> pairList;
	// OpenCV的に必要なもの
	IplImage *eig_img, *temp_img, *prev_pyramid, *curr_pyramid;
//	CvPoint2D32f *nextCorners;
	char* status;
	CvTermCriteria criteria;
	
	static const int maxCorner = 50;
	static const double quality = 0.05;
	static const double minDist = 5;
	
public:
	motionCue();
	~motionCue();
	void initialize(Mat& img, const CvRect rect);
	void update(Mat& img, const CvRect rect);
	void calcOpticalFlow(Mat& img);
	double evaluateMotion(const int x, const int y,const CvRect rect, const Point center);
	
};


class kaToracker  {
private:
	vector<particle> particleList;
	
//	Mat previousImage;
	Mat currentImage;
	
	int width, height;
	int nParticle;
	
	// トラッキング矩形
	CvRect trackingRect;
	
	// トラッキング中心
	CvPoint center;
	
	// 乱数ジェネレータ
	RNG rng;
	
	// カラーヒストグラム
	colorSubspace colorSubs;
	
	// モーション
	motionCue motion;
	
	double calcLikelihood(const int x, const int y);
	void setLikelihood(particle& p);
	
	void resampling();
	void predict();
	void updateWeight();
	void calcCenterPoint();
	
	bool isYellow(const int x, const int y);
	
	inline int checkWidth(int cand_width);
	inline int checkHeight(int cand_height);
	inline bool inImage(const int x, const int y);
	
	
public:
	kaToracker();
	
	// 初期化	
	void initialize(Mat& img, CvRect rect, const int num_particle);
	
	// 次のフレームを受け取ってトラッカーを更新
	void update(Mat& img);
	
	// 結果の矩形を取得
	CvRect getRectangle();
	
	// トラッキング結果矩形を描画した画像を取得する.
	Mat getResultImage();
	
	// パーティクルと矩形を描画した画像を取得する.
	Mat getParticleImage();

};
