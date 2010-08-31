/*
 *  ParticleFilter.cpp
 *  KaToracker
 *
 *  Created by Tetsuro Kato on 10/08/13.
 *  Copyright 2010 Tetsuro Kato. All rights reserved.
 *
 */

#include "ParticleFilter.h"


kaToracker::kaToracker()
{
	this->rng = cv::RNG();
}

void kaToracker::initialize(Mat& img, CvRect rect, const int num_particle)
{
//	this->previousImage = img.clone();
	this->currentImage	= img.clone();
	
	this->width = img.cols;
	this->height = img.rows;
	this->nParticle = num_particle;
	this->trackingRect = rect;
	
	// カラーヒストグラム
	Mat subImage = img.colRange(rect.x, rect.x+rect.width).rowRange(rect.y, rect.y+rect.height);
	colorSubs.update(subImage);
	
	// モーション
	motion.initialize(img, rect);
	
	particle max_particle = particle();
	for (int y=trackingRect.y; y<trackingRect.y+trackingRect.height ; y++) {
		for (int x=trackingRect.x; x<trackingRect.x+trackingRect.width; x++) {
			double temp = calcLikelihood(x, y);
			if(temp > max_particle.weight){
				max_particle.x = x;
				max_particle.y = y;
				max_particle.weight = temp;
			}
		}
	}
	
	for (int i=0; i<nParticle; i++) {
		this->particleList.push_back(max_particle);
	}
}


void kaToracker::resampling()
{
	// 累積重みの計算
	double* cumulation = new double[this->nParticle];
	cumulation[0] = particleList[0].weight;
	for (int i=1; i<nParticle; i++) {
		cumulation[i] = cumulation[i-1] + particleList[i].weight;
	}
	
	// 現在のパーティクルリストを一時退避
	vector<particle> temp(particleList); // コピー
	for (int i=0; i<nParticle; i++) {
		// 重いやつはひっかかりやすくなるってことかな
		// でも重みを1.0にセットするってのはどうなんだろうかね
		const double weight = rng.uniform(0.0, 1.0) * cumulation[nParticle - 1];
		int n=0;
		while (cumulation[++n] < weight);
		particleList[i] = temp[n]; // 代入演算子こわいなぁ
		particleList[i].weight = 1.0;
	}
	
	delete [] cumulation;
}

void kaToracker::predict()
{
//	const double variance = sqrt(13.0); // ルートとった方がいいかも
	const double variance = 13.0; // ルートとった方がいいかも
	
	for (int i=0; i<nParticle; i++) {
		int vx = static_cast<int>(rng.gaussian(variance));
		int vy = static_cast<int>(rng.gaussian(variance));
		
		particleList[i].x += vx;
		particleList[i].y += vy;
	}

}

void kaToracker::updateWeight()
{
	// まずはmotionを更新
	motion.calcOpticalFlow(this->currentImage);
	// 尤度（＝重み）を再計算する
	double sumWeight = 0.0;
	for (int i=0; i<nParticle; i++) {
		particleList[i].weight = calcLikelihood(particleList[i].x, particleList[i].y);
		sumWeight = particleList[i].weight;
	}
	// 重みを正規化しておく
	for (int i=0; i<nParticle; i++) {
		particleList[i].weight = (particleList[i].weight / sumWeight) * nParticle;
	}
}

// 中心点を計算する
void kaToracker::calcCenterPoint()
{
	double x, y, weight;
	x = y = weight = 0.0;
	
	// 重み付き和
	for (int i=0; i<nParticle; i++) {
		x += particleList[i].x * particleList[i].weight;
		y += particleList[i].y * particleList[i].weight;
		weight += particleList[i].weight;
	}
	
	// 正規化
	center.x = checkWidth(static_cast<int> (x / weight));
	center.y = checkHeight(static_cast<int> (y / weight));
}

// 尤度の計算
// x,yはパーティクルの座標
double kaToracker::calcLikelihood(int x, int y)
{
	const int margin = 15;
	int x0 = checkWidth(x-margin);
	int x1 = checkWidth(x+margin);
	int y0 = checkHeight(y-margin);
	int y1 = checkHeight(y+margin);
	
	Mat targetRect = currentImage.colRange(x0, x1).rowRange(y0,y1);
	double dist = colorSubs.calcDistance(targetRect);
	// ほんとはmotionCueからも加算される
	int rect_x = checkWidth(x-trackingRect.width);
	int rect_y = checkHeight(y-trackingRect.height);
	Mat neighborRegion = currentImage.colRange(rect_x, rect_x+trackingRect.width).rowRange(rect_y, rect_y+trackingRect.height);
	double coef = 0.01; // モーションの項に乗じる係数
	double motionDist = coef * motion.evaluateMotion(x, y, trackingRect, this->center);
	dist += motionDist;
	double sigma = 0.1; // 分散はこれくらいが良いby 論文
	double likellihood = exp(-dist/sigma);
	return likellihood;
}

void kaToracker::update(Mat& img)
{
	// ディープコピーばっかりして！！
//	previousImage = currentImage.clone();
	currentImage = img.clone();
	
	resampling();
	this->predict();
	updateWeight();
	calcCenterPoint();
	// 求めた矩形を使ってモーションを更新
	motion.update(currentImage, trackingRect);
}

CvRect kaToracker::getRectangle()
{
	// とりあえず適当に実装
/*	int x1 = checkWidth(center.x);
	int y1 = checkHeight(center.y);
	int x2 = checkWidth(x1 + 50);
	int y2 = checkHeight(y1 + 50);*/
	trackingRect.x = checkWidth(center.x - trackingRect.width / 2);
	trackingRect.y = checkHeight(center.y - trackingRect.height / 2);
		
	return trackingRect;
}

Mat kaToracker::getResultImage()
{
	Mat dst = currentImage.clone();
	CvRect rect = getRectangle();
	rectangle(dst, cvPoint(rect.x, rect.y), cvPoint(rect.x+rect.width, rect.y+rect.height), CV_RGB(250, 250, 50));
	
	return dst;
}

Mat kaToracker::getParticleImage()
{
	Mat dst = getResultImage();
	for (int i=0; i<nParticle; i++) {
		circle(dst, cvPoint(particleList[i].x, particleList[i].y), 2, CV_RGB(255,0,0));
	}
	return dst;
	
}

// 黄色か判定する.
bool kaToracker::isYellow(const int x, const int y)
{
	
//	printf("%d,%d,%d",currentImage.data[currentImage.step*y + x*3 + 0],
//		   currentImage.data[currentImage.step*y + x*3 + 1],
//		   currentImage.data[currentImage.step*y + x*3 + 2]);
	
	bool yellow =
	(uchar)currentImage.data[currentImage.step*y + x*3 + 0] < 50
	&& (uchar)currentImage.data[currentImage.step*y + x*3 + 1] < 50
	&& (uchar)currentImage.data[currentImage.step*y + x*3 + 2] < 50;
	
	return yellow;
}

int kaToracker::checkWidth(int cand_width)
{
	return min(max(0, cand_width), width);
}

int kaToracker::checkHeight(int cand_height)
{
	return min(max(0, cand_height), height);
}

bool kaToracker::inImage(const int x, const int y)
{
	if (x < 0 || width <= x) {
		return false;
	}
	if (y < 0 || height <= y) {
		return false;
	}
	
	return true;
}

