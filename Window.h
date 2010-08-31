/*
 *  Window.h
 *  KaToracker
 *
 *  Created by Tetsuro Kato on 10/08/14.
 *  Copyright 2010 Tetsuro Kato. All rights reserved.
 *
 */
#pragma once

#include <opencv.hpp>
#include <cstring>

using namespace cv;

class window
{
private:
	string name;
	Mat* pImg;
	VideoCapture* capture;
	
public:
	bool selected;
	int width, height;
	CvRect rect;
	CvPoint origin;
	
	window(string windowName);

	void initialize(VideoCapture* capture);
	void run(bool isVideoFile);
	CvRect getSelectedWindow();
};