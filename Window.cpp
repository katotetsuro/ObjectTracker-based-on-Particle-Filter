/*
 *  Window.cpp
 *  KaToracker
 *
 *  Created by Tetsuro Kato on 10/08/14.
 *  Copyright 2010 Tetsuro Kato. All rights reserved.
 *
 */

#include "Window.h"

using namespace cv;

void on_mouse(int event, int x, int y, int flags, void* param)
{
	window* w = (window*)param;
	
	CvRect* rect = &w->rect;
	CvPoint* origin = &w->origin;
	
	if (w->selected) {
		rect->x = min(x, origin->x);
		rect->y = min(y, origin->y);
		rect->width = rect->x + abs(x - origin->x);
		rect->height = rect->y + abs(y - origin->y);
		
		rect->x = max(rect->x, 0);
		rect->y = max(rect->y, 0);
		rect->width = min(rect->width, w->width);
		rect->height = min(rect->height, w->height); //この時点で原点と右下点を指定しているイメージ
		rect->width -= rect->x;
		rect->height -= rect->y; // ここでx,y,width,heightに変換
	}
	
	switch (event) {
		case CV_EVENT_LBUTTONDOWN:
			// 初期化
			origin->x = x;
			origin->y = y;
			rect->x = x;
			rect->y = y;
			w->selected = true;
			break;
			
		case CV_EVENT_LBUTTONUP:
			w->selected = false;
			break;
			
		default:
			break;
	}
}


window::window(string windowName)
{
	name = windowName;
	selected = false;
	
	rect = cvRect(0, 0, 0, 0);
	origin = cvPoint(0, 0);
}

void window::initialize(VideoCapture* capture)
{
	this->capture = capture;
	Mat img;
	if (!capture->isOpened()) {
		printf("failed to capture.¥n");
		return;
	}
	
	*capture >> img;
	this->width = img.cols;
	this->height = img.rows;
	
	namedWindow(name, CV_WINDOW_AUTOSIZE);
	cvSetMouseCallback(name.c_str(), (CvMouseCallback)(&on_mouse), this);
}

void window::run(bool isVideoFile)
{
	printf("マウスで矩形を選択して、よければrを押してください\n");
	Mat frame;
	*capture >> frame;
	do{
		if (!isVideoFile) {
			*capture >> frame;
		}
		Mat showImg = frame.clone();
		rectangle(showImg, cvPoint(rect.x,rect.y), cvPoint(rect.x+rect.width, rect.y+rect.height), CV_RGB(255,255,0));
		imshow(name, showImg);
		
	}while (waitKey(10)!='r');
}

