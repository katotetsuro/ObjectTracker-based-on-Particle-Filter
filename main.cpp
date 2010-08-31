#include <iostream>

#include <opencv.hpp>
#include "ParticleFilter.h"
#include "Window.h"

using namespace cv;
using namespace std;

int main (int argc, char * const argv[]) {
	VideoCapture capture;
	if (argc>1) {
		string file(argv[1]);
		capture.open(file);
	}else {
		capture.open(0);
	}
	
	if (!capture.isOpened()) {
		cout << "camera not found..." << endl;
		return -1;
	}
	
	// 矩形をゲット
	window selectWin = window("select rectangle to track");
	selectWin.initialize(&capture);
	selectWin.run(argc>1);
	
	Mat src, dst;
	const char* w = "window";
	namedWindow(w, CV_WINDOW_AUTOSIZE);
	kaToracker tracker = kaToracker();
	capture >> src;
	tracker.initialize(src, selectWin.rect, 200);
	do {
		capture >> src;
		
		tracker.update(src);
//		dst = tracker.getResultImage();
		dst = tracker.getParticleImage();
		imshow(w, dst);
		
	} while (waitKey(10)!='q');
	
    return 0;
}
