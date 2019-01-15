#include <bits/stdc++.h>
#include <cv.h>
#include <highgui.h>
#include "opencv2/opencv.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/ml.hpp"
using namespace std;

int main(){
	string s;
	s = "ls train/ | grep ^Left >L.txt";
	system(s.c_str());
	ifstream ls_res1_file("L.txt");
	s = "ls train/ | grep ^Up >U.txt";
	system(s.c_str());
	ifstream ls_res2_file("U.txt");
	s = "ls train/ | grep ^Right >R.txt";
	system(s.c_str());
	ifstream ls_res3_file("R.txt");
	/*while(ls_res1_file.good()){
		string imName;
		getline(ls_res1_file,imName);
		cout<<imName<<endl;
		if(imName.empty())continue;
		imName = "train/"+imName;
		cv::Mat img = cv::imread(imName,1);
		if(img.rows==227 && img.cols==227)continue;
		/*cv::imshow("img",img);
		cv::waitKey(0);*/
	/*	img = img(cv::Rect(0,0,640,360));
		cv::Mat img2(640,640,CV_8UC3,cv::Scalar(0));
		cv::Mat rect2= img2(cv::Rect(0,140,640,360));
		img.copyTo(rect2);
		cv::resize(img2,img2,cv::Size(227,227));
		/*cv::imshow("img",img2);
		cv::waitKey(0);*/
	/*	cv::imwrite(imName,img2);
	}*/
	while(ls_res2_file.good()){
		string imName;
		getline(ls_res2_file,imName);
		cout<<imName<<endl;
		if(imName.empty())continue;
		imName = "train/"+imName;
		cv::Mat img = cv::imread(imName,1);
		if(img.rows==227 && img.cols==227)continue;
		/*cv::imshow("img",img);
		cv::waitKey(0);*/
		img = img(cv::Rect(0,0,640,360));
		cv::Mat img2(640,640,CV_8UC3,cv::Scalar(0));
		cv::Mat rect2= img2(cv::Rect(0,140,640,360));
		img.copyTo(rect2);
		cv::resize(img2,img2,cv::Size(227,227));
		/*cv::imshow("img",img2);
		cv::waitKey(0);*/
		cv::imwrite(imName,img2);
	}
	while(ls_res3_file.good()){
		string imName;
		getline(ls_res3_file,imName);
		cout<<imName<<endl;
		if(imName.empty())continue;
		imName = "train/"+imName;
		cv::Mat img = cv::imread(imName,1);
		if(img.rows==227 && img.cols==227)continue;
		/*cv::imshow("img",img);
		cv::waitKey(0);*/
		img = img(cv::Rect(0,0,640,360));
		cv::Mat img2(640,640,CV_8UC3,cv::Scalar(0));
		cv::Mat rect2= img2(cv::Rect(0,140,640,360));
		img.copyTo(rect2);
		cv::resize(img2,img2,cv::Size(227,227));
		/*cv::imshow("img",img2);
		cv::waitKey(0);*/
		cv::imwrite(imName,img2);
	}
	return 0;
}