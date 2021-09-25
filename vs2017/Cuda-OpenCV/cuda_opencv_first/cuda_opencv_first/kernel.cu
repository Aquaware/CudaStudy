#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <opencv2/opencv.hpp>

#ifdef _DEBUG
#pragma comment(lib, "opencv_world453d.lib")
#else
#pragma comment(lib, "opencv_world453.lib")
#endif

#include <stdio.h>
#include <iostream>
#include <time.h>

__global__
void rgbToGray(uchar3 *rgb, unsigned char* gray) {
	int index = blockIdx.x*blockDim.x + threadIdx.x;
	gray[index] = (unsigned char)(0.299f * rgb[index].x
		+ 0.587f * (float)rgb[index].y
		+ 0.114f * (float)rgb[index].z);
	//if (index >= 1000 && index < 1020) {
	//	printf("index: %d  color: %d, %d, %d => gray: %d\n", index, rgb[index].x, rgb[index].y, rgb[index].z, gray[index]);
	//}
}

int main(void) {
	cv::Mat image = cv::imread("C:\\Users\\docs9\\Dropbox\\PC\\Desktop\\test.bmp", 1);
	if (image.empty()) {
		std::cerr << "Error : cannot find input image" << std::endl;
		return 1;
	}

	int width = image.cols;
	int height = image.rows;
	int pixels = width * height;
	std::cout << "size : " << width << " x " << height << std::endl;

	unsigned char* gray = new unsigned char[pixels];
	uchar3* d_rgb;
	unsigned char* d_gray;
	int rgb_bytes = sizeof(uchar3) * pixels;
	int gray_bytes = sizeof(unsigned char) * pixels;

	cudaMalloc((void**)&d_rgb, rgb_bytes);
	cudaMalloc((void**)&d_gray, gray_bytes);
	cudaMemcpy(d_rgb, image.data, rgb_bytes, cudaMemcpyHostToDevice);
	rgbToGray << <width * height, 1 >> > (d_rgb, d_gray);
	cudaMemcpy(gray, d_gray, gray_bytes, cudaMemcpyDeviceToHost);
	cv::Mat1b out(height, width, gray);

	cv::imshow("", out);
	cv::waitKey(0);
	cv::destroyAllWindows();
	cudaFree(d_rgb);
	cudaFree(d_gray);
	delete gray;
}
