

#ifndef FILM_GRAIN_RENDERING_H
#define FILM_GRAIN_RENDERING_H
	#include <vector>
	#include <cmath>
	#include <iostream>
	#include <stdio.h>
	#include <random>
	#include <chrono>
	#include <sys/time.h>
	#include <fstream>

	#ifdef _OPENMP
	#include <omp.h>
	#endif
	
	#include "matrix.h"
	#include "pseudo_random_number_generator.h"

	#define PIXEL_WISE 0
	#define GRAIN_WISE 1

	#define MAX_GREY_LEVEL 255
	#define EPSILON_GREY_LEVEL 0.1

	template <typename T>
	struct filmGrainOptionsStruct {
		T muR;
		T sigmaR;
		T sigmaFilter;
		unsigned int NmonteCarlo;
		unsigned int algorithmID;
		float s;
		float xA;
		float yA;
		float xB;
		float yB;
		unsigned int mOut;
		unsigned int nOut;
		unsigned int grainSeed;
	};


	matrix<float>* boolean_model(float lambda, float r, float stdGrain);

	int choose_rendering_algorithm(const std::string& inputFile, float muR, float sigmaR);
	float sqDistance(const float x1, const float y1, const float x2, const float y2);
	float render_pixel(float *imgIn, int yOut, int xOut, unsigned int mIn, unsigned int nIn, unsigned int mOut, unsigned int nOut,
	unsigned int offset, unsigned int nMonteCarlo, float grainRadius, float grainSigma, float sigmaFilter,
	float xA, float yA, float xB, float yB, float *lambdaList,
	float *expLambdaList, float *xGaussianList, float *yGaussianList);
	
	matrix<float> *film_grain_rendering_pixel_wise(matrix<float> *imgIn, filmGrainOptionsStruct<float> filmGrainOptions);
	

	matrix<float> *film_grain_rendering_grain_wise(matrix<float> *imgIn, filmGrainOptionsStruct<float> filmGrainOptions);

#endif
