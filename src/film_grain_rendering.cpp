
#include "libtiff_io.h"
#include "pseudo_random_number_generator.h"
#include "film_grain_rendering.h"


/**
 * 
 */
 /**
* @brief Choose algorithm to use
*
* @param File name of the file containing empirical timings
* @return 0 if pixel-wise algorithm, 1 if grain-wise algorithm
*/
int choose_rendering_algorithm(const std::string& inputFile, float muR, float sigmaR)
{
	std::string line;
    std::ifstream myfile (inputFile);

    //get number of lines in input file
    int nbLines = 0;
    while (std::getline(myfile, line))
        ++nbLines;

	//read file data
    float *switchPoints = new float[2*nbLines];

    float a,b;
    int cntr=0;
    FILE *fid = fopen ( (const char*)(inputFile.c_str()),"r");
    if(fid != NULL)
    {
        while ( !feof(fid) )
        {
            if (fscanf(fid, "%f %f\n", &a, &b) != EOF)
            {
	            switchPoints[cntr++] = a;
	        	switchPoints[cntr++] = b;
            }
            else
            {
            	delete switchPoints;
            	std::cout << "Error in choose_rendering_algorithm, failure to read the switch point file." << std::endl;
				return(PIXEL_WISE);
            }

        }
    }
    else
    {
    	delete switchPoints;
    	std::cout << "Error in choose_rendering_algorithm, failure to read the switch point file." << std::endl;
		return(PIXEL_WISE);
    }
	fclose (fid);

	//choose best algorithm
	for (int i=0; i<nbLines; i++)
	{
		if ( (muR) <= switchPoints[2*i])	//we have found the right radius
		{
			//the standard deviation is small, so we use the pixel-wise algorithm
			if ( (sigmaR) <= switchPoints[2*i+1] )
			{
				delete switchPoints;
				return(PIXEL_WISE);
			}
			else //the standard deviation is large, so we use the grain-wise algorithm
			{
				delete switchPoints;
				return(GRAIN_WISE);
			}
		}
	}
	delete switchPoints;
	return(PIXEL_WISE);
}


/**
 * 
 */
 /**
* @brief Square distance 
*
* @param lambda parameter of the Poisson process
* @param x1, y1 : x, y coordinates of the first point
* @param x2, y2 : x, y coordinates of the second point
* @return squared Euclidean distance
*/
double sqDistance(const double x1, const double y1, const double x2, const double y2)
{
	return((x1-x2)*(x1-x2) + (y1-y2)*(y1-y2));
}

/**
 * 
 */
 /**
* @brief Render one pixel in the pixel-wise algorithm
*
* @param imgIn input image
* @param yOut, xOut : coordinates of the output pixel
* @param mIn, nIn : input image size
* @param mOut, nOut : output image size
* @param offset : offset to put into the pseudo-random number generator
* @param nMonteCarlo : number of iterations in the Monte Carlo simulation
* @param grainRadius : average grain radius
* @param sigmaR : standard deviation of the grain radius
* @param sigmaFilter : standard deviation of the blur kernel
* @param (xA,yA), (xB,yB) : limits of image to render
* @return output value of the pixel
*/
float render_pixel(float *imgIn, int yOut, int xOut, unsigned int mIn, unsigned int nIn, unsigned int mOut, unsigned int nOut,
	unsigned int offset, unsigned int nMonteCarlo, float grainRadius, float sigmaR, float sigmaFilter,
	float xA, float yA, float xB, float yB, float *lambdaList,
	float *expLambdaList, float *xGaussianList, float *yGaussianList)
{
	float normalQuantile = 3.0902;//2.3263;	//standard normal quantile for alpha=0.999
	float logNormalQuantile;
	float grainRadiusSq = grainRadius*grainRadius;
	float maxRadius = grainRadius;
	float mu=0.0,sigma=0.0,sigmaSq;
	float currRadius,currGrainRadiusSq;

	float ag = 1/ceil(1/grainRadius);
	float sX = ((float)(nOut-1))/((float)(xB-xA)); 
	float sY = ((float)(mOut-1))/((float)(yB-yA));
	
	//random generator for Monte Carlo
	noise_prng pMonteCarlo;
	mysrand(&pMonteCarlo, 2016u*offset);
	noise_prng p,pTest;
	mysrand(&pTest, 2016u*offset);
	
	float pixOut = 0.0;

	//conversion from output grid (xOut,yOut) to input grid (xIn,yIn)
	//we inspect the middle of the output pixel (1/2)
	//the size of a pixel is (xB-xA)/nOut
	float xIn = xA + (xOut+(float)0.5 ) * ((xB-xA)/(nOut)); 	//((float)xOut)/sX;	//float xIn = ((float)xOut);
	float yIn = yA + (yOut+(float)0.5 ) * ((yB-yA)/(mOut));   //((float)yOut)/sY;//float yIn = ((float)yOut);

	//calculate the mu and sigma for the lognormal distribution
	if (sigmaR > 0.0)
	{
		sigma = sqrt(log( (sigmaR/grainRadius)*(sigmaR/grainRadius) + (float)1.0));
		sigmaSq = sigma*sigma;
		mu = log(grainRadius)-sigmaSq/((float)2.0);
		logNormalQuantile = exp(mu + sigma*normalQuantile);
		maxRadius = logNormalQuantile;
	}

	//loop over the number of Monte Carlo simulations
	for (unsigned int i=0; i<nMonteCarlo; i++)
	{

		float xGaussian = xIn + sigmaFilter*(xGaussianList[i])/sX;
		float yGaussian = yIn + sigmaFilter*(yGaussianList[i])/sY;

		//determine the bounding boxes around the current shifted pixel
		// this was where the floating point precision was insufficient (for the additions and subtractions)
		unsigned int minX = (unsigned int)floor( ( (double)xGaussian - (double)maxRadius)/((double)ag));
		unsigned int maxX = (unsigned int)floor( ( (double)xGaussian + (double)maxRadius)/((double)ag));
		unsigned int minY = (unsigned int)floor( ( (double)yGaussian - (double)maxRadius)/((double)ag));
		unsigned int maxY = (unsigned int)floor( ( (double)yGaussian + (double)maxRadius)/((double)ag));

		bool ptCovered = false; // used to break all for loops
		for(unsigned int ncx = minX; ncx <= maxX; ncx++) /* x-cell number */
		{
			if(ptCovered == true)
				break;
			for(unsigned int ncy = minY; ncy <= maxY; ncy++) /* y-cell number */
			{
				if(ptCovered == true)
					break;
				// cell corner in pixel coordinates
				// note : we put everything to do with the cell positions to double precision, because
				// there can be MANY of them (depending on the grain size), and floating point precision is not sufficient
				double cellCornerX = ((double)ag)*((double)ncx);
		        double cellCornerY = ((double)ag)*((double)ncy);

				/* seed cell = (x/w, y/w) */
				unsigned int seed = cellseed(ncx,ncy, offset);
				mysrand(&p, seed);

				// Compute the Poisson parameters for the pixel that contains (x,y)
				float u = imgIn[  (unsigned int)(fmin(fmax(floor(cellCornerY),0.0),(float)(mIn-1)))*nIn+
				(unsigned int)(fmin(fmax(floor(cellCornerX),0.0),(float)(nIn-1)))];
				int uInd = (int)floor(u*((float)(MAX_GREY_LEVEL+EPSILON_GREY_LEVEL)));
				float currLambda = lambdaList[uInd];
				float currExpLambda = expLambdaList[uInd];

				// Draw number of points in the cell
        		unsigned int Ncell = my_rand_poisson(&p, currLambda, currExpLambda);
        		for(unsigned int k=0; k<Ncell; k++)
        		{
					//draw the grain centre
					double xCentreGrain = cellCornerX + ((double)ag)*((double)myrand_uniform_0_1(&p));
			        double yCentreGrain = cellCornerY + ((double)ag)*((double)myrand_uniform_0_1(&p));

					//draw the grain radius
					if (sigmaR>0.0)
					{
						//draw a random Gaussian radius, and convert it to log-normal
						currRadius = fmin(exp(mu + sigma*myrand_gaussian_0_1(&p)),maxRadius);
						currGrainRadiusSq = currRadius*currRadius;
					}
					else if (sigmaR ==  0.0)
						currGrainRadiusSq = grainRadiusSq;
					else
					{
						std::cout << "Error, the standard deviation of the grain should be positive." << std::endl;
					}

					// test distance
					if(sqDistance(xCentreGrain,yCentreGrain, xGaussian,yGaussian) < (double)currGrainRadiusSq)
					{
						pixOut = pixOut+(float)1.0;
						ptCovered = true;
						break;
					}
				}
			}
		}
		ptCovered = false;
	}
	return(pixOut/((float)nMonteCarlo));
}



/**
 * 
 */
 /**
* @brief Pixel-wise film grain rendering algorithm
*
* @param imgIn input image
* @param filmGrainOptions : film grain rendering options
* @return output, rendered image
*/
matrix<float>* film_grain_rendering_pixel_wise(matrix<float> *imgIn, filmGrainOptionsStruct<float> filmGrainOptions)
{
	/*********************************************/
	/***************   PARAMETERS   **************/
	/*********************************************/

	//parameters for film grain
    float grainRadius = filmGrainOptions.muR;//grainRadius/pixelSize;
    float grainStd = filmGrainOptions.sigmaR;
	float sigmaFilter = filmGrainOptions.sigmaFilter;

	std::cout << "min : " << imgIn->min() << std::endl;
	std::cout << "max : " << imgIn->max() << std::endl;

	/*********************************************/
	/***   MONTE CARLO TRANSLATION VECTORS   *****/
	/*********************************************/

	int NmonteCarlo = filmGrainOptions.NmonteCarlo;
	std::random_device rd;
    std::mt19937 rndGenerator(rd());

	//draw the random (gaussian) translation vectors
    float *xGaussianList = new float[NmonteCarlo];
    float *yGaussianList = new float[NmonteCarlo];
	std::normal_distribution<float> normalDistribution(0.0,sigmaFilter);
	for (int i=0; i<NmonteCarlo; i++)
	{
		xGaussianList[i] = (float)normalDistribution(rndGenerator);
		yGaussianList[i] = (float)normalDistribution(rndGenerator);
	}

	//pre-calculate lambda and exp(-lambda) for each possible grey-level
	float *lambdaList = new float[ MAX_GREY_LEVEL +1];
	float *expLambdaList = new float[ MAX_GREY_LEVEL +1];
	for (int i=0; i<=MAX_GREY_LEVEL; i++)
	{
		float u = ((float)i)/( (float) ( (float)MAX_GREY_LEVEL + (float)EPSILON_GREY_LEVEL) );
		float ag = 1/ceil(1/grainRadius);
		float lambdaTemp = -((ag*ag) /
			( pi*(grainRadius*grainRadius + grainStd*grainStd))) * log(1.0f-u);
		lambdaList[i] = lambdaTemp;
		expLambdaList[i] = exp(-lambdaTemp);
	}

	//create list of temporary images, and set them to 0
	matrix<float> *imgOut = new matrix<float>( filmGrainOptions.mOut, filmGrainOptions.nOut);
	std::cout<< "image output size : " << imgOut->get_ncols() << " x " << imgOut->get_nrows() << std::endl;

	unsigned int i,j;
	float *ptrTemp,pixTemp;
	#pragma omp parallel for schedule(dynamic, 2) private(i,j,ptrTemp,pixTemp) shared(imgOut)
	for (i=0; i<(unsigned int)imgOut->get_nrows(); i++)
	{
		for (j=0; j<(unsigned int)imgOut->get_ncols(); j++)
		{
			ptrTemp = (float*)imgIn->get_ptr();
			pixTemp = render_pixel(ptrTemp, i, j, (unsigned int)imgIn->get_nrows(), (unsigned int)imgIn->get_ncols(),
				(unsigned int)imgOut->get_nrows(), (unsigned int)imgOut->get_ncols(),
				filmGrainOptions.grainSeed, NmonteCarlo, grainRadius,grainStd,sigmaFilter,
				filmGrainOptions.xA, filmGrainOptions.yA,
				filmGrainOptions.xB, filmGrainOptions.yB,
				lambdaList, expLambdaList,
				xGaussianList,yGaussianList);
			
			imgOut->set_value(i,j,pixTemp);
		}
	}

	delete xGaussianList;
	delete yGaussianList;
	return(imgOut);
}



/**
 * 
 */
 /**
* @brief Generate local Boolean model information 
*
* @param lambda parameter of the Poisson process
* @param r average grain radius
* @param stdGrain standard deviation of the grain radii
* @param distributionType 'constant' or 'log-normal' grain radii
* @param xPixel x coordinate of the cell
* @param yPixel y coordinate of the cell
* @return list of grains : [xCentre yCentre radius]
*/
matrix<float>* boolean_model(float lambda, float r, float stdGrain)
{

    std::random_device rd;
    std::mt19937 rndGenerator(rd());
    //draw the number of points from a Poisson distribution
	std::poisson_distribution<int> poissonDistribution(lambda);
	int nDots = poissonDistribution(rndGenerator);
	matrix<float> * grainModelOut = new matrix<float>(nDots,4);

	std::uniform_real_distribution<float> uniformDistribution(0.0,1.0);
	for (unsigned int i=0; i<( (unsigned int)nDots); i++)
	{
		grainModelOut->set_value(i,0,uniformDistribution(rndGenerator));
		grainModelOut->set_value(i,1,uniformDistribution(rndGenerator));
	}

	//set radius of each grain
    if ( stdGrain == 0.0)	//constant radius
	{
        for (unsigned int i=0; i<( (unsigned int)nDots); i++)
		{
			grainModelOut->set_value(i,2,r);
		}
	}
    else 	//random radius (with log-normal distribution)
	{
        float sigmaSquare = log( (stdGrain/r)*(stdGrain/r) + 1);
        float sigma = sqrt( sigmaSquare);
        float mu = log(r)-sigmaSquare/(2.0);
		std::lognormal_distribution<float> logDistribution(mu,sigma);

		for (unsigned int i=0; i<( (unsigned int)nDots); i++)
		{
			grainModelOut->set_value(i,2,logDistribution(rndGenerator));
		}
	}
    return(grainModelOut);
}


/**
 * 
 */
 /**
* @brief Grain-wise film grain rendering algorithm
*
* @param imgIn input image
* @param filmGrainOptions : film grain rendering options
* @return output, rendered image
*/
matrix<float>* film_grain_rendering_grain_wise(matrix<float> *imgIn,
	filmGrainOptionsStruct<float> filmGrainOptions)
{
	float lambda, E;

	/*********************************************/
	/***************   PARAMETERS   **************/
	/*********************************************/

	//parameters for film grain
    float grainRadius = filmGrainOptions.muR;//grainRadius/pixelSize;
    float grainStd = filmGrainOptions.sigmaR;
    float grainVar = grainStd*grainStd;//grainRadius;
	float sigmaFilter = filmGrainOptions.sigmaFilter;
 	float sX = ((float)filmGrainOptions.nOut)/((float)(filmGrainOptions.xB-filmGrainOptions.xA)); 
	float sY = ((float)filmGrainOptions.mOut)/((float)(filmGrainOptions.yB-filmGrainOptions.yA));
	std::cout<< "sX : " << sX << std::endl;
	std::cout<< "sY : " << sY << std::endl;

	std::cout << "min : " << imgIn->min() << std::endl;
	std::cout << "max : " << imgIn->max() << std::endl;

	/*********************************************/
	/***   MONTE CARLO TRANSLATION VECTORS   *****/
	/*********************************************/

	int NmonteCarlo = filmGrainOptions.NmonteCarlo;
	std::random_device rd;
    std::mt19937 rndGenerator(rd());

	//draw the random (gaussian) translation vectors
    matrix<float> * X = new matrix<float>(NmonteCarlo,1);
    matrix<float> * Y = new matrix<float>(NmonteCarlo,1);
	noise_prng pMonteCarlo;
	mysrand(&pMonteCarlo, 2016u*(filmGrainOptions.grainSeed));
	for (int i=0; i<NmonteCarlo; i++)
	{
		X->set_value(i,0,sigmaFilter*(float)myrand_gaussian_0_1(&pMonteCarlo));   //(float)normalDistribution(rndGenerator)
		Y->set_value(i,0,sigmaFilter*(float)myrand_gaussian_0_1(&pMonteCarlo));
	}


	//optionally, save the image of values of lambda
	matrix<float> *imgLambda = new matrix<float>(imgIn->get_nrows(),imgIn->get_ncols());
	//create output image
	matrix<float> *imgOut = new matrix<float>( filmGrainOptions.mOut, filmGrainOptions.nOut );
	imgOut->set_to_zero();
	//create list of temporary images, and set them to 0. We use these to store the result of each Monte Carlo evaluation
	matrix<bool>** imgTempPtr = new matrix<bool>* [NmonteCarlo];
	for (int i=0; i<NmonteCarlo; i++)
	{
		imgTempPtr[i] = new matrix<bool>( filmGrainOptions.mOut, filmGrainOptions.nOut);
		(imgTempPtr[i])->set_to_zero();
	}

	//determine grain centres for the whole image
	matrix<float> *modelTemp;
	int i,j,k,x,y,minBBx,maxBBx,minBBy,maxBBy, nGrain;
	float yGrainTemp,xGrainTemp,rGrainTemp,rGrainTempSq;

	#pragma omp parallel for private(i,j,k,E,lambda,modelTemp,x,y,minBBx,maxBBx,minBBy,maxBBy,yGrainTemp,xGrainTemp,rGrainTemp,rGrainTempSq,nGrain) shared(imgTempPtr)
	for (i=(int)floor(filmGrainOptions.yA); i< (int)ceil(filmGrainOptions.yB); i++)
	{
		for (j=(int)floor(filmGrainOptions.xA); j< (int)ceil(filmGrainOptions.xB); j++)
		{
			//determine lambda
		    E = ((*imgIn)(i,j));
		    lambda = 1/(pi*(grainVar + grainRadius*grainRadius))*log(1/(1-E));
		    modelTemp = boolean_model(lambda,grainRadius,grainStd);
	
			//for each image of the monte carlo simulation
			for (k=0; k<NmonteCarlo; k++)
			{
				//add this grain to the kth image
				for (nGrain=0; nGrain< modelTemp->get_nrows(); nGrain++)
				{
					//get the current grain info, in the INPUT image coordinate system
					xGrainTemp = ( (*modelTemp)(nGrain, 0) + j) - (*X)(k,0)/sX;
					yGrainTemp = ( (*modelTemp)(nGrain, 1) + i) - (*Y)(k,0)/sY;
					rGrainTemp = (*modelTemp)(nGrain, 2);
					rGrainTempSq = rGrainTemp*rGrainTemp;
					//cells to check, in the OUTPUT image coordinate system
					minBBx = (int) ceil( (float)  ( xGrainTemp*sX -(float)rGrainTemp*sX) );
					maxBBx = (int) floor( (float) ( xGrainTemp*sX +(float)rGrainTemp*sX) );
					minBBy = (int) ceil( (float)  ( yGrainTemp*sY -(float)rGrainTemp*sY) );
					maxBBy = (int) floor( (float) ( yGrainTemp*sY +(float)rGrainTemp*sY) );

					//loop over bounding boxes for each grain
					for (x=minBBx; x<=maxBBx; x++)
					{
						for (y=minBBy; y<=maxBBy; y++)
						{
							//check if the coordinates (x,y) are in the bounds of the output image
							if ( ((float)y - floor( (filmGrainOptions.yA) *sY) >= 0) &&
								 ((float)y - floor( (filmGrainOptions.yA) *sY) < imgOut->get_nrows() ) &&
								 ((float)x  - floor( (filmGrainOptions.xA) *sX) >=0 ) &&
								 ((float)x  - floor( (filmGrainOptions.xA) *sX) < imgOut->get_ncols() )
								 )
							{
								if (  ( ( ( (float)y/sY)-yGrainTemp)*( ( (float)y/sY)-yGrainTemp) +
									( ( (float)x/sX) -xGrainTemp)*( ( (float)x/sX)-xGrainTemp)  <= rGrainTempSq) == true)
								{
								    (imgTempPtr[k])->set_value( (int)( (float)y - floor( (filmGrainOptions.yA) *sY)),
								    (int)fmin(fmax( (float)x  - floor( (filmGrainOptions.xA) *sX) , 0),imgOut->get_ncols()-1), (bool) 1 );
								}
							}
						}
					}
				}
			}
			//save lambda
			imgLambda->set_value(i,j,lambda);
			delete modelTemp;
		}
	}

	//copy the information to the output image
	for (int i=0; i< imgOut->get_nrows(); i++)
	{
		for (int j=0; j< imgOut->get_ncols(); j++)
		{
			float valTemp = (float)0.0;
			for (int k=0; k<NmonteCarlo; k++)
			{
				valTemp = valTemp + (float)( (*(imgTempPtr[k]))(i,j) );
			}
			imgOut->set_value(i , j , valTemp );//imgOut->set_value(i , j , imgLambda->get_value(i,j) );//
		}
	}

	imgOut->divide((float)NmonteCarlo);

	delete imgLambda;
	//delete temporary images
	for (int i=0; i<NmonteCarlo; i++)
	{
		delete imgTempPtr[i];
	}
	delete imgTempPtr;
	return(imgOut);
}

