#include "trackers/klttracker.h"

#include <cmath>
#include <fstream>
#include <string>

#include <opencv2/opencv.hpp>
#include <opencv2/gpu/gpu.hpp>

#include <cuda.h>
//#include <cublas.h>
#include <cuda_runtime.h>
using namespace cv;
using namespace std;
#define PI 3.14159265

#define persA 0.06
#define persB 40
#define minDist 10

//#define persA 0.01
//#define persB 20
//#define minDist 5

Mat corners,prePts,nextPts,status,eigenvec;
cv::gpu::GoodFeaturesToTrackDetector_GPU detector;
cv::gpu::PyrLKOpticalFlow tracker;
cv::gpu::GpuMat gpuGray, gpuPreGray, gpuCorners, gpuPrePts, gpuNextPts,gpuStatus,gpuEigenvec,gpuDenseX,gpuDenseY,gpuDenseXC,gpuDenseYC;
Mat denseX,denseY,denseRGB;

typedef struct
{
	float val, x, y;
}CoVal, CoVal_p;

__device__ int d_kernelh[1], d_kernelw[1], d_cocount[1];
__device__ float d_persa[1], d_persb[1];

__device__ unsigned char *d_neighbor,* d_rgbmap;
__device__ unsigned int* d_isnewmat;
__device__ float* d_distmat, *d_curvec, *d_kernel, *d_confmap,*d_group;
__device__ ofv* d_ofvec;
__device__ CoVal* d_covec;
//__device__ float d_confmap2[1280 * 720

__global__ void crossMult(unsigned int* dst,float* vertical,float* horizon,int h,int w)
{
	int x = threadIdx.x,y=blockIdx.x;
	if (x < w&&y<h)
	{
		float xv = vertical[y * 2], yv = vertical[y*2+1],xh=horizon[x*2],yh=horizon[x*2+1];
		float dx = xv - xh, dy = yv - yh;
		float dist = abs(dx) + abs(dy);
		if (dist < minDist)
			atomicAdd(dst + y, 1);
	}
}
__global__ void applyKernel(CoVal* d_covec, float* d_kernel, float* d_confmap, int w, int h)
{
	
	int idx = blockIdx.x;
	if (idx < *d_cocount)
	{
		float cx = d_covec[idx].x, cy = d_covec[idx].y, val = d_covec[idx].val;
		if (val != 0)
		{
			for (int steper = 0; steper < 10; steper++)
			{
				int i = threadIdx.y, j = threadIdx.x * 10 + steper;
				int kw = *d_kernelw, kh = *d_kernelh;

				if (i < kh&&j < kw)
				{
					int xk = (j - kw / 2) + 0.5, yk = (i - kh / 2) + 0.5;
					int xmap = cx + xk, ymap = cy + yk;
					if (xmap >= 0 && ymap >= 0 && xmap < w&&ymap < h)
					{
						float addval = val*d_kernel[i*kw + j];
						float oldval = atomicAdd(d_confmap + (ymap*w + xmap), addval);
					}
				}
			}
			//printf("%d\n", idx);
		}
	}
}
__global__ void float2RGB(unsigned char* d_rgbmap, float* d_confmap, float max, float min)
{
	int x = blockIdx.x, y = threadIdx.x;
	int w = gridDim.x, h = blockDim.x;
	int offset = y*w + x;
	float val = d_confmap[offset];
	unsigned char r = 0, g = 0, b = 255;
	/*
	if (val < 0)
	{
		r = abs(val) / abs(min) * 255;
	}
	else if (val > 0)
	{
		b = abs(val) / abs(max) * 255;
	}
	*/
	//b = (1 - r);
	//printf("%f,%f\n", val,max);
	//if (val < 0)
	
		float denseval = (-val) / max ;
		int indcator = (denseval>0.1);
		float alpha = indcator*denseval + (1 - indcator)*0.1;
		//int indcator = r > 255;
		//r = indcator * 255 + (1 - indcator)*(denseval);
		//r = denseval*255+0.5;
		//b = 255 - r;
		//g = 0;
		d_rgbmap[offset * 3] = d_rgbmap[offset * 3] * alpha;
		d_rgbmap[offset * 3 + 1] = d_rgbmap[offset * 3 + 1] * alpha;
		d_rgbmap[offset * 3 + 2] = d_rgbmap[offset * 3 + 2] * alpha;
}
__global__ void applyKernelIter(CoVal* d_covec, float* d_kernel, float* d_confmap, int w, int h)
{

	int idx = blockIdx.x*1000+threadIdx.x;
	int kw = *d_kernelw, kh = *d_kernelh;
	//int i = threadIdx.y + steper * 10, j = threadIdx.x;
	if (idx < *d_cocount)
	{
		float cx = d_covec[idx].x, cy = d_covec[idx].y, val = d_covec[idx].val;
		for (int i = 0; i < kh; i++)
		{
			for (int j = 0; j < kw; j++)
			{
				int xk = (j - kw / 2), yk = (i - kh / 2);
				xk = xk + 0.5, yk = yk + 0.5;
				int xmap = cx + xk, ymap = cy + yk;
				if (xmap >= 0 && ymap >= 0 && xmap < w && ymap < h)
				{
					float addval = val*d_kernel[i*kw + j];
					atomicAdd(d_confmap + (ymap*w + xmap), addval);

				}
			}
		}
	}

}
__global__ void searchNeighbor(unsigned char* d_neighbor, ofv* d_ofvec,CoVal* d_covec, int nFeatures)
{
	int r = blockIdx.x, c = threadIdx.x;
	if (r != c)
	{
		
		float dx = abs(d_ofvec[r].x1 - d_ofvec[c].x1), dy = abs(d_ofvec[r].y1 - d_ofvec[c].y1);
		int yidx = d_ofvec[r].idx, xidx = d_ofvec[c].idx;
		float dist = sqrt(dx*dx + dy*dy);
		float xmid = (d_ofvec[r].x1 + d_ofvec[c].x1) / 2, ymid = (d_ofvec[r].y1 + d_ofvec[c].y1) / 2;
		//applyKernel<<<2,2>>>(0,0,x1,y1);
		//float mindist = ymid*(0.16) + 50;
		//printf("%f\n", mindist);
		if (dx < ymid*(persA)+persB && dy < ymid*(persA*1.5) + persB*1.5)
		{
			d_neighbor[yidx*nFeatures + xidx] = 1;
			
			float vx0 = d_ofvec[r].x1 - d_ofvec[r].x0, vx1 = d_ofvec[c].x1 - d_ofvec[c].x0,
				vy0 = d_ofvec[r].y1 - d_ofvec[r].y0, vy1 = d_ofvec[c].y1 - d_ofvec[c].y0;
			float norm0 = sqrt(vx0*vx0 + vy0*vy0), norm1 = sqrt(vx1*vx1 + vy1*vy1);
			float cosine = (vx0*vx1 + vy0*vy1);// / norm0 / norm1;
			if (cosine < 0)
			{
				//printf("%f\n", cosine);
				int arrpos = atomicAdd(d_cocount, 1);
				d_covec[arrpos].x = xmid, d_covec[arrpos].y = ymid, d_covec[arrpos].val = cosine;
			}
			//else
				//printf("%d:%f\n", arrpos, cosine);
		}
		else
			d_neighbor[yidx*nFeatures + xidx] = 0;

	}
}

int KLTtracker::init(int bsize,int w,int h)
{
	int nDevices;
	cudaGetDeviceCount(&nDevices);
	for (int i = 0; i < nDevices; i++) {
		cudaDeviceProp prop;
		cudaGetDeviceProperties(&prop, i);
		printf("Device Number: %d\n", i);
		printf("  Device name: %s\n", prop.name);
		printf("  Memory Clock Rate (KHz): %d\n",
			prop.memoryClockRate);
		printf("  Memory Bus Width (bits): %d\n",
			prop.memoryBusWidth);
		printf("  Peak Memory Bandwidth (GB/s): %f\n\n",
			2.0*prop.memoryClockRate*(prop.memoryBusWidth / 8) / 1.0e6);
		std::cout << "maxgridDim" << prop.maxGridSize[0] << "," << prop.maxGridSize[1] << "," << prop.maxGridSize[2] << std::endl;
		std::cout<<"maxThreadsPerBlock:"<<prop.maxThreadsPerBlock<<std::endl;
		std::cout << prop.major << "," << prop.minor << std::endl;
	}

    nFeatures = 1000;
    nSearch=3000;
	trackBuff = std::vector<FeatBuff>(nFeatures);
    isTracking=new int[nFeatures];
	for (int i=0;i<nFeatures;i++)
	{
		trackBuff[i].init(1,100);
        isTracking[i]=0;
	}
	frame_width = w;
	frame_height = h;
	frameidx=0;
	dirvec = new float[nFeatures];
	memset(dirvec, 0, nFeatures*sizeof(float));

    goodNewPts.init(1,nSearch);
    detector= gpu::GoodFeaturesToTrackDetector_GPU(nSearch,0.0001,3,3);
    tracker = gpu::PyrLKOpticalFlow();
    tracker.winSize=Size(3,3);
    tracker.maxLevel=3;
    tracker.iters=10;
    gpuGray=gpu::GpuMat(frame_height, frame_width, CV_8UC1 );
    gpuPreGray=gpu::GpuMat(frame_height, frame_width, CV_8UC1 );

	cudaMalloc(&d_isnewmat, nSearch*sizeof(unsigned int));
	h_isnewmat = (unsigned int*)malloc(nSearch*sizeof(unsigned int));


	h_curvec = (float*)malloc(nFeatures*2*sizeof(float));
	cudaMalloc(&d_curvec, nFeatures * 2 * sizeof(float));
	cudaMalloc(&d_ofvec, nFeatures* sizeof(ofv));
	ofvBuff.init(1, nFeatures);

	cudaMemcpyToSymbol(d_persa, &persa, sizeof(float));
	cudaMemcpyToSymbol(d_persb, &persb, sizeof(float));
	std::cout << persa << " * a + " << persb << std::endl;
	cudaMalloc(&d_neighbor, nFeatures*nFeatures);
	h_neighbor = (unsigned char*)malloc(nFeatures*nFeatures);
	memset(h_neighbor, 0, nFeatures*nFeatures);

	kernelw = 80, kernelh = 80, cocount = 0;
	cudaMemcpyToSymbol(d_kernelw, &kernelw, sizeof(int));
	cudaMemcpyToSymbol(d_kernelh, &kernelh, sizeof(int));
	cudaMemcpyToSymbol(d_cocount, &cocount, sizeof(int));

	h_kernel = (float *)malloc(kernelw*kernelh*sizeof(float));
	rgbkernel = (unsigned char *)malloc(kernelw*kernelh * 3);
	cudaMalloc(&d_kernel, kernelw*kernelh*sizeof(float));
	genkernel(h_kernel,kernelw,kernelh);
	cudaMemcpy(d_kernel, h_kernel, kernelw*kernelh*sizeof(float), cudaMemcpyHostToDevice);

	cudaMalloc(&d_covec, nFeatures*nFeatures*sizeof(CoVal));
	cudaMalloc(&d_confmap, h*w*sizeof(float));
	cudaMemset(d_confmap, 0, h*w*sizeof(float));
	h_confmap = (float *)malloc(h*w*sizeof(float));
	cudaMalloc(&d_rgbmap, h*w*3);
	h_rgbmap = (unsigned char *)malloc(h*w * 3);
	alpha = 0.5;
	std::cout << "inited" << std::endl;
	gt_inited = false;

	return 1;
}
void KLTtracker::genkernel(float* ptr, int w, int h)
{
	float cx = w / 2, cy = h / 2;
	double theta = 0, sigma_x = w/2, sigma_y=h/2;

	double gaussA = cos(theta) * cos(theta) / 2 / (sigma_x*sigma_x) + sin(theta)*sin(theta) / 2 / (sigma_y*sigma_y)
		, gaussB = -sin(2 * theta) / 4 / (sigma_x*sigma_x) + sin(2 * theta) / 4 / (sigma_y*sigma_y)
		, gaussC = sin(theta) *sin(theta) / 2 / (sigma_x*sigma_x) + cos(theta)*cos(theta) / 2 / (sigma_y*sigma_y);
	for (int i = 0; i < h; i++)
	{
		for (int j = 0; j < w; j++)
		{
			float x = j - cx;
			float y = i - cy;
			float power = -(gaussA * x*x + 2 * gaussB * x*y + gaussC * y*y)*6;
			float val = exp(power);
			ptr[i*w + j] = val;
			rgbkernel[(i*w + j) * 3] = val * 255;
			rgbkernel[(i*w + j) * 3 + 1] = val * 255;
			rgbkernel[(i*w + j) * 3 + 2] = val * 255;
		}
	}
}
int KLTtracker::selfinit(unsigned char* framedata)
{
	curframedata=framedata;
    Mat curframe(frame_height,frame_width,CV_8UC1,framedata);
    gpuGray.upload(curframe);
    gpuPreGray.upload(curframe);
    detector(gpuGray, gpuCorners);
    gpuCorners.download(corners);
    gpuCorners.copyTo(gpuPrePts);
    for (int k = 0; k < nFeatures; k++)
    {
        Vec2f p = corners.at<Vec2f>(k);
		pttmp.x = p[0];//(PntT)(p[0] + 0.5);
		pttmp.y = p[1];//(PntT)(p[1]+ 0.5);
        pttmp.t = frameidx;
        trackBuff[k].updateAFrame(&pttmp);
        isTracking[k]=1;
    }
	return true;
}
bool KLTtracker::checkTrackMoving(FeatBuff &strk)
{
	bool isTrkValid = true;

    int Movelen=7,minlen=5,startidx=max(strk.len-Movelen,0);
    if(strk.len>Movelen)
    {
        double maxdist = .0, dtmp = .0,totlen=.0;

		FeatPts* aptr = strk.getPtr(startidx), *bptr = aptr;
        //int xa=(*aptr),ya=(*(aptr+1)),xb=*(strk.cur_frame_ptr),yb=*(strk.cur_frame_ptr+1);
        PntT xa=aptr->x,ya=aptr->y,xb=strk.cur_frame_ptr->x,yb=strk.cur_frame_ptr->y;
        double displc=sqrt( pow(xb-xa, 2.0) + pow(yb-ya, 2.0));
        /*
        for(int posi=0;posi<strk.len;posi++)
        {
            bptr=strk.getPtr(posi);
            xb=bptr->x,yb=bptr->y;
            dtmp = sqrt( pow(xb-xa, 2.0) + pow(yb-ya, 2.0));
            totlen+=dtmp;
            if (dtmp > maxdist && posi>=startidx) maxdist = dtmp;
            xa=xb,ya=yb;
        }
        */
        //if(strk.len>100&&totlen*0.5>displc){strk.isCurved=true;}
        if((strk.len -startidx)*0.2>displc)
        {
            isTrkValid = false;
        }
        //if (maxdist < 1.4 && strk.len>30){isTrkValid = false;}
        //if (maxdist <=0.1 && strk.len>=minlen){isTrkValid = false;}
    }
	return isTrkValid;
}

int KLTtracker::updateAframe(unsigned char* framedata, unsigned char* rgbdata, int fidx)
{
    frameidx=fidx;
	curframedata=framedata;
    gpuGray.copyTo(gpuPreGray);
    Mat curframe(frame_height,frame_width,CV_8UC1,framedata);
    gpuGray.upload(curframe);

    tracker.sparse(gpuPreGray, gpuGray, gpuPrePts, gpuNextPts, gpuStatus, &gpuEigenvec);
    gpuStatus.download(status);
    gpuNextPts.download(nextPts);
    gpuPrePts.download(prePts);
    detector(gpuGray, gpuCorners);
    gpuCorners.download(corners);

	
    goodNewPts.clear();
	
	for (int k = 0; k < nFeatures; k++)
	{
		h_curvec[k*2] = trackBuff[k].cur_frame_ptr->x;
		h_curvec[k*2+1] = trackBuff[k].cur_frame_ptr->y;
	}
	
	cudaMemcpy(d_curvec, h_curvec, nFeatures*2*sizeof(float), cudaMemcpyHostToDevice);
	//dim3 gridsize(nSearch, 1, 1);
	//dim3 blocksize(nFeatures, 1, 1);
	float* d_corner = (float*)gpuCorners.data;
	cudaMemset(d_isnewmat, 0, nSearch*sizeof(unsigned int));
	crossMult << <nSearch, nFeatures >> >(d_isnewmat, d_corner, d_curvec, nSearch, nFeatures);
	cudaMemcpy(h_isnewmat, d_isnewmat, nSearch*sizeof(unsigned int), cudaMemcpyDeviceToHost);

    for(int i=0;i<nSearch;i++)
    {
		if (h_isnewmat[i] == 0)
        {
            goodNewPts.updateAFrame(&i);
        }
    }

	ofvBuff.clear();
    int addidx=0,counter=0;
	for (int k = 0; k < nFeatures; k++)
	{
        int statusflag = status.at<int>(k);
        Vec2f trkp = nextPts.at<Vec2f>(k);
        bool lost=false;
        if ( statusflag)
		{
            Vec2f pre=prePts.at<Vec2f>(k),cur=nextPts.at<Vec2f>(k);
            int prex=trackBuff[k].cur_frame_ptr->x, prey=trackBuff[k].cur_frame_ptr->y;
			pttmp.x = trkp[0];//(PntT)(trkp[0] + 0.5);
			pttmp.y = trkp[1];//(PntT)(trkp[1] + 0.5);
            pttmp.t = frameidx;
            trackBuff[k].updateAFrame(&pttmp);
            double trkdist=abs(prex-pttmp.x)+abs(prey-pttmp.y),ofdist=abs(pre[0]-cur[0])+abs(pre[1]-cur[1]);
			dirvec[k] = 0.5*dirvec[k] + 0.5*sgn(pttmp.y-prey);
			//std::cout << dirvec[k] << std::endl;
            isTracking[k]=1;
			bool isMoving=checkTrackMoving(trackBuff[k]);
			if (!isMoving||(trackBuff[k].len>1 && trkdist>10))
			{
                //trackBuff[k].clear();
                lost=true;
                isTracking[k]=0;
			}
		}
        else
        {
            //trackBuff[k].clear();
            counter++;
            lost=true;
            isTracking[k]=0;
		}
        if(lost)
        {
            trackBuff[k].clear();
			dirvec[k] = 0;
            if(addidx<goodNewPts.len)
            {
                int newidx=*(goodNewPts.getPtr(addidx++));
                Vec2f cnrp = corners.at<Vec2f>(newidx);
				pttmp.x = cnrp[0];// (PntT)(cnrp[0] + 0.5);
				pttmp.y = cnrp[1];// (PntT)(cnrp[1] + 0.5);
                pttmp.t = frameidx;
                trackBuff[k].updateAFrame(&pttmp);
                nextPts.at<Vec2f>(k)=cnrp;
                isTracking[k]=1;
            }
        }
		else
		{
			if (trackBuff[k].len > 10)
			{
				ofvtmp.x0 = trackBuff[k].getPtr(trackBuff[k].len - 5)->x;
				ofvtmp.y0 = trackBuff[k].getPtr(trackBuff[k].len - 5)->y;
				ofvtmp.x1 = trackBuff[k].cur_frame_ptr->x;
				ofvtmp.y1 = trackBuff[k].cur_frame_ptr->y;
				ofvtmp.len = trackBuff[k].len;
				ofvtmp.idx = k;
				ofvBuff.updateAFrame(&ofvtmp);
			}
		}
	}
	
	cocount = 0;
	if (ofvBuff.len > 0)
	{
		cudaMemcpyToSymbol(d_cocount, &cocount, sizeof(int));
		cudaMemset(d_ofvec, 0, nFeatures* sizeof(ofv));
		cudaMemcpy(d_ofvec, ofvBuff.data, ofvBuff.len*sizeof(ofv), cudaMemcpyHostToDevice);
		cudaMemset(d_neighbor, 0, nFeatures*nFeatures);
		searchNeighbor << <ofvBuff.len, ofvBuff.len >> >(d_neighbor, d_ofvec, d_covec, nFeatures);
		cudaMemcpy(h_neighbor, d_neighbor, nFeatures*nFeatures, cudaMemcpyDeviceToHost);
		cudaMemcpyFromSymbol(&cocount, d_cocount, sizeof(int));

		cudaMemset(d_confmap, 0, frame_width*frame_height*sizeof(float));
		applyKernelIter << <cocount / 1000 + 1, 1000 >> >(d_covec, d_kernel, d_confmap, frame_width, frame_height);

		//dim3 blocksize(kernelw/10, kernelh, 1);
		//applyKernel <<<cocount, blocksize >>>(d_covec, d_kernel, d_confmap, frame_width, frame_height);

		cudaMemcpy(h_confmap, d_confmap, frame_width*frame_height*sizeof(float), cudaMemcpyDeviceToHost);
		//Mat tmpmat(frame_height, frame_width, CV_32FC1, h_confmap);

		float* tmpptr = h_confmap, tmpmax = 0, tmpmin = 0;


		for (int i = 0; i < frame_width*frame_height; i++)
		{
			float tmpval = -*(tmpptr);
			//std::cout << *(tmpptr) << std::endl;
			if (tmpval>tmpmax)tmpmax = tmpval;
			//if (tmpval<tmpmin)tmpmin = tmpmin
			tmpptr++;
		}
		//std::cout << cocount << "," << tmpmin << "," << tmpmax << std::endl;
		/*
		tmpptr = h_confmap;
		unsigned char* tmprgbptr = rgbdata;
		for (int i = 0; i < frame_width*frame_height; i++)
		{
		float tmpval = *(tmpptr);
		unsigned char r = abs(tmpval - tmpmin) / abs(tmpmax - tmpmin) * 255,b=255-r;
		tmprgbptr[i * 3] = tmprgbptr[i * 3] * (1 - alpha) + alpha*r;
		tmprgbptr[i * 3 + 2] = tmprgbptr[i * 3 + 2] * (1 - alpha) + alpha*b;
		tmpptr++;
		}
		*/
		cudaMemcpy(d_rgbmap, rgbdata, frame_width*frame_height * 3, cudaMemcpyHostToDevice);
		float2RGB << <frame_width, frame_height >> >(d_rgbmap, d_confmap, tmpmax, tmpmin - 1);
		cudaMemcpy(rgbdata, d_rgbmap, frame_width*frame_height * 3, cudaMemcpyDeviceToHost);
	}
    gpuPrePts.upload(nextPts);
	return 1;
}
