#ifndef KLTTRACKER_H
#define KLTTRACKER_H
#include "trackers/tracker.h"
//#include "trackers/klt_c/klt.h"

#include <vector>
class KLTtracker : public Tracker
{
public:
    int *isTracking;
	int frameidx;
    int nFeatures,nSearch; /*** get frature number ***/
	unsigned char* preframedata,* bgdata,*curframedata;
    Buff<int> goodNewPts;
    std::vector<FeatBuff> trackBuff;
    FeatPts pttmp;
	ofv ofvtmp;
    float* dirvec;
	

	/**cuda **/
	float persa,persb;
	float alpha;
	Buff<ofv> ofvBuff;
	unsigned char *h_neighbor,*rgbkernel,*h_rgbmap;
	unsigned int* h_isnewmat;
	float* h_distmat, *h_curvec, *h_kernel,* h_confmap;
	int kernelw, kernelh, cocount;
	
	void genkernel(float* ptr, int w, int h);
	int init(int bsize,int w,int h);
	int selfinit(unsigned char* framedata);
    int updateAframe(unsigned char* framedata,unsigned char* rgbdata,int fidx);
	void updateFGMask(unsigned char* bgptr);
	bool checkTrackMoving(FeatBuff &strk);
	bool checkCurve(FeatBuff &strk);

	int endTraking();
};
#endif
