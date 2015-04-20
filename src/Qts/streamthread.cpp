#include "Qts/streamthread.h"

#include "Qts/viewqt.h"
#include "Qts/modelsqt.h"
#include "trackers/utils.h"
#include <iostream>
#include <fstream>
//#include <stdlib.h>

//#include <direct.h>
#include "Qts/mainwindow.h"
#include <opencv2/opencv.hpp>
#include <ctime>

#include <QMessageBox>
#include <QDir>
#include <QFile>
#include <QTextStream>
#include <QStringList>

using namespace cv;

VideoCapture cap;
Mat frame;
float fps=0;
char strbuff[100];
QDir qdirtmp;
Mat gray;
VideoWriter vwriter;
Mat renderframe;

StreamThread::StreamThread(QObject *parent) : QThread(parent)
{
    restart = false;
    abort = false;
    pause = false;
    bufflen=0;
    trkscene=NULL;
    framebuff=NULL;
    inited=false;
	firsttime = true;
	tracker = new KLTtracker();
}
StreamThread::~StreamThread()
{
    abort = true;
	vwriter.release();
    cv0.wakeAll();
    wait();
}

bool StreamThread::init()
{
    restart=false,abort=false,pause=false;
    bufflen=0;

    if(!cap.isOpened())
    {
        cap.open(vidfname);
        std::cout<<"reopened"<<std::endl;
    }
    cap.set(CV_CAP_PROP_POS_FRAMES,0);
    frameidx=0;
    cap>>frame;
    fps=0;
    delay=25;
    bufflen=delay+10;
    cvtColor(frame,frame,CV_BGR2RGB);
    framewidth=frame.size[1],frameheight=frame.size[0];
    cvtColor(frame,gray,CV_BGR2GRAY);
	
    if(framebuff==NULL)
    {
        framebuff = new FrameBuff();
        framebuff->init(frame.elemSize(),framewidth,frameheight,bufflen);
    }
    else
        framebuff->clear();
	
    frameByteSize=frame.size[0]*frame.size[1]*frame.elemSize();
    framebuff->updateAFrame(frame.data);
    frameptr=framebuff->cur_frame_ptr;
	parsefname();
	setUpPers();
	tracker->persa = persa;
	tracker->persb = persb;
	if (firsttime){
		tracker->init(10, framewidth, frameheight);
		tracker->selfinit(gray.data);
	}
	//vwriter.open(vidid.toStdString() + "out.avi", -1, 25, Size(framewidth, frameheight));

    inited=true;
	firsttime = false;
    return cap.isOpened();
}
void StreamThread::parsefname()
{
	QFileInfo qvidfileinfo(vidfname.data());
	baseDirname = qvidfileinfo.path();
	vidid = qvidfileinfo.baseName();
	vidid = vidid + "_" + qvidfileinfo.completeSuffix();
	gtdir = baseDirname + "/" + vidid + "/";
	qdirstr = baseDirname + "/" + vidid + "/";
}
void StreamThread::setUpPers()
{
	QString savefname = qdirstr + "pers.txt";
	QFile qinfile(savefname);
	if (qinfile.open(QIODevice::ReadOnly | QIODevice::Text))
	{
		QTextStream qinstream(&qinfile);
		qinstream.readLine();
		QString line = qinstream.readLine();
		QStringList list = line.split(",", QString::SkipEmptyParts);
		persa = list[0].toFloat(), persb = list[1].toFloat();
	}
}
void StreamThread::streaming()
{
    //forever
    {
		QImage img;
		QPainter painter(&img);
        if(init())
        {
			
            emit initSig();
            frameidx=0;
            int fcounter=0;
            std::clock_t start = std::clock();
            double duration;
			img = QImage(framewidth, frameheight, QImage::Format_RGB888);
			renderframe = Mat(frameheight, framewidth,CV_8UC3);
            while(!frame.empty())
            {
				if (abort)
					break;
                cap >> frame;
                if(frame.empty())
                    break;
                cvtColor(frame,gray,CV_BGR2GRAY);
                cvtColor(frame,frame,CV_BGR2RGB);
				tracker->updateAframe(gray.data, frame.data, frameidx);
                framebuff->updateAFrame(frame.data);
                //framebuff->updateAFrame(tracker->kltdense);
                frameptr=framebuff->cur_frame_ptr;
                frameidx++;
                trkscene->update();
				/*
				painter.begin(&img);
				trkscene->render(&painter);
				painter.end();
				rgbimg = img.convertToFormat(QImage::Format_RGB888);
				memcpy(renderframe.data, rgbimg.bits(), rgbimg.byteCount());
				cvtColor(renderframe, renderframe, CV_BGR2RGB);
				vwriter << renderframe;
				*/
                fcounter++;
                duration = ( std::clock() - start ) / (double) CLOCKS_PER_SEC;
                if(duration>=1)
                {
                    fps=fcounter/duration;
                    start=std::clock() ;
                    fcounter=0;
                }
				
            }
			vwriter.release();
        }
        else
        {
            //emit debug( "init Failed");
        }
        trkscene->clear();
        inited=false;
    }
}
void StreamThread::run()
{
    streaming();
}

void StreamThread::streamStart(std::string & filename)
{
    QMutexLocker locker(&mutex);
    //QMessageBox::question(NULL, "Test", "msg",QMessageBox::Ok);
    if (!isRunning()) {
        vidfname=filename;
        start(InheritPriority);
    }
    else
    {
        restart=true;
    }
}
