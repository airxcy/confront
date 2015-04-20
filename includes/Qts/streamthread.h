#ifndef STREAMTHREAD
#define STREAMTHREAD

#include "trackers/buffers.h"
#include "trackers/klttracker.h"

#include <QThread>
#include <QMutex>
#include <QImage>
#include <QWaitCondition>
#include <opencv2/core/core.hpp>

class TrkScene;
class StreamThread : public QThread
{
    Q_OBJECT
public:
    //StreamThread(){};
    StreamThread(QObject *parent = 0);
    ~StreamThread();
    void streaming();
	void parsefname();
	void setUpPers();

public:
    std::string vidfname;
    QString qdirstr,baseDirname,vidid,gtdir;
    unsigned char * frameptr;
    int framewidth,frameheight,frameidx,frameByteSize;
    double fps;
	float persa, persb;
    QMutex mutex;
    QWaitCondition cv0;
	QImage rgbimg;
//    NoTracker* tracker;
    TrkScene* trkscene;
    int delay,bufflen;
    FrameBuff* framebuff;

    KLTtracker* tracker;
public slots:
    void streamStart(std::string &filename);
    bool init();
signals:
    void initSig();

protected:
    void run();
public:
	bool restart, abort, pause, inited, firsttime;
};

#endif // STREAMTHREAD

