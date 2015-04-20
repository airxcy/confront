#include "Qts/viewqt.h"
#include "Qts/modelsqt.h"
#include "Qts/streamthread.h"

#include <iostream>

#include <QPainter>
#include <QBrush>
#include <QPixmap>
#include <cmath>
#include <QGraphicsSceneEvent>
#include <QMimeData>
#include <QByteArray>
#include <QFont>
char viewstrbuff[200];
void DefaultScene::mousePressEvent ( QGraphicsSceneMouseEvent * event )
{
    emit clicked(event);
}
void DefaultScene::drawBackground(QPainter * painter, const QRectF & rect)
{
    QPen pen;
    QFont txtfont("Roman",40);
    txtfont.setBold(true);
    pen.setColor(QColor(255,255,255));
    pen.setCapStyle(Qt::RoundCap);
    pen.setJoinStyle(Qt::RoundJoin);
    pen.setWidth(10);
    painter->setPen(QColor(243,134,48,150));
    painter->setFont(txtfont);
    painter->drawText(rect, Qt::AlignCenter,"打开文件\nOpen File");
}
TrkScene::TrkScene(const QRectF & sceneRect, QObject * parent):QGraphicsScene(sceneRect, parent)
{
    streamThd=NULL;
}
TrkScene::TrkScene(qreal x, qreal y, qreal width, qreal height, QObject * parent):QGraphicsScene( x, y, width, height, parent)
{
    streamThd=NULL;
}
void TrkScene::drawBackground(QPainter * painter, const QRectF & rect)
{
    //std::cout<<streamThd->inited<<std::endl;
    if(streamThd!=NULL&&streamThd->inited)
    {
        updateFptr(streamThd->frameptr, streamThd->frameidx);
    }
    painter->setBrush(bgBrush);
    painter->drawRect(rect);

    if(streamThd!=NULL&&streamThd->inited)
    {
        painter->setPen(Qt::red);
        painter->setFont(QFont("System",20,2));
        painter->drawText(rect, Qt::AlignLeft|Qt::AlignTop,QString::number(streamThd->fps));
		std::vector<FeatBuff>& klttrkvec = streamThd->tracker->trackBuff;
		int nFeature = streamThd->tracker->nFeatures;
		unsigned char * neighbormat = streamThd->tracker->h_neighbor;
		int framew = streamThd->tracker->frame_width, frameh = streamThd->tracker->frame_height;
		int len = klttrkvec.size();
		//unsigned char r, g = feat_colos[i % 6][1], b = feat_colos[i % 6][2];
		float* dirvec = streamThd->tracker->dirvec;
        for(int i=0;i<klttrkvec.size();i++)
        {

			FeatBuff& klttrk = klttrkvec[i];
            //unsigned char r=feat_colos[i%6][0],g=feat_colos[i%6][1],b=feat_colos[i%6][2];
			unsigned char r = (dirvec[i]) * 127.0+127.0, g = 255 - r, b = 0;
            double x0,y0,x1,y1;

			bool connected=false;
			for (int j = i+1; j < nFeature; j++)
			{
				if (neighbormat[i*nFeature + j])
				{
					connected = true;
					int xj = klttrkvec[j].cur_frame_ptr->x, yj = klttrkvec[j].cur_frame_ptr->y;
					x1 = klttrkvec[i].cur_frame_ptr->x, y1 = klttrkvec[i].cur_frame_ptr->y;
					double dist = abs(xj - x1) + abs(yj - y1);
					//unsigned char r = (feat_colos[i % 6][0] + feat_colos[j % 6][0]) / 2,g = (feat_colos[i % 6][1] + feat_colos[j % 6][1]) / 2,b = (feat_colos[i % 6][2] + feat_colos[j % 6][2]) / 2;
					unsigned char r1 = (dirvec[j]) * 127.0 + 127.0, g1 = 255-r1, b1 =0;
					r = (r1 + r) / 2, g = (g1 + g) / 2, b = 0;
					painter->setPen(QColor(r, g, b, 30));
					painter->drawLine(x1, y1, xj, yj);
				}	
			}
			//float light = 0.5;
			//if (connected)
			{

				int startidx = std::max(1, klttrk.len - 10);
				for (int j = startidx; j < klttrk.len; j++)
				{
					x1 = klttrk.getPtr(j)->x, y1 = klttrk.getPtr(j)->y;
					x0 = klttrk.getPtr(j - 1)->x, y0 = klttrk.getPtr(j - 1)->y;
					int denseval = ((j - startidx) * 20 + 30);
					int indcator = (denseval) > 255;
					int alpha = indcator * 255 + (1 - indcator)*(denseval);
					//r = r*0.5+0.5*((y1 > y0) * 255), g = 255 - r, b = 0;
					painter->setPen(QPen(QColor(r, g, b, alpha*0.5), 2));//,alpha));
					painter->drawLine(x0, y0, x1, y1);

				}
				//light = 0.2;
			}

        }
		int kernelw = streamThd->tracker->kernelw, kernelh = streamThd->tracker->kernelh;
		//painter->setBrush(streamThd->rgbimg);
		painter->setBrush(QImage(streamThd->tracker->rgbkernel, kernelw, kernelh, QImage::Format_RGB888));
		painter->drawRect(0, 0, kernelw, kernelh);
		/*
		painter->setPen(QPen(QColor(255, 255,255), 2));//,alpha));
		for (int hi = 2; hi < height() - 2; hi += 20)
		{
			painter->drawLine(0, hi, hi*(0.06) + 40, hi);
		}
		*/
    }

    //update();
    //views().at(0)->update();
}
void TrkScene::updateFptr(unsigned char * fptr,int fidx)
{
    bgBrush.setTextureImage(QImage(fptr,streamThd->framewidth,streamThd->frameheight,QImage::Format_RGB888));
    frameidx=fidx;
    //std::cout<<frameidx<<std::endl;
}
void TrkScene::clear()
{
    bgBrush.setStyle(Qt::NoBrush);
}
