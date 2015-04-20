/********************************************************************************
** Form generated from reading UI file 'kltgpu.ui'
**
** Created by: Qt User Interface Compiler version 5.4.1
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_KLTGPU_H
#define UI_KLTGPU_H

#include <QtCore/QVariant>
#include <QtWidgets/QAction>
#include <QtWidgets/QApplication>
#include <QtWidgets/QButtonGroup>
#include <QtWidgets/QHeaderView>
#include <QtWidgets/QMainWindow>
#include <QtWidgets/QMenuBar>
#include <QtWidgets/QStatusBar>
#include <QtWidgets/QToolBar>
#include <QtWidgets/QWidget>

QT_BEGIN_NAMESPACE

class Ui_kltgpuClass
{
public:
    QMenuBar *menuBar;
    QToolBar *mainToolBar;
    QWidget *centralWidget;
    QStatusBar *statusBar;

    void setupUi(QMainWindow *kltgpuClass)
    {
        if (kltgpuClass->objectName().isEmpty())
            kltgpuClass->setObjectName(QStringLiteral("kltgpuClass"));
        kltgpuClass->resize(600, 400);
        menuBar = new QMenuBar(kltgpuClass);
        menuBar->setObjectName(QStringLiteral("menuBar"));
        kltgpuClass->setMenuBar(menuBar);
        mainToolBar = new QToolBar(kltgpuClass);
        mainToolBar->setObjectName(QStringLiteral("mainToolBar"));
        kltgpuClass->addToolBar(mainToolBar);
        centralWidget = new QWidget(kltgpuClass);
        centralWidget->setObjectName(QStringLiteral("centralWidget"));
        kltgpuClass->setCentralWidget(centralWidget);
        statusBar = new QStatusBar(kltgpuClass);
        statusBar->setObjectName(QStringLiteral("statusBar"));
        kltgpuClass->setStatusBar(statusBar);

        retranslateUi(kltgpuClass);

        QMetaObject::connectSlotsByName(kltgpuClass);
    } // setupUi

    void retranslateUi(QMainWindow *kltgpuClass)
    {
        kltgpuClass->setWindowTitle(QApplication::translate("kltgpuClass", "kltgpu", 0));
    } // retranslateUi

};

namespace Ui {
    class kltgpuClass: public Ui_kltgpuClass {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_KLTGPU_H
