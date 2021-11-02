#include "mainwindow.h"
#include <QApplication>
#include "selectimportingoptions.h"
#include "progressbarupdate.h"


int main(int argc, char *argv[])
{
    QApplication a(argc, argv);

    MainWindow w;
    w.setWindowTitle("MRImage3D");
    w.setWindow(0);
    w.show();
    return a.exec();
}

