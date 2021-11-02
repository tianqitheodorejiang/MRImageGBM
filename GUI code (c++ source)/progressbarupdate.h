#ifndef PROGRESSBARUPDATE_H
#define PROGRESSBARUPDATE_H

#include "QThread"
#include "QObject"

class ProgressBarUpdate : public QThread
{
    Q_OBJECT
public:
    explicit ProgressBarUpdate(QObject *parent = 0);
    void run();
    int expectedDuration = 0;
signals:
    void progressPercentUpdated(int i);
    void progressLabelUpdated(QString text);
    void nextStep();
    void loadingFailed();
    void lockWindow(bool lock);
public slots:
    //void cleanUpandClose();


};

#endif // PROGRESSBARUPDATE_H
