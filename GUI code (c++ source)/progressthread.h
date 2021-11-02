#ifndef PROGRESSTHREAD_H
#define PROGRESSTHREAD_H
#include "QtCore"

class ProgressThread : public QThread
{
public:
    ProgressThread();
    void run();
};

#endif // PROGRESSTHREAD_H
