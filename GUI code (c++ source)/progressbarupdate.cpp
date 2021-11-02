#include "progressbarupdate.h"
#include "QtCore"
#include "QDateTime"
#include "QObject"

ProgressBarUpdate::ProgressBarUpdate(QObject *parent) :
    QThread(parent)
{
}
void print2(QString string, bool nextLine = true)
{
    QTextStream out(stdout);
    out << string;
    if (nextLine)
    {
        out << Qt::endl;
    }

}

void ProgressBarUpdate::run()
{
    emit lockWindow(true);
    int progress;
    int initTime = (int)QDateTime::currentMSecsSinceEpoch();
    while (true)
        {
            QFile progressLabel("C:/ProgramData/MRImage3D/metadata/progress_label.txt");
            if (progressLabel.open(QIODevice::ReadWrite))
            {
                QTextStream stream(&progressLabel);
                QString labelText = stream.readLine();
                emit progressLabelUpdated(labelText);
                progressLabel.close();
            }
            QFile progressAmt("C:/ProgramData/MRImage3D/metadata/progress_percent.txt");
            if (progressAmt.open(QIODevice::ReadWrite))
            {
                QTextStream stream2(&progressAmt);
                progress = stream2.readLine().toInt();
                if (progress == 100)  break;
                int timeSpent = (int)QDateTime::currentMSecsSinceEpoch()-initTime;
                progress = (int)((timeSpent*100)/expectedDuration);
                if (progress <= 95) emit progressPercentUpdated(progress);
                progressAmt.close();
            }

            msleep(100);

        }
    QFile success_file("C:/ProgramData/MRImage3D/metadata/load_in_success.txt");
    while (true)
    {
        if (success_file.open(QIODevice::ReadWrite))
        {
            QTextStream stream(&success_file);
            QString success = stream.readLine();

            if (success == "true")
            {
                int endTime = (int)QDateTime::currentMSecsSinceEpoch();
                //write the start time out
                QFile loadStart("C:/ProgramData/MRImage3D/metadata/last_start.txt");
                if (loadStart.exists()) loadStart.remove();
                while (true)
                {
                    if (loadStart.open(QIODevice::ReadWrite))
                    {
                        QTextStream stream(&loadStart);
                        stream<<initTime;
                        loadStart.close();
                        break;
                    }
                }
                //write the end time out
                QFile runEnd("C:/ProgramData/MRImage3D/metadata/last_end.txt");
                if (runEnd.exists()) runEnd.remove();
                while (true)
                {
                    if (runEnd.open(QIODevice::ReadWrite))
                    {
                        QTextStream stream(&runEnd);
                        stream<<endTime;
                        runEnd.close();
                        break;
                    }
                }
                emit progressPercentUpdated(100);
                emit nextStep();
                emit lockWindow(false);
                break;
            }
            else if (success == "false")
            {
                emit loadingFailed();
                emit lockWindow(false);
                break;
            }

        }
    }

}
