#include "mainwindow.h"
#include "ui_mainwindow.h"
#include "QFileDialog"
#include "QFile"
#include "QTextStream"
#include "QDir"
#include "QProcess"
#include "iostream"
#include "string"
#include "qmessagebox.h"
#include "QErrorMessage"
#include "QTextStream"
#include "QStringRef"
#include "QDragEnterEvent"
#include "QDragLeaveEvent"
#include "QDragMoveEvent"
#include "QDropEvent"
#include "QMimeData"
#include "QShortcut"
#include "QDateTime"

//Constants
QStringList g_valid_file_endings = {".dcm",".nii.gz",".h5"};
QStringList g_valid_folder_endings = {".dcm",".nii.gz",".png",".h5"};
QString g_acceptable_file_types = "Dicom format image (*.dcm);;Nii compressed image (*.nii.gz);;3D array stored in h5 format (*.h5);;All files (*.*)";
int g_start = 0;
int g_end = 0;
bool g_window_locked = false;
bool g_earlier_scan = false;
bool g_later_scan = false;
bool g_loaded_earlier_scan = false;
bool g_loaded_later_scan = false;

bool g_tumor_eval = false;


//utility functions
int find_min(std::vector<int> list){
    int min = list[0];
    foreach(int number, list)
    {
        if (number < min)
        {
            min=number;
        }
    }
    return min;
}

void print(QString string, bool nextLine = true)
{
    QTextStream out(stdout);
    out << string;
    if (nextLine)
    {
        out << Qt::endl;
    }

}

//checks if a QString has a certain ending
bool hasEnding (QString const &fullString, QStringList valid_endings) {
    std::vector<int> includes;

    foreach(QString ending,valid_endings)
    {
        QStringRef stringEnding(&fullString, fullString.length() - ending.length(), ending.length());
        includes.push_back(QStringRef::compare(stringEnding, ending, Qt::CaseInsensitive));
    }
    if (std::find(includes.begin(), includes.end(), 0) != includes.end())
    {
        return true;
    }
    else
    {
        return false;
    }
}

void delay(int amt)
{
    int dieTime = QDateTime::currentSecsSinceEpoch()+amt;
    while (QDateTime::currentSecsSinceEpoch() < dieTime)
        QCoreApplication::processEvents(QEventLoop::AllEvents, 100);
}


//Default window init stuff
MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::MainWindow)
{
    ui->setupUi(this);
    setAcceptDrops(true);
    new QShortcut(QKeySequence(Qt::CTRL + Qt::Key_Q), this, SLOT(close()));
    new QShortcut(QKeySequence(Qt::CTRL + Qt::Key_R), this, SLOT(on_actionReturn_to_start_page_triggered()));
    progressBarUpdating = new ProgressBarUpdate(this);
    connect(progressBarUpdating,SIGNAL(progressPercentUpdated(int)),this,SLOT(onProgressPercentUpdated(int)));
    connect(progressBarUpdating,SIGNAL(progressLabelUpdated(QString)),this,SLOT(onProgressLabelUpdated(QString)));
    connect(progressBarUpdating,SIGNAL(nextStep()),this,SLOT(onNextStep()));
    connect(progressBarUpdating,SIGNAL(loadingFailed()),this,SLOT(onLoadingFailed()));
    connect(progressBarUpdating,SIGNAL(lockWindow(bool)),this,SLOT(onLockWindow(bool)));
}

MainWindow::~MainWindow()
{
    progressBarUpdating->wait();
    delete ui;
}

void MainWindow::Import_images(QString path)
{   QFile file("C:/ProgramData/MRImage3D/metadata/import_file_name.txt");
    QDir dir("C:/ProgramData/MRImage3D/metadata");
    if (!dir.exists())
        dir.mkpath(".");
    if (file.exists())
        file.remove();

    if (file.open(QIODevice::ReadWrite))
    {
        QTextStream stream(&file);
        stream << path;
        file.close();
    }
    load_data();
}




//functions for each open button
void MainWindow::on_import_file_clicked()
{   try
    {
        QString import_path = QFileDialog::getOpenFileName(this,tr("Select File"),"C://",g_acceptable_file_types);
        //checks to see if the C:/ProgramData/MRImage3D/metadata folder exists, and if not makes it
        QDir dir("C:/ProgramData/MRImage3D/metadata");
        if (!dir.exists())
            dir.mkpath(".");

        //checks to see if the selected file is valid
        bool file_valid = false;
        if (hasEnding(import_path, g_valid_file_endings)){
            file_valid = true;
        }

        //If the file is valid, checks if info file exists, and if so delete it, then remake it (to only have a single path), then write the path out
        if (file_valid == true)
        {
            Import_images(import_path);
        }
        else if (!import_path.isNull())
        {
            QMessageBox folderinvalid(QMessageBox::Warning, "File type unsupported", "Please select a file that is a supported data type.", QMessageBox::Ok, this);
            folderinvalid.setWindowFlags(Qt::Dialog | Qt::CustomizeWindowHint | Qt::WindowTitleHint | Qt::WindowCloseButtonHint);
            folderinvalid.exec();
        }
    }
    catch(...)
    {
        ;
    }
}

void MainWindow::on_import_folder_clicked()
{   try
    {
        //Get the user to select the folder from the file explorer
        QString import_path =  QFileDialog::getExistingDirectory(this,tr("Select Folder"),"C://");
        QDir directory(import_path);
        QStringList items = directory.entryList(QStringList()<<"*.dcm"<<"*.nii.gz"<<"*.png",QDir::Files);

        //Checks to see if the directory contains the required files
        bool valid_folder = false;
        foreach(QString item_path,items)
        {
            if (hasEnding(item_path, g_valid_folder_endings))
            {
                valid_folder = true;
            }
        }

        //If it does, then write the path to a .txt, then opens a dialog
        if (valid_folder == true)
        {
            Import_images(import_path);
        }
        else if (!import_path.isNull())
        {
            QMessageBox folderinvalid(QMessageBox::Warning, "File Type Unsupported", "Please chooose a folder that contains files of a supported data type.", QMessageBox::Ok, this);
            folderinvalid.setWindowFlags(Qt::Dialog | Qt::CustomizeWindowHint | Qt::WindowTitleHint | Qt::WindowCloseButtonHint);
            folderinvalid.exec();
        }
    }
    catch(...)
    {
        ;
    }
}

//For switching windows
void MainWindow::setWindow(int index)
{
    ui->stackedWidget->setCurrentIndex(index);
}



//After the imports
void MainWindow::load_data()
{
    if (!(ui->stackedWidget->currentIndex() == 1)) setWindow(2);
    setAcceptDrops(false);

    QProcess *process = new QProcess(this);
    QString file("./scripts/load_data.exe");
    process->startDetached(file);

    QFile progressLabel("C:/ProgramData/MRImage3D/metadata/progress_label.txt");
    progressLabel.remove();
    while (true)
    {
        if (progressLabel.open(QIODevice::ReadWrite))
        {
            QTextStream stream(&progressLabel);
            stream << "Analyzing raw data...";
            progressLabel.close();
            break;
        }
    }
    QFile progressAmt("C:/ProgramData/MRImage3D/metadata/progress_percent.txt");
    progressAmt.remove();
    while (true)
    {
        if (progressAmt.open(QIODevice::ReadWrite))
        {
            QTextStream stream(&progressAmt);
            stream << "0";
            progressAmt.close();
            break;
        }
    }
    int startTime;
    int endTime;
    //get the start time of starting the exe
    QFile loadStart("C:/ProgramData/MRImage3D/metadata/last_load_start.txt");
    if (loadStart.exists())
    {
        while (true)
        {
            if (loadStart.open(QIODevice::ReadWrite))
            {
                QTextStream stream(&loadStart);
                startTime = stream.readLine().toFloat();
                loadStart.close();
                break;
            }
        }
    }
    else
    {
        startTime = 0;
    }
    //get the finish time of finishing the exe
    QFile runEnd("C:/ProgramData/MRImage3D/metadata/last_load_end.txt");
    if (runEnd.exists())
    {
        while (true)
        {
            if (runEnd.open(QIODevice::ReadWrite))
            {
                QTextStream stream(&runEnd);
                endTime = stream.readLine().toFloat();
                runEnd.close();
                break;
            }
        }
    }
    else
    {
        endTime = 10000;
    }
    int duration = endTime-startTime;
    progressBarUpdating->expectedDuration = duration;
    progressBarUpdating->start();

}

//After the imports
void MainWindow::process_images()
{
    QProcess *process = new QProcess(this);
    QString file("./scripts/process_images.exe");
    process->startDetached(file);

    QFile progressLabel("C:/ProgramData/MRImage3D/metadata/progress_label.txt");
    progressLabel.remove();
    while (true)
    {
        if (progressLabel.open(QIODevice::ReadWrite))
        {
            QTextStream stream(&progressLabel);
            stream << "Loading in voxel data...";
            progressLabel.close();
            break;
        }
    }
    QFile progressAmt("C:/ProgramData/MRImage3D/metadata/progress_percent.txt");
    progressAmt.remove();
    while (true)
    {
        if (progressAmt.open(QIODevice::ReadWrite))
        {
            QTextStream stream(&progressAmt);
            stream << "0";
            progressAmt.close();
            break;
        }
    }
    int startTime;
    int endTime;
    //get the start time of starting the exe
    QFile loadStart("C:/ProgramData/MRImage3D/metadata/last_process_start.txt");
    if (loadStart.exists())
    {
        while (true)
        {
            if (loadStart.open(QIODevice::ReadWrite))
            {
                QTextStream stream(&loadStart);
                startTime = stream.readLine().toFloat();
                loadStart.close();
                break;
            }
        }
    }
    else
    {
        startTime = 0;
    }
    //get the finish time of finishing the exe
    QFile runEnd("C:/ProgramData/MRImage3D/metadata/last_process_end.txt");
    if (runEnd.exists())
    {
        while (true)
        {
            if (runEnd.open(QIODevice::ReadWrite))
            {
                QTextStream stream(&runEnd);
                endTime = stream.readLine().toFloat();
                runEnd.close();
                break;
            }
        }
    }
    else
    {
        endTime = 140000;
    }
    int duration = endTime-startTime;
    progressBarUpdating->expectedDuration = duration;
    progressBarUpdating->start();

}


void MainWindow::treatment_evaluation()
{
    g_tumor_eval = true;
    QProcess *process = new QProcess(this);
    QString file("./scripts/treatment_evaluation.exe");
    process->startDetached(file);

    QFile progressLabel("C:/ProgramData/MRImage3D/metadata/progress_label.txt");
    progressLabel.remove();
    while (true)
    {
        if (progressLabel.open(QIODevice::ReadWrite))
        {
            QTextStream stream(&progressLabel);
            stream << "Loading in voxel data...";
            progressLabel.close();
            break;
        }
    }
    QFile progressAmt("C:/ProgramData/MRImage3D/metadata/progress_percent.txt");
    progressAmt.remove();
    while (true)
    {
        if (progressAmt.open(QIODevice::ReadWrite))
        {
            QTextStream stream(&progressAmt);
            stream << "0";
            progressAmt.close();
            break;
        }
    }
    int startTime;
    int endTime;
    //get the start time of starting the exe
    QFile loadStart("C:/ProgramData/MRImage3D/metadata/last_eval_start.txt");
    if (loadStart.exists())
    {
        while (true)
        {
            if (loadStart.open(QIODevice::ReadWrite))
            {
                QTextStream stream(&loadStart);
                startTime = stream.readLine().toFloat();
                loadStart.close();
                break;
            }
        }
    }
    else
    {
        startTime = 0;
    }
    //get the finish time of finishing the exe
    QFile runEnd("C:/ProgramData/MRImage3D/metadata/last_eval_end.txt");
    if (runEnd.exists())
    {
        while (true)
        {
            if (runEnd.open(QIODevice::ReadWrite))
            {
                QTextStream stream(&runEnd);
                endTime = stream.readLine().toFloat();
                runEnd.close();
                break;
            }
        }
    }
    else
    {
        endTime = 140000;
    }
    int duration = endTime-startTime;
    progressBarUpdating->expectedDuration = duration;
    progressBarUpdating->start();

}

//for the progress bar
void MainWindow::onProgressPercentUpdated(int i)
{
    if (ui->stackedWidget->currentIndex() == 2)
    {
        ui->progressBarLoading->setValue(i);
    }
    else if (ui->stackedWidget->currentIndex() == 3)
    {
        ui->progressBar->setValue(i);
        ui->progressBar_2->setValue(i);
    }
    else if (ui->stackedWidget->currentIndex() == 1)
    {
        if (g_earlier_scan)
        {
            ui->progressBarLoading_2->setValue(i);
        }
        else if (g_later_scan)
        {
            ui->progressBarLoading_3->setValue(i);
        }
    }
}

void MainWindow::onProgressLabelUpdated(QString text)
{
    if (ui->stackedWidget->currentIndex() == 2)
    {
        ui->currentProcessLabelAnalyzing->setText(text);
    }
    else if (ui->stackedWidget->currentIndex() == 3)
    {
        ui->currentProcessLabel->setText(text);
    }
    else if (ui->stackedWidget->currentIndex() == 1)
    {
        if (g_earlier_scan)
        {
            ui->currentProcessLabelAnalyzing_2->setText(text);;
        }
        else if (g_later_scan)
        {
            ui->currentProcessLabelAnalyzing_3->setText(text);;
        }
    }
}

void MainWindow::onLockWindow(bool lock)
{
    g_window_locked = lock;
}


//After the initial loading, onto import options
void MainWindow::onNextStep()
{
    if (ui->stackedWidget->currentIndex() == 2)
    {
        ui->currentProcessLabelAnalyzing->setText("Successfully loaded in data.");
        int startTime;
        int endTime;

        //get the start time of starting the exe
        QFile loadStart("C:/ProgramData/MRImage3D/metadata/last_start.txt");
        if (loadStart.exists())
        {
            while (true)
            {
                if (loadStart.open(QIODevice::ReadWrite))
                {
                    QTextStream stream(&loadStart);
                    startTime = stream.readLine().toFloat();
                    loadStart.close();
                    break;
                }
            }
        }

        //get the finish time of finishing the exe
        QFile runEnd("C:/ProgramData/MRImage3D/metadata/last_end.txt");
        if (runEnd.exists())
        {
            while (true)
            {
                if (runEnd.open(QIODevice::ReadWrite))
                {
                    QTextStream stream(&runEnd);
                    endTime = stream.readLine().toFloat();
                    runEnd.close();
                    break;
                }
            }
        }

        QFile loadStartOut("C:/ProgramData/MRImage3D/metadata/last_load_start.txt");
        if (loadStartOut.exists()) loadStartOut.remove();
        while (true)
        {
            if (loadStartOut.open(QIODevice::ReadWrite))
            {
                QTextStream stream(&loadStartOut);
                stream<<startTime;
                loadStartOut.close();
                break;
            }
        }
        //write the end time out
        QFile runEndOut("C:/ProgramData/MRImage3D/metadata/last_load_end.txt");
        if (runEndOut.exists()) runEndOut.remove();
        while (true)
        {
            if (runEndOut.open(QIODevice::ReadWrite))
            {
                QTextStream stream(&runEndOut);
                stream<<endTime;
                runEndOut.close();
                break;
            }
        }


        //Opens dialog window to select processing options
        int accepted = 0;
        SelectImportingOptions import_options_dialog;
        import_options_dialog.setWindowTitle("MRImage3D - Image Processing Options");
        import_options_dialog.setWindowFlags(windowFlags() & ~Qt::WindowContextHelpButtonHint);
        import_options_dialog.setModal(true);
        import_options_dialog.move(this->geometry().center()-import_options_dialog.rect().center());

        accepted = import_options_dialog.exec();

        //If the dialog has been accepted, switch the window then run the python exe's, then switch another window
        if (accepted == 1)
        {
            setWindow(3);
            setAcceptDrops(false);
            process_images();
        }
        else if (accepted == 10)
        {
            close();
        }
        else if (accepted == 20)
        {
            on_actionReturn_to_start_page_triggered();
        }
        else
        {
            setWindow(0);
            setAcceptDrops(true);
        }
    }
    else if (ui->stackedWidget->currentIndex() == 3)
    {
        ui->currentProcessLabel->setText("Processing finished.");
        int startTime;
        int endTime;

        //get the start time of starting the exe
        QFile loadStart("C:/ProgramData/MRImage3D/metadata/last_start.txt");
        if (loadStart.exists())
        {
            while (true)
            {
                if (loadStart.open(QIODevice::ReadWrite))
                {
                    QTextStream stream(&loadStart);
                    startTime = stream.readLine().toFloat();
                    loadStart.close();
                    break;
                }
            }
        }

        //get the finish time of finishing the exe
        QFile runEnd("C:/ProgramData/MRImage3D/metadata/last_end.txt");
        if (runEnd.exists())
        {
            while (true)
            {
                if (runEnd.open(QIODevice::ReadWrite))
                {
                    QTextStream stream(&runEnd);
                    endTime = stream.readLine().toFloat();
                    runEnd.close();
                    break;
                }
            }
        }
        if (g_tumor_eval == true)
        {
            QFile loadStartOut("C:/ProgramData/MRImage3D/metadata/last_eval_start.txt");
            if (loadStartOut.exists()) loadStartOut.remove();
            while (true)
            {
                if (loadStartOut.open(QIODevice::ReadWrite))
                {
                    QTextStream stream(&loadStartOut);
                    stream<<startTime;
                    loadStartOut.close();
                    break;
                }
            }
            //write the end time out
            QFile runEndOut("C:/ProgramData/MRImage3D/metadata/last_eval_end.txt");
            if (runEndOut.exists()) runEndOut.remove();
            while (true)
            {
                if (runEndOut.open(QIODevice::ReadWrite))
                {
                    QTextStream stream(&runEndOut);
                    stream<<endTime;
                    runEndOut.close();
                    break;
                }
            }

            setWindow(4);

        }
        else
        {
            QFile loadStartOut("C:/ProgramData/MRImage3D/metadata/last_process_start.txt");
            if (loadStartOut.exists()) loadStartOut.remove();
            while (true)
            {
                if (loadStartOut.open(QIODevice::ReadWrite))
                {
                    QTextStream stream(&loadStartOut);
                    stream<<startTime;
                    loadStartOut.close();
                    break;
                }
            }
            //write the end time out
            QFile runEndOut("C:/ProgramData/MRImage3D/metadata/last_process_end.txt");
            if (runEndOut.exists()) runEndOut.remove();
            while (true)
            {
                if (runEndOut.open(QIODevice::ReadWrite))
                {
                    QTextStream stream(&runEndOut);
                    stream<<endTime;
                    runEndOut.close();
                    break;
                }
            }

            setWindow(4);
        }
    }
    else if (ui->stackedWidget->currentIndex() == 1)
    {
        int startTime;
        int endTime;

        //get the start time of starting the exe
        QFile loadStart("C:/ProgramData/MRImage3D/metadata/last_start.txt");
        if (loadStart.exists())
        {
            while (true)
            {
                if (loadStart.open(QIODevice::ReadWrite))
                {
                    QTextStream stream(&loadStart);
                    startTime = stream.readLine().toFloat();
                    loadStart.close();
                    break;
                }
            }
        }

        //get the finish time of finishing the exe
        QFile runEnd("C:/ProgramData/MRImage3D/metadata/last_end.txt");
        if (runEnd.exists())
        {
            while (true)
            {
                if (runEnd.open(QIODevice::ReadWrite))
                {
                    QTextStream stream(&runEnd);
                    endTime = stream.readLine().toFloat();
                    runEnd.close();
                    break;
                }
            }
        }
        if (g_earlier_scan)
        {
            ui->currentProcessLabelAnalyzing_2->setText("Successfully loaded in data.");
            QFile loadStartOut("C:/ProgramData/MRImage3D/metadata/last_load_start.txt");
            if (loadStartOut.exists()) loadStartOut.remove();
            while (true)
            {
                if (loadStartOut.open(QIODevice::ReadWrite))
                {
                    QTextStream stream(&loadStartOut);
                    stream<<startTime;
                    loadStartOut.close();
                    break;
                }
            }
            //write the end time out
            QFile runEndOut("C:/ProgramData/MRImage3D/metadata/last_load_end.txt");
            if (runEndOut.exists()) runEndOut.remove();
            while (true)
            {
                if (runEndOut.open(QIODevice::ReadWrite))
                {
                    QTextStream stream(&runEndOut);
                    stream<<endTime;
                    runEndOut.close();
                    break;
                }
            }
            QFile pixel_array("C:/ProgramData/MRImage3D/metadata/pixel_array.h5");
            pixel_array.rename("C:/ProgramData/MRImage3D/metadata/pixel_array.h5", "C:/ProgramData/MRImage3D/metadata/pixel_array_earlier_scan.h5");

            setAcceptDrops(true);

            g_earlier_scan = false;
            g_loaded_earlier_scan = true;
            if (g_loaded_later_scan && g_loaded_earlier_scan)
            {
                ui->earlier_scan_label->setText("Scan 1 loaded. Please select desired import options to begin treatment evaluation");
                ui->earlier_scan_label->setVisible(true);
                //Opens dialog window to select processing options
                int accepted = 0;
                SelectImportingOptions import_options_dialog1;
                import_options_dialog1.setWindowTitle("MRImage3D - Image Processing Options - Scan 1");
                import_options_dialog1.setWindowFlags(windowFlags() & ~Qt::WindowContextHelpButtonHint);
                import_options_dialog1.setModal(true);
                import_options_dialog1.move(this->geometry().center()-import_options_dialog1.rect().center());

                accepted = import_options_dialog1.exec();

                QFile import_options1("C:/ProgramData/MRImage3D/metadata/import_options.txt");
                import_options1.rename("C:/ProgramData/MRImage3D/metadata/import_options.txt", "C:/ProgramData/MRImage3D/metadata/import_options_earlier.txt");



                //If the dialog has been accepted, switch the window then run the python exe's, then switch another window


                if (accepted == 10)
                {
                    close();
                }

                //Opens dialog window to select processing options
                accepted = 0;
                SelectImportingOptions import_options_dialog2;
                import_options_dialog2.setWindowTitle("MRImage3D - Image Processing Options - Scan 2");
                import_options_dialog2.setWindowFlags(windowFlags() & ~Qt::WindowContextHelpButtonHint);
                import_options_dialog2.setModal(true);
                import_options_dialog2.move(this->geometry().center()-import_options_dialog2.rect().center());

                accepted = import_options_dialog2.exec();

                QFile import_options2("C:/ProgramData/MRImage3D/metadata/import_options.txt");
                import_options2.rename("C:/ProgramData/MRImage3D/metadata/import_options.txt", "C:/ProgramData/MRImage3D/metadata/import_options_later.txt");


                //If the dialog has been accepted, switch the window then run the python exe's, then switch another window


                if (accepted == 10)
                {
                    close();
                }


                setWindow(3);
                setAcceptDrops(false);
                treatment_evaluation();
            }
            else
            {
                ui->earlier_scan_label->setVisible(true);
            }
        }
        else if (g_later_scan)
        {
            ui->currentProcessLabelAnalyzing_3->setText("Successfully loaded in data.");
            QFile loadStartOut("C:/ProgramData/MRImage3D/metadata/last_load_start.txt");
            if (loadStartOut.exists()) loadStartOut.remove();
            while (true)
            {
                if (loadStartOut.open(QIODevice::ReadWrite))
                {
                    QTextStream stream(&loadStartOut);
                    stream<<startTime;
                    loadStartOut.close();
                    break;
                }
            }
            //write the end time out
            QFile runEndOut("C:/ProgramData/MRImage3D/metadata/last_load_end.txt");
            if (runEndOut.exists()) runEndOut.remove();
            while (true)
            {
                if (runEndOut.open(QIODevice::ReadWrite))
                {
                    QTextStream stream(&runEndOut);
                    stream<<endTime;
                    runEndOut.close();
                    break;
                }
            }
            QFile pixel_array("C:/ProgramData/MRImage3D/metadata/pixel_array.h5");
            pixel_array.rename("C:/ProgramData/MRImage3D/metadata/pixel_array.h5", "C:/ProgramData/MRImage3D/metadata/pixel_array_later_scan.h5");

            setAcceptDrops(true);

            g_later_scan = false;
            g_loaded_later_scan = true;
            if (g_loaded_later_scan && g_loaded_earlier_scan)
            {
                ui->later_scan_label->setText("Scan 2 loaded. Please select desired import options to begin treatment evaluation");
                ui->later_scan_label->setVisible(true);
                //Opens dialog window to select processing options
                int accepted = 0;
                SelectImportingOptions import_options_dialog1;
                import_options_dialog1.setWindowTitle("MRImage3D - Image Processing Options - Scan 1");
                import_options_dialog1.setWindowFlags(windowFlags() & ~Qt::WindowContextHelpButtonHint);
                import_options_dialog1.setModal(true);
                import_options_dialog1.move(this->geometry().center()-import_options_dialog1.rect().center());

                accepted = import_options_dialog1.exec();

                QFile import_options1("C:/ProgramData/MRImage3D/metadata/import_options.txt");
                import_options1.rename("C:/ProgramData/MRImage3D/metadata/import_options.txt", "C:/ProgramData/MRImage3D/metadata/import_options_earlier.txt");



                //If the dialog has been accepted, switch the window then run the python exe's, then switch another window


                if (accepted == 10)
                {
                    close();
                }

                //Opens dialog window to select processing options
                accepted = 0;
                SelectImportingOptions import_options_dialog2;
                import_options_dialog2.setWindowTitle("MRImage3D - Image Processing Options - Scan 2");
                import_options_dialog2.setWindowFlags(windowFlags() & ~Qt::WindowContextHelpButtonHint);
                import_options_dialog2.setModal(true);
                import_options_dialog2.move(this->geometry().center()-import_options_dialog2.rect().center());

                accepted = import_options_dialog2.exec();

                QFile import_options2("C:/ProgramData/MRImage3D/metadata/import_options.txt");
                import_options2.rename("C:/ProgramData/MRImage3D/metadata/import_options.txt", "C:/ProgramData/MRImage3D/metadata/import_options_later.txt");
                //If the dialog has been accepted, switch the window then run the python exe's, then switch another window


                if (accepted == 10)
                {
                    close();
                }


                setWindow(3);
                setAcceptDrops(false);
                treatment_evaluation();

            }
            else
            {
                ui->later_scan_label->setVisible(true);
            }
        }
    }
}

void MainWindow::onLoadingFailed()
{
    if (ui->stackedWidget->currentIndex() == 2)
    {
        ui->currentProcessLabelAnalyzing->setText("Data loading failed.");
        setWindow(0);
        setAcceptDrops(true);
        QMessageBox dataLoadFailed(QMessageBox::Warning, "Data Invalid", "Failed to load the selected data. The data was either corrupt or did not represent a 3D array of voxels.", QMessageBox::Ok, this);
        dataLoadFailed.setWindowFlags(Qt::Dialog | Qt::CustomizeWindowHint | Qt::WindowTitleHint | Qt::WindowCloseButtonHint);
        dataLoadFailed.exec();
    }
    else if (ui->stackedWidget->currentIndex() == 1)
    {
        if (g_earlier_scan)
        {
            ui->currentProcessLabelAnalyzing_2->setText("Data loading failed.");

            g_earlier_scan = false;
            QMessageBox dataLoadFailed(QMessageBox::Warning, "Data Invalid", "Failed to load the selected data. The data was either corrupt or did not represent a 3D array of voxels.", QMessageBox::Ok, this);
            dataLoadFailed.setWindowFlags(Qt::Dialog | Qt::CustomizeWindowHint | Qt::WindowTitleHint | Qt::WindowCloseButtonHint);
            dataLoadFailed.exec();
        }
        if (g_later_scan)
        {
            ui->currentProcessLabelAnalyzing_3->setText("Data loading failed.");
            g_later_scan = false;
            QMessageBox dataLoadFailed(QMessageBox::Warning, "Data Invalid", "Failed to load the selected data. The data was either corrupt or did not represent a 3D array of voxels.", QMessageBox::Ok, this);
            dataLoadFailed.setWindowFlags(Qt::Dialog | Qt::CustomizeWindowHint | Qt::WindowTitleHint | Qt::WindowCloseButtonHint);
            dataLoadFailed.exec();
        }
    }
}

//post operation buttons
void MainWindow::on_saveAsButton_clicked()
{
    QString save_path = QFileDialog::getSaveFileName(this,tr("Save As"),"C://","MRImage3D files (*.mr3d);;All files (*.*)");
    QFile importSettings("C:/ProgramData/MRImage3D/metadata/import_options.txt");
    int images;
    while (true)
    {
        if (importSettings.open(QIODevice::ReadWrite))
        {

            QTextStream stream(&importSettings);
            int nb_line = 0;
            while(!stream.atEnd())
            {
                images = stream.readLine().toInt();
                if(nb_line == 6)
                    break;
                nb_line++;
            }
            importSettings.close();
            break;
        }
    }
    try
    {
        if (images == 2)
        {
            if (!save_path.toStdString().empty())
            {
                QFile saveFile(save_path);
                if (saveFile.exists()) saveFile.remove();
                QFile pathesDict("C:/ProgramData/MRImage3D/metadata/processed_data.mr3d");
                QString save_path_replaced = save_path;
                save_path_replaced.replace("/", ".").replace(":","_");
                while (true)
                {
                    if (pathesDict.open(QIODevice::ReadWrite))
                    {
                        QTextStream stream(&pathesDict);
                        QString dict = stream.readLine();
                        dict.replace("temp_processed_storage", save_path_replaced+"."+QFileInfo(save_path).fileName());
                        pathesDict.resize(0);
                        stream<<dict;
                        pathesDict.close();
                        break;
                    }
                }

                QDir tempDir("C:/ProgramData/MRImage3D/metadata/temp_processed_storage");
                QDir newDir("C:/ProgramData/MRImage3D/metadata/" + save_path_replaced+"."+QFileInfo(save_path).fileName());

                tempDir.rename(tempDir.absolutePath(), newDir.absolutePath());
                QDir temp_imgs_dir("C:/ProgramData/MRImage3D/metadata/" + save_path_replaced+"."+QFileInfo(save_path).fileName());
                QDir temp_imgs_dir_brain("C:/ProgramData/MRImage3D/metadata/" + save_path_replaced+"."+QFileInfo(save_path).fileName()+"/images/brain_seg");
                QDir temp_imgs_dir_tumor("C:/ProgramData/MRImage3D/metadata/" + save_path_replaced+"."+QFileInfo(save_path).fileName()+"/images/tumor_seg");
                QDir dest_dir(save_path);
                QDir imgs_path(dest_dir.absolutePath() + "/images");
                if (!imgs_path.exists())
                {
                    imgs_path.mkpath("./tumor_seg");
                    imgs_path.mkpath("./brain_seg");
                }
                else
                {
                    imgs_path.removeRecursively();
                    imgs_path.mkpath("./tumor_seg");
                    imgs_path.mkpath("./brain_seg");
                }
                QStringList brain_images = temp_imgs_dir_brain.entryList(QStringList()<<"*.png",QDir::Files);

                QStringList tumor_images = temp_imgs_dir_tumor.entryList(QStringList()<<"*.png",QDir::Files);

                foreach(QString image, brain_images)
                {
                    QFile::copy(temp_imgs_dir_brain.path()+"/"+image, imgs_path.path()+"/brain_seg/"+image);
                }

                foreach(QString image, tumor_images)
                {
                    QFile::copy(temp_imgs_dir_tumor.path()+"/"+image, imgs_path.path()+"/tumor_seg/"+image);
                }
                QFile::copy("C:/ProgramData/MRImage3D/metadata/processed_data.mr3d", dest_dir.path()+"/"+QFileInfo(save_path).fileName());
                setWindow(5);
            }
        }
        else if (images==0)
        {
            if (!save_path.toStdString().empty())
            {
                QFile saveFile(save_path);
                if (saveFile.exists()) saveFile.remove();
                QFile pathesDict("C:/ProgramData/MRImage3D/metadata/processed_data.mr3d");
                QString save_path_replaced = save_path;
                save_path_replaced.replace("/", ".").replace(":","_");
                while (true)
                {
                    if (pathesDict.open(QIODevice::ReadWrite))
                    {
                        QTextStream stream(&pathesDict);
                        QString dict = stream.readLine();
                        dict.replace("temp_processed_storage", save_path_replaced);
                        pathesDict.resize(0);
                        stream<<dict;
                        pathesDict.close();
                        break;
                    }
                }

                QDir tempDir("C:/ProgramData/MRImage3D/metadata/temp_processed_storage");
                QDir newDir("C:/ProgramData/MRImage3D/metadata/" + save_path_replaced);

                tempDir.rename(tempDir.absolutePath(), newDir.absolutePath());

                QFile::copy("C:/ProgramData/MRImage3D/metadata/processed_data.mr3d", save_path);
                setWindow(5);
            }
        }
    }
    catch(...)
    {
        QMessageBox savingFailed(QMessageBox::Warning, "Saving Location Invalid", "Failed to save into " + save_path + ". Please check that the file is not currently in use and try again.", QMessageBox::Ok, this);
        savingFailed.setWindowFlags(Qt::Dialog | Qt::CustomizeWindowHint | Qt::WindowTitleHint | Qt::WindowCloseButtonHint);
        savingFailed.exec();
    }
}

//if the user drags in an item
void MainWindow::dragEnterEvent(QDragEnterEvent * event)
{
    if (ui->stackedWidget->currentIndex() == 1) event->accept();
    if (event->mimeData()->hasUrls())
    {
        QList<QUrl> urls = event->mimeData()->urls();
        foreach(QUrl url,urls)
        {
            //checks whether the dragged item is a folder
            const QString path = url.path();
            QStringRef cut_path(&path,1,path.length()-1);
            QString actual_path = cut_path.toString();

            QFileInfo fileInfo(actual_path);
            //if it is a folder, check the items inside, and if are valid, accept the items
            if (fileInfo.isDir())
            {
                QDir actual_folder(actual_path);
                QStringList files = actual_folder.entryList(QStringList()<<"*.dcm"<<"*.nii.gz"<<"*.png",QDir::Files);
                foreach(QString file, files)
                {
                    if (hasEnding(file,g_valid_folder_endings))
                    {
                        event->accept();
                    }
                }
                //if there are more than one folder, reject it (only one scan at a time can be processed)
                if (urls.length() > 1)
                {
                    event->ignore();
                }
            }
            else if (fileInfo.isFile())
            {
                if (urls.length() == 1) //if it's only one, so a single file image
                {
                    if (hasEnding(url.fileName(),g_valid_file_endings))
                    {
                        event->accept();
                    }
                }
                else if (urls.length() > 1) //it shouldn't ever be 0, but just in case
                {
                    if (hasEnding(url.fileName(),g_valid_folder_endings))
                    {
                        event->accept();
                    }
                }
            }
        }
    }
    else event->ignore();
}

void MainWindow::dragMoveEvent(QDragMoveEvent * event)
{
    print("moving");
    bool tumor_eval_valid = true;
    if (ui->stackedWidget->currentIndex() == 1)//tumor treatment evaluation
    {
        //earlier scan
        print(QString::number(ui->centralwidget->mapToGlobal(event->pos()).x()));
        print(QString::number(ui->centralwidget->mapToGlobal(event->pos()).y()));
        if ((MainWindow::mapToGlobal(event->pos()).x() >= ui->frame_6->mapToGlobal(ui->frame_7->pos()).x() && MainWindow::mapToGlobal(event->pos()).x() <= ui->frame_6->mapToGlobal(ui->frame_7->pos()).x()+ui->frame_7->width()) && (MainWindow::mapToGlobal(event->pos()).y() >= ui->frame_6->mapToGlobal(ui->frame_7->pos()).y() && MainWindow::mapToGlobal(event->pos()).y() <= ui->frame_6->mapToGlobal(ui->frame_7->pos()).y()+ui->frame_7->height()))
        {
            event->accept();
            print("7");
        }
        else if ((MainWindow::mapToGlobal(event->pos()).x() >= ui->frame_6->mapToGlobal(ui->frame_8->pos()).x() && MainWindow::mapToGlobal(event->pos()).x() <= ui->frame_6->mapToGlobal(ui->frame_8->pos()).x()+ui->frame_8->width()) && (MainWindow::mapToGlobal(event->pos()).y() >= ui->frame_6->mapToGlobal(ui->frame_8->pos()).y() && MainWindow::mapToGlobal(event->pos()).y() <= ui->frame_6->mapToGlobal(ui->frame_8->pos()).y()+ui->frame_8->height()))
        {
            event->accept();
            print("8");
        }
        else
        {
            print("oof");
            event->ignore();
            tumor_eval_valid = false;
        }
    }

    if (event->mimeData()->hasUrls() && tumor_eval_valid==true)
    {
        QList<QUrl> urls = event->mimeData()->urls();
        foreach(QUrl url,urls)
        {
            //checks whether the dragged item is a folder
            const QString path = url.path();
            QStringRef cut_path(&path,1,path.length()-1);
            QString actual_path = cut_path.toString();

            QFileInfo fileInfo(actual_path);
            //if it is a folder, check the items inside, and if are valid, accept the items
            if (fileInfo.isDir())
            {
                QDir actual_folder(actual_path);
                QStringList files = actual_folder.entryList(QStringList()<<"*.dcm"<<"*.nii.gz"<<"*.png",QDir::Files);
                foreach(QString file, files)
                {
                    if (hasEnding(file,g_valid_folder_endings))
                    {
                        event->accept();
                    }
                }
                //if there are more than one folder, reject it (only one scan at a time can be processed)
                if (urls.length() > 1)
                {
                    event->ignore();
                }
            }
            else if (fileInfo.isFile())
            {
                if (urls.length() == 1) //if it's only one, so a single file image
                {
                    if (hasEnding(url.fileName(),g_valid_file_endings))
                    {
                        event->accept();
                    }
                }
                else if (urls.length() > 1) //it shouldn't ever be 0, but just in case
                {
                    if (hasEnding(url.fileName(),g_valid_folder_endings))
                    {
                        event->accept();
                    }
                }
            }
        }
    }
    else event->ignore();
}

void MainWindow::dropEvent(QDropEvent * event)
{
     bool tumor_eval_valid = true;
     if (ui->stackedWidget->currentIndex() == 1)//tumor treatment evaluation
     {
         //earlier scan
         print(QString::number(ui->centralwidget->mapToGlobal(event->pos()).x()));
         print(QString::number(ui->centralwidget->mapToGlobal(event->pos()).y()));
         if ((MainWindow::mapToGlobal(event->pos()).x() >= ui->frame_6->mapToGlobal(ui->frame_7->pos()).x() && MainWindow::mapToGlobal(event->pos()).x() <= ui->frame_6->mapToGlobal(ui->frame_7->pos()).x()+ui->frame_7->width()) && (MainWindow::mapToGlobal(event->pos()).y() >= ui->frame_6->mapToGlobal(ui->frame_7->pos()).y() && MainWindow::mapToGlobal(event->pos()).y() <= ui->frame_6->mapToGlobal(ui->frame_7->pos()).y()+ui->frame_7->height()))
         {
             event->accept();
             g_earlier_scan = true;
             ui->label_4->setVisible(false);
             ui->import_file_2->setVisible(false);
             ui->import_folder_2->setVisible(false);
             ui->progressBarLoading_2->setVisible(true);
             ui->currentProcessLabelAnalyzing_2->setVisible(true);
         }
         else if ((MainWindow::mapToGlobal(event->pos()).x() >= ui->frame_6->mapToGlobal(ui->frame_8->pos()).x() && MainWindow::mapToGlobal(event->pos()).x() <= ui->frame_6->mapToGlobal(ui->frame_8->pos()).x()+ui->frame_8->width()) && (MainWindow::mapToGlobal(event->pos()).y() >= ui->frame_6->mapToGlobal(ui->frame_8->pos()).y() && MainWindow::mapToGlobal(event->pos()).y() <= ui->frame_6->mapToGlobal(ui->frame_8->pos()).y()+ui->frame_8->height()))
         {
             event->accept();
             g_later_scan = true;
             ui->label_9->setVisible(false);
             ui->import_file_6->setVisible(false);
             ui->import_folder_6->setVisible(false);
             ui->progressBarLoading_3->setVisible(true);
             ui->currentProcessLabelAnalyzing_3->setVisible(true);
         }
         else
         {
             event->ignore();
             tumor_eval_valid = false;
         }
     }


    if (event->mimeData()->hasUrls() && tumor_eval_valid)
    {
        bool multiple_file_drop = false;
        QString multiple_file_path = "";
        QList<QUrl> urls = event->mimeData()->urls();
        foreach(QUrl url,urls)
        {
            //checks whether the dragged item is a folder
            const QString path = url.path();
            QStringRef cut_path(&path,1,path.length()-1);
            QString actual_path = cut_path.toString();

            QFileInfo fileInfo(actual_path);
            //if it is a folder, check the items inside, and if are valid, accept the items
            if (fileInfo.isDir())
            {
                QDir actual_folder(actual_path);
                QStringList files = actual_folder.entryList(QStringList()<<"*.dcm"<<"*.nii.gz"<<"*.png",QDir::Files);
                foreach(QString file, files)
                {
                    if (hasEnding(file,g_valid_folder_endings))
                    {
                        event->accept();
                    }
                }
                //if there are more than one folder, reject it (only one scan at a time can be processed)
                if (urls.length() > 1)
                {
                    event->ignore();
                }
                //otherwise, import
                else
                {
                    Import_images(actual_path);
                }
            }
            //if its a file
            else if (fileInfo.isFile())
            {
                if (hasEnding(url.fileName(),g_valid_folder_endings))
                {
                    if (urls.length() == 1) //if it's only one, so a single file image, it cant be png
                    {
                        if (hasEnding(url.fileName(),g_valid_file_endings))
                        {
                            event->accept();
                            Import_images(actual_path);
                        }
                    }
                    else if (urls.length() > 1) //it shouldn't ever be 0, but just in case
                    {
                        event->accept();
                        QDir actual_folder(actual_path);
                        actual_folder.cdUp();
                        multiple_file_path = actual_folder.absolutePath();
                        multiple_file_drop = true;
                    }
                }
            }
        //if there is more than one file selected, so the user selected all the files in a folder
        }
        if (multiple_file_drop)
        {
            Import_images(multiple_file_path);
        }
    }
}

void MainWindow::on_openAnotherButton_clicked()
{
    setWindow(0);
    setAcceptDrops(true);
}

void MainWindow::on_closeButton_clicked()
{
    close();
}

//Menu items (the folder selctions are all gonna be the same)
void MainWindow::on_action_dcm_folder_triggered()
{
    setWindow(0);
    setAcceptDrops(true);
    on_import_folder_clicked();
}

void MainWindow::on_action_png_folder_triggered()
{
    setWindow(0);
    setAcceptDrops(true);
    on_import_folder_clicked();
}

void MainWindow::on_action_nii_folder_triggered()
{
    setWindow(0);
    setAcceptDrops(true);
    on_import_folder_clicked();
}

//import file
void MainWindow::on_action_dcm_file_triggered()
{
    setWindow(0);
    setAcceptDrops(true);
    g_acceptable_file_types = "Dicom format image (*.dcm);;All files (*.*)";
    on_import_file_clicked();
    g_acceptable_file_types = "Dicom format image (*.dcm);;Nii compressed image (*.nii.gz);;3D array stored in h5 format (*.h5);;All files (*.*)";
}

void MainWindow::on_action_import_nii_file_triggered()
{
    setWindow(0);
    setAcceptDrops(true);
    g_acceptable_file_types = "Nii compressed image (*.nii.gz);;All files (*.*)";
    on_import_file_clicked();
    g_acceptable_file_types = "Dicom format image (*.dcm);;Nii compressed image (*.nii.gz);;3D array stored in h5 format (*.h5);;All files (*.*)";
}

void MainWindow::on_action_h5_3D_array_triggered()
{
    setWindow(0);
    setAcceptDrops(true);
    g_acceptable_file_types = "3D array stored in h5 format (*.h5);;All files (*.*)";
    on_import_file_clicked();
    g_acceptable_file_types = "Dicom format image (*.dcm);;Nii compressed image (*.nii.gz);;3D array stored in h5 format (*.h5);;All files (*.*)";
}

void MainWindow::on_actionExit_triggered()
{
    close();
}

void MainWindow::on_actionReturn_to_start_page_triggered()
{
    if (!g_window_locked)
    {
        setAcceptDrops(true);
        setWindow(0);
    }
    else
    {
        QMessageBox dataLoadFailed(QMessageBox::Warning, "Cannot Return Now", "Please wait for the current process to complete then try again.", QMessageBox::Ok, this);
        dataLoadFailed.setWindowFlags(Qt::Dialog | Qt::CustomizeWindowHint | Qt::WindowTitleHint | Qt::WindowCloseButtonHint);
        dataLoadFailed.exec();
    }
}

void MainWindow::on_action_h5_folder_triggered()
{
    setWindow(0);
    setAcceptDrops(true);
    on_import_folder_clicked();
}

void MainWindow::on_comboBox_currentIndexChanged(int index)
{
    if (index == 1 || index==3)
    {
        ui->comboBox_2->setCurrentIndex(1);
        setWindow(5);
        ui->currentProcessLabelAnalyzing_2->setVisible(false);
        ui->progressBarLoading_2->setVisible(false);
        ui->currentProcessLabelAnalyzing_3->setVisible(false);
        ui->progressBarLoading_3->setVisible(false);
        ui->earlier_scan_label->setVisible(false);
        ui->later_scan_label->setVisible(false);
        int accepted = 0;
        SelectImportingOptions import_options_dialog;
        import_options_dialog.setWindowTitle("MRImage3D - Image Processing Options");
        import_options_dialog.setWindowFlags(windowFlags() & ~Qt::WindowContextHelpButtonHint);
        import_options_dialog.setModal(true);
        import_options_dialog.move(this->geometry().center()-import_options_dialog.rect().center());

        accepted = import_options_dialog.exec();
    }
}

void MainWindow::on_comboBox_2_currentIndexChanged(int index)
{
    if (index == 0)
    {
        ui->comboBox->setCurrentIndex(0);
        setWindow(0);
    }
}


