#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include "progressbarupdate.h"
#include "QDialog"
#include "selectimportingoptions.h"

QT_BEGIN_NAMESPACE
namespace Ui { class MainWindow; }
QT_END_NAMESPACE

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    MainWindow(QWidget *parent = nullptr);
    void setWindow(int index);
    void Import_images(QString path);

    void load_data();
    void process_images();
    void treatment_evaluation();
    ProgressBarUpdate *progressBarUpdating;
    //void close();

    ~MainWindow();

private slots:
    void on_import_folder_clicked();

    void on_import_file_clicked();

    void on_saveAsButton_clicked();

    void on_openAnotherButton_clicked();

    void on_closeButton_clicked();

    void on_action_dcm_folder_triggered();

    void on_action_png_folder_triggered();

    void on_action_nii_folder_triggered();

    void on_action_dcm_file_triggered();

    void on_action_import_nii_file_triggered();

    void on_action_h5_3D_array_triggered();

    void on_actionExit_triggered();

    void on_actionReturn_to_start_page_triggered();

    void on_action_h5_folder_triggered();


    void on_comboBox_2_currentIndexChanged(int index);

    void on_comboBox_currentIndexChanged(int index);


private:
    Ui::MainWindow *ui;

protected:
    void dragEnterEvent(QDragEnterEvent * event);
    void dragMoveEvent(QDragMoveEvent * event);
    void dropEvent(QDropEvent * event);

public slots:
    void onProgressPercentUpdated(int);
    void onProgressLabelUpdated(QString);
    void onNextStep();
    void onLoadingFailed();
    void onLockWindow(bool lock);


};
#endif // MAINWINDOW_H
