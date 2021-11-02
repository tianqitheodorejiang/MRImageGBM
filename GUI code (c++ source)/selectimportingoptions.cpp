#include "selectimportingoptions.h"
#include "ui_selectimportingoptions.h"
#include "iostream"
#include "QFile"
#include "QTextStream"
#include "QDir"
#include "mainwindow.h"
#include "ui_mainwindow.h"
#include "QShortcut"


//Running list of selected parameters
std::vector<int> g_info = {0,2,2,2,2,2,2};
int g_model_count = 3;
QString g_currentConfirmTip = "Confirm selections and proceed to image processing";

void SelectImportingOptions::closeAll()
{
    QDialog::done(10);
    close();
}

void SelectImportingOptions::mainReturn()
{
    QDialog::done(20);
    close();
}

//default init stuff
SelectImportingOptions::SelectImportingOptions(QWidget *parent) :
    QDialog(parent),
    ui(new Ui::SelectImportingOptions)
{
    ui->setupUi(this);
    g_info = {0,2,2,2,2,2,2};
    g_model_count = 3;
    g_currentConfirmTip = "Confirm selections and proceed to image processing";
    new QShortcut(QKeySequence(Qt::CTRL + Qt::Key_Q), this, SLOT(closeAll()));
    new QShortcut(QKeySequence(Qt::CTRL + Qt::Key_R), this, SLOT(mainReturn()));
}
SelectImportingOptions::~SelectImportingOptions()
{
    delete ui;
}


//confirm and cancel
void SelectImportingOptions::on_confirmButton_clicked()
{
    //accepts and closes the dialog
    QDialog::accept();

    //writes the info data into a .txt file
    QFile file("C:/ProgramData/MRImage3D/metadata/import_options.txt");
    QDir dir("C:/ProgramData/MRImage3D/metadata");
    if (!dir.exists())
        dir.mkpath(".");
    if (file.exists())
        file.remove();

    if (file.open(QIODevice::ReadWrite))
    {
        QTextStream stream( &file );
        foreach(int value, g_info)
        {
        stream << value << Qt::endl;
        }
        file.close();
    }
}
void SelectImportingOptions::on_cancelButton_clicked()
{
    QDialog::reject();
}


//buttons in the dialog
void SelectImportingOptions::on_scanType_currentIndexChanged(int index)
{
    g_info[0] = index;
    //disable the checkboxes unsupported options
    if (index == 0)
    {
        //Flair
        ui->tumorSeg->setEnabled(true);
        ui->brainSeg->setEnabled(true);
        ui->axialModel->setEnabled(true);
        ui->corticalModel->setEnabled(true);
        ui->sagittalModel->setEnabled(true);
        ui->confirmButton->setEnabled(true);
        ui->tumorSeg->setChecked(true);
        ui->brainSeg->setChecked(true);
        ui->axialModel->setChecked(true);
        ui->corticalModel->setChecked(true);
        ui->sagittalModel->setChecked(true);
        ui->confirmButton->setChecked(true);

        //setting the tool tips
        ui->confirmButton->setToolTip("Confirm selections and proceed to image processing");
        g_currentConfirmTip = "Confirm selections and proceed to image processing";
        ui->tumorSeg->setToolTip("Use tumor segmentation");
        ui->brainSeg->setToolTip("Use brain segmentation");
        ui->axialModel->setToolTip("Use axial model");
        ui->sagittalModel->setToolTip("Use sagittal model");
        ui->corticalModel->setToolTip("Use cortical model");
    }
    else if (index == 1)
    {
        //T1
        ui->tumorSeg->setEnabled(false);
        ui->brainSeg->setEnabled(true);
        ui->axialModel->setEnabled(true);
        ui->corticalModel->setEnabled(true);
        ui->sagittalModel->setEnabled(true);
        ui->confirmButton->setEnabled(true);
        ui->tumorSeg->setChecked(false);
        ui->brainSeg->setChecked(true);
        ui->axialModel->setChecked(true);
        ui->corticalModel->setChecked(true);
        ui->sagittalModel->setChecked(true);
        ui->confirmButton->setChecked(true);

        //setting the tool tips
        ui->confirmButton->setToolTip("Confirm selections and proceed to image processing");
        g_currentConfirmTip = "Confirm selections and proceed to image processing";
        ui->tumorSeg->setToolTip("Currently the only supported format for tumor segmentation is flair, but T1 and T2 support may come with a new version");
        ui->brainSeg->setToolTip("Use brain segmentation");
        ui->axialModel->setToolTip("Use axial model");
        ui->sagittalModel->setToolTip("Use sagittal model");
        ui->corticalModel->setToolTip("Use cortical model");
    }
    else if (index == 2)
    {
        //T2 (currently not yet supported)
        ui->tumorSeg->setEnabled(false);
        ui->brainSeg->setEnabled(false);
        ui->axialModel->setEnabled(false);
        ui->corticalModel->setEnabled(false);
        ui->sagittalModel->setEnabled(false);
        ui->confirmButton->setEnabled(false);
        ui->tumorSeg->setChecked(false);
        ui->brainSeg->setChecked(false);
        ui->axialModel->setChecked(false);
        ui->corticalModel->setChecked(false);
        ui->sagittalModel->setChecked(false);
        ui->confirmButton->setChecked(false);

        //setting the tool tips
        ui->confirmButton->setToolTip("T2 support is currently unavailable, although it may come with a new version");
        g_currentConfirmTip = "T2 support is currently unavailable, although it may come with a new version";
        ui->tumorSeg->setToolTip("T2 support is currently unavailable, although it may come with a new version");
        ui->brainSeg->setToolTip("T2 support is currently unavailable, although it may come with a new version");
        ui->axialModel->setToolTip("T2 support is currently unavailable, although it may come with a new version");
        ui->sagittalModel->setToolTip("T2 support is currently unavailable, although it may come with a new version");
        ui->corticalModel->setToolTip("T2 support is currently unavailable, although it may come with a new version");
    }
}

//Check that at least one model was selected
void SelectImportingOptions::check_selections_not_empty()
{
    if (g_model_count == 0)
    {
        ui->confirmButton->setEnabled(false);
        ui->confirmButton->setToolTip("Please select at least one model");
    }
    else
    {
        ui->confirmButton->setEnabled(true);
        ui->confirmButton->setToolTip(g_currentConfirmTip);
    }
}

//For each one, check to see if at least one segmentation was ticked
void SelectImportingOptions::on_brainSeg_stateChanged(int arg1)
{
    g_info[1] = arg1;
}

void SelectImportingOptions::on_tumorSeg_stateChanged(int arg1)
{
    g_info[2] = arg1;
}

//For each one, check to see if at least one model was ticked
void SelectImportingOptions::on_axialModel_stateChanged(int arg1)
{
    g_info[3] = arg1;
    if (arg1 == 0)
    {
        g_model_count = g_model_count - 1;
    }
    else
    {
        g_model_count = g_model_count + 1;
    }
    check_selections_not_empty();
}

void SelectImportingOptions::on_corticalModel_stateChanged(int arg1)
{
    g_info[4] = arg1;
    if (arg1 == 0)
    {
        g_model_count = g_model_count - 1;
    }
    else
    {
        g_model_count = g_model_count + 1;
    }
    check_selections_not_empty();
}

void SelectImportingOptions::on_sagittalModel_stateChanged(int arg1)
{
    g_info[5] = arg1;
    if (arg1 == 0)
    {
        g_model_count = g_model_count - 1;
    }
    else
    {
        g_model_count = g_model_count + 1;
    }
    check_selections_not_empty();
}




void SelectImportingOptions::on_images_stateChanged(int arg1)
{
    g_info[6] = arg1;
}
