#ifndef SELECTIMPORTINGOPTIONS_H
#define SELECTIMPORTINGOPTIONS_H

#include <QDialog>

namespace Ui {
class SelectImportingOptions;
}

class SelectImportingOptions : public QDialog
{
    Q_OBJECT

public:
    explicit SelectImportingOptions(QWidget *parent = nullptr);
    void check_selections_not_empty();
    ~SelectImportingOptions();

private slots:
    void closeAll();

    void mainReturn();

    void on_confirmButton_clicked();

    void on_cancelButton_clicked();

    void on_sagittalModel_stateChanged(int arg1);

    void on_corticalModel_stateChanged(int arg1);

    void on_axialModel_stateChanged(int arg1);

    void on_tumorSeg_stateChanged(int arg1);

    void on_brainSeg_stateChanged(int arg1);

    void on_scanType_currentIndexChanged(int index);

    void on_images_stateChanged(int arg1);

private:
    Ui::SelectImportingOptions *ui;
};

#endif // SELECTIMPORTINGOPTIONS_H
