:: ================================================================
:: ENTRENAMIENTO TUCKER S1->S2 (rdim 1-12, 1/2/3 cohortes)
:: ================================================================

@echo off
set "SCRIPT_DIR=%~dp0"
pushd "%SCRIPT_DIR%"

set BATCH=128
set LR=0.0005
set EPOCHS=500
set EDIM=200
set DR=1.0
set INPUT_DO=0.3
set HIDDEN_DO1=0.4
set HIDDEN_DO2=0.5
set LS=0.1

:: Datasets
set DATASETS=dataset_2021_hacia_s2_binario dataset_2020_2021_hacia_s2_binario dataset_2019_2020_2021_hacia_s2_binario

for %%D in (%DATASETS%) do (
    for /L %%R in (1,1,12) do (
        ECHO.
        ECHO ===============================================================
        ECHO Dataset: %%D  -  rdim=%%R  edim=%EDIM%
        ECHO ===============================================================

        python scripts\main\main.py ^
            --dataset %%D ^
            --output_prefix s2_%%D_rdim%%R ^
            --edim %EDIM% ^
            --rdim %%R ^
            --num_iterations %EPOCHS% ^
            --batch_size %BATCH% ^
            --lr %LR% ^
            --dr %DR% ^
            --input_dropout %INPUT_DO% ^
            --hidden_dropout1 %HIDDEN_DO1% ^
            --hidden_dropout2 %HIDDEN_DO2% ^
            --label_smoothing %LS%

        ECHO ---------------------------------------------------------------
        ECHO %%D rdim=%%R completado.
        ECHO ---------------------------------------------------------------
    )
)

ECHO ===============================================================
ECHO TODOS LOS ENTRENAMIENTOS S2 FINALIZADOS.
ECHO ===============================================================

popd
pause
