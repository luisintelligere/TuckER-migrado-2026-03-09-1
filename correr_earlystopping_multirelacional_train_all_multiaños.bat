@echo off
REM ===========================================================
REM     ENTRENAMIENTO MULTIRELACIONAL (rdim = 1 al 12)
REM ===========================================================

ECHO ===========================================================
ECHO       INICIANDO ENTRENAMIENTO TuckER MULTIRELACIONAL
ECHO ===========================================================
ECHO.

SETLOCAL ENABLEDELAYEDEXPANSION

REM --- Parámetros base ---
SET DATASET=20121_train_all_multirelacional
SET EDIM=75
SET EPOCHS=600
SET BATCH=128
SET LR=0.003
SET PATIENCE=500

REM --- Bucle sobre rdim = 1 al 12 ---
FOR %%R IN (1 2 3 4 5 6 7 8 9 10 12 13 14 15 16) DO (
    ECHO.
    ECHO 🔹 ENTRENANDO MODELO CON RDIM=%%R ...
    ECHO -------------------------------------

    python main_guardado.py ^
        --dataset %DATASET% ^
        --output_prefix Experimento_rdim%%R_20121_multirelacional ^
        --edim %EDIM% ^
        --rdim %%R ^
        --num_iterations %EPOCHS% ^
        --batch_size %BATCH% ^
        --lr %LR% ^
        --patience %PATIENCE% ^
        --cuda True ^
        --input_dropout 0.2 ^
        --hidden_dropout1 0.1 ^
        --hidden_dropout2 0.2 ^
        --label_smoothing 0.1

    ECHO.
    ECHO ✅ Modelo con RDIM=%%R completado.
    ECHO -------------------------------------
)

ECHO ===========================================================
ECHO    ENTRENAMIENTO MULTIRELACIONAL FINALIZADO COMPLETAMENTE
ECHO ===========================================================
pause
