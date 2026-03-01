@echo off
REM ===========================================================
REM   ENTRENAMIENTO MULTIRELACIONAL (edim = varios, rdim = varios)
REM ===========================================================

ECHO ===========================================================
ECHO       INICIANDO ENTRENAMIENTO TuckER MULTIRELACIONAL
ECHO ===========================================================
ECHO.

SETLOCAL ENABLEDELAYEDEXPANSION

REM --- Parámetros base ---
SET DATASET=20121_train_all_multirelacional
SET EPOCHS=600
SET BATCH=128
SET LR=0.003
SET PATIENCE=500

REM --- Valores de EDIM a recorrer ---
SET EDIMS=4 10 20 30 40 50 60

REM --- Valores de RDIM a recorrer ---
SET RDIMS=1 2 3 4 5 6 7 8 9 10 12 13 14 15 16

REM --- Bucle anidado: por cada EDIM y RDIM ---
FOR %%E IN (%EDIMS%) DO (
    ECHO.
    ECHO ===========================================================
    ECHO       🔸 ENTRENANDO MODELOS PARA EDIM=%%E
    ECHO ===========================================================
    ECHO.

    REM Crear carpeta de salida por edim si no existe
    IF NOT EXIST "results\edim%%E" (
        mkdir "results\edim%%E"
    )

    FOR %%R IN (%RDIMS%) DO (
        ECHO.
        ECHO 🔹 ENTRENANDO MODELO CON EDIM=%%E Y RDIM=%%R ...
        ECHO ----------------------------------------------------

        python main_guardado.py ^
            --dataset %DATASET% ^
            --output_prefix results/edim%%E/Experimento_rdim%%R_20121_multirelacional ^
            --edim %%E ^
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
        ECHO ✅ Modelo con EDIM=%%E y RDIM=%%R completado.
        ECHO ----------------------------------------------------
    )

    ECHO ===========================================================
    ECHO     🔸 FINALIZADOS TODOS LOS RDIM PARA EDIM=%%E
    ECHO ===========================================================
    ECHO.
)

ECHO ===========================================================
ECHO     ENTRENAMIENTO MULTIRELACIONAL COMPLETADO PARA TODOS
ECHO ===========================================================
pause
