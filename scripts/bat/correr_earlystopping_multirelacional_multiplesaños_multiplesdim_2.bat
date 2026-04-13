:: ================================================================
:: ENTRENAMIENTO MULTIPLE DE MODELOS TUCKER (5D - NOTAS + PUNTAJE)
:: ================================================================

@echo off
set "SCRIPT_DIR=%~dp0"
set "REPO_ROOT=%SCRIPT_DIR%..\.."
pushd "%REPO_ROOT%"
ECHO ===============================================================
ECHO            INICIANDO ENTRENAMIENTOS MULTIDIMENSIONALES (5D)
ECHO ===============================================================

:: Configuración general
set DATASET=dataset_2019_2020_2021_multirelacional_4d
:: ⚠️ RUTAS ACTUALIZADAS A TUS NUEVOS ARCHIVOS 5D
set EMB_INIT=notebooks/Experimento_warm_start/embeddings_5d_multirelacional/embeddings_inicializados_multi_5d.pt
set VOCAB_INIT=notebooks/Experimento_warm_start/embeddings_5d_multirelacional/vocabulario_multi_5d.json

set BATCH=128
set LR=0.003
set PATIENCE=400
set EPOCHS=1000 

:: Lista de dimensiones de relación a entrenar
set RDIMS=1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16

:: Bucle principal
for %%R in (%RDIMS%) do (
    ECHO.
    ECHO ===============================================================
    ECHO Entrenando modelo con dimension de relaciones = %%R
    ECHO ===============================================================

    python scripts\main\main_original_warm_start_gemini_2_earlystopping_gemini.py ^
        --dataset %DATASET% ^
        --output_prefix Experimento_warm_start_rdim%%R_1000epochs_earlystopping_2019_2020_2021_multirelacional_patience%PATIENCE%_5d_2 ^
        --edim 5 ^
        --rdim %%R ^
        --num_iterations %EPOCHS% ^
        --batch_size %BATCH% ^
        --lr %LR% ^
        --init_embeddings %EMB_INIT% ^
        --init_vocab %VOCAB_INIT% ^
        --patience %PATIENCE%
    ECHO ---------------------------------------------------------------
    ECHO Modelo con rdim=%%R completado.
    ECHO ---------------------------------------------------------------
)

ECHO ===============================================================
ECHO TODOS LOS ENTRENAMIENTOS FINALIZADOS.
ECHO ===============================================================

popd
pause