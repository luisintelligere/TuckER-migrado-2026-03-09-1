:: ================================================================
:: ENTRENAMIENTO MULTIPLE DE MODELOS TUCKER CON DISTINTAS DIMENSIONES
:: ================================================================

@echo off
set "SCRIPT_DIR=%~dp0"
set "REPO_ROOT=%SCRIPT_DIR%..\.."
pushd "%REPO_ROOT%"
ECHO ===============================================================
ECHO            INICIANDO ENTRENAMIENTOS MULTIDIMENSIONALES
ECHO ===============================================================

:: Configuración general
set DATASET=dataset_2019_2020_fundamentales
set EMB_INIT=notebooks/Experimento_warm_start/embeddings_inicializados_normalizado_2019_2020.pt
set VOCAB_INIT=notebooks/Experimento_warm_start/vocabulario_normalizado_2019_2020.json
set BATCH=128
set LR=0.003
set PATIENCE=300
set EPOCHS=1000 
:: Lista de dimensiones de relación a entrenar
set RDIMS=6 10  12  14  16

:: Bucle principal
for %%R in (%RDIMS%) do (
    ECHO.
    ECHO ===============================================================
    ECHO Entrenando modelo con dimension de relaciones = %%R
    ECHO ===============================================================

    python scripts\main\main_original_warm_start_gemini_2_earlystopping_gemini_softmax.py ^
        --dataset %DATASET% ^
        --output_prefix Experimento_warm_start_rdim%%R_1000epochs_earlystopping_2019_2020_patience%PATIENCE% ^
        --edim 4 ^
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