:: ================================================================
:: ENTRENAMIENTO TUCKER 5D (S1 -> S2)
:: ================================================================

@echo off
ECHO ===============================================================
ECHO            INICIANDO ENTRENAMIENTO 2019-2 (5D)
ECHO ===============================================================

:: 1. Dataset del Segundo Semestre
set DATASET=dataset_20192_fundamentales

:: 2. Archivos generados por el script de inicialización
:: (Están en la carpeta raíz de Experimento_warm_start según tu última indicación)
set EMB_INIT=notebooks/Experimento_warm_start/embeddings_5d_final/embeddings_inicializados_normalizados_5d.pt
set VOCAB_INIT=notebooks/Experimento_warm_start/embeddings_5d_final/vocabulario_5d.json

set BATCH=128
set LR=0.003
set PATIENCE=400
set EPOCHS=1000 

:: Lista de dimensiones
set RDIMS=1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16

:: Bucle principal
for %%R in (%RDIMS%) do (
    ECHO.
    ECHO ===============================================================
    ECHO Entrenando modelo rdim=%%R (5D)
    ECHO ===============================================================

    python main_original_warm_start_gemini_2_earlystopping_gemini.py ^
        --dataset %DATASET% ^
        --output_prefix Experimento_warm_start_rdim%%R_1000epochs_earlystopping_2019_2_patience%PATIENCE%_balanceado_5d ^
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

pause