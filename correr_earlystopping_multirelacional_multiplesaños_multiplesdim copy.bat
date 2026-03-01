:: ================================================================
:: ENTRENAMIENTO MULTIPLE DE MODELOS TUCKER (5D - NOTAS + PUNTAJE)
:: ================================================================

@echo off
ECHO ===============================================================
ECHO            INICIANDO ENTRENAMIENTOS MULTIDIMENSIONALES (5D)
ECHO ===============================================================

:: Configuración general
set DATASET=dataset_2019_2020_2021_binario_4d
:: ⚠️ RUTAS ACTUALIZADAS A TUS NUEVOS ARCHIVOS 5D
set EMB_INIT=notebooks/Experimento_warm_start/embeddings_5d_3cohortes/embeddings_inicializados_normalizados_5d_3cohortes.pt
set VOCAB_INIT=notebooks/Experimento_warm_start/embeddings_5d_3cohortes/vocabulario_5d_3cohortes.json

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

    python main_original_warm_start_gemini_2_earlystopping_gemini.py ^
        --dataset %DATASET% ^
        --output_prefix Experimento_warm_start_rdim%%R_1000epochs_earlystopping_2019_2020_binario_patience%PATIENCE%_5d_balanceado ^
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