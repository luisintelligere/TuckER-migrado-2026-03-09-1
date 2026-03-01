:: Este script ejecuta el entrenamiento con "warm start" para varios RDIM.
@echo off

ECHO =================================================================
ECHO               INICIANDO ENTRENAMIENTOS WARM START
ECHO =================================================================
ECHO.

:: ---- Lista de RDIM a entrenar ----
SET RDIMS=30

:: ---- Parámetros fijos ----
SET DATASET=dataset_20122_fundamentales
SET EDIM=4
SET NUM_EPOCHS=1000
SET BATCH=128
SET LR=0.003
SET PATIENCE=250
SET INIT_EMB=notebooks/Experimento_warm_start/embeddings_inicializados_normalizado.pt
SET INIT_VOCAB=notebooks/Experimento_warm_start/vocabulario_normalizado.json

FOR %%R IN (%RDIMS%) DO (
    ECHO -------------------------------------------------------------
    ECHO  🔹 Entrenando con RDIM=%%R
    ECHO -------------------------------------------------------------

    python main_original_warm_start_gemini_2_earlystopping_gemini.py ^
        --dataset %DATASET% ^
        --output_prefix Experimento_warm_start_1000epochs_earlystopping_rdim%%R_normalizado ^
        --edim %EDIM% ^
        --rdim %%R ^
        --num_iterations %NUM_EPOCHS% ^
        --batch_size %BATCH% ^
        --lr %LR% ^
        --init_embeddings %INIT_EMB% ^
        --init_vocab %INIT_VOCAB% ^
        --patience %PATIENCE%

    ECHO.
    ECHO ✅ Finalizado RDIM=%%R
    ECHO.
)

ECHO =================================================================
ECHO                   TODOS LOS ENTRENAMIENTOS LISTOS
ECHO =================================================================
pause
