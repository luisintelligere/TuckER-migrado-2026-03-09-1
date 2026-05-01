:: ================================================================
:: ENTRENAMIENTO TUCKER S1->S2 WARM START 4D (1 cohorte, rdim 1-12)
:: ================================================================

@echo off
set "SCRIPT_DIR=%~dp0"
pushd "%SCRIPT_DIR%"

set DATASET=dataset_2021_hacia_s2_binario
set EMB_INIT=data/%DATASET%/embeddings_4d/embeddings_inicializados_4d.pt
set VOCAB_INIT=data/%DATASET%/embeddings_4d/vocabulario_4d.json

set BATCH=128
set LR=0.003
set PATIENCE=400
set EPOCHS=1000
set EDIM=4
set DR=1.0
set INPUT_DO=0.3
set HIDDEN_DO1=0.4
set HIDDEN_DO2=0.5
set LS=0.1

for /L %%R in (1,1,12) do (
    ECHO.
    ECHO ===============================================================
    ECHO Dataset: %DATASET%  -  rdim=%%R  edim=%EDIM% (warm start 4D)
    ECHO ===============================================================

    python scripts\main\main_original_warm_start_gemini_2_earlystopping_gemini.py ^
        --dataset %DATASET% ^
        --output_prefix s2_ws4d_%DATASET%_rdim%%R ^
        --edim %EDIM% ^
        --rdim %%R ^
        --num_iterations %EPOCHS% ^
        --batch_size %BATCH% ^
        --lr %LR% ^
        --dr %DR% ^
        --input_dropout %INPUT_DO% ^
        --hidden_dropout1 %HIDDEN_DO1% ^
        --hidden_dropout2 %HIDDEN_DO2% ^
        --label_smoothing %LS% ^
        --init_embeddings %EMB_INIT% ^
        --init_vocab %VOCAB_INIT% ^
        --patience %PATIENCE%

    ECHO ---------------------------------------------------------------
    ECHO rdim=%%R completado.
    ECHO ---------------------------------------------------------------
)

ECHO ===============================================================
ECHO TODOS LOS ENTRENAMIENTOS S2 WARM START FINALIZADOS.
ECHO ===============================================================

popd
pause
