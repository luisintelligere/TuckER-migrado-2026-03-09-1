:: Este script activa el entorno de Conda y ejecuta el entrenamiento con "warm start".
@echo off
set "SCRIPT_DIR=%~dp0"
set "REPO_ROOT=%SCRIPT_DIR%..\.."
pushd "%REPO_ROOT%"

ECHO =================================================================
ECHO               INICIANDO ENTRENAMIENTO CON WARM START
ECHO =================================================================


:: 2. Ejecutar el entrenamiento con los embeddings inicializados
ECHO.
ECHO ==> Ejecutando el script de entrenamiento...
python scripts\main\main_original_warm_start_gemini_2_earlystopping_gemini.py ^
    --dataset dataset_2019_2020_multinivel_filtrado_estado ^
    --output_prefix Experimento_warm_start_1000epochs_earlystopping_rdim4_multirelaciona_20192020_patience500 ^
    --edim 4 ^
    --rdim 4 ^
    --num_iterations 1000 ^
    --batch_size 128 ^
    --lr 0.003 ^
    --init_embeddings notebooks/Experimento_warm_start_multirelacional/embeddings_inicializados/2019-2020/embeddings_inicializados.pt ^
    --init_vocab notebooks/Experimento_warm_start_multirelacional/embeddings_inicializados/2019-2020/vocabulario.json ^
    --patience 500

ECHO.
ECHO =============================================================
ECHO                ENTRENAMIENTO FINALIZADO.
ECHO =============================================================

:: Mantiene la ventana abierta para que veas el mensaje final.
popd
pause