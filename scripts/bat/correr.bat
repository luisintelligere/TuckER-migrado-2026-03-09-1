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
python scripts\main\main_original_warm_start_gemini_2.py ^
    --dataset dataset_20122_fundamentales ^
    --output_prefix Experimento_warm_start_1000epochs ^
    --edim 4 ^
    --rdim 2 ^
    --num_iterations 1000 ^
    --batch_size 128 ^
    --lr 0.003 ^
    --init_embeddings notebooks/Experimento_warm_start/embeddings_inicializados.pt ^
    --init_vocab notebooks/Experimento_warm_start/vocabulario.json

ECHO.
ECHO =============================================================
ECHO               ENTRENAMIENTO FINALIZADO.
ECHO =============================================================

:: Mantiene la ventana abierta para que veas el mensaje final.
popd
pause