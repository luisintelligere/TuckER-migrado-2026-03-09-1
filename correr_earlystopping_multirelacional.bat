:: Este script activa el entorno de Conda y ejecuta el entrenamiento con "warm start".
@echo off

ECHO =================================================================
ECHO               INICIANDO ENTRENAMIENTO CON WARM START
ECHO =================================================================


:: 2. Ejecutar el entrenamiento con los embeddings inicializados
ECHO.
ECHO ==> Ejecutando el script de entrenamiento...
python main_original_warm_start_gemini_2_earlystopping_gemini.py ^
    --dataset dataset_20122_multinivel_filtrado_estado ^
    --output_prefix Experimento_warm_start_1000epochs_earlystopping_rdim10_multirelacional ^
    --edim 4 ^
    --rdim 10 ^
    --num_iterations 1000 ^
    --batch_size 128 ^
    --lr 0.003 ^
    --init_embeddings notebooks/Experimento_warm_start/embeddings_inicializados.pt ^
    --init_vocab notebooks/Experimento_warm_start/vocabulario.json ^
    --patience 250

ECHO.
ECHO =============================================================
ECHO                ENTRENAMIENTO FINALIZADO.
ECHO =============================================================

:: Mantiene la ventana abierta para que veas el mensaje final.
pause