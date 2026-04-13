@echo off
set "SCRIPT_DIR=%~dp0"
set "REPO_ROOT=%SCRIPT_DIR%..\.."
pushd "%REPO_ROOT%"
setlocal EnableDelayedExpansion

REM === RUTAS / CONFIG ===
set PYTHON=python
set SCRIPT=scripts\main\main_original_warm_start_gemini_2_earlystopping_gemini.py

REM OJO: pon aquí el nombre EXACTO del dataset (carpeta dentro de data/)
REM Ejemplos posibles:
REM   dataset_20192020_multirrelacional
REM   dataset_20122_fundamentales_multirrelacional
set DATASET=dataset_2019_2020_multinivel_filtrado_estado

REM Hiperparámetros
set EPOCHS=1000
set PATIENCE=500
set LR=0.003
set BATCH=128

REM === BUCLES: edim en {4,10} y rdim=1..14 ===
for %%E in (4 8 10 12) do (
  for /L %%R in (1,1,14) do (
    set "RUN=Experimento_no_warm_start_rdim%%R_edim%%E_%EPOCHS%epochs_earlystopping_20192020_patience%PATIENCE%"
    echo.
    echo ==============================
    echo   Ejecutando: edim=%%E rdim=%%R
    echo   Output: !RUN!
    echo ==============================

    %PYTHON% %SCRIPT% ^
      --dataset "%DATASET%" ^
      --num_iterations %EPOCHS% ^
      --batch_size %BATCH% ^
      --lr %LR% ^
      --edim %%E ^
      --rdim %%R ^
      --patience %PATIENCE% ^
      --output_prefix "!RUN!"
      REM Importante: SIN --init_embeddings ni --init_vocab  -> NO warm-start
  )
)

echo.
echo Listo.
popd
endlocal
