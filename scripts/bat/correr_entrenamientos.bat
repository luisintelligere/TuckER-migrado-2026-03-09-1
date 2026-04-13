:: Este es un script de lotes para entrenar el modelo TuckER en dos datasets.
@echo off
set "SCRIPT_DIR=%~dp0"
set "REPO_ROOT=%SCRIPT_DIR%..\.."
pushd "%REPO_ROOT%"
ECHO =============================================================
ECHO                INICIANDO ENTRENAMIENTO
ECHO =============================================================

:: Activa el entorno de Conda.
ECHO Activando el entorno 'tucker_env'...
call conda activate tucker_env

:: --- Entrenamiento para el semestre 20121 ---
ECHO.
ECHO Entrenando con los datos de 20121...
python scripts\main\main.py --dataset 20121_multirelacional --output_prefix 20121_wn18 --num_iterations 500 --batch_size 128 --lr 0.005 --dr 0.995 --edim 200 --rdim 30 --input_dropout 0.2 --hidden_dropout1 0.1 --hidden_dropout2 0.2 --label_smoothing 0.1

:: --- Entrenamiento para el semestre 20122 ---
ECHO.
ECHO Entrenando con los datos de 20122...
python scripts\main\main.py --dataset 20122_multirelacional --output_prefix 20122_wn18 --num_iterations 500 --batch_size 128 --lr 0.005 --dr 0.995 --edim 200 --rdim 30 --input_dropout 0.2 --hidden_dropout1 0.1 --hidden_dropout2 0.2 --label_smoothing 0.1

ECHO.
ECHO =============================================================
ECHO                  TODOS LOS ENTRENAMIENTOS HAN FINALIZADO
ECHO =============================================================

:: Desactiva el entorno al final.
call conda deactivate

:: Pausa la ventana para que puedas ver el mensaje final.
popd
pause