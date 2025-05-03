@echo off
REM Script de execução para sistema de detecção de ataques spoofing de replay
REM Autores: André Thiago de Souza, Felipe de Lima dos Santos, 
REM          Juliano Gaona Proença, Matheus Henrique Reich Favarin Zagonel

echo ===== Sistema de Detecção de Ataques Spoofing de Replay =====

REM Configuração de caminhos - MODIFIQUE CONFORME SUA ESTRUTURA DE DIRETÓRIOS
set ASV2019_PA_DIR=E:\ASV 2019 DATA\PA
set ASV2019_LA_DIR=E:\ASV 2019 DATA\LA
set ASV2021_DIR=E:\ASV 2021 DATA\PA
set OUTPUT_DIR=E:\Experimentos\ASVspoof

REM Verificar se Python está instalado
python --version >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo Erro: Python não encontrado. Instale o Python 3.8 ou superior.
    exit /b 1
)

REM Verificar e criar diretório de saída
if not exist "%OUTPUT_DIR%" mkdir "%OUTPUT_DIR%"

REM Menu de opções
:menu
echo.
echo 1. Converter protocolos para formato de labels
echo 2. Extrair características (LA)
echo 3. Extrair características (PA)
echo 4. Treinar modelo (LA)
echo 5. Treinar modelo (PA)
echo 6. Avaliar modelo (LA)
echo 7. Avaliar modelo (PA)
echo 8. Avaliar generalização (cross-dataset)
echo 9. Executar fluxo completo (LA)
echo 10. Executar fluxo completo (PA)
echo 11. Instalar dependências
echo 0. Sair
echo.

set /p opcao=Escolha uma opção: 

if "%opcao%"=="1" goto convert_protocols
if "%opcao%"=="2" goto extract_features_la
if "%opcao%"=="3" goto extract_features_pa
if "%opcao%"=="4" goto train_la
if "%opcao%"=="5" goto train_pa
if "%opcao%"=="6" goto evaluate_la
if "%opcao%"=="7" goto evaluate_pa
if "%opcao%"=="8" goto cross_dataset
if "%opcao%"=="9" goto full_pipeline_la
if "%opcao%"=="10" goto full_pipeline_pa
if "%opcao%"=="11" goto install_deps
if "%opcao%"=="0" goto end

echo Opção inválida. Tente novamente.
goto menu

:convert_protocols
echo === Convertendo protocolos para formato de labels ===
python convert_protocols.py --train-protocol "%ASV2019_PA_DIR%\ASVspoof2019_PA_cm_protocols\ASVspoof2019.PA.cm.train.trl.txt" --dev-protocol "%ASV2019_PA_DIR%\ASVspoof2019_PA_cm_protocols\ASVspoof2019.PA.cm.dev.trl.txt" --eval-protocol "%ASV2019_PA_DIR%\ASVspoof2019_PA_cm_protocols\ASVspoof2019.PA.cm.eval.trl.txt" --output-dir "%ASV2019_PA_DIR%\labels"
python convert_protocols.py --train-protocol "%ASV2019_LA_DIR%\ASVspoof2019_LA_cm_protocols\ASVspoof2019.LA.cm.train.trl.txt" --dev-protocol "%ASV2019_LA_DIR%\ASVspoof2019_LA_cm_protocols\ASVspoof2019.LA.cm.dev.trl.txt" --eval-protocol "%ASV2019_LA_DIR%\ASVspoof2019_LA_cm_protocols\ASVspoof2019.LA.cm.eval.trl.txt" --output-dir "%ASV2019_LA_DIR%\labels"
goto menu

:extract_features_la
echo === Extraindo características (LA) ===
python feature_extraction.py --train-audio-dir "%ASV2019_LA_DIR%\ASVspoof2019_LA_train\flac" --dev-audio-dir "%ASV2019_LA_DIR%\ASVspoof2019_LA_dev\flac" --eval-audio-dir "%ASV2019_LA_DIR%\ASVspoof2019_LA_eval\flac" --output-dir "%OUTPUT_DIR%\LA\features" --audio-ext ".flac"
goto menu

:extract_features_pa
echo === Extraindo características (PA) ===
python feature_extraction.py --train-audio-dir "%ASV2019_PA_DIR%\ASVspoof2019_PA_train\flac" --dev-audio-dir "%ASV2019_PA_DIR%\ASVspoof2019_PA_dev\flac" --eval-audio-dir "%ASV2019_PA_DIR%\ASVspoof2019_PA_eval\flac" --output-dir "%OUTPUT_DIR%\PA\features" --audio-ext ".flac"
goto menu

:train_la
echo === Treinando modelo (LA) ===
python train.py --train-features-dir "%OUTPUT_DIR%\LA\features\train" --dev-features-dir "%OUTPUT_DIR%\LA\features\dev" --train-labels-file "%ASV2019_LA_DIR%\labels\train_labels.txt" --dev-labels-file "%ASV2019_LA_DIR%\labels\dev_labels.txt" --save-dir "%OUTPUT_DIR%\LA\checkpoints" --num-epochs 50
goto menu

:train_pa
echo === Treinando modelo (PA) ===
python train.py --train-features-dir "%OUTPUT_DIR%\PA\features\train" --dev-features-dir "%OUTPUT_DIR%\PA\features\dev" --train-labels-file "%ASV2019_PA_DIR%\labels\train_labels.txt" --dev-labels-file "%ASV2019_PA_DIR%\labels\dev_labels.txt" --save-dir "%OUTPUT_DIR%\PA\checkpoints" --num-epochs 50
goto menu

:evaluate_la
echo === Avaliando modelo (LA) ===
python test.py --test-features-dir "%OUTPUT_DIR%\LA\features\eval" --test-labels-file "%ASV2019_LA_DIR%\labels\eval_labels.txt" --checkpoint "%OUTPUT_DIR%\LA\checkpoints\best_model.pth" --results-dir "%OUTPUT_DIR%\LA\results" --analyze-generalization
goto menu

:evaluate_pa
echo === Avaliando modelo (PA) ===
python test.py --test-features-dir "%OUTPUT_DIR%\PA\features\eval" --test-labels-file "%ASV2019_PA_DIR%\labels\eval_labels.txt" --checkpoint "%OUTPUT_DIR%\PA\checkpoints\best_model.pth" --results-dir "%OUTPUT_DIR%\PA\results" --analyze-generalization
goto menu

:cross_dataset
echo === Avaliando generalização (cross-dataset) ===
python test.py --test-features-dir "%OUTPUT_DIR%\PA\features\eval" --test-labels-file "%ASV2019_PA_DIR%\labels\eval_labels.txt" --checkpoint "%OUTPUT_DIR%\LA\checkpoints\best_model.pth" --results-dir "%OUTPUT_DIR%\cross_dataset\LA_to_PA" --analyze-generalization
goto menu

:full_pipeline_la
echo === Executando fluxo completo (LA) ===
call :convert_protocols
call :extract_features_la
call :train_la
call :evaluate_la
echo === Fluxo completo (LA) concluído ===
goto menu

:full_pipeline_pa
echo === Executando fluxo completo (PA) ===
call :convert_protocols
call :extract_features_pa
call :train_pa
call :evaluate_pa
echo === Fluxo completo (PA) concluído ===
goto menu

:install_deps
echo === Instalando dependências ===
pip install torch torchaudio numpy scipy scikit-learn scikit-image matplotlib pandas tqdm pydub soundfile librosa
goto menu

:end
echo Saindo...
exit /b 0