@echo off
REM Script de execução aprimorado para sistema de detecção de ataques spoofing de replay
REM Com suporte a segmentação de dados e treinamento otimizado
REM Autores: André Thiago de Souza, Felipe de Lima dos Santos, 
REM          Juliano Gaona Proença, Matheus Henrique Reich Favarin Zagonel

echo ===== Sistema de Detecção de Ataques Spoofing de Replay =====
echo ===== Versão Otimizada com Segmentação de Dados =====

REM Configuração de caminhos - MODIFIQUE CONFORME SUA ESTRUTURA DE DIRETÓRIOS
set ASV2019_PA_DIR=E:\ASV 2019 DATA\PA
set ASV2019_LA_DIR=E:\ASV 2019 DATA\LA
set ASV2021_DIR=E:\ASV 2021 DATA\PA
set OUTPUT_DIR=E:\Experimentos\ASVspoof

REM Configurações de segmentação
set DEFAULT_STRATEGY=adaptive
set DEFAULT_RATIO=0.3
set BENCHMARK_DIR=%OUTPUT_DIR%\benchmark
set SEGMENTATION_CACHE=%OUTPUT_DIR%\segmentation_cache

REM Verificar se Python está instalado
python --version >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo Erro: Python não encontrado. Instale o Python 3.8 ou superior.
    exit /b 1
)

REM Verificar e criar diretórios necessários
if not exist "%OUTPUT_DIR%" mkdir "%OUTPUT_DIR%"
if not exist "%BENCHMARK_DIR%" mkdir "%BENCHMARK_DIR%"
if not exist "%SEGMENTATION_CACHE%" mkdir "%SEGMENTATION_CACHE%"

REM Menu principal
:menu
echo.
echo ============= MENU PRINCIPAL =============
echo === OPERAÇÕES BÁSICAS ===
echo 1.  Converter protocolos para formato de labels
echo 2.  Extrair características (LA)
echo 3.  Extrair características (PA)
echo 4.  Instalar dependências
echo.
echo === ANÁLISE DE SEGMENTAÇÃO ===
echo 5.  Benchmark de estratégias de segmentação
echo 6.  Análise rápida de segmentação (LA)
echo 7.  Análise rápida de segmentação (PA)
echo 8.  Comparar estratégias de segmentação
echo.
echo === TREINAMENTO OTIMIZADO ===
echo 9.  Treinar modelo otimizado (LA) - Segmentação
echo 10. Treinar modelo otimizado (PA) - Segmentação
echo 11. Treinar modelo progressivo (LA)
echo 12. Treinar modelo progressivo (PA)
echo 13. Treinar modelo inteligente (LA)
echo 14. Treinar modelo inteligente (PA)
echo.
echo === TREINAMENTO PADRÃO ===
echo 15. Treinar modelo padrão (LA)
echo 16. Treinar modelo padrão (PA)
echo.
echo === AVALIAÇÃO ===
echo 17. Avaliar modelo (LA)
echo 18. Avaliar modelo (PA)
echo 19. Avaliar generalização (cross-dataset)
echo.
echo === FLUXOS COMPLETOS ===
echo 20. Fluxo completo otimizado (LA)
echo 21. Fluxo completo otimizado (PA)
echo 22. Fluxo completo padrão (LA)
echo 23. Fluxo completo padrão (PA)
echo.
echo === CONFIGURAÇÕES ===
echo 24. Configurar parâmetros de segmentação
echo 25. Visualizar configurações atuais
echo.
echo 0.  Sair
echo ========================================
echo.

set /p opcao=Escolha uma opção: 

if "%opcao%"=="1" goto convert_protocols
if "%opcao%"=="2" goto extract_features_la
if "%opcao%"=="3" goto extract_features_pa
if "%opcao%"=="4" goto install_deps
if "%opcao%"=="5" goto benchmark_segmentation
if "%opcao%"=="6" goto quick_analysis_la
if "%opcao%"=="7" goto quick_analysis_pa
if "%opcao%"=="8" goto compare_strategies
if "%opcao%"=="9" goto train_optimized_la
if "%opcao%"=="10" goto train_optimized_pa
if "%opcao%"=="11" goto train_progressive_la
if "%opcao%"=="12" goto train_progressive_pa
if "%opcao%"=="13" goto train_intelligent_la
if "%opcao%"=="14" goto train_intelligent_pa
if "%opcao%"=="15" goto train_standard_la
if "%opcao%"=="16" goto train_standard_pa
if "%opcao%"=="17" goto evaluate_la
if "%opcao%"=="18" goto evaluate_pa
if "%opcao%"=="19" goto cross_dataset
if "%opcao%"=="20" goto full_pipeline_optimized_la
if "%opcao%"=="21" goto full_pipeline_optimized_pa
if "%opcao%"=="22" goto full_pipeline_standard_la
if "%opcao%"=="23" goto full_pipeline_standard_pa
if "%opcao%"=="24" goto configure_segmentation
if "%opcao%"=="25" goto show_config
if "%opcao%"=="0" goto end

echo Opção inválida. Tente novamente.
goto menu

REM ===============================
REM OPERAÇÕES BÁSICAS
REM ===============================

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

:install_deps
echo === Instalando dependências ===
echo Instalando dependências básicas...
pip install torch torchaudio numpy scipy scikit-learn scikit-image matplotlib pandas tqdm pydub soundfile librosa
echo.
echo Instalando dependências para segmentação...
pip install umap-learn seaborn joblib
echo.
echo Dependências instaladas com sucesso!
pause
goto menu

REM ===============================
REM ANÁLISE DE SEGMENTAÇÃO
REM ===============================

:benchmark_segmentation
echo === Benchmark de Estratégias de Segmentação ===
echo.
echo Escolha o dataset para benchmark:
echo 1. LA (Logical Access)
echo 2. PA (Physical Access)
echo.
set /p dataset_choice=Escolha o dataset (1 ou 2): 

if "%dataset_choice%"=="1" (
    set FEATURES_DIR=%OUTPUT_DIR%\LA\features\train
    set LABELS_FILE=%ASV2019_LA_DIR%\labels\train_labels.txt
    set BENCHMARK_OUTPUT=%BENCHMARK_DIR%\LA
) else if "%dataset_choice%"=="2" (
    set FEATURES_DIR=%OUTPUT_DIR%\PA\features\train
    set LABELS_FILE=%ASV2019_PA_DIR%\labels\train_labels.txt
    set BENCHMARK_OUTPUT=%BENCHMARK_DIR%\PA
) else (
    echo Opção inválida.
    goto menu
)

echo Executando benchmark completo...
python segmentation_integration.py --features-dir "%FEATURES_DIR%" --labels-file "%LABELS_FILE%" --output-dir "%BENCHMARK_OUTPUT%" --mode benchmark
echo.
echo Benchmark concluído! Resultados salvos em: %BENCHMARK_OUTPUT%
pause
goto menu

:quick_analysis_la
echo === Análise Rápida de Segmentação (LA) ===
python segmentation_integration.py --features-dir "%OUTPUT_DIR%\LA\features\train" --labels-file "%ASV2019_LA_DIR%\labels\train_labels.txt" --output-dir "%BENCHMARK_DIR%\LA_quick" --mode benchmark --quick
pause
goto menu

:quick_analysis_pa
echo === Análise Rápida de Segmentação (PA) ===
python segmentation_integration.py --features-dir "%OUTPUT_DIR%\PA\features\train" --labels-file "%ASV2019_PA_DIR%\labels\train_labels.txt" --output-dir "%BENCHMARK_DIR%\PA_quick" --mode benchmark --quick
pause
goto menu

:compare_strategies
echo === Comparar Estratégias de Segmentação ===
echo.
echo Escolha o dataset:
echo 1. LA (Logical Access)
echo 2. PA (Physical Access)
echo.
set /p dataset_choice=Escolha o dataset (1 ou 2): 

if "%dataset_choice%"=="1" (
    set FEATURES_DIR=%OUTPUT_DIR%\LA\features\train
    set LABELS_FILE=%ASV2019_LA_DIR%\labels\train_labels.txt
    set COMPARE_OUTPUT=%BENCHMARK_DIR%\LA_comparison
) else if "%dataset_choice%"=="2" (
    set FEATURES_DIR=%OUTPUT_DIR%\PA\features\train
    set LABELS_FILE=%ASV2019_PA_DIR%\labels\train_labels.txt
    set COMPARE_OUTPUT=%BENCHMARK_DIR%\PA_comparison
) else (
    echo Opção inválida.
    goto menu
)

python segmentation_integration.py --features-dir "%FEATURES_DIR%" --labels-file "%LABELS_FILE%" --output-dir "%COMPARE_OUTPUT%" --mode all
pause
goto menu

REM ===============================
REM TREINAMENTO OTIMIZADO
REM ===============================

:train_optimized_la
echo === Treinando modelo otimizado (LA) com segmentação ===
echo Estratégia: %DEFAULT_STRATEGY%
echo Ratio: %DEFAULT_RATIO%
python enhanced_train_with_segmentation.py --train-features-dir "%OUTPUT_DIR%\LA\features\train" --dev-features-dir "%OUTPUT_DIR%\LA\features\dev" --train-labels-file "%ASV2019_LA_DIR%\labels\train_labels.txt" --dev-labels-file "%ASV2019_LA_DIR%\labels\dev_labels.txt" --save-dir "%OUTPUT_DIR%\LA\checkpoints_optimized" --optimization-mode segmentation --segmentation-strategy %DEFAULT_STRATEGY% --sample-ratio %DEFAULT_RATIO% --num-epochs 50
pause
goto menu

:train_optimized_pa
echo === Treinando modelo otimizado (PA) com segmentação ===
echo Estratégia: %DEFAULT_STRATEGY%
echo Ratio: %DEFAULT_RATIO%
python enhanced_train_with_segmentation.py --train-features-dir "%OUTPUT_DIR%\PA\features\train" --dev-features-dir "%OUTPUT_DIR%\PA\features\dev" --train-labels-file "%ASV2019_PA_DIR%\labels\train_labels.txt" --dev-labels-file "%ASV2019_PA_DIR%\labels\dev_labels.txt" --save-dir "%OUTPUT_DIR%\PA\checkpoints_optimized" --optimization-mode segmentation --segmentation-strategy %DEFAULT_STRATEGY% --sample-ratio %DEFAULT_RATIO% --num-epochs 50
pause
goto menu

:train_progressive_la
echo === Treinamento Progressivo (LA) ===
echo.
set /p stages=Número de estágios (padrão 5): 
if "%stages%"=="" set stages=5

python enhanced_train_with_segmentation.py --train-features-dir "%OUTPUT_DIR%\LA\features\train" --dev-features-dir "%OUTPUT_DIR%\LA\features\dev" --train-labels-file "%ASV2019_LA_DIR%\labels\train_labels.txt" --dev-labels-file "%ASV2019_LA_DIR%\labels\dev_labels.txt" --save-dir "%OUTPUT_DIR%\LA\checkpoints_progressive" --optimization-mode progressive --progressive-stages %stages% --num-epochs 50
pause
goto menu

:train_progressive_pa
echo === Treinamento Progressivo (PA) ===
echo.
set /p stages=Número de estágios (padrão 5): 
if "%stages%"=="" set stages=5

python enhanced_train_with_segmentation.py --train-features-dir "%OUTPUT_DIR%\PA\features\train" --dev-features-dir "%OUTPUT_DIR%\PA\features\dev" --train-labels-file "%ASV2019_PA_DIR%\labels\train_labels.txt" --dev-labels-file "%ASV2019_PA_DIR%\labels\dev_labels.txt" --save-dir "%OUTPUT_DIR%\PA\checkpoints_progressive" --optimization-mode progressive --progressive-stages %stages% --num-epochs 50
pause
goto menu

:train_intelligent_la
echo === Treinamento Inteligente (LA) ===
echo.
set /p update_freq=Frequência de atualização (padrão 10): 
if "%update_freq%"=="" set update_freq=10

python enhanced_train_with_segmentation.py --train-features-dir "%OUTPUT_DIR%\LA\features\train" --dev-features-dir "%OUTPUT_DIR%\LA\features\dev" --train-labels-file "%ASV2019_LA_DIR%\labels\train_labels.txt" --dev-labels-file "%ASV2019_LA_DIR%\labels\dev_labels.txt" --save-dir "%OUTPUT_DIR%\LA\checkpoints_intelligent" --optimization-mode intelligent --intelligent-update-freq %update_freq% --sample-ratio %DEFAULT_RATIO% --num-epochs 50
pause
goto menu

:train_intelligent_pa
echo === Treinamento Inteligente (PA) ===
echo.
set /p update_freq=Frequência de atualização (padrão 10): 
if "%update_freq%"=="" set update_freq=10

python enhanced_train_with_segmentation.py --train-features-dir "%OUTPUT_DIR%\PA\features\train" --dev-features-dir "%OUTPUT_DIR%\PA\features\dev" --train-labels-file "%ASV2019_PA_DIR%\labels\train_labels.txt" --dev-labels-file "%ASV2019_PA_DIR%\labels\dev_labels.txt" --save-dir "%OUTPUT_DIR%\PA\checkpoints_intelligent" --optimization-mode intelligent --intelligent-update-freq %update_freq% --sample-ratio %DEFAULT_RATIO% --num-epochs 50
pause
goto menu

REM ===============================
REM TREINAMENTO PADRÃO
REM ===============================

:train_standard_la
echo === Treinando modelo padrão (LA) ===
python train.py --train-features-dir "%OUTPUT_DIR%\LA\features\train" --dev-features-dir "%OUTPUT_DIR%\LA\features\dev" --train-labels-file "%ASV2019_LA_DIR%\labels\train_labels.txt" --dev-labels-file "%ASV2019_LA_DIR%\labels\dev_labels.txt" --save-dir "%OUTPUT_DIR%\LA\checkpoints" --num-epochs 50
pause
goto menu

:train_standard_pa
echo === Treinando modelo padrão (PA) ===
python train.py --train-features-dir "%OUTPUT_DIR%\PA\features\train" --dev-features-dir "%OUTPUT_DIR%\PA\features\dev" --train-labels-file "%ASV2019_PA_DIR%\labels\train_labels.txt" --dev-labels-file "%ASV2019_PA_DIR%\labels\dev_labels.txt" --save-dir "%OUTPUT_DIR%\PA\checkpoints" --num-epochs 50
pause
goto menu

REM ===============================
REM AVALIAÇÃO
REM ===============================

:evaluate_la
echo === Avaliando modelo (LA) ===
echo.
echo Escolha o modelo para avaliar:
echo 1. Modelo padrão
echo 2. Modelo otimizado
echo 3. Modelo progressivo
echo 4. Modelo inteligente
echo.
set /p model_choice=Escolha o modelo (1-4): 

if "%model_choice%"=="1" (
    set CHECKPOINT_DIR=%OUTPUT_DIR%\LA\checkpoints
    set RESULTS_DIR=%OUTPUT_DIR%\LA\results
) else if "%model_choice%"=="2" (
    set CHECKPOINT_DIR=%OUTPUT_DIR%\LA\checkpoints_optimized
    set RESULTS_DIR=%OUTPUT_DIR%\LA\results_optimized
) else if "%model_choice%"=="3" (
    set CHECKPOINT_DIR=%OUTPUT_DIR%\LA\checkpoints_progressive
    set RESULTS_DIR=%OUTPUT_DIR%\LA\results_progressive
) else if "%model_choice%"=="4" (
    set CHECKPOINT_DIR=%OUTPUT_DIR%\LA\checkpoints_intelligent
    set RESULTS_DIR=%OUTPUT_DIR%\LA\results_intelligent
) else (
    echo Opção inválida.
    goto menu
)

python test.py --test-features-dir "%OUTPUT_DIR%\LA\features\eval" --test-labels-file "%ASV2019_LA_DIR%\labels\eval_labels.txt" --checkpoint "%CHECKPOINT_DIR%\best_model.pth" --results-dir "%RESULTS_DIR%" --analyze-generalization
pause
goto menu

:evaluate_pa
echo === Avaliando modelo (PA) ===
echo.
echo Escolha o modelo para avaliar:
echo 1. Modelo padrão
echo 2. Modelo otimizado
echo 3. Modelo progressivo
echo 4. Modelo inteligente
echo.
set /p model_choice=Escolha o modelo (1-4): 

if "%model_choice%"=="1" (
    set CHECKPOINT_DIR=%OUTPUT_DIR%\PA\checkpoints
    set RESULTS_DIR=%OUTPUT_DIR%\PA\results
) else if "%model_choice%"=="2" (
    set CHECKPOINT_DIR=%OUTPUT_DIR%\PA\checkpoints_optimized
    set RESULTS_DIR=%OUTPUT_DIR%\PA\results_optimized
) else if "%model_choice%"=="3" (
    set CHECKPOINT_DIR=%OUTPUT_DIR%\PA\checkpoints_progressive
    set RESULTS_DIR=%OUTPUT_DIR%\PA\results_progressive
) else if "%model_choice%"=="4" (
    set CHECKPOINT_DIR=%OUTPUT_DIR%\PA\checkpoints_intelligent
    set RESULTS_DIR=%OUTPUT_DIR%\PA\results_intelligent
) else (
    echo Opção inválida.
    goto menu
)

python test.py --test-features-dir "%OUTPUT_DIR%\PA\features\eval" --test-labels-file "%ASV2019_PA_DIR%\labels\eval_labels.txt" --checkpoint "%CHECKPOINT_DIR%\best_model.pth" --results-dir "%RESULTS_DIR%" --analyze-generalization
pause
goto menu

:cross_dataset
echo === Avaliando generalização (cross-dataset) ===
python test.py --test-features-dir "%OUTPUT_DIR%\PA\features\eval" --test-labels-file "%ASV2019_PA_DIR%\labels\eval_labels.txt" --checkpoint "%OUTPUT_DIR%\LA\checkpoints_optimized\best_model.pth" --results-dir "%OUTPUT_DIR%\cross_dataset\LA_to_PA" --analyze-generalization --cross-dataset
pause
goto menu

REM ===============================
REM FLUXOS COMPLETOS
REM ===============================

:full_pipeline_optimized_la
echo === Fluxo completo otimizado (LA) ===
echo Este fluxo incluirá: conversão, extração, benchmark, treinamento otimizado e avaliação
pause
call :convert_protocols
call :extract_features_la
call :benchmark_segmentation
call :train_optimized_la
call :evaluate_la
echo === Fluxo completo otimizado (LA) concluído ===
pause
goto menu

:full_pipeline_optimized_pa
echo === Fluxo completo otimizado (PA) ===
echo Este fluxo incluirá: conversão, extração, benchmark, treinamento otimizado e avaliação
pause
call :convert_protocols
call :extract_features_pa
call :benchmark_segmentation
call :train_optimized_pa
call :evaluate_pa
echo === Fluxo completo otimizado (PA) concluído ===
pause
goto menu

:full_pipeline_standard_la
echo === Fluxo completo padrão (LA) ===
call :convert_protocols
call :extract_features_la
call :train_standard_la
call :evaluate_la
echo === Fluxo completo padrão (LA) concluído ===
pause
goto menu

:full_pipeline_standard_pa
echo === Fluxo completo padrão (PA) ===
call :convert_protocols
call :extract_features_pa
call :train_standard_pa
call :evaluate_pa
echo === Fluxo completo padrão (PA) concluído ===
pause
goto menu

REM ===============================
REM CONFIGURAÇÕES
REM ===============================

:configure_segmentation
echo === Configurar Parâmetros de Segmentação ===
echo.
echo Configurações atuais:
echo Estratégia: %DEFAULT_STRATEGY%
echo Ratio: %DEFAULT_RATIO%
echo.
echo Estratégias disponíveis:
echo 1. adaptive (recomendado)
echo 2. kmeans
echo 3. dbscan
echo 4. stratified
echo 5. diversity
echo.
set /p strategy_choice=Escolha a estratégia (1-5): 

if "%strategy_choice%"=="1" set DEFAULT_STRATEGY=adaptive
if "%strategy_choice%"=="2" set DEFAULT_STRATEGY=kmeans
if "%strategy_choice%"=="3" set DEFAULT_STRATEGY=dbscan
if "%strategy_choice%"=="4" set DEFAULT_STRATEGY=stratified
if "%strategy_choice%"=="5" set DEFAULT_STRATEGY=diversity

echo.
set /p new_ratio=Ratio de amostragem (0.1-0.9, atual: %DEFAULT_RATIO%): 
if not "%new_ratio%"=="" set DEFAULT_RATIO=%new_ratio%

echo.
echo Novas configurações:
echo Estratégia: %DEFAULT_STRATEGY%
echo Ratio: %DEFAULT_RATIO%
echo.
pause
goto menu

:show_config
echo === Configurações Atuais ===
echo.
echo Caminhos:
echo   ASV2019 PA: %ASV2019_PA_DIR%
echo   ASV2019 LA: %ASV2019_LA_DIR%
echo   ASV2021: %ASV2021_DIR%
echo   Saída: %OUTPUT_DIR%
echo.
echo Segmentação:
echo   Estratégia: %DEFAULT_STRATEGY%
echo   Ratio: %DEFAULT_RATIO%
echo   Cache: %SEGMENTATION_CACHE%
echo   Benchmark: %BENCHMARK_DIR%
echo.
echo Verificando dependências...
python -c "import torch, librosa, sklearn, numpy; print('✓ Dependências básicas OK')" 2>nul || echo "✗ Algumas dependências faltando"
python -c "import umap, seaborn; print('✓ Dependências de segmentação OK')" 2>nul || echo "✗ Dependências de segmentação faltando"
echo.
pause
goto menu

REM ===============================
REM UTILITÁRIOS
REM ===============================

:end
echo.
echo ===== Relatório de Sessão =====
echo Diretório de saída: %OUTPUT_DIR%
echo.
echo Arquivos gerados podem incluir:
echo - Características extraídas
echo - Modelos treinados
echo - Resultados de avaliação
echo - Relatórios de benchmark
echo - Gráficos de análise
echo.
echo Obrigado por usar o sistema!
pause
exit /b 0