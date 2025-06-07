@echo off
SETLOCAL ENABLEDELAYEDEXPANSION

:MENU
cls
echo ============= MENU PRINCIPAL =============       
echo === OPERAÇÔES BÁSICAS ===
echo 1.  Converter protocolos para formato de labels  
echo 2.  Extrair características (LA)
echo 3.  Extrair características (PA)
echo 4.  Instalar dependências
echo === ANÁLISE DE SEGMENTAÇÃO ===
echo 5.  Benchmark de estratégias de segmentação   
echo 6.  Análise rápida de segmentação (LA)       
echo 7.  Análise rápida de segmentação (PA)       
echo 8.  Comparar estratégias de segmentação       
echo === TREINAMENTO OTIMIZADO ===
echo 9.  Treinar modelo otimizado (LA) - Segmentação
echo 10. Treinar modelo otimizado (PA) - Segmentação
echo 11. Treinar modelo progressivo (LA)
echo 12. Treinar modelo progressivo (PA)
echo 13. Treinar modelo inteligente (LA)
echo 14. Treinar modelo inteligente (PA)
echo === TREINAMENTO PADRÃO ===
echo 15. Treinar modelo padrão (LA)
echo 16. Treinar modelo padrão (PA)
echo === AVALIAÇÃO ===
echo 17. Avaliar modelo (LA)
echo 18. Avaliar modelo (PA)
echo 19. Avaliar generalização (cross-dataset)
echo === FLUXOS COMPLETOS ===
echo 20. Fluxo completo otimizado (LA)
echo 21. Fluxo completo otimizado (PA)
echo 22. Fluxo completo padrão (LA)
echo 23. Fluxo completo padrão (PA)
echo.
echo 0. Sair
echo.

set /p choice="Escolha uma opção: "

if "%choice%"=="1" goto CONVERT_PROTOCOLS
if "%choice%"=="2" goto EXTRACT_FEATURES_LA
if "%choice%"=="3" goto EXTRACT_FEATURES_PA
if "%choice%"=="4" goto INSTALL_DEPS
if "%choice%"=="5" goto BENCHMARK_SEGMENTATION
if "%choice%"=="6" goto ANALYZE_SEGMENTATION_LA
if "%choice%"=="7" goto ANALYZE_SEGMENTATION_PA
if "%choice%"=="8" goto COMPARE_SEGMENTATION
if "%choice%"=="9" goto TRAIN_OPTIMIZED_LA
if "%choice%"=="10" goto TRAIN_OPTIMIZED_PA
if "%choice%"=="11" goto TRAIN_PROGRESSIVE_LA
if "%choice%"=="12" goto TRAIN_PROGRESSIVE_PA
if "%choice%"=="13" goto TRAIN_INTELLIGENT_LA
if "%choice%"=="14" goto TRAIN_INTELLIGENT_PA
if "%choice%"=="15" goto TRAIN_STANDARD_LA
if "%choice%"=="16" goto TRAIN_STANDARD_PA
if "%choice%"=="17" goto EVALUATE_LA
if "%choice%"=="18" goto EVALUATE_PA
if "%choice%"=="19" goto EVALUATE_CROSS_DATASET
if "%choice%"=="20" goto FULL_FLOW_OPTIMIZED_LA
if "%choice%"=="21" goto FULL_FLOW_OPTIMIZED_PA
if "%choice%"=="22" goto FULL_FLOW_STANDARD_LA
if "%choice%"=="23" goto FULL_FLOW_STANDARD_PA
if "%choice%"=="0" goto END

echo Opção inválida. Pressione qualquer tecla para tentar novamente...
pause > nul
goto MENU

:CONVERT_PROTOCOLS
echo Executando: Converter protocolos para formato de labels
python convert_protocols.py
pause
goto MENU

:EXTRACT_FEATURES_LA
echo Executando: Extrair características (LA) com 30%% do dataset
python main.py --task extract --dataset ASVspoof2019 --output-dir output/experiment1 --sample-proportion 0.5
pause
goto MENU

:EXTRACT_FEATURES_PA
echo Executando: Extrair características (PA) com 30%% do dataset
python main.py --task extract --dataset ASVspoof2019PA --output-dir output/experiment1 --sample-proportion 0.5
pause
goto MENU

:INSTALL_DEPS
echo Executando: Instalar dependências
python setup.py --install-deps
pause
goto MENU

:BENCHMARK_SEGMENTATION
echo Opção não implementada neste script de exemplo.
pause
goto MENU

:ANALYZE_SEGMENTATION_LA
echo Opção não implementada neste script de exemplo.
pause
goto MENU

:ANALYZE_SEGMENTATION_PA
echo Opção não implementada neste script de exemplo.
pause
goto MENU

:COMPARE_SEGMENTATION
echo Opção não implementada neste script de exemplo.
pause
goto MENU

:TRAIN_OPTIMIZED_LA
echo Executando: Treinar modelo otimizado (LA)
python main.py --task train --dataset ASVspoof2019 --output-dir output/experiment1 --sample-proportion 0.5
pause
goto MENU

:TRAIN_OPTIMIZED_PA
echo Executando: Treinar modelo otimizado (PA)
python main.py --task train --dataset ASVspoof2019PA --output-dir output/experiment1 --sample-proportion 0.5
pause
goto MENU

:TRAIN_PROGRESSIVE_LA
echo Opção não implementada neste script de exemplo.
pause
goto MENU

:TRAIN_PROGRESSIVE_PA
echo Opção não implementada neste script de exemplo.
pause
goto MENU

:TRAIN_INTELLIGENT_LA
echo Opção não implementada neste script de exemplo.
pause
goto MENU

:TRAIN_INTELLIGENT_PA
echo Opção não implementada neste script de exemplo.
pause
goto MENU

:TRAIN_STANDARD_LA
echo Executando: Treinar modelo padrão (LA)
python main.py --task train --dataset ASVspoof2019 --output-dir output/experiment1 --sample-proportion 1.0
pause
goto MENU

:TRAIN_STANDARD_PA
echo Executando: Treinar modelo padrão (PA)
python main.py --task train --dataset ASVspoof2019PA --output-dir output/experiment1 --sample-proportion 1.0
pause
goto MENU

:EVALUATE_LA
echo Executando: Avaliar modelo (LA)
python main.py --task test --dataset ASVspoof2019 --output-dir output/experiment1 --sample-proportion 1.0
pause
goto MENU

:EVALUATE_PA
echo Executando: Avaliar modelo (PA)
python main.py --task test --dataset ASVspoof2019PA --output-dir output/experiment1 --sample-proportion 1.0
pause
goto MENU

:EVALUATE_CROSS_DATASET
echo Executando: Avaliar generalização (cross-dataset)
python main.py --task test --dataset ASVspoof2019 --cross-dataset --cross-dataset-name ASVspoof2021 --output-dir output/experiment1 --sample-proportion 1.0
pause
goto MENU

:FULL_FLOW_OPTIMIZED_LA
echo Executando: Fluxo completo otimizado (LA)
python main.py --task all --dataset ASVspoof2019 --output-dir output/experiment1 --sample-proportion 0.5
pause
goto MENU

:FULL_FLOW_OPTIMIZED_PA
echo Executando: Fluxo completo otimizado (PA)
python main.py --task all --dataset ASVspoof2019PA --output-dir output/experiment1 --sample-proportion 0.5
pause
goto MENU

:FULL_FLOW_STANDARD_LA
echo Executando: Fluxo completo padrão (LA)
python main.py --task all --dataset ASVspoof2019 --output-dir output/experiment1 --sample-proportion 1.0
pause
goto MENU

:FULL_FLOW_STANDARD_PA
echo Executando: Fluxo completo padrão (PA)
python main.py --task all --dataset ASVspoof2019PA --output-dir output/experiment1 --sample-proportion 1.0
pause
goto MENU


:END
echo Saindo...
ENDLOCAL
exit /b 0
