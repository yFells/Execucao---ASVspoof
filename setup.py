#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script de configuração inicial para o sistema de detecção de ataques spoofing de replay.
Verifica as dependências e cria a estrutura de diretórios necessária.

Autores: André Thiago de Souza, Felipe de Lima dos Santos, 
         Juliano Gaona Proença, Matheus Henrique Reich Favarin Zagonel
"""

import os
import sys
import subprocess
import argparse
import platform


def check_dependencies():
    """
    Verifica se as dependências necessárias estão instaladas.
    
    Returns:
        Lista de pacotes que precisam ser instalados
    """
    required_packages = [
        'torch', 'torchaudio', 'numpy', 'scipy', 'librosa', 'scikit-learn', 
        'scikit-image', 'matplotlib', 'pandas', 'tqdm', 'pydub', 'soundfile'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    return missing_packages


def install_dependencies(missing_packages):
    """
    Instala as dependências faltantes.
    
    Args:
        missing_packages: Lista de pacotes a serem instalados
        
    Returns:
        True se a instalação foi bem-sucedida, False caso contrário
    """
    if not missing_packages:
        print("Todas as dependências já estão instaladas.")
        return True
    
    print(f"Instalando dependências: {', '.join(missing_packages)}")
    
    try:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install'] + missing_packages)
        print("Dependências instaladas com sucesso!")
        return True
    except subprocess.CalledProcessError:
        print("Erro ao instalar dependências.")
        return False


def create_directory_structure(base_dir):
    """
    Cria a estrutura de diretórios necessária.
    
    Args:
        base_dir: Diretório base
        
    Returns:
        True se a criação foi bem-sucedida, False caso contrário
    """
    try:
        # Diretórios para LA
        os.makedirs(os.path.join(base_dir, 'LA', 'features', 'train'), exist_ok=True)
        os.makedirs(os.path.join(base_dir, 'LA', 'features', 'dev'), exist_ok=True)
        os.makedirs(os.path.join(base_dir, 'LA', 'features', 'eval'), exist_ok=True)
        os.makedirs(os.path.join(base_dir, 'LA', 'checkpoints'), exist_ok=True)
        os.makedirs(os.path.join(base_dir, 'LA', 'results'), exist_ok=True)
        
        # Diretórios para PA
        os.makedirs(os.path.join(base_dir, 'PA', 'features', 'train'), exist_ok=True)
        os.makedirs(os.path.join(base_dir, 'PA', 'features', 'dev'), exist_ok=True)
        os.makedirs(os.path.join(base_dir, 'PA', 'features', 'eval'), exist_ok=True)
        os.makedirs(os.path.join(base_dir, 'PA', 'checkpoints'), exist_ok=True)
        os.makedirs(os.path.join(base_dir, 'PA', 'results'), exist_ok=True)
        
        # Diretórios para cross-dataset
        os.makedirs(os.path.join(base_dir, 'cross_dataset'), exist_ok=True)
        
        print(f"Estrutura de diretórios criada em: {base_dir}")
        return True
    except Exception as e:
        print(f"Erro ao criar estrutura de diretórios: {str(e)}")
        return False


def verify_datasets(asvspoof2019_pa_dir, asvspoof2019_la_dir, asvspoof2021_dir=None):
    """
    Verifica se os datasets estão presentes nas localizações especificadas.
    
    Args:
        asvspoof2019_pa_dir: Diretório do dataset ASVspoof 2019 PA
        asvspoof2019_la_dir: Diretório do dataset ASVspoof 2019 LA
        asvspoof2021_dir: Diretório do dataset ASVspoof 2021 (opcional)
        
    Returns:
        Dicionário com informações sobre os datasets
    """
    datasets_info = {
        'ASVspoof2019_PA': {'present': False, 'files': 0},
        'ASVspoof2019_LA': {'present': False, 'files': 0},
        'ASVspoof2021': {'present': False, 'files': 0}
    }
    
    # Verificar ASVspoof 2019 PA
    pa_train_dir = os.path.join(asvspoof2019_pa_dir, 'ASVspoof2019_PA_train', 'flac')
    pa_dev_dir = os.path.join(asvspoof2019_pa_dir, 'ASVspoof2019_PA_dev', 'flac')
    pa_eval_dir = os.path.join(asvspoof2019_pa_dir, 'ASVspoof2019_PA_eval', 'flac')
    
    if os.path.exists(pa_train_dir) and os.path.exists(pa_dev_dir) and os.path.exists(pa_eval_dir):
        datasets_info['ASVspoof2019_PA']['present'] = True
        
        # Contar arquivos
        train_files = sum(1 for _ in os.listdir(pa_train_dir) if _.endswith('.flac'))
        dev_files = sum(1 for _ in os.listdir(pa_dev_dir) if _.endswith('.flac'))
        eval_files = sum(1 for _ in os.listdir(pa_eval_dir) if _.endswith('.flac'))
        
        datasets_info['ASVspoof2019_PA']['files'] = train_files + dev_files + eval_files
    
    # Verificar ASVspoof 2019 LA
    la_train_dir = os.path.join(asvspoof2019_la_dir, 'ASVspoof2019_LA_train', 'flac')
    la_dev_dir = os.path.join(asvspoof2019_la_dir, 'ASVspoof2019_LA_dev', 'flac')
    la_eval_dir = os.path.join(asvspoof2019_la_dir, 'ASVspoof2019_LA_eval', 'flac')
    
    if os.path.exists(la_train_dir) and os.path.exists(la_dev_dir) and os.path.exists(la_eval_dir):
        datasets_info['ASVspoof2019_LA']['present'] = True
        
        # Contar arquivos
        train_files = sum(1 for _ in os.listdir(la_train_dir) if _.endswith('.flac'))
        dev_files = sum(1 for _ in os.listdir(la_dev_dir) if _.endswith('.flac'))
        eval_files = sum(1 for _ in os.listdir(la_eval_dir) if _.endswith('.flac'))
        
        datasets_info['ASVspoof2019_LA']['files'] = train_files + dev_files + eval_files
    
    # Verificar ASVspoof 2021, se fornecido
    if asvspoof2021_dir:
        flac_dir = os.path.join(asvspoof2021_dir, 'flac')
        
        if os.path.exists(flac_dir):
            datasets_info['ASVspoof2021']['present'] = True
            
            # Contar arquivos
            files = sum(1 for _ in os.listdir(flac_dir) if _.endswith('.flac'))
            
            datasets_info['ASVspoof2021']['files'] = files
    
    return datasets_info


def check_for_ffmpeg():
    """
    Verifica se o ffmpeg está instalado.
    
    Returns:
        True se o ffmpeg estiver instalado, False caso contrário
    """
    try:
        # Verificar se o ffmpeg está no PATH
        subprocess.check_call(['ffmpeg', '-version'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        # ffmpeg não está instalado ou não está no PATH
        return False


def main():
    """
    Função principal.
    """
    # Configurar parser de argumentos
    parser = argparse.ArgumentParser(description="Configuração inicial para detecção de ataques de replay")
    
    parser.add_argument('--output-dir', type=str, default="E:/Experimentos/ASVspoof",
                        help='Diretório de saída para experimentos')
    parser.add_argument('--asv2019-pa-dir', type=str, default="E:/ASV 2019 DATA/PA",
                        help='Diretório do dataset ASVspoof 2019 PA')
    parser.add_argument('--asv2019-la-dir', type=str, default="E:/ASV 2019 DATA/LA",
                        help='Diretório do dataset ASVspoof 2019 LA')
    parser.add_argument('--asv2021-dir', type=str, default="E:/ASV 2021 DATA/PA",
                        help='Diretório do dataset ASVspoof 2021')
    parser.add_argument('--install-deps', action='store_true',
                        help='Instalar dependências')
    
    # Analisar argumentos
    args = parser.parse_args()
    
    # Exibir informações do sistema
    print("=== Configuração do Sistema ===")
    print(f"Sistema operacional: {platform.system()} {platform.release()}")
    print(f"Python: {sys.version}")
    
    # Verificar dependências
    missing_packages = check_dependencies()
    
    if missing_packages:
        print(f"\nDependências faltantes: {', '.join(missing_packages)}")
        
        if args.install_deps:
            install_dependencies(missing_packages)
        else:
            print("Use --install-deps para instalar as dependências automaticamente.")
    else:
        print("\nTodas as dependências estão instaladas!")
    
    # Verificar se o ffmpeg está instalado
    if check_for_ffmpeg():
        print("FFmpeg está instalado e disponível no PATH.")
    else:
        print("FFmpeg não está instalado ou não está no PATH.")
        print("É recomendado instalar o FFmpeg para manipulação de áudio.")
        
        if platform.system() == 'Windows':
            print("Instruções para instalar o FFmpeg no Windows:")
            print("1. Baixe o FFmpeg de https://ffmpeg.org/download.html")
            print("2. Extraia os arquivos para uma pasta (ex: C:\\ffmpeg)")
            print("3. Adicione a pasta bin ao PATH do sistema")
        elif platform.system() == 'Linux':
            print("Instale o FFmpeg usando seu gerenciador de pacotes:")
            print("  sudo apt-get install ffmpeg  (Ubuntu/Debian)")
            print("  sudo yum install ffmpeg      (CentOS/RHEL)")
        elif platform.system() == 'Darwin':  # macOS
            print("Instale o FFmpeg usando Homebrew:")
            print("  brew install ffmpeg")
    
    # Criar estrutura de diretórios
    create_directory_structure(args.output_dir)
    
    # Verificar datasets
    print("\n=== Verificação de Datasets ===")
    datasets_info = verify_datasets(args.asv2019_pa_dir, args.asv2019_la_dir, args.asv2021_dir)
    
    for dataset, info in datasets_info.items():
        if info['present']:
            print(f"{dataset}: Encontrado ({info['files']} arquivos)")
        else:
            print(f"{dataset}: Não encontrado")
    
    # Instruções finais
    print("\n=== Próximos Passos ===")
    print("1. Execute convert_protocols.py para converter os arquivos de protocolo para o formato de labels")
    print("2. Execute feature_extraction_fix.py para extrair características dos arquivos de áudio")
    print("3. Execute train.py para treinar o modelo")
    print("4. Execute test.py para avaliar o modelo")
    print("5. Ou use run_asvspoof.bat para executar todas as etapas de forma interativa")
    
    print("\nConfiguração concluída!")
    return 0


if __name__ == "__main__":
    sys.exit(main())