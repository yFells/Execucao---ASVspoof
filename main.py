#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script principal para orquestrar todo o fluxo de execução do projeto de detecção de ataques de replay.

Autores: André Thiago de Souza, Felipe de Lima dos Santos, 
         Juliano Gaona Proença, Matheus Henrique Reich Favarin Zagonel
"""

import os
import argparse
import subprocess
import json
import shutil
from datetime import datetime


def parse_args():
    """
    Analisa argumentos de linha de comando.
    
    Returns:
        Argumentos analisados
    """
    parser = argparse.ArgumentParser(description="Detecção de ataques de replay em sistemas ASV")
    
    # Argumentos gerais
    parser.add_argument('--task', type=str, required=True, choices=['extract', 'train', 'test', 'all'],
                        help='Tarefa a ser executada')
    parser.add_argument('--config', type=str, default='config.json',
                        help='Arquivo de configuração')
    parser.add_argument('--output-dir', type=str, default='output',
                        help='Diretório de saída para resultados')
    parser.add_argument('--gpu', type=int, default=0,
                        help='Índice da GPU a ser utilizada')
    
    # Argumentos específicos para extração de características
    parser.add_argument('--dataset', type=str, default='ASVspoof2019',
                        help='Nome do conjunto de dados')
    parser.add_argument('--audio-ext', type=str, default='.flac',
                        help='Extensão dos arquivos de áudio')
    
    # Argumentos específicos para treinamento
    parser.add_argument('--resume', action='store_true',
                        help='Continuar treinamento a partir de checkpoint')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Caminho para checkpoint para continuar treinamento')
    parser.add_argument('--experiment-name', type=str, default=None,
                        help='Nome do experimento')
    
    # Argumentos específicos para teste
    parser.add_argument('--test-only', action='store_true',
                        help='Realizar apenas teste, sem treinamento')
    parser.add_argument('--cross-dataset', action='store_true',
                        help='Realizar análise cruzada entre conjuntos de dados')
    parser.add_argument('--cross-dataset-name', type=str, default='ASVspoof2021',
                        help='Nome do conjunto de dados para análise cruzada')
    
    # Argumentos para amostragem do dataset
    parser.add_argument('--sample-proportion', type=float, default=1.0,
                        help='Proporção do dataset a ser usado para treinamento e validação (0.0 a 1.0)')

    return parser.parse_args()


def load_config(config_path):
    """
    Carrega configurações a partir de arquivo JSON.
    
    Args:
        config_path: Caminho para o arquivo de configuração
        
    Returns:
        Dicionário com configurações
    """
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    return config


def setup_experiment_dir(output_dir, experiment_name=None):
    """
    Configura diretório para o experimento.
    
    Args:
        output_dir: Diretório base para saída
        experiment_name: Nome do experimento (opcional)
        
    Returns:
        Caminho para o diretório do experimento
    """
    # Criar diretório base, se não existir
    os.makedirs(output_dir, exist_ok=True)
    
    # Definir nome do experimento, se não fornecido
    if experiment_name is None:
        experiment_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Criar diretório para o experimento
    experiment_dir = os.path.join(output_dir, experiment_name)
    os.makedirs(experiment_dir, exist_ok=True)
    
    # Criar subdiretórios
    checkpoints_dir = os.path.join(experiment_dir, 'checkpoints')
    results_dir = os.path.join(experiment_dir, 'results')
    logs_dir = os.path.join(experiment_dir, 'logs')
    
    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)
    
    return experiment_dir


def extract_features(args, config, experiment_dir):
    """
    Executa processo de extração de características.
    
    Args:
        args: Argumentos de linha de comando
        config: Configurações
        experiment_dir: Diretório do experimento
        
    Returns:
        Código de retorno do processo
    """
    print("Iniciando extração de características...")
    
    # Preparar diretórios
    dataset_config = config['datasets'][args.dataset]
    output_features_dir = os.path.join(experiment_dir, 'features')
    
    # Criar diretórios para características
    os.makedirs(output_features_dir, exist_ok=True)
    os.makedirs(os.path.join(output_features_dir, 'train'), exist_ok=True)
    os.makedirs(os.path.join(output_features_dir, 'dev'), exist_ok=True)
    os.makedirs(os.path.join(output_features_dir, 'eval'), exist_ok=True)
    
    # Construir comando para executar script de extração
    # Passar os caminhos dos arquivos de rótulo e a proporção de amostragem
    extract_cmd = [
        'python', 'feature_extraction.py',
        '--train-audio-dir', dataset_config['train_audio_dir'],
        '--dev-audio-dir', dataset_config['dev_audio_dir'],
        '--eval-audio-dir', dataset_config['eval_audio_dir'],
        '--output-dir', output_features_dir,
        '--audio-ext', args.audio_ext,
        '--train-labels-file', dataset_config['train_labels_file'],
        '--dev-labels-file', dataset_config['dev_labels_file'],
        '--eval-labels-file', dataset_config['eval_labels_file'], # Garantindo que este argumento seja passado
        '--sample-proportion', str(args.sample_proportion)
    ]
    
    # Executar comando
    print(" ".join(extract_cmd))
    return subprocess.call(extract_cmd)


def train_model(args, config, experiment_dir):
    """
    Executa processo de treinamento do modelo.
    
    Args:
        args: Argumentos de linha de comando
        config: Configurações
        experiment_dir: Diretório do experimento
        
    Returns:
        Código de retorno do processo
    """
    print("Iniciando treinamento do modelo...")
    
    # Preparar diretórios
    dataset_config = config['datasets'][args.dataset]
    features_dir = os.path.join(experiment_dir, 'features')
    checkpoints_dir = os.path.join(experiment_dir, 'checkpoints')
    
    # Construir comando para executar script de treinamento
    train_cmd = [
        'python', 'train.py',
        '--train-features-dir', os.path.join(features_dir, 'train'),
        '--dev-features-dir', os.path.join(features_dir, 'dev'),
        '--train-labels-file', dataset_config['train_labels_file'],
        '--dev-labels-file', dataset_config['dev_labels_file'],
        '--batch-size', str(config['training']['batch_size']),
        '--num-epochs', str(config['training']['num_epochs']),
        '--learning-rate', str(config['training']['learning_rate']),
        '--weight-decay', str(config['training']['weight_decay']),
        '--patience', str(config['training']['patience']),
        '--save-dir', checkpoints_dir,
        '--segment-length', str(config['segmentation']['segment_length']),
        '--stride', str(config['segmentation']['stride']),
        '--num-workers', str(config['training']['num_workers']),
        '--sample-proportion', str(args.sample_proportion) # Passa sample_proportion
    ]
    
    # Adicionar checkpoint, se fornecido
    if args.resume and args.checkpoint:
        train_cmd.extend(['--checkpoint', args.checkpoint])
    
    # Definir variável de ambiente para GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    
    # Executar comando
    print(" ".join(train_cmd))
    return subprocess.call(train_cmd)


def test_model(args, config, experiment_dir):
    """
    Executa processo de teste e avaliação do modelo.
    
    Args:
        args: Argumentos de linha de comando
        config: Configurações
        experiment_dir: Diretório do experimento
        
    Returns:
        Código de retorno do processo
    """
    print("Iniciando teste e avaliação do modelo...")
    
    # Preparar diretórios
    dataset_config = config['datasets'][args.dataset]
    features_dir = os.path.join(experiment_dir, 'features')
    checkpoints_dir = os.path.join(experiment_dir, 'checkpoints')
    results_dir = os.path.join(experiment_dir, 'results')
    
    # Obter caminho para o melhor modelo
    best_model_path = os.path.join(checkpoints_dir, 'best_model.pth')
    if args.checkpoint:
        best_model_path = args.checkpoint
    
    # Construir comando para executar script de teste
    test_cmd = [
        'python', 'test.py',
        '--test-features-dir', os.path.join(features_dir, 'eval'),
        '--test-labels-file', dataset_config['eval_labels_file'],
        '--checkpoint', best_model_path,
        '--batch-size', str(config['testing']['batch_size']),
        '--results-dir', results_dir,
        '--segment-length', str(config['segmentation']['segment_length']),
        '--stride', str(config['segmentation']['stride']),
        '--num-workers', str(config['testing']['num_workers']),
        '--sample-proportion', str(args.sample_proportion) # Passa sample_proportion para test.py para loaders de treino/dev em análise de generalização
    ]
    
    # Adicionar análise de generalização, se solicitado
    if config['testing']['analyze_generalization']:
        test_cmd.append('--analyze-generalization')
        test_cmd.extend([
            '--train-features-dir', os.path.join(features_dir, 'train'),
            '--train-labels-file', dataset_config['train_labels_file'],
            '--dev-features-dir', os.path.join(features_dir, 'dev'),
            '--dev-labels-file', dataset_config['dev_labels_file']
        ])
    
    # Adicionar análise cruzada entre conjuntos de dados, se solicitado
    if args.cross_dataset:
        cross_dataset_config = config['datasets'][args.cross_dataset_name]
        cross_features_dir = os.path.join(experiment_dir, 'cross_features')
        
        # Extrair características para o conjunto de dados cruzado, se necessário
        os.makedirs(cross_features_dir, exist_ok=True)
        
        # Adicionar análise cruzada ao comando
        test_cmd.append('--cross-dataset')
        test_cmd.extend([
            '--cross-dataset-features-dir', cross_features_dir,
            '--cross-dataset-labels-file', cross_dataset_config['eval_labels_file']
        ])
    
    # Definir variável de ambiente para GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    
    # Executar comando
    print(" ".join(test_cmd))
    return subprocess.call(test_cmd)


def run_all(args, config, experiment_dir):
    """
    Executa todo o fluxo do projeto: extração, treinamento e teste.
    
    Args:
        args: Argumentos de linha de comando
        config: Configurações
        experiment_dir: Diretório do experimento
        
    Returns:
        True se todas as etapas foram bem-sucedidas, False caso contrário
    """
    # Extrair características
    extract_ret = extract_features(args, config, experiment_dir)
    if extract_ret != 0:
        print("Erro na extração de características!")
        return False
    
    # Treinar modelo
    train_ret = train_model(args, config, experiment_dir)
    if train_ret != 0:
        print("Erro no treinamento do modelo!")
        return False
    
    # Testar modelo
    test_ret = test_model(args, config, experiment_dir)
    if test_ret != 0:
        print("Erro no teste do modelo!")
        return False
    
    print("Todas as etapas concluídas com sucesso!")
    return True


def save_experiment_config(args, config, experiment_dir):
    """
    Salva configuração do experimento.
    
    Args:
        args: Argumentos de linha de comando
        config: Configurações
        experiment_dir: Diretório do experimento
    """
    # Criar cópia do arquivo de configuração
    config_path = os.path.join(experiment_dir, 'config.json')
    shutil.copy2(args.config, config_path)
    
    # Salvar argumentos de linha de comando
    args_dict = vars(args)
    args_path = os.path.join(experiment_dir, 'args.json')
    
    with open(args_path, 'w') as f:
        json.dump(args_dict, f, indent=4)


def main():
    """
    Função principal.
    """
    # Analisar argumentos
    args = parse_args()
    
    # Carregar configurações
    config = load_config(args.config)
    
    # Configurar diretório do experimento
    experiment_dir = setup_experiment_dir(args.output_dir, args.experiment_name)
    print(f"Diretório do experimento: {experiment_dir}")
    
    # Salvar configuração do experimento
    save_experiment_config(args, config, experiment_dir)
    
    # Executar tarefa solicitada
    if args.task == 'extract':
        extract_features(args, config, experiment_dir)
    elif args.task == 'train':
        train_model(args, config, experiment_dir)
    elif args.task == 'test':
        test_model(args, config, experiment_dir)
    elif args.task == 'all':
        run_all(args, config, experiment_dir)
    else:
        print(f"Tarefa desconhecida: {args.task}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())

