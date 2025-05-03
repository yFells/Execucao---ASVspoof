#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script para teste e avaliação do modelo treinado para detecção de ataques de replay.

Autores: André Thiago de Souza, Felipe de Lima dos Santos, 
         Juliano Gaona Proença, Matheus Henrique Reich Favarin Zagonel
"""

import os
import argparse
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

# Importar módulos locais
from feature_extraction import FeatureExtractor
from bidirectional_segmentation import BidirectionalDataLoader
from model import MultiPatternModel
from evaluation import ReplayAttackEvaluator
from generalization_analysis import GeneralizationAnalyzer


def load_model(checkpoint_path, device):
    """
    Carrega modelo a partir de checkpoint.
    
    Args:
        checkpoint_path: Caminho para o checkpoint
        device: Dispositivo para carregar o modelo
        
    Returns:
        Modelo carregado e pesos para fusão de pontuações
    """
    # Criar modelo
    model = MultiPatternModel(input_channels=1, hidden_size=512, num_classes=2).to(device)
    
    # Carregar checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Carregar estado do modelo
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Extrair pesos para fusão de pontuações
    score_weights = checkpoint.get('score_weights', {'lbp': 0.33, 'glcm': 0.33, 'lpq': 0.34})
    
    return model, score_weights


def prepare_dataloader(features_dir, labels_file, batch_size=32, segment_length=400, stride=200, num_workers=4):
    """
    Prepara DataLoader para teste.
    
    Args:
        features_dir: Diretório com características
        labels_file: Arquivo com rótulos
        batch_size: Tamanho do lote
        segment_length: Comprimento do segmento em frames
        stride: Tamanho do salto entre segmentos consecutivos
        num_workers: Número de processos para carregamento paralelo
        
    Returns:
        DataLoader para teste
    """
    # Criar DataLoader
    test_loader = BidirectionalDataLoader(
        features_dir, labels_file, batch_size, segment_length, stride, num_workers, shuffle=False
    ).get_dataloader()
    
    return test_loader


def extract_features(audio_dir, features_dir, audio_ext='.flac'):
    """
    Extrai características dos arquivos de áudio.
    
    Args:
        audio_dir: Diretório com arquivos de áudio
        features_dir: Diretório para salvar características
        audio_ext: Extensão dos arquivos de áudio
        
    Returns:
        Dicionário com características extraídas
    """
    # Criar diretório para salvar características, se não existir
    os.makedirs(features_dir, exist_ok=True)
    
    # Inicializar extrator de características
    extractor = FeatureExtractor(
        sample_rate=16000, n_mfcc=30, n_cqcc=30, n_mels=257,
        window_size=0.025, hop_size=0.010, pre_emphasis=0.97
    )
    
    # Extrair características
    features = extractor.batch_feature_extraction(audio_dir, features_dir, audio_ext)
    
    return features


def evaluate_model(model, dataloader, score_weights, device, results_dir=None):
    """
    Avalia o modelo no conjunto de dados.
    
    Args:
        model: Modelo treinado
        dataloader: DataLoader com dados para avaliação
        score_weights: Pesos para fusão de pontuações
        device: Dispositivo para execução
        results_dir: Diretório para salvar resultados (opcional)
        
    Returns:
        Resultados da avaliação
    """
    # Criar avaliador
    evaluator = ReplayAttackEvaluator(model, device, score_weights)
    
    # Avaliar modelo
    evaluation_results = evaluator.evaluate(dataloader)
    
    # Exibir resultados
    print("\nResultados da Avaliação:")
    print("Equal Error Rate (EER):")
    for score_type, result in evaluation_results['eer'].items():
        print(f"  {score_type}: {result['value']:.4f} (limiar: {result['threshold']:.4f})")
    
    print("\ntandem Detection Cost Function (t-DCF):")
    for score_type, result in evaluation_results['tdcf'].items():
        print(f"  {score_type}: {result['value']:.4f} (limiar: {result['threshold']:.4f})")
    
    print("\nArea Under Curve (AUC):")
    for score_type, auc_value in evaluation_results['auc'].items():
        print(f"  {score_type}: {auc_value:.4f}")
    
    print("\nTempo de Inferência:")
    print(f"  Média: {evaluation_results['inference_time']['avg'] * 1000:.2f} ms")
    print(f"  Desvio Padrão: {evaluation_results['inference_time']['std'] * 1000:.2f} ms")
    
    # Plotar curvas ROC e distribuições de pontuações
    if results_dir:
        # Criar diretório para resultados, se não existir
        os.makedirs(results_dir, exist_ok=True)
        
        # Plotar curvas ROC
        evaluator.plot_roc_curves(dataloader, os.path.join(results_dir, 'roc_curves.png'))
        
        # Plotar distribuições de pontuações
        evaluator.plot_score_distributions(dataloader, 'fused', os.path.join(results_dir, 'score_distributions.png'))
        
        # Salvar resultados em arquivo JSON
        results_path = os.path.join(results_dir, 'evaluation_results.json')
        
        # Converter valores para tipos serializáveis
        results_json = {}
        for metric, values in evaluation_results.items():
            if metric == 'inference_time':
                results_json[metric] = {
                    'avg': float(values['avg']),
                    'std': float(values['std'])
                }
            elif isinstance(values, dict):
                results_json[metric] = {}
                for score_type, result in values.items():
                    if isinstance(result, dict):
                        results_json[metric][score_type] = {
                            key: float(value) for key, value in result.items()
                        }
                    else:
                        results_json[metric][score_type] = float(result)
        
        # Salvar resultados
        with open(results_path, 'w') as f:
            json.dump(results_json, f, indent=4)
        
        print(f"\nResultados salvos em: {results_path}")
    
    return evaluation_results


def analyze_generalization(model, train_dataloader, test_dataloader, 
                           train_dataset_name, test_dataset_name, 
                           score_weights, device, results_dir=None):
    """
    Analisa a capacidade de generalização do modelo para ataques desconhecidos.
    
    Args:
        model: Modelo treinado
        train_dataloader: DataLoader com dados de treinamento
        test_dataloader: DataLoader com dados de teste
        train_dataset_name: Nome do conjunto de dados de treinamento
        test_dataset_name: Nome do conjunto de dados de teste
        score_weights: Pesos para fusão de pontuações
        device: Dispositivo para execução
        results_dir: Diretório para salvar resultados (opcional)
        
    Returns:
        Resultados da análise de generalização
    """
    # Criar analisador
    analyzer = GeneralizationAnalyzer(model, device, score_weights)
    
    # Realizar análise cruzada entre conjuntos de dados
    cross_dataset_results = analyzer.cross_dataset_analysis(
        train_dataloader, test_dataloader, train_dataset_name, test_dataset_name,
        os.path.join(results_dir, 'cross_dataset') if results_dir else None
    )
    
    # Realizar ablação para avaliar contribuição de componentes
    if results_dir:
        ablation_dir = os.path.join(results_dir, 'ablation')
        os.makedirs(ablation_dir, exist_ok=True)
        
        # Realizar estudo de ablação
        ablation_results = analyzer.ablation_study(test_dataloader)
        
        # Salvar resultados
        ablation_path = os.path.join(ablation_dir, 'ablation_results.csv')
        ablation_results.to_csv(ablation_path)
        
        # Plotar resultados
        plt.figure(figsize=(12, 8))
        ablation_results[['eer', 'tdcf']].plot(kind='bar', figsize=(12, 8))
        plt.title('Estudo de Ablação')
        plt.ylabel('Valor')
        plt.xlabel('Componente Removido')
        plt.xticks(rotation=45, ha='right')
        plt.grid(True, axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(ablation_dir, 'ablation_results.png'), dpi=300, bbox_inches='tight')
    
    return cross_dataset_results


def parse_args():
    """
    Analisa argumentos de linha de comando.
    
    Returns:
        Argumentos analisados
    """
    parser = argparse.ArgumentParser(description="Teste e avaliação de modelo para detecção de ataques de replay")
    
    # Argumentos para entrada de dados
    parser.add_argument('--test-audio-dir', type=str, default='data/ASVspoof2019/eval/audio',
                        help='Diretório com arquivos de áudio de teste')
    parser.add_argument('--test-features-dir', type=str, default='data/ASVspoof2019/eval/features',
                        help='Diretório para salvar características de teste')
    parser.add_argument('--test-labels-file', type=str, default='data/ASVspoof2019/eval/labels.txt',
                        help='Arquivo com rótulos de teste')
    parser.add_argument('--audio-ext', type=str, default='.flac',
                        help='Extensão dos arquivos de áudio')
    
    # Argumentos para modelo
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Caminho para checkpoint do modelo')
    
    # Argumentos para extração de características
    parser.add_argument('--extract-features', action='store_true',
                        help='Extrair características antes do teste')
    
    # Argumentos para avaliação
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Tamanho do lote')
    parser.add_argument('--segment-length', type=int, default=400,
                        help='Comprimento do segmento em frames')
    parser.add_argument('--stride', type=int, default=200,
                        help='Tamanho do salto entre segmentos consecutivos')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Número de processos para carregamento paralelo')
    parser.add_argument('--results-dir', type=str, default='results',
                        help='Diretório para salvar resultados')
    
    # Argumentos para análise de generalização
    parser.add_argument('--analyze-generalization', action='store_true',
                        help='Analisar capacidade de generalização do modelo')
    parser.add_argument('--train-features-dir', type=str, default='data/ASVspoof2019/train/features',
                        help='Diretório com características de treinamento para análise de generalização')
    parser.add_argument('--train-labels-file', type=str, default='data/ASVspoof2019/train/labels.txt',
                        help='Arquivo com rótulos de treinamento para análise de generalização')
    parser.add_argument('--dev-features-dir', type=str, default='data/ASVspoof2019/dev/features',
                        help='Diretório com características de validação para análise de generalização')
    parser.add_argument('--dev-labels-file', type=str, default='data/ASVspoof2019/dev/labels.txt',
                        help='Arquivo com rótulos de validação para análise de generalização')
    parser.add_argument('--cross-dataset', action='store_true',
                        help='Realizar análise cruzada entre conjuntos de dados')
    parser.add_argument('--cross-dataset-features-dir', type=str, default='data/ASVspoof2021/eval/features',
                        help='Diretório com características do outro conjunto de dados')
    parser.add_argument('--cross-dataset-labels-file', type=str, default='data/ASVspoof2021/eval/labels.txt',
                        help='Arquivo com rótulos do outro conjunto de dados')
    
    return parser.parse_args()


def main():
    """
    Função principal para teste e avaliação do modelo.
    """
    # Analisar argumentos
    args = parse_args()
    
    # Verificar GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Usando dispositivo: {device}")
    
    # Extrair características, se solicitado
    if args.extract_features:
        print(f"Extraindo características de {args.test_audio_dir}...")
        test_features = extract_features(args.test_audio_dir, args.test_features_dir, args.audio_ext)
        print(f"Características extraídas para {len(test_features)} arquivos")
    
    # Preparar DataLoader para teste
    print("Preparando DataLoader para teste...")
    test_loader = prepare_dataloader(
        args.test_features_dir, args.test_labels_file,
        args.batch_size, args.segment_length, args.stride, args.num_workers
    )
    
    # Carregar modelo
    print(f"Carregando modelo de {args.checkpoint}...")
    model, score_weights = load_model(args.checkpoint, device)
    model.eval()
    
    # Avaliar modelo
    print("Avaliando modelo...")
    test_results = evaluate_model(model, test_loader, score_weights, device, args.results_dir)
    
    # Analisar generalização, se solicitado
    if args.analyze_generalization:
        print("\nAnalisando capacidade de generalização do modelo...")
        
        # Preparar DataLoader para treinamento (para análise de generalização)
        train_loader = prepare_dataloader(
            args.train_features_dir, args.train_labels_file,
            args.batch_size, args.segment_length, args.stride, args.num_workers
        )
        
        # Analisar generalização
        generalization_dir = os.path.join(args.results_dir, 'generalization')
        os.makedirs(generalization_dir, exist_ok=True)
        
        generalization_results = analyze_generalization(
            model, train_loader, test_loader, 'Treinamento', 'Teste',
            score_weights, device, generalization_dir
        )
        
        # Realizar análise cruzada entre conjuntos de dados, se solicitado
        if args.cross_dataset:
            print("\nRealizando análise cruzada entre conjuntos de dados...")
            
            # Preparar DataLoader para o outro conjunto de dados
            cross_dataset_loader = prepare_dataloader(
                args.cross_dataset_features_dir, args.cross_dataset_labels_file,
                args.batch_size, args.segment_length, args.stride, args.num_workers
            )
            
            # Analisar generalização cruzada
            cross_dataset_dir = os.path.join(args.results_dir, 'cross_dataset')
            os.makedirs(cross_dataset_dir, exist_ok=True)
            
            cross_dataset_results = analyze_generalization(
                model, train_loader, cross_dataset_loader, 'ASVspoof2019', 'ASVspoof2021',
                score_weights, device, cross_dataset_dir
            )
    
    print("\nAvaliação concluída!")


if __name__ == "__main__":
    main()
