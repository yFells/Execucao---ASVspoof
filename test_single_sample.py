#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script para testar uma única amostra de áudio com o modelo treinado de detecção de ataques spoofing de replay.

Autores: André Thiago de Souza, Felipe de Lima dos Santos, 
         Juliano Gaona Proença, Matheus Henrique Reich Favarin Zagonel
"""

import os
import sys
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from feature_extraction_fix import FeatureExtractor
from model import MultiPatternModel


def load_model(checkpoint_path, device):
    """
    Carrega o modelo a partir de um checkpoint.
    
    Args:
        checkpoint_path: Caminho para o arquivo de checkpoint
        device: Dispositivo para carregar o modelo (CPU ou GPU)
        
    Returns:
        Modelo carregado
    """
    try:
        # Criar modelo
        model = MultiPatternModel(input_channels=1, hidden_size=512, num_classes=2).to(device)
        
        # Carregar checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Carregar estado do modelo
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Colocar o modelo em modo de avaliação
        model.eval()
        
        print(f"Modelo carregado de: {checkpoint_path}")
        
        # Extrair pesos para fusão de pontuações
        score_weights = checkpoint.get('score_weights', {'lbp': 0.33, 'glcm': 0.33, 'lpq': 0.34})
        
        return model, score_weights
    except Exception as e:
        print(f"Erro ao carregar o modelo: {str(e)}")
        return None, None


def extract_features(audio_path):
    """
    Extrai características de um arquivo de áudio.
    
    Args:
        audio_path: Caminho para o arquivo de áudio
        
    Returns:
        Dicionário com características extraídas
    """
    try:
        # Criar extrator de características
        extractor = FeatureExtractor()
        
        # Extrair características
        features = extractor.extract_hybrid_features(audio_path)
        
        if features is None:
            print(f"Erro: Não foi possível extrair características de {audio_path}")
            return None
        
        return features
    except Exception as e:
        print(f"Erro ao extrair características: {str(e)}")
        return None


def predict(model, features, score_weights, device):
    """
    Faz previsão com o modelo para as características extraídas.
    
    Args:
        model: Modelo carregado
        features: Características extraídas
        score_weights: Pesos para fusão de pontuações
        device: Dispositivo para execução (CPU ou GPU)
        
    Returns:
        Resultado da previsão
    """
    try:
        # Preparar dados para o modelo
        lbp = torch.FloatTensor(features['lbp']).unsqueeze(0).unsqueeze(0).to(device)  # [1, 1, H, W]
        glcm = torch.FloatTensor(features['glcm']).unsqueeze(0).to(device)  # [1, D]
        lpq = torch.FloatTensor(features['lpq']).unsqueeze(0).unsqueeze(0).to(device)  # [1, 1, H, W]
        
        # Fazer previsão
        with torch.no_grad():
            # Obter pontuações
            scores = model.get_scores(lbp, glcm, lpq)
            
            # Calcular pontuação fundida
            fused_score = (
                score_weights['lbp'] * scores['lbp_score'] +
                score_weights['glcm'] * scores['glcm_score'] +
                score_weights['lpq'] * scores['lpq_score']
            )
            
            # Transferir para CPU e converter para numpy
            scores_np = {
                'lbp': scores['lbp_score'].cpu().numpy()[0],
                'glcm': scores['glcm_score'].cpu().numpy()[0],
                'lpq': scores['lpq_score'].cpu().numpy()[0],
                'fused': fused_score.cpu().numpy()[0]
            }
            
            return scores_np
    except Exception as e:
        print(f"Erro ao fazer previsão: {str(e)}")
        return None


def interpret_result(scores):
    """
    Interpreta o resultado da previsão.
    
    Args:
        scores: Pontuações do modelo
        
    Returns:
        Interpretação do resultado
    """
    # Limiar de decisão (ajustar conforme necessário)
    threshold = 0.5
    
    # Verificar se a pontuação fundida está acima do limiar
    is_spoof = scores['fused'] >= threshold
    
    # Calcular confiança
    confidence = abs(scores['fused'] - 0.5) * 2  # Normalizar para 0-1
    
    # Formatar pontuações
    formatted_scores = {k: f"{v:.4f}" for k, v in scores.items()}
    
    return {
        'prediction': 'SPOOF' if is_spoof else 'GENUINE',
        'confidence': f"{confidence:.2%}",
        'scores': formatted_scores
    }


def plot_features(features, output_path=None):
    """
    Plota as características extraídas.
    
    Args:
        features: Características extraídas
        output_path: Caminho para salvar o gráfico (opcional)
    """
    plt.figure(figsize=(18, 10))
    
    # Plotar espectrograma Mel
    plt.subplot(2, 3, 1)
    plt.imshow(features['mel_spectrogram'], aspect='auto', origin='lower')
    plt.title('Espectrograma Mel')
    plt.xlabel('Tempo')
    plt.ylabel('Frequência')
    plt.colorbar(format='%+2.0f dB')
    
    # Plotar MFCC
    plt.subplot(2, 3, 2)
    plt.imshow(features['mfcc'], aspect='auto', origin='lower')
    plt.title('MFCC')
    plt.xlabel('Tempo')
    plt.ylabel('Coeficiente')
    plt.colorbar()
    
    # Plotar CQCC
    plt.subplot(2, 3, 3)
    plt.imshow(features['cqcc'], aspect='auto', origin='lower')
    plt.title('CQCC')
    plt.xlabel('Tempo')
    plt.ylabel('Coeficiente')
    plt.colorbar()
    
    # Plotar LBP
    plt.subplot(2, 3, 4)
    plt.imshow(features['lbp'], aspect='auto', origin='lower', cmap='gray')
    plt.title('LBP')
    plt.xlabel('Tempo')
    plt.ylabel('Frequência')
    plt.colorbar()
    
    # Plotar LPQ
    plt.subplot(2, 3, 5)
    plt.imshow(features['lpq'], aspect='auto', origin='lower', cmap='gray')
    plt.title('LPQ')
    plt.xlabel('Tempo')
    plt.ylabel('Frequência')
    plt.colorbar()
    
    plt.tight_layout()
    
    # Salvar gráfico, se solicitado
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def main():
    """
    Função principal.
    """
    # Configurar parser de argumentos
    parser = argparse.ArgumentParser(description="Teste de detecção de ataques spoofing de replay para uma única amostra")
    
    parser.add_argument('--audio', type=str, required=True,
                        help='Caminho para o arquivo de áudio')
    parser.add_argument('--model', type=str, required=True,
                        help='Caminho para o arquivo de checkpoint do modelo')
    parser.add_argument('--plot', action='store_true',
                        help='Plotar características extraídas')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Diretório para salvar resultados')
    
    # Analisar argumentos
    args = parser.parse_args()
    
    # Verificar se o arquivo de áudio existe
    if not os.path.exists(args.audio):
        print(f"Erro: Arquivo de áudio não encontrado: {args.audio}")
        return 1
    
    # Verificar se o arquivo de checkpoint existe
    if not os.path.exists(args.model):
        print(f"Erro: Arquivo de checkpoint não encontrado: {args.model}")
        return 1
    
    # Criar diretório de saída, se solicitado
    if args.output_dir and not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    # Verificar GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Usando dispositivo: {device}")
    
    # Carregar modelo
    model, score_weights = load_model(args.model, device)
    if model is None:
        return 1
    
    # Extrair características
    print(f"Extraindo características de: {args.audio}")
    features = extract_features(args.audio)
    if features is None:
        return 1
    
    # Plotar características, se solicitado
    if args.plot:
        plot_path = os.path.join(args.output_dir, 'features.png') if args.output_dir else None
        plot_features(features, plot_path)
    
    # Fazer previsão
    print("Fazendo previsão...")
    scores = predict(model, features, score_weights, device)
    if scores is None:
        return 1
    
    # Interpretar resultado
    result = interpret_result(scores)
    
    # Exibir resultado
    print("\n=== Resultado ===")
    print(f"Arquivo: {args.audio}")
    print(f"Previsão: {result['prediction']}")
    print(f"Confiança: {result['confidence']}")
    print("\nPontuações:")
    for key, value in result['scores'].items():
        print(f"  {key}: {value}")
    
    # Salvar resultado, se solicitado
    if args.output_dir:
        result_path = os.path.join(args.output_dir, 'result.txt')
        with open(result_path, 'w') as f:
            f.write(f"Arquivo: {args.audio}\n")
            f.write(f"Previsão: {result['prediction']}\n")
            f.write(f"Confiança: {result['confidence']}\n")
            f.write("\nPontuações:\n")
            for key, value in result['scores'].items():
                f.write(f"  {key}: {value}\n")
        
        print(f"\nResultado salvo em: {result_path}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
