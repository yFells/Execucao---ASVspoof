#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script para integração da segmentação de dados no pipeline de treinamento,
demonstrando como usar diferentes estratégias para acelerar o treinamento.

Autores: André Thiago de Souza, Felipe de Lima dos Santos, 
         Juliano Gaona Proença, Matheus Henrique Reich Favarin Zagonel
"""

import os
import argparse
import json
import time
import numpy as np
import matplotlib.pyplot as plt
from data_segmentation import (
    DataSegmentationManager, 
    ProgressiveSegmentation,
    IntelligentDataSampler,
    create_segmented_dataset,
    create_segmented_training_pipeline
)


def benchmark_strategies(features_dir, labels_file, output_dir):
    """
    Executa benchmark completo das estratégias de segmentação.
    
    Args:
        features_dir: Diretório com características
        labels_file: Arquivo com rótulos
        output_dir: Diretório para salvar resultados
    """
    print("=== Benchmark das Estratégias de Segmentação ===")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Criar gerenciador
    manager = DataSegmentationManager(features_dir, labels_file, 
                                    os.path.join(output_dir, "cache"))
    
    # Estratégias a testar
    strategies = ['stratified', 'kmeans', 'dbscan', 'diversity', 'adaptive']
    sample_ratios = [0.1, 0.2, 0.3, 0.4, 0.5]
    
    results = []
    timing_results = []
    
    print("\nTestando estratégias...")
    
    for strategy in strategies:
        print(f"\n--- Testando estratégia: {strategy} ---")
        
        for ratio in sample_ratios:
            print(f"Ratio: {ratio}")
            
            try:
                # Medir tempo de execução
                start_time = time.time()
                selected_indices, stats = manager.apply_segmentation(
                    strategy, sample_ratio=ratio, save_results=False
                )
                execution_time = time.time() - start_time
                
                # Salvar resultados
                result = {
                    'strategy': strategy,
                    'sample_ratio': ratio,
                    'execution_time': execution_time,
                    'selected_count': stats['selected_count'],
                    'compression_ratio': stats['compression_ratio'],
                    'genuine_preservation': stats['distribution_preservation']['genuine_preservation'],
                    'spoof_preservation': stats['distribution_preservation']['spoof_preservation'],
                    'distribution_balance': abs(
                        stats['original_distribution']['genuine_ratio'] - 
                        stats['selected_distribution']['genuine_ratio']
                    )
                }
                
                results.append(result)
                timing_results.append({
                    'strategy': strategy,
                    'ratio': ratio,
                    'time': execution_time
                })
                
                print(f"  Tempo: {execution_time:.2f}s")
                print(f"  Compressão: {stats['compression_ratio']:.1%}")
                print(f"  Preservação G/S: {stats['distribution_preservation']['genuine_preservation']:.1%}/{stats['distribution_preservation']['spoof_preservation']:.1%}")
                
            except Exception as e:
                print(f"  Erro: {str(e)}")
    
    # Salvar resultados detalhados
    import pandas as pd
    
    results_df = pd.DataFrame(results)
    timing_df = pd.DataFrame(timing_results)
    
    results_path = os.path.join(output_dir, "benchmark_results.csv")
    timing_path = os.path.join(output_dir, "timing_results.csv")
    
    results_df.to_csv(results_path, index=False)
    timing_df.to_csv(timing_path, index=False)
    
    print(f"\nResultados salvos em:")
    print(f"  - Resultados: {results_path}")
    print(f"  - Tempos: {timing_path}")
    
    # Plotar análises
    plot_benchmark_results(results_df, timing_df, output_dir)
    
    return results_df, timing_df


def plot_benchmark_results(results_df, timing_df, output_dir):
    """
    Plota resultados do benchmark.
    
    Args:
        results_df: DataFrame com resultados
        timing_df: DataFrame com tempos de execução
        output_dir: Diretório para salvar gráficos
    """
    print("\nGerando gráficos de análise...")
    
    # Configurar estilo
    plt.style.use('seaborn-v0_8')
    
    # Gráfico 1: Tempo de Execução por Estratégia
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Tempo de execução
    ax1 = axes[0, 0]
    for strategy in timing_df['strategy'].unique():
        strategy_data = timing_df[timing_df['strategy'] == strategy]
        ax1.plot(strategy_data['ratio'], strategy_data['time'], 
                marker='o', label=strategy, linewidth=2)
    ax1.set_xlabel('Taxa de Amostragem')
    ax1.set_ylabel('Tempo de Execução (s)')
    ax1.set_title('Tempo de Execução por Estratégia')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Eficiência de compressão
    ax2 = axes[0, 1]
    for strategy in results_df['strategy'].unique():
        strategy_data = results_df[results_df['strategy'] == strategy]
        ax2.plot(strategy_data['sample_ratio'], strategy_data['compression_ratio'], 
                marker='s', label=strategy, linewidth=2)
    ax2.set_xlabel('Taxa de Amostragem')
    ax2.set_ylabel('Taxa de Compressão')
    ax2.set_title('Eficiência de Compressão')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Preservação de classes
    ax3 = axes[0, 2]
    for strategy in results_df['strategy'].unique():
        strategy_data = results_df[results_df['strategy'] == strategy]
        genuine_preservation = strategy_data['genuine_preservation']
        spoof_preservation = strategy_data['spoof_preservation']
        avg_preservation = (genuine_preservation + spoof_preservation) / 2
        ax3.plot(strategy_data['sample_ratio'], avg_preservation, 
                marker='^', label=strategy, linewidth=2)
    ax3.set_xlabel('Taxa de Amostragem')
    ax3.set_ylabel('Preservação Média das Classes')
    ax3.set_title('Preservação das Classes Originais')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Balance da distribuição
    ax4 = axes[1, 0]
    import seaborn as sns
    sns.boxplot(data=results_df, x='strategy', y='distribution_balance', ax=ax4)
    ax4.set_xlabel('Estratégia')
    ax4.set_ylabel('Desbalanceamento')
    ax4.set_title('Preservação do Balanceamento Original')
    ax4.tick_params(axis='x', rotation=45)
    
    # Eficiência vs Qualidade (scatter)
    ax5 = axes[1, 1]
    scatter = ax5.scatter(results_df['execution_time'], 
                         (results_df['genuine_preservation'] + results_df['spoof_preservation'])/2,
                         c=results_df['compression_ratio'], 
                         s=60, alpha=0.7, cmap='viridis')
    ax5.set_xlabel('Tempo de Execução (s)')
    ax5.set_ylabel('Preservação Média')
    ax5.set_title('Eficiência vs Qualidade')
    plt.colorbar(scatter, ax=ax5, label='Taxa de Compressão')
    
    # Ranking de estratégias
    ax6 = axes[1, 2]
    # Calcular score combinado (menor é melhor)
    results_df['combined_score'] = (
        results_df['execution_time'] / results_df['execution_time'].max() +
        results_df['distribution_balance'] +
        (1 - (results_df['genuine_preservation'] + results_df['spoof_preservation'])/2)
    )
    
    strategy_scores = results_df.groupby('strategy')['combined_score'].mean().sort_values()
    ax6.barh(range(len(strategy_scores)), strategy_scores.values)
    ax6.set_yticks(range(len(strategy_scores)))
    ax6.set_yticklabels(strategy_scores.index)
    ax6.set_xlabel('Score Combinado (menor = melhor)')
    ax6.set_title('Ranking Geral das Estratégias')
    
    plt.tight_layout()
    
    # Salvar gráfico
    plot_path = os.path.join(output_dir, 'benchmark_analysis.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Gráfico salvo em: {plot_path}")
    
    plt.show()


def demonstrate_progressive_training(features_dir, labels_file, output_dir):
    """
    Demonstra treinamento progressivo com segmentação.
    
    Args:
        features_dir: Diretório com características
        labels_file: Arquivo com rótulos
        output_dir: Diretório para salvar resultados
    """
    print("\n=== Demonstração de Treinamento Progressivo ===")
    
    # Criar segmentação progressiva
    progressive = ProgressiveSegmentation(features_dir, labels_file, n_stages=5)
    
    # Testar diferentes estratégias
    strategies = ['complexity', 'diversity', 'balanced']
    
    for strategy in strategies:
        print(f"\n--- Estratégia: {strategy} ---")
        
        # Criar estágios
        stages = progressive.create_progressive_stages(strategy=strategy)
        cumulative_stages = progressive.get_cumulative_stages()
        
        # Plotar crescimento dos estágios
        stage_sizes = [len(stage) for stage in cumulative_stages]
        
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(stage_sizes) + 1), stage_sizes, 
                marker='o', linewidth=3, markersize=8)
        plt.xlabel('Estágio')
        plt.ylabel('Número Total de Amostras')
        plt.title(f'Crescimento Progressivo - Estratégia: {strategy}')
        plt.grid(True, alpha=0.3)
        
        # Adicionar anotações
        for i, size in enumerate(stage_sizes):
            plt.annotate(f'{size}', (i+1, size), 
                        textcoords="offset points", xytext=(0,10), ha='center')
        
        # Salvar gráfico
        plot_path = os.path.join(output_dir, f'progressive_{strategy}.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Gráfico salvo em: {plot_path}")
        
        plt.show()


def demonstrate_intelligent_sampling(features_dir, labels_file, output_dir):
    """
    Demonstra amostragem inteligente baseada em dificuldade.
    
    Args:
        features_dir: Diretório com características
        labels_file: Arquivo com rótulos
        output_dir: Diretório para salvar resultados
    """
    print("\n=== Demonstração de Amostragem Inteligente ===")
    
    # Criar amostrador inteligente
    sampler = IntelligentDataSampler(features_dir, labels_file)
    
    # Carregar dados para análise
    manager = DataSegmentationManager(features_dir, labels_file)
    features, labels, file_names = manager.load_data()
    
    # Calcular dificuldade das amostras
    difficulty_scores = sampler.calculate_sample_difficulty(features, labels)
    
    # Plotar distribuição de dificuldade
    plt.figure(figsize=(12, 8))
    
    # Subplot 1: Histograma de dificuldade
    plt.subplot(2, 2, 1)
    plt.hist(difficulty_scores, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    plt.xlabel('Pontuação de Dificuldade')
    plt.ylabel('Frequência')
    plt.title('Distribuição de Dificuldade das Amostras')
    plt.grid(True, alpha=0.3)
    
    # Subplot 2: Dificuldade por classe
    plt.subplot(2, 2, 2)
    genuine_diff = difficulty_scores[labels == 0]
    spoof_diff = difficulty_scores[labels == 1]
    
    plt.boxplot([genuine_diff, spoof_diff], labels=['Genuine', 'Spoof'])
    plt.ylabel('Pontuação de Dificuldade')
    plt.title('Dificuldade por Classe')
    plt.grid(True, alpha=0.3)
    
    # Subplot 3: Comparação de estratégias de amostragem
    plt.subplot(2, 2, 3)
    ratios = [0.1, 0.2, 0.3, 0.4, 0.5]
    
    random_preservations = []
    intelligent_preservations = []
    
    for ratio in ratios:
        # Amostragem aleatória
        n_samples = int(len(features) * ratio)
        random_indices = np.random.choice(len(features), n_samples, replace=False)
        random_genuine = np.sum(labels[random_indices] == 0)
        random_preservation = random_genuine / np.sum(labels == 0)
        random_preservations.append(random_preservation)
        
        # Amostragem inteligente
        intelligent_indices = sampler.adaptive_sampling(ratio, difficulty_weight=0.7)
        intelligent_genuine = np.sum(labels[intelligent_indices] == 0)
        intelligent_preservation = intelligent_genuine / np.sum(labels == 0)
        intelligent_preservations.append(intelligent_preservation)
    
    plt.plot(ratios, random_preservations, 'o-', label='Aleatória', linewidth=2)
    plt.plot(ratios, intelligent_preservations, 's-', label='Inteligente', linewidth=2)
    plt.xlabel('Taxa de Amostragem')
    plt.ylabel('Preservação de Classe Genuine')
    plt.title('Comparação: Aleatória vs Inteligente')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Subplot 4: Análise temporal de complexidade
    plt.subplot(2, 2, 4)
    # Simular como a dificuldade muda ao longo do tempo (por índice de arquivo)
    sorted_indices = np.argsort([int(name.split('_')[-1]) if '_' in name else 0 for name in file_names])
    rolling_difficulty = []
    window_size = len(file_names) // 20  # 20 janelas
    
    for i in range(0, len(sorted_indices), window_size):
        window_indices = sorted_indices[i:i+window_size]
        if len(window_indices) > 0:
            avg_difficulty = np.mean(difficulty_scores[window_indices])
            rolling_difficulty.append(avg_difficulty)
    
    plt.plot(range(len(rolling_difficulty)), rolling_difficulty, 'g-', linewidth=2, marker='o')
    plt.xlabel('Janela Temporal')
    plt.ylabel('Dificuldade Média')
    plt.title('Evolução da Dificuldade')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Salvar gráfico
    plot_path = os.path.join(output_dir, 'intelligent_sampling_analysis.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Análise de amostragem inteligente salva em: {plot_path}")
    
    plt.show()


def estimate_training_speedup(original_size, segmented_size, complexity_factor=2.0):
    """
    Estima o ganho de velocidade no treinamento com dados segmentados.
    
    Args:
        original_size: Tamanho original do dataset
        segmented_size: Tamanho do dataset segmentado
        complexity_factor: Fator de complexidade do algoritmo (padrão: quadrático)
        
    Returns:
        Estimativa de speedup e tempo economizado
    """
    # Calcular speedup baseado na complexidade do algoritmo
    original_complexity = original_size ** complexity_factor
    segmented_complexity = segmented_size ** complexity_factor
    
    speedup = original_complexity / segmented_complexity
    time_reduction = 1 - (1 / speedup)
    
    return speedup, time_reduction


def create_recommendation_report(benchmark_results, output_dir):
    """
    Cria relatório com recomendações baseado nos resultados do benchmark.
    
    Args:
        benchmark_results: DataFrame com resultados do benchmark
        output_dir: Diretório para salvar o relatório
    """
    print("\n=== Gerando Relatório de Recomendações ===")
    
    # Analisar resultados
    best_overall = benchmark_results.groupby('strategy').agg({
        'execution_time': 'mean',
        'compression_ratio': 'mean',
        'genuine_preservation': 'mean',
        'spoof_preservation': 'mean',
        'distribution_balance': 'mean'
    }).round(4)
    
    # Calcular scores normalizados
    best_overall['time_score'] = 1 - (best_overall['execution_time'] / best_overall['execution_time'].max())
    best_overall['quality_score'] = (best_overall['genuine_preservation'] + best_overall['spoof_preservation']) / 2
    best_overall['balance_score'] = 1 - best_overall['distribution_balance']
    best_overall['efficiency_score'] = best_overall['compression_ratio']
    
    # Score combinado
    best_overall['final_score'] = (
        0.2 * best_overall['time_score'] +
        0.3 * best_overall['quality_score'] +
        0.3 * best_overall['balance_score'] +
        0.2 * best_overall['efficiency_score']
    )
    
    # Ordenar por score final
    best_overall = best_overall.sort_values('final_score', ascending=False)
    
    # Criar relatório
    report = []
    report.append("# RELATÓRIO DE RECOMENDAÇÕES - SEGMENTAÇÃO DE DADOS\n")
    report.append("## Resumo Executivo\n")
    report.append("Este relatório analisa diferentes estratégias de segmentação de dados para")
    report.append("acelerar o treinamento de modelos de detecção de ataques de replay.\n")
    
    report.append("## Ranking das Estratégias\n")
    for i, (strategy, row) in enumerate(best_overall.iterrows(), 1):
        report.append(f"### {i}. {strategy.upper()}")
        report.append(f"- **Score Final**: {row['final_score']:.3f}")
        report.append(f"- **Tempo Médio**: {row['execution_time']:.2f}s")
        report.append(f"- **Compressão Média**: {row['compression_ratio']:.1%}")
        report.append(f"- **Preservação Genuine**: {row['genuine_preservation']:.1%}")
        report.append(f"- **Preservação Spoof**: {row['spoof_preservation']:.1%}")
        report.append(f"- **Equilíbrio de Distribuição**: {1-row['distribution_balance']:.1%}\n")
    
    report.append("## Recomendações por Cenário\n")
    
    # Recomendações específicas
    fastest = best_overall.index[best_overall['time_score'].argmax()]
    most_balanced = best_overall.index[best_overall['balance_score'].argmax()]
    highest_quality = best_overall.index[best_overall['quality_score'].argmax()]
    
    report.append(f"### Para Prototipagem Rápida")
    report.append(f"**Recomendação**: {fastest}")
    report.append(f"- Menor tempo de execução")
    report.append(f"- Ideal para testes rápidos e iterações iniciais")
    report.append(f"- Taxa de compressão: {best_overall.loc[fastest, 'compression_ratio']:.1%}\n")
    
    report.append(f"### Para Preservação de Qualidade")
    report.append(f"**Recomendação**: {highest_quality}")
    report.append(f"- Melhor preservação das características originais")
    report.append(f"- Ideal para treinamento de modelos finais")
    report.append(f"- Preservação média: {best_overall.loc[highest_quality, 'quality_score']:.1%}\n")
    
    report.append(f"### Para Datasets Desbalanceados")
    report.append(f"**Recomendação**: {most_balanced}")
    report.append(f"- Melhor preservação do balanceamento original")
    report.append(f"- Ideal quando a proporção de classes é crítica")
    report.append(f"- Score de balanceamento: {best_overall.loc[most_balanced, 'balance_score']:.1%}\n")
    
    report.append("## Estimativas de Speedup\n")
    
    # Calcular estimativas de speedup para diferentes tamanhos
    dataset_sizes = [1000, 5000, 10000, 50000, 100000]
    report.append("| Tamanho Original | Compressão 30% | Speedup | Tempo Economizado |")
    report.append("|------------------|----------------|---------|-------------------|")
    
    for size in dataset_sizes:
        segmented_size = int(size * 0.3)
        speedup, time_reduction = estimate_training_speedup(size, segmented_size)
        report.append(f"| {size:,} | {segmented_size:,} | {speedup:.1f}x | {time_reduction:.1%} |")
    
    report.append("\n## Diretrizes de Implementação\n")
    report.append("### 1. Escolha da Estratégia")
    report.append("- **Pequenos datasets (<5k)**: Stratified Sampling")
    report.append("- **Datasets médios (5k-50k)**: K-Means Clustering")
    report.append("- **Grandes datasets (>50k)**: Adaptive Segmentation")
    report.append("- **Datasets complexos**: Intelligent Sampling\n")
    
    report.append("### 2. Taxa de Compressão Recomendada")
    report.append("- **Prototipagem**: 10-20% dos dados originais")
    report.append("- **Desenvolvimento**: 30-50% dos dados originais")
    report.append("- **Treinamento final**: 70-90% dos dados originais\n")
    
    report.append("### 3. Monitoramento")
    report.append("- Acompanhar métricas de qualidade do modelo")
    report.append("- Validar preservação de características críticas")
    report.append("- Ajustar estratégia baseado no desempenho\n")
    
    report.append("## Código de Implementação\n")
    report.append("```python")
    report.append("# Exemplo de uso das estratégias recomendadas")
    report.append("from data_segmentation import DataSegmentationManager")
    report.append("")
    report.append("# Criar gerenciador")
    report.append("manager = DataSegmentationManager(features_dir, labels_file)")
    report.append("")
    report.append("# Aplicar estratégia recomendada")
    report.append(f"indices, stats = manager.apply_segmentation('{best_overall.index[0]}', sample_ratio=0.3)")
    report.append("")
    report.append("# Criar dataset segmentado")
    report.append("create_segmented_dataset(original_dir, output_dir, indices)")
    report.append("```")
    
    # Salvar relatório
    report_path = os.path.join(output_dir, 'segmentation_recommendations.md')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report))
    
    print(f"Relatório de recomendações salvo em: {report_path}")
    
    return report_path


def main():
    """
    Função principal para demonstração completa da segmentação de dados.
    """
    parser = argparse.ArgumentParser(description="Demonstração de Segmentação de Dados")
    
    parser.add_argument('--features-dir', type=str, required=True,
                        help='Diretório com características extraídas')
    parser.add_argument('--labels-file', type=str, required=True,
                        help='Arquivo com rótulos')
    parser.add_argument('--output-dir', type=str, default='segmentation_analysis',
                        help='Diretório para salvar resultados')
    parser.add_argument('--mode', type=str, default='all',
                        choices=['benchmark', 'progressive', 'intelligent', 'all'],
                        help='Modo de análise a executar')
    parser.add_argument('--quick', action='store_true',
                        help='Executar análise rápida (menos estratégias)')
    
    args = parser.parse_args()
    
    # Criar diretório de saída
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("=== ANÁLISE DE SEGMENTAÇÃO DE DADOS ===")
    print(f"Características: {args.features_dir}")
    print(f"Rótulos: {args.labels_file}")
    print(f"Saída: {args.output_dir}")
    print(f"Modo: {args.mode}")
    
    # Executar análises baseado no modo
    if args.mode in ['benchmark', 'all']:
        print("\n" + "="*50)
        print("EXECUTANDO BENCHMARK DE ESTRATÉGIAS")
        print("="*50)
        
        results_df, timing_df = benchmark_strategies(
            args.features_dir, args.labels_file, args.output_dir
        )
        
        # Gerar relatório de recomendações
        create_recommendation_report(results_df, args.output_dir)
    
    if args.mode in ['progressive', 'all']:
        print("\n" + "="*50)
        print("DEMONSTRANDO TREINAMENTO PROGRESSIVO")
        print("="*50)
        
        demonstrate_progressive_training(
            args.features_dir, args.labels_file, args.output_dir
        )
    
    if args.mode in ['intelligent', 'all']:
        print("\n" + "="*50)
        print("DEMONSTRANDO AMOSTRAGEM INTELIGENTE")
        print("="*50)
        
        demonstrate_intelligent_sampling(
            args.features_dir, args.labels_file, args.output_dir
        )
    
    print("\n" + "="*50)
    print("ANÁLISE CONCLUÍDA")
    print("="*50)
    print(f"Todos os resultados salvos em: {args.output_dir}")


# Exemplo de uso direto para integração com pipeline existente
def quick_segmentation_example():
    """
    Exemplo rápido de como usar a segmentação no pipeline existente.
    """
    print("=== EXEMPLO RÁPIDO DE SEGMENTAÇÃO ===")
    
    # Configurações de exemplo
    config_file = "config.json"
    features_dir = "features/train"
    labels_file = "labels/train_labels.txt"
    
    # 1. Análise rápida das estratégias
    manager = DataSegmentationManager(features_dir, labels_file)
    
    # 2. Aplicar estratégia adaptativa (recomendada para uso geral)
    print("Aplicando segmentação adaptativa...")
    selected_indices, stats = manager.apply_segmentation('adaptive', sample_ratio=0.3)
    
    print(f"Resultado da segmentação:")
    print(f"- Dados originais: {stats['original_count']}")
    print(f"- Dados selecionados: {stats['selected_count']}")
    print(f"- Compressão: {stats['compression_ratio']:.1%}")
    print(f"- Speedup estimado: {estimate_training_speedup(stats['original_count'], stats['selected_count'])[0]:.1f}x")
    
    # 3. Criar dataset segmentado para uso no treinamento
    output_features_dir = "features/train_segmented"
    output_labels_file = "labels/train_segmented.txt"
    
    _, _, file_names = manager.load_data()
    create_segmented_dataset(
        features_dir, labels_file, output_features_dir, output_labels_file,
        selected_indices, file_names
    )
    
    print(f"\nDataset segmentado criado:")
    print(f"- Características: {output_features_dir}")
    print(f"- Rótulos: {output_labels_file}")
    print(f"\nPara usar no treinamento, modifique os caminhos no script train.py:")
    print(f"--train-features-dir {output_features_dir}")
    print(f"--train-labels-file {output_labels_file}")


if __name__ == "__main__":
    # Verificar se argumentos foram fornecidos
    import sys
    
    if len(sys.argv) == 1:
        # Se nenhum argumento foi fornecido, executar exemplo rápido
        quick_segmentation_example()
    else:
        # Executar com argumentos da linha de comando
        main()