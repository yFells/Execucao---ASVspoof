#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Módulo para análise da capacidade de generalização do modelo para ataques desconhecidos
em sistemas de verificação automática de locutor.

Autores: André Thiago de Souza, Felipe de Lima dos Santos, 
         Juliano Gaona Proença, Matheus Henrique Reich Favarin Zagonel
"""

import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from tqdm import tqdm
from collections import defaultdict

from evaluation import ReplayAttackEvaluator


class GeneralizationAnalyzer:
    """
    Analisador para avaliar a capacidade de generalização do modelo para ataques desconhecidos.
    """
    
    def __init__(self, model, device=None, score_weights=None):
        """
        Inicializa o analisador.
        
        Args:
            model: Modelo treinado para detecção
            device: Dispositivo para execução (CPU ou GPU)
            score_weights: Pesos para fusão das pontuações dos diferentes ramos
        """
        self.model = model
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.evaluator = ReplayAttackEvaluator(model, device, score_weights)
    
    def extract_embeddings(self, dataloader, layer_name='fc1'):
        """
        Extrai embeddings de uma camada específica do modelo.
        
        Args:
            dataloader: DataLoader com dados para extração
            layer_name: Nome da camada para extração de embeddings
            
        Returns:
            Embeddings, pontuações e rótulos
        """
        self.model.eval()
        
        all_embeddings = []
        all_scores = {
            'lbp': [],
            'glcm': [],
            'lpq': [],
            'fused': []
        }
        all_labels = []
        all_attack_types = []
        
        # Registrar hook para extrair embeddings
        embeddings = []
        
        def hook_fn(module, input, output):
            embeddings.append(output.detach().cpu().numpy())
        
        # Encontrar o módulo apropriado para o hook
        # Aqui assumimos uma implementação específica do modelo
        if layer_name == 'fc1':
            hook = self.model.lbp_branch.resnet.fc1.register_forward_hook(hook_fn)
        else:
            raise ValueError(f"Camada {layer_name} não suportada para extração de embeddings")
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Extraindo embeddings"):
                # Desempacotar batch
                if len(batch) == 4:  # (lbp, glcm, lpq, labels)
                    lbp, glcm, lpq, labels = batch
                    attack_types = None
                else:  # (lbp, glcm, lpq, labels, attack_types)
                    lbp, glcm, lpq, labels, attack_types = batch
                
                # Mover dados para o dispositivo
                lbp = lbp.to(self.device)
                glcm = glcm.to(self.device)
                lpq = lpq.to(self.device)
                
                # Obter pontuações
                scores = self.model.get_scores(lbp, glcm, lpq)
                
                # Calcular pontuação fundida
                if self.evaluator.score_weights is not None:
                    fused_scores = (
                        self.evaluator.score_weights['lbp'] * scores['lbp_score'] +
                        self.evaluator.score_weights['glcm'] * scores['glcm_score'] +
                        self.evaluator.score_weights['lpq'] * scores['lpq_score']
                    )
                else:
                    # Pesos iguais
                    fused_scores = (scores['lbp_score'] + scores['glcm_score'] + scores['lpq_score']) / 3
                
                # Armazenar pontuações, rótulos e tipos de ataque
                all_scores['lbp'].extend(scores['lbp_score'].cpu().numpy())
                all_scores['glcm'].extend(scores['glcm_score'].cpu().numpy())
                all_scores['lpq'].extend(scores['lpq_score'].cpu().numpy())
                all_scores['fused'].extend(fused_scores.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
                if attack_types is not None:
                    all_attack_types.extend(attack_types)
        
        # Coletar os embeddings
        all_embeddings = np.vstack(embeddings)
        
        # Remover hook
        hook.remove()
        
        # Converter pontuações para arrays numpy
        for key in all_scores:
            all_scores[key] = np.array(all_scores[key])
        
        # Converter rótulos para array numpy
        all_labels = np.array(all_labels)
        
        return {
            'embeddings': all_embeddings,
            'scores': all_scores,
            'labels': all_labels,
            'attack_types': all_attack_types if all_attack_types else None
        }
    
    def visualize_embeddings(self, embeddings, labels, attack_types=None, method='tsne', save_path=None):
        """
        Visualiza embeddings usando redução de dimensionalidade.
        
        Args:
            embeddings: Embeddings extraídos
            labels: Rótulos dos dados
            attack_types: Tipos de ataque (opcional)
            method: Método de redução de dimensionalidade ('tsne' ou 'pca')
            save_path: Caminho para salvar o gráfico (opcional)
            
        Returns:
            Figura com visualização de embeddings
        """
        # Redução de dimensionalidade
        if method == 'tsne':
            print("Aplicando t-SNE para redução de dimensionalidade...")
            reducer = TSNE(n_components=2, random_state=42)
        else:  # pca
            print("Aplicando PCA para redução de dimensionalidade...")
            reducer = PCA(n_components=2, random_state=42)
        
        # Aplicar redução de dimensionalidade
        reduced_embeddings = reducer.fit_transform(embeddings)
        
        # Criar DataFrame para plotagem
        df = pd.DataFrame({
            'x': reduced_embeddings[:, 0],
            'y': reduced_embeddings[:, 1],
            'label': ['Genuine' if l == 0 else 'Spoof' for l in labels]
        })
        
        # Adicionar tipo de ataque, se fornecido
        if attack_types is not None:
            df['attack_type'] = attack_types
        
        # Criar figura
        plt.figure(figsize=(12, 10))
        
        # Plotar embeddings
        if attack_types is not None:
            # Colorir por tipo de ataque
            sns.scatterplot(
                data=df, x='x', y='y', hue='attack_type', style='label',
                palette='tab10', s=100, alpha=0.7
            )
        else:
            # Colorir apenas por rótulo (genuine/spoof)
            sns.scatterplot(
                data=df, x='x', y='y', hue='label',
                palette={'Genuine': 'green', 'Spoof': 'red'}, s=100, alpha=0.7
            )
        
        # Configurar o gráfico
        plt.title(f'Visualização de Embeddings ({method.upper()})')
        plt.xlabel(f'Componente 1')
        plt.ylabel(f'Componente 2')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        # Salvar o gráfico, se solicitado
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return plt.gcf()
    
    def analyze_attack_types(self, dataloader, results_dir=None):
        """
        Analisa o desempenho do modelo para diferentes tipos de ataques.
        
        Args:
            dataloader: DataLoader com dados para análise
            results_dir: Diretório para salvar resultados (opcional)
            
        Returns:
            DataFrame com análise por tipo de ataque
        """
        # Extrair embeddings, pontuações e tipos de ataque
        data = self.extract_embeddings(dataloader)
        embeddings = data['embeddings']
        scores = data['scores']
        labels = data['labels']
        attack_types = data['attack_types']
        
        # Verificar se temos informações de tipo de ataque
        if attack_types is None:
            raise ValueError("Informações de tipo de ataque não disponíveis no dataloader")
        
        # Calcular EER e t-DCF para cada tipo de ataque
        attack_analysis = defaultdict(dict)
        unique_attacks = np.unique([at for at in attack_types if at != 'genuine'])
        
        for attack in unique_attacks:
            # Selecionar apenas amostras genuínas e do tipo de ataque atual
            attack_indices = [i for i, at in enumerate(attack_types) if at == attack or at == 'genuine']
            attack_labels = [1 if at == attack else 0 for at in [attack_types[i] for i in attack_indices]]
            
            # Extrair pontuações para o tipo de ataque atual
            attack_scores = {
                score_type: scores[score_type][attack_indices]
                for score_type in scores
            }
            
            # Calcular métricas para cada tipo de pontuação
            for score_type in attack_scores:
                eer, eer_threshold = self.evaluator.compute_eer(attack_scores[score_type], attack_labels)
                tdcf, tdcf_threshold = self.evaluator.compute_tdcf(attack_scores[score_type], attack_labels)
                
                attack_analysis[attack][f'{score_type}_eer'] = eer
                attack_analysis[attack][f'{score_type}_tdcf'] = tdcf
        
        # Converter para DataFrame
        attack_df = pd.DataFrame.from_dict(attack_analysis, orient='index')
        
        # Ordenar por EER fundido
        attack_df = attack_df.sort_values('fused_eer')
        
        # Exibir resumo
        print("Análise por tipo de ataque:")
        print(attack_df)
        
        # Plotar resultados
        if results_dir and not os.path.exists(results_dir):
            os.makedirs(results_dir)
        
        # Plotar EER por tipo de ataque
        plt.figure(figsize=(12, 8))
        
        # Selecionar colunas de EER
        eer_cols = [col for col in attack_df.columns if 'eer' in col]
        eer_df = attack_df[eer_cols].copy()
        
        # Renomear colunas para melhor visualização
        eer_df.columns = [col.replace('_eer', '') for col in eer_df.columns]
        
        # Plotar
        eer_df.plot(kind='bar', figsize=(12, 8))
        plt.title('Equal Error Rate (EER) por Tipo de Ataque')
        plt.ylabel('EER')
        plt.xlabel('Tipo de Ataque')
        plt.xticks(rotation=45, ha='right')
        plt.grid(True, axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        if results_dir:
            plt.savefig(os.path.join(results_dir, 'eer_by_attack_type.png'), dpi=300, bbox_inches='tight')
        
        # Plotar visualização de embeddings
        self.visualize_embeddings(
            embeddings, labels, attack_types, method='tsne',
            save_path=os.path.join(results_dir, 'embeddings_visualization.png') if results_dir else None
        )
        
        return attack_df
    
    def cross_attack_analysis(self, train_dataloader, test_dataloader, train_attacks, test_attacks):
        """
        Realiza análise cruzada entre diferentes tipos de ataques.
        
        Args:
            train_dataloader: DataLoader com dados de treinamento
            test_dataloader: DataLoader com dados de teste
            train_attacks: Lista de tipos de ataque no conjunto de treinamento
            test_attacks: Lista de tipos de ataque no conjunto de teste
            
        Returns:
            DataFrame com resultados da análise cruzada
        """
        # Avaliar no conjunto de treinamento
        print("Avaliando no conjunto de treinamento...")
        train_results = self.evaluator.evaluate(train_dataloader)
        
        # Avaliar no conjunto de teste
        print("Avaliando no conjunto de teste...")
        test_results = self.evaluator.evaluate(test_dataloader)
        
        # Extrair embeddings para visualização
        train_data = self.extract_embeddings(train_dataloader)
        test_data = self.extract_embeddings(test_dataloader)
        
        # Criar DataFrame com resultados
        results = {
            'train_eer': train_results['eer']['fused']['value'],
            'test_eer': test_results['eer']['fused']['value'],
            'train_tdcf': train_results['tdcf']['fused']['value'],
            'test_tdcf': test_results['tdcf']['fused']['value'],
            'train_attacks': train_attacks,
            'test_attacks': test_attacks,
            'unseen_attacks': [a for a in test_attacks if a not in train_attacks]
        }
        
        # Calcular a diferença de desempenho
        results['eer_degradation'] = results['test_eer'] - results['train_eer']
        results['tdcf_degradation'] = results['test_tdcf'] - results['train_tdcf']
        
        # Exibir resultados
        print("\nAnálise Cruzada de Ataques:")
        print(f"Ataques de Treinamento: {', '.join(train_attacks)}")
        print(f"Ataques de Teste: {', '.join(test_attacks)}")
        print(f"Ataques Desconhecidos: {', '.join(results['unseen_attacks'])}")
        print(f"EER Treinamento: {results['train_eer']:.4f}")
        print(f"EER Teste: {results['test_eer']:.4f}")
        print(f"Degradação de EER: {results['eer_degradation']:.4f}")
        print(f"t-DCF Treinamento: {results['train_tdcf']:.4f}")
        print(f"t-DCF Teste: {results['test_tdcf']:.4f}")
        print(f"Degradação de t-DCF: {results['tdcf_degradation']:.4f}")
        
        return results
    
    def cross_dataset_analysis(self, train_dataloader, test_dataloader, train_dataset_name, test_dataset_name, results_dir=None):
        """
        Realiza análise cruzada entre diferentes conjuntos de dados.
        
        Args:
            train_dataloader: DataLoader com dados de treinamento
            test_dataloader: DataLoader com dados de teste
            train_dataset_name: Nome do conjunto de dados de treinamento
            test_dataset_name: Nome do conjunto de dados de teste
            results_dir: Diretório para salvar resultados (opcional)
            
        Returns:
            Dicionário com resultados da análise cruzada
        """
        # Extrair embeddings para ambos os conjuntos
        print(f"Extraindo embeddings para {train_dataset_name}...")
        train_data = self.extract_embeddings(train_dataloader)
        
        print(f"Extraindo embeddings para {test_dataset_name}...")
        test_data = self.extract_embeddings(test_dataloader)
        
        # Encontrar pesos ótimos usando o conjunto de treinamento
        print("Encontrando pesos ótimos para fusão...")
        best_weights, best_eer = self.evaluator.find_optimal_fusion_weights(train_dataloader)
        print(f"Pesos ótimos encontrados: {best_weights} (EER = {best_eer:.4f})")
        
        # Atualizar pesos do avaliador
        self.evaluator.score_weights = best_weights
        
        # Avaliar no conjunto de treinamento
        print(f"Avaliando no conjunto de treinamento ({train_dataset_name})...")
        train_results = self.evaluator.evaluate(train_dataloader)
        
        # Avaliar no conjunto de teste
        print(f"Avaliando no conjunto de teste ({test_dataset_name})...")
        test_results = self.evaluator.evaluate(test_dataloader)
        
        # Criar dicionário com resultados
        results = {
            'train_dataset': train_dataset_name,
            'test_dataset': test_dataset_name,
            'best_weights': best_weights,
            'train_eer': train_results['eer']['fused']['value'],
            'test_eer': test_results['eer']['fused']['value'],
            'train_tdcf': train_results['tdcf']['fused']['value'],
            'test_tdcf': test_results['tdcf']['fused']['value'],
        }
        
        # Calcular a diferença de desempenho
        results['eer_degradation'] = results['test_eer'] - results['train_eer']
        results['tdcf_degradation'] = results['test_tdcf'] - results['train_tdcf']
        
        # Exibir resultados
        print("\nAnálise Cruzada de Conjuntos de Dados:")
        print(f"Conjunto de Treinamento: {train_dataset_name}")
        print(f"Conjunto de Teste: {test_dataset_name}")
        print(f"EER Treinamento: {results['train_eer']:.4f}")
        print(f"EER Teste: {results['test_eer']:.4f}")
        print(f"Degradação de EER: {results['eer_degradation']:.4f}")
        print(f"t-DCF Treinamento: {results['train_tdcf']:.4f}")
        print(f"t-DCF Teste: {results['test_tdcf']:.4f}")
        print(f"Degradação de t-DCF: {results['tdcf_degradation']:.4f}")
        
        # Plotar visualização conjunta de embeddings
        if results_dir:
            if not os.path.exists(results_dir):
                os.makedirs(results_dir)
            
            # Combinar embeddings para visualização
            combined_embeddings = np.vstack([train_data['embeddings'], test_data['embeddings']])
            combined_labels = np.concatenate([train_data['labels'], test_data['labels']])
            combined_sources = np.concatenate([
                np.array([train_dataset_name] * len(train_data['labels'])), 
                np.array([test_dataset_name] * len(test_data['labels']))
            ])
            
            # Criar DataFrame para visualização
            df = pd.DataFrame({
                'embeddings': list(combined_embeddings),
                'label': ['Genuine' if l == 0 else 'Spoof' for l in combined_labels],
                'source': combined_sources
            })
            
            # Redução de dimensionalidade
            print("Aplicando t-SNE para visualização conjunta...")
            tsne = TSNE(n_components=2, random_state=42)
            reduced_embeddings = tsne.fit_transform(combined_embeddings)
            
            # Adicionar coordenadas reduzidas ao DataFrame
            df['x'] = reduced_embeddings[:, 0]
            df['y'] = reduced_embeddings[:, 1]
            
            # Plotar
            plt.figure(figsize=(14, 10))
            
            # Usar diferentes formas para diferentes conjuntos e cores para rótulos
            for source, marker in zip([train_dataset_name, test_dataset_name], ['o', 'X']):
                for label, color in zip(['Genuine', 'Spoof'], ['green', 'red']):
                    mask = (df['source'] == source) & (df['label'] == label)
                    plt.scatter(
                        df.loc[mask, 'x'], df.loc[mask, 'y'],
                        marker=marker, c=color, s=100, alpha=0.7,
                        label=f"{source} - {label}"
                    )
            
            plt.title('Visualização de Embeddings para Análise Cross-Dataset')
            plt.xlabel('Componente 1')
            plt.ylabel('Componente 2')
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.tight_layout()
            
            plt.savefig(os.path.join(results_dir, 'cross_dataset_embeddings.png'), dpi=300, bbox_inches='tight')
        
        return results
    
    def ablation_study(self, dataloader, components=['lbp', 'glcm', 'lpq', 'bidirectional', 'attention']):
        """
        Realiza estudo de ablação para avaliar a contribuição de cada componente do sistema.
        
        Args:
            dataloader: DataLoader com dados para avaliação
            components: Lista de componentes para análise
            
        Returns:
            DataFrame com resultados do estudo de ablação
        """
        # Resultados do modelo completo
        print("Avaliando modelo completo...")
        full_model_results = self.evaluator.evaluate(dataloader)
        
        # Inicializar dicionário para armazenar resultados
        ablation_results = {
            'full_model': {
                'eer': full_model_results['eer']['fused']['value'],
                'tdcf': full_model_results['tdcf']['fused']['value']
            }
        }
        
        # Exemplo de análise para componentes individuais
        # (Na implementação real, seria necessário modificar o modelo para remover componentes)
        for component in components:
            print(f"Analisando contribuição do componente '{component}'...")
            
            # Aqui, simularemos a remoção de componentes ajustando os pesos
            # Em uma implementação real, modificaríamos a arquitetura do modelo
            if component == 'lbp':
                temp_weights = self.evaluator.score_weights.copy()
                temp_weights['lbp'] = 0.0
                temp_weights['glcm'] = temp_weights['glcm'] / (temp_weights['glcm'] + temp_weights['lpq'])
                temp_weights['lpq'] = temp_weights['lpq'] / (temp_weights['glcm'] + temp_weights['lpq'])
                
                self.evaluator.score_weights = temp_weights
                ablation_result = self.evaluator.evaluate(dataloader)
                ablation_results['without_lbp'] = {
                    'eer': ablation_result['eer']['fused']['value'],
                    'tdcf': ablation_result['tdcf']['fused']['value']
                }
                
            elif component == 'glcm':
                temp_weights = self.evaluator.score_weights.copy()
                temp_weights['glcm'] = 0.0
                temp_weights['lbp'] = temp_weights['lbp'] / (temp_weights['lbp'] + temp_weights['lpq'])
                temp_weights['lpq'] = temp_weights['lpq'] / (temp_weights['lbp'] + temp_weights['lpq'])
                
                self.evaluator.score_weights = temp_weights
                ablation_result = self.evaluator.evaluate(dataloader)
                ablation_results['without_glcm'] = {
                    'eer': ablation_result['eer']['fused']['value'],
                    'tdcf': ablation_result['tdcf']['fused']['value']
                }
                
            elif component == 'lpq':
                temp_weights = self.evaluator.score_weights.copy()
                temp_weights['lpq'] = 0.0
                temp_weights['lbp'] = temp_weights['lbp'] / (temp_weights['lbp'] + temp_weights['glcm'])
                temp_weights['glcm'] = temp_weights['glcm'] / (temp_weights['lbp'] + temp_weights['glcm'])
                
                self.evaluator.score_weights = temp_weights
                ablation_result = self.evaluator.evaluate(dataloader)
                ablation_results['without_lpq'] = {
                    'eer': ablation_result['eer']['fused']['value'],
                    'tdcf': ablation_result['tdcf']['fused']['value']
                }
        
        # Converter para DataFrame
        ablation_df = pd.DataFrame.from_dict(ablation_results, orient='index')
        
        # Calcular degradação em relação ao modelo completo
        ablation_df['eer_degradation'] = ablation_df['eer'] - ablation_df.loc['full_model', 'eer']
        ablation_df['tdcf_degradation'] = ablation_df['tdcf'] - ablation_df.loc['full_model', 'tdcf']
        
        # Ordenar por degradação de EER
        ablation_df = ablation_df.sort_values('eer_degradation', ascending=False)
        
        # Exibir resultados
        print("\nResultados do Estudo de Ablação:")
        print(ablation_df)
        
        return ablation_df


# Exemplo de uso
if __name__ == "__main__":
    import torch
    from model import MultiPatternModel
    
    # Verificar GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Usando dispositivo: {device}")
    
    # Criar modelo de exemplo
    model = MultiPatternModel().to(device)
    
    # Criar analisador
    analyzer = GeneralizationAnalyzer(model, device)
    
    # Exemplo de saída para funções principais
    print("Funções principais implementadas:")
    print("- extract_embeddings: Extrai embeddings de uma camada específica do modelo")
    print("- visualize_embeddings: Visualiza embeddings usando redução de dimensionalidade")
    print("- analyze_attack_types: Analisa o desempenho para diferentes tipos de ataques")
    print("- cross_attack_analysis: Realiza análise cruzada entre diferentes tipos de ataques")
    print("- cross_dataset_analysis: Realiza análise cruzada entre diferentes conjuntos de dados")
    print("- ablation_study: Realiza estudo de ablação para avaliar a contribuição de cada componente")
