#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Módulo para avaliação da robustez do sistema de detecção de ataques de replay
usando conjuntos de dados padrão (ASVspoof 2019 e 2021).

Autores: André Thiago de Souza, Felipe de Lima dos Santos, 
         Juliano Gaona Proença, Matheus Henrique Reich Favarin Zagonel
"""

import os
import numpy as np
import torch
import time
from sklearn.metrics import roc_curve, auc, confusion_matrix
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd


class ReplayAttackEvaluator:
    """
    Avaliador para sistemas de detecção de ataques de replay.
    """
    
    def __init__(self, model, device=None, score_weights=None):
        """
        Inicializa o avaliador.
        
        Args:
            model: Modelo treinado para detecção
            device: Dispositivo para execução (CPU ou GPU)
            score_weights: Pesos para fusão das pontuações dos diferentes ramos
        """
        self.model = model
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Configurar pesos padrão para fusão de pontuações, se não fornecidos
        if score_weights is None:
            self.score_weights = {'lbp': 0.33, 'glcm': 0.33, 'lpq': 0.34}
        else:
            self.score_weights = score_weights
    
    def compute_scores(self, dataloader):
        """
        Calcula as pontuações para todas as amostras no dataloader.
        
        Args:
            dataloader: DataLoader com dados para avaliação
            
        Returns:
            Dicionário com pontuações e rótulos
        """
        self.model.eval()
        
        all_scores = {
            'lbp': [],
            'glcm': [],
            'lpq': [],
            'fused': []
        }
        all_labels = []
        
        inference_times = []
        
        with torch.no_grad():
            for lbp, glcm, lpq, labels in tqdm(dataloader, desc="Computando pontuações"):
                # Mover dados para o dispositivo
                lbp = lbp.to(self.device)
                glcm = glcm.to(self.device)
                lpq = lpq.to(self.device)
                
                # Medir tempo de inferência
                start_time = time.time()
                
                # Obter pontuações
                scores = self.model.get_scores(lbp, glcm, lpq)
                
                # Calcular tempo de inferência
                inference_time = time.time() - start_time
                inference_times.append(inference_time)
                
                # Calcular pontuação fundida
                fused_scores = (
                    self.score_weights['lbp'] * scores['lbp_score'] +
                    self.score_weights['glcm'] * scores['glcm_score'] +
                    self.score_weights['lpq'] * scores['lpq_score']
                )
                
                # Armazenar pontuações e rótulos
                all_scores['lbp'].extend(scores['lbp_score'].cpu().numpy())
                all_scores['glcm'].extend(scores['glcm_score'].cpu().numpy())
                all_scores['lpq'].extend(scores['lpq_score'].cpu().numpy())
                all_scores['fused'].extend(fused_scores.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Converter para arrays numpy
        for key in all_scores:
            all_scores[key] = np.array(all_scores[key])
        all_labels = np.array(all_labels)
        
        # Calcular estatísticas de tempo de inferência
        avg_inference_time = np.mean(inference_times)
        std_inference_time = np.std(inference_times)
        
        return {
            'scores': all_scores,
            'labels': all_labels,
            'avg_inference_time': avg_inference_time,
            'std_inference_time': std_inference_time
        }
    
    def compute_eer(self, scores, labels):
        """
        Calcula a Equal Error Rate (EER).
        
        Args:
            scores: Array com pontuações de classificação
            labels: Array com rótulos verdadeiros
            
        Returns:
            EER e limiar correspondente
        """
        # Calcular taxas de falsa aceitação e falsa rejeição
        fpr, tpr, thresholds = roc_curve(labels, scores)
        fnr = 1 - tpr
        
        # Encontrar o ponto onde FAR = FRR (EER)
        eer_threshold_idx = np.nanargmin(np.absolute(fpr - fnr))
        eer = np.mean([fpr[eer_threshold_idx], fnr[eer_threshold_idx]])
        eer_threshold = thresholds[eer_threshold_idx]
        
        return eer, eer_threshold
    
    def compute_tdcf(self, scores, labels, p_target=0.05, c_miss=1, c_fa=1):
        """
        Calcula a tandem Detection Cost Function (t-DCF).
        
        Args:
            scores: Array com pontuações de classificação
            labels: Array com rótulos verdadeiros
            p_target: Probabilidade a priori do alvo (spoof)
            c_miss: Custo de perda (miss)
            c_fa: Custo de falso alarme
            
        Returns:
            t-DCF mínima e limiar correspondente
        """
        # Calcular taxas de falsa aceitação e falsa rejeição
        fpr, tpr, thresholds = roc_curve(labels, scores)
        fnr = 1 - tpr
        
        # Calcular t-DCF para cada limiar
        p_non_target = 1 - p_target
        c_0 = 0  # Custo quando não há erro
        
        tdcf_values = []
        for i in range(len(thresholds)):
            c_1 = c_miss * fnr[i] * p_target
            c_2 = c_fa * fpr[i] * p_non_target
            
            tdcf = c_0 + c_1 + c_2
            tdcf_values.append(tdcf)
        
        # Normalizar pelo t-DCF default
        tdcf_default = min(c_miss * p_target, c_fa * p_non_target)
        tdcf_norm = np.array(tdcf_values) / tdcf_default
        
        # Encontrar t-DCF mínima
        min_tdcf_idx = np.argmin(tdcf_norm)
        min_tdcf = tdcf_norm[min_tdcf_idx]
        min_tdcf_threshold = thresholds[min_tdcf_idx]
        
        return min_tdcf, min_tdcf_threshold
    
    def evaluate(self, dataloader):
        """
        Avalia o modelo em um conjunto de dados.
        
        Args:
            dataloader: DataLoader com dados para avaliação
            
        Returns:
            Resultados da avaliação
        """
        # Computar pontuações
        results = self.compute_scores(dataloader)
        scores = results['scores']
        labels = results['labels']
        
        # Inicializar dicionário de resultados
        evaluation_results = {
            'eer': {},
            'tdcf': {},
            'auc': {},
            'inference_time': {
                'avg': results['avg_inference_time'],
                'std': results['std_inference_time']
            }
        }
        
        # Calcular métricas para cada tipo de pontuação
        for score_type in scores:
            # EER
            eer, eer_threshold = self.compute_eer(scores[score_type], labels)
            evaluation_results['eer'][score_type] = {
                'value': eer,
                'threshold': eer_threshold
            }
            
            # t-DCF
            tdcf, tdcf_threshold = self.compute_tdcf(scores[score_type], labels)
            evaluation_results['tdcf'][score_type] = {
                'value': tdcf,
                'threshold': tdcf_threshold
            }
            
            # AUC
            fpr, tpr, _ = roc_curve(labels, scores[score_type])
            roc_auc = auc(fpr, tpr)
            evaluation_results['auc'][score_type] = roc_auc
        
        return evaluation_results
    
    def plot_roc_curves(self, dataloader, save_path=None):
        """
        Plota as curvas ROC para cada tipo de pontuação.
        
        Args:
            dataloader: DataLoader com dados para avaliação
            save_path: Caminho para salvar o gráfico (opcional)
            
        Returns:
            Figura com curvas ROC
        """
        # Computar pontuações
        results = self.compute_scores(dataloader)
        scores = results['scores']
        labels = results['labels']
        
        # Criar figura
        plt.figure(figsize=(10, 8))
        
        # Cores para cada tipo de pontuação
        colors = {
            'lbp': 'blue',
            'glcm': 'green',
            'lpq': 'red',
            'fused': 'black'
        }
        
        # Plotar curva ROC para cada tipo de pontuação
        for score_type, color in colors.items():
            fpr, tpr, _ = roc_curve(labels, scores[score_type])
            roc_auc = auc(fpr, tpr)
            
            plt.plot(
                fpr, tpr, color=color, lw=2,
                label=f'{score_type.upper()} (AUC = {roc_auc:.4f})'
            )
            
            # Calcular e marcar o ponto de EER
            eer, eer_threshold = self.compute_eer(scores[score_type], labels)
            eer_idx = np.nanargmin(np.absolute(fpr - (1 - tpr)))
            plt.scatter(
                fpr[eer_idx], tpr[eer_idx], marker='o', color=color,
                label=f'{score_type.upper()} EER = {eer:.4f}'
            )
        
        # Configurar o gráfico
        plt.plot([0, 1], [1, 0], 'k--', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Taxa de Falso Positivo')
        plt.ylabel('Taxa de Verdadeiro Positivo')
        plt.title('Curvas ROC para Detecção de Ataques de Replay')
        plt.legend(loc="lower right")
        plt.grid(True)
        
        # Salvar o gráfico, se solicitado
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return plt.gcf()
    
    def plot_score_distributions(self, dataloader, score_type='fused', save_path=None):
        """
        Plota as distribuições de pontuações para as classes genuína e spoof.
        
        Args:
            dataloader: DataLoader com dados para avaliação
            score_type: Tipo de pontuação a ser plotada
            save_path: Caminho para salvar o gráfico (opcional)
            
        Returns:
            Figura com distribuições de pontuações
        """
        # Computar pontuações
        results = self.compute_scores(dataloader)
        scores = results['scores'][score_type]
        labels = results['labels']
        
        # Separar pontuações por classe
        genuine_scores = scores[labels == 0]
        spoof_scores = scores[labels == 1]
        
        # Criar figura
        plt.figure(figsize=(10, 6))
        
        # Plotar histogramas
        plt.hist(
            genuine_scores, bins=50, alpha=0.5, color='green',
            label=f'Genuíno (n={len(genuine_scores)})'
        )
        plt.hist(
            spoof_scores, bins=50, alpha=0.5, color='red',
            label=f'Spoof (n={len(spoof_scores)})'
        )
        
        # Calcular EER e marcar o limiar
        eer, eer_threshold = self.compute_eer(scores, labels)
        plt.axvline(
            x=eer_threshold, color='black', linestyle='--',
            label=f'Limiar EER = {eer_threshold:.4f} (EER = {eer:.4f})'
        )
        
        # Configurar o gráfico
        plt.xlabel('Pontuação')
        plt.ylabel('Contagem')
        plt.title(f'Distribuição de Pontuações ({score_type.upper()})')
        plt.legend()
        plt.grid(True)
        
        # Salvar o gráfico, se solicitado
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return plt.gcf()
    
    def find_optimal_fusion_weights(self, dataloader, grid_size=11):
        """
        Encontra os pesos ótimos para fusão de pontuações usando busca em grade.
        
        Args:
            dataloader: DataLoader com dados para avaliação
            grid_size: Tamanho da grade para busca (default: 11 = passos de 0.1)
            
        Returns:
            Pesos ótimos para fusão
        """
        # Computar pontuações
        results = self.compute_scores(dataloader)
        scores = results['scores']
        labels = results['labels']
        
        # Inicializar variáveis para acompanhar os melhores pesos
        best_eer = 1.0
        best_weights = {'lbp': 0.33, 'glcm': 0.33, 'lpq': 0.34}
        
        # Criar grade de pesos
        weight_values = np.linspace(0, 1, grid_size)
        
        # Iterar sobre todas as combinações possíveis de pesos
        for w_lbp in weight_values:
            for w_glcm in weight_values:
                w_lpq = 1.0 - w_lbp - w_glcm
                
                # Verificar se a combinação é válida (soma = 1)
                if 0 <= w_lpq <= 1:
                    # Calcular pontuações fundidas
                    fused_scores = (
                        w_lbp * scores['lbp'] +
                        w_glcm * scores['glcm'] +
                        w_lpq * scores['lpq']
                    )
                    
                    # Calcular EER
                    eer, _ = self.compute_eer(fused_scores, labels)
                    
                    # Atualizar se encontrarmos um EER melhor
                    if eer < best_eer:
                        best_eer = eer
                        best_weights = {'lbp': w_lbp, 'glcm': w_glcm, 'lpq': w_lpq}
        
        return best_weights, best_eer
    
    def analyze_error_cases(self, dataloader, save_path=None):
        """
        Analisa casos de erro e identifica padrões.
        
        Args:
            dataloader: DataLoader com dados para avaliação
            save_path: Caminho para salvar a análise (opcional)
            
        Returns:
            DataFrame com análise de erros
        """
        # Computar pontuações
        results = self.compute_scores(dataloader)
        scores = results['scores']
        labels = results['labels']
        
        # Usar pontuações fundidas para identificar erros
        fused_scores = scores['fused']
        
        # Calcular EER e obter limiar
        eer, eer_threshold = self.compute_eer(fused_scores, labels)
        
        # Identificar predições e erros
        predictions = (fused_scores >= eer_threshold).astype(int)
        errors = (predictions != labels)
        
        # Falsos positivos (genuíno classificado como spoof)
        false_positives = (predictions == 1) & (labels == 0)
        
        # Falsos negativos (spoof classificado como genuíno)
        false_negatives = (predictions == 0) & (labels == 1)
        
        # Confusão frequente por tipo
        print(f"Limiar EER: {eer_threshold:.4f}")
        print(f"Taxa de erro: {errors.mean():.4f}")
        print(f"Taxa de falsos positivos: {false_positives.mean():.4f}")
        print(f"Taxa de falsos negativos: {false_negatives.mean():.4f}")
        
        # Criar DataFrame com informações detalhadas
        error_analysis = pd.DataFrame({
            'label': labels,
            'prediction': predictions,
            'error': errors,
            'false_positive': false_positives,
            'false_negative': false_negatives,
            'score_lbp': scores['lbp'],
            'score_glcm': scores['glcm'],
            'score_lpq': scores['lpq'],
            'score_fused': fused_scores
        })
        
        # Salvar análise, se solicitado
        if save_path:
            error_analysis.to_csv(save_path, index=False)
        
        # Matriz de confusão
        cm = confusion_matrix(labels, predictions)
        print("Matriz de confusão:")
        print(cm)
        
        return error_analysis
    
    def evaluate_cross_dataset(self, train_dataloader, eval_dataloader, find_optimal_weights=True):
        """
        Avalia o modelo em cenário cross-dataset.
        
        Args:
            train_dataloader: DataLoader com dados de treinamento
            eval_dataloader: DataLoader com dados de avaliação
            find_optimal_weights: Se True, encontra pesos ótimos para fusão usando dados de treinamento
            
        Returns:
            Resultados da avaliação
        """
        # Se solicitado, encontrar pesos ótimos para fusão
        if find_optimal_weights:
            best_weights, best_eer = self.find_optimal_fusion_weights(train_dataloader)
            print(f"Pesos ótimos encontrados: {best_weights} (EER = {best_eer:.4f})")
            
            # Atualizar pesos
            self.score_weights = best_weights
        
        # Avaliar no conjunto de avaliação
        evaluation_results = self.evaluate(eval_dataloader)
        
        # Retornar resultados
        return evaluation_results


# Exemplo de uso
if __name__ == "__main__":
    import torch
    from model import MultiPatternModel
    
    # Verificar GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Usando dispositivo: {device}")
    
    # Criar modelo de exemplo
    model = MultiPatternModel().to(device)
    
    # Criar avaliador
    evaluator = ReplayAttackEvaluator(model, device)
    
    # Exemplo de saída para funções principais
    print("Funções principais implementadas:")
    print("- compute_scores: Calcula pontuações para todas as amostras")
    print("- compute_eer: Calcula Equal Error Rate")
    print("- compute_tdcf: Calcula tandem Detection Cost Function")
    print("- evaluate: Avalia o modelo completo")
    print("- plot_roc_curves: Plota curvas ROC")
    print("- plot_score_distributions: Plota distribuições de pontuações")
    print("- find_optimal_fusion_weights: Encontra pesos ótimos para fusão")
    print("- analyze_error_cases: Analisa casos de erro")
    print("- evaluate_cross_dataset: Avalia o modelo em cenário cross-dataset")
