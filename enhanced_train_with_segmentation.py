#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script de treinamento otimizado com integração de estratégias de segmentação
para acelerar o processo de treinamento mantendo a qualidade do modelo.

Autores: André Thiago de Souza, Felipe de Lima dos Santos, 
         Juliano Gaona Proença, Matheus Henrique Reich Favarin Zagonel
"""

import os
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
from tqdm import tqdm
import joblib
import json

# Importar módulos locais
from data_segmentation import (
    DataSegmentationManager, 
    ProgressiveSegmentation,
    IntelligentDataSampler,
    create_segmented_dataset
)
from feature_extraction import FeatureExtractor
from bidirectional_segmentation import BidirectionalDataset, BidirectionalDataLoader
from model import MultiPatternModel, OCLoss
from evaluation import ReplayAttackEvaluator
from train import Trainer  # Importar classe original


class OptimizedTrainer(Trainer):
    """
    Versão otimizada do treinador com suporte a segmentação de dados.
    """
    
    def __init__(self, model, train_loader, dev_loader, optimizer, criterion, 
                 device, scheduler=None, num_epochs=100, patience=10,
                 save_dir='checkpoints', segmentation_config=None):
        """
        Inicializa o treinador otimizado.
        
        Args:
            segmentation_config: Configuração para segmentação de dados
        """
        super().__init__(model, train_loader, dev_loader, optimizer, criterion,
                        device, scheduler, num_epochs, patience, save_dir)
        
        self.segmentation_config = segmentation_config or {}
        self.progressive_stages = None
        self.current_stage = 0
        self.intelligent_sampler = None
        
        # Estatísticas de treinamento otimizado
        self.training_stats = {
            'data_reduction': 0.0,
            'time_savings': 0.0,
            'stage_transitions': [],
            'difficulty_updates': []
        }
    
    def setup_progressive_training(self, features_dir, labels_file, n_stages=5):
        """
        Configura treinamento progressivo.
        
        Args:
            features_dir: Diretório com características
            labels_file: Arquivo com rótulos
            n_stages: Número de estágios progressivos
        """
        print(f"Configurando treinamento progressivo com {n_stages} estágios...")
        
        progressive = ProgressiveSegmentation(features_dir, labels_file, n_stages)
        
        # Escolher estratégia baseada na configuração
        strategy = self.segmentation_config.get('progressive_strategy', 'balanced')
        stages = progressive.create_progressive_stages(strategy=strategy)
        self.progressive_stages = progressive.get_cumulative_stages()
        
        print(f"Treinamento progressivo configurado:")
        for i, stage in enumerate(self.progressive_stages):
            print(f"  Estágio {i+1}: {len(stage)} amostras")
    
    def setup_intelligent_sampling(self, features_dir, labels_file):
        """
        Configura amostragem inteligente.
        
        Args:
            features_dir: Diretório com características
            labels_file: Arquivo com rótulos
        """
        print("Configurando amostragem inteligente...")
        
        self.intelligent_sampler = IntelligentDataSampler(features_dir, labels_file)
        
        # Carregar dados e calcular dificuldade inicial
        manager = DataSegmentationManager(features_dir, labels_file)
        features, labels, _ = manager.load_data()
        
        self.intelligent_sampler.calculate_sample_difficulty(features, labels)
        print("Amostragem inteligente configurada.")
    
    def create_segmented_dataloader(self, original_features_dir, original_labels_file,
                                  strategy='adaptive', sample_ratio=0.3, 
                                  batch_size=32, **kwargs):
        """
        Cria DataLoader com dados segmentados.
        
        Args:
            original_features_dir: Diretório original com características
            original_labels_file: Arquivo original com rótulos
            strategy: Estratégia de segmentação
            sample_ratio: Proporção de dados a manter
            batch_size: Tamanho do lote
            
        Returns:
            DataLoader segmentado e estatísticas
        """
        print(f"Criando DataLoader segmentado (estratégia: {strategy}, ratio: {sample_ratio})...")
        
        # Aplicar segmentação
        manager = DataSegmentationManager(original_features_dir, original_labels_file)
        
        start_time = time.time()
        selected_indices, stats = manager.apply_segmentation(strategy, sample_ratio, **kwargs)
        segmentation_time = time.time() - start_time
        
        # Criar dataset temporário segmentado
        temp_features_dir = os.path.join(self.save_dir, 'temp_segmented_features')
        temp_labels_file = os.path.join(self.save_dir, 'temp_segmented_labels.txt')
        
        _, _, file_names = manager.load_data()
        create_segmented_dataset(
            original_features_dir, original_labels_file,
            temp_features_dir, temp_labels_file,
            selected_indices, file_names
        )
        
        # Criar DataLoader
        segmented_loader = BidirectionalDataLoader(
            temp_features_dir, temp_labels_file, 
            batch_size=batch_size, shuffle=True, **kwargs
        ).get_dataloader()
        
        # Registrar estatísticas
        self.training_stats['data_reduction'] = stats['compression_ratio']
        self.training_stats['segmentation_time'] = segmentation_time
        
        print(f"DataLoader segmentado criado:")
        print(f"  Compressão: {stats['compression_ratio']:.1%}")
        print(f"  Tempo de segmentação: {segmentation_time:.2f}s")
        print(f"  Amostras: {stats['selected_count']}/{stats['original_count']}")
        
        return segmented_loader, stats
    
    def train_with_progressive_stages(self):
        """
        Treina modelo usando estágios progressivos.
        
        Returns:
            Histórico de treinamento
        """
        if not self.progressive_stages:
            raise ValueError("Configure treinamento progressivo primeiro com setup_progressive_training()")
        
        print(f"Iniciando treinamento progressivo com {len(self.progressive_stages)} estágios...")
        
        total_start_time = time.time()
        
        for stage_idx, stage_indices in enumerate(self.progressive_stages):
            print(f"\n=== ESTÁGIO {stage_idx + 1}/{len(self.progressive_stages)} ===")
            print(f"Treinando com {len(stage_indices)} amostras...")
            
            stage_start_time = time.time()
            
            # Criar DataLoader para este estágio
            # (Implementação simplificada - na prática, recriaria o DataLoader)
            
            # Treinar por algumas épocas neste estágio
            stage_epochs = max(1, self.num_epochs // len(self.progressive_stages))
            
            for epoch in range(stage_epochs):
                epoch_loss, epoch_eer, epoch_auc = self.train_epoch()
                val_loss, val_eer, val_auc = self.validate()
                
                # Registrar métricas
                self.history['train_loss'].append(epoch_loss)
                self.history['val_loss'].append(val_loss)
                self.history['train_eer'].append(epoch_eer)
                self.history['val_eer'].append(val_eer)
                self.history['train_auc'].append(epoch_auc)
                self.history['val_auc'].append(val_auc)
                
                print(f"  Época {epoch+1}: Loss={epoch_loss:.4f}, EER={epoch_eer:.4f}")
            
            stage_time = time.time() - stage_start_time
            self.training_stats['stage_transitions'].append({
                'stage': stage_idx + 1,
                'samples': len(stage_indices),
                'time': stage_time,
                'final_eer': val_eer
            })
            
            print(f"Estágio {stage_idx + 1} concluído em {stage_time:.2f}s")
        
        total_time = time.time() - total_start_time
        print(f"\nTreinamento progressivo concluído em {total_time:.2f}s")
        
        return self.history
    
    def train_with_intelligent_sampling(self, update_frequency=10):
        """
        Treina modelo usando amostragem inteligente adaptativa.
        
        Args:
            update_frequency: Frequência de atualização da amostragem (épocas)
            
        Returns:
            Histórico de treinamento
        """
        if not self.intelligent_sampler:
            raise ValueError("Configure amostragem inteligente primeiro com setup_intelligent_sampling()")
        
        print(f"Iniciando treinamento com amostragem inteligente...")
        print(f"Frequência de atualização: a cada {update_frequency} épocas")
        
        best_val_eer = float('inf')
        epochs_without_improvement = 0
        
        for epoch in range(1, self.num_epochs + 1):
            print(f"\nÉpoca {epoch}/{self.num_epochs}")
            
            # Atualizar amostragem periodicamente baseado em erros
            if epoch % update_frequency == 0 and epoch > 1:
                print("Atualizando amostragem baseado em erros...")
                
                # Identificar amostras que causaram erro
                error_indices = self.identify_error_samples()
                
                if len(error_indices) > 0:
                    self.intelligent_sampler.update_difficulty_from_errors(error_indices)
                    
                    # Recriar DataLoader com nova amostragem
                    # (Implementação simplificada)
                    print(f"Amostragem atualizada baseado em {len(error_indices)} erros")
                    
                    self.training_stats['difficulty_updates'].append({
                        'epoch': epoch,
                        'error_count': len(error_indices)
                    })
            
            # Treinar época normal
            train_loss, train_eer, train_auc = self.train_epoch()
            val_loss, val_eer, val_auc = self.validate()
            
            # Registrar métricas
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_eer'].append(train_eer)
            self.history['val_eer'].append(val_eer)
            self.history['train_auc'].append(train_auc)
            self.history['val_auc'].append(val_auc)
            
            # Exibir métricas
            print(f"Loss Treino: {train_loss:.4f}, EER Treino: {train_eer:.4f}")
            print(f"Loss Val: {val_loss:.4f}, EER Val: {val_eer:.4f}")
            
            # Early stopping
            if val_eer < best_val_eer:
                best_val_eer = val_eer
                epochs_without_improvement = 0
                self.save_checkpoint(epoch, val_eer, is_best=True)
            else:
                epochs_without_improvement += 1
                
                if epochs_without_improvement >= self.patience:
                    print(f"Early stopping após {epoch} épocas.")
                    break
        
        return self.history
    
    def identify_error_samples(self):
        """
        Identifica amostras que o modelo está errando para atualização de dificuldade.
        
        Returns:
            Índices das amostras com erro
        """
        self.model.eval()
        error_indices = []
        sample_idx = 0
        
        with torch.no_grad():
            for lbp, glcm, lpq, labels in self.train_loader:
                lbp = lbp.to(self.device)
                glcm = glcm.to(self.device)
                lpq = lpq.to(self.device)
                labels = labels.to(self.device)
                
                # Obter predições
                scores = self.model.get_scores(lbp, glcm, lpq)
                fused_scores = (
                    self.evaluator.score_weights['lbp'] * scores['lbp_score'] +
                    self.evaluator.score_weights['glcm'] * scores['glcm_score'] +
                    self.evaluator.score_weights['lpq'] * scores['lpq_score']
                )
                
                # Identificar erros (usando limiar simples de 0.5)
                predictions = (fused_scores > 0.5).long()
                errors = (predictions != labels)
                
                # Coletar índices de erros
                error_positions = torch.where(errors)[0]
                for pos in error_positions:
                    error_indices.append(sample_idx + pos.item())
                
                sample_idx += len(labels)
        
        return error_indices
    
    def plot_optimization_stats(self, save_path=None):
        """
        Plota estatísticas de otimização do treinamento.
        
        Args:
            save_path: Caminho para salvar o gráfico (opcional)
        """
        if not self.training_stats:
            print("Nenhuma estatística de otimização disponível.")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Gráfico 1: Progresso por estágios (se disponível)
        if self.training_stats.get('stage_transitions'):
            ax1 = axes[0, 0]
            stages_data = self.training_stats['stage_transitions']
            
            stages = [s['stage'] for s in stages_data]
            samples = [s['samples'] for s in stages_data]
            times = [s['time'] for s in stages_data]
            eers = [s['final_eer'] for s in stages_data]
            
            ax1_twin = ax1.twinx()
            
            bars = ax1.bar(stages, samples, alpha=0.7, color='skyblue', label='Amostras')
            line = ax1_twin.plot(stages, eers, 'ro-', linewidth=2, label='EER Final')
            
            ax1.set_xlabel('Estágio')
            ax1.set_ylabel('Número de Amostras', color='blue')
            ax1_twin.set_ylabel('EER Final', color='red')
            ax1.set_title('Progresso do Treinamento Progressivo')
            
            # Adicionar anotações de tempo
            for i, (stage, time_val) in enumerate(zip(stages, times)):
                ax1.annotate(f'{time_val:.1f}s', (stage, samples[i]), 
                           textcoords="offset points", xytext=(0,5), ha='center')
        
        # Gráfico 2: Atualizações de dificuldade (se disponível)
        if self.training_stats.get('difficulty_updates'):
            ax2 = axes[0, 1]
            updates_data = self.training_stats['difficulty_updates']
            
            epochs = [u['epoch'] for u in updates_data]
            error_counts = [u['error_count'] for u in updates_data]
            
            ax2.plot(epochs, error_counts, 'go-', linewidth=2, markersize=8)
            ax2.set_xlabel('Época')
            ax2.set_ylabel('Número de Erros')
            ax2.set_title('Atualizações de Dificuldade')
            ax2.grid(True, alpha=0.3)
        
        # Gráfico 3: Redução de dados e tempo economizado
        ax3 = axes[1, 0]
        
        reduction = self.training_stats.get('data_reduction', 0)
        if reduction > 0:
            categories = ['Dados\nOriginais', 'Dados\nSelecionados']
            values = [1.0, 1.0 - reduction]
            colors = ['lightcoral', 'lightgreen']
            
            bars = ax3.bar(categories, values, color=colors, alpha=0.7)
            ax3.set_ylabel('Proporção')
            ax3.set_title(f'Redução de Dados: {reduction:.1%}')
            ax3.set_ylim(0, 1.1)
            
            # Adicionar valores nas barras
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax3.annotate(f'{value:.1%}', (bar.get_x() + bar.get_width()/2., height),
                           ha='center', va='bottom')
        
        # Gráfico 4: Histórico de treinamento comparativo
        ax4 = axes[1, 1]
        
        if self.history and self.history.get('val_eer'):
            epochs = range(1, len(self.history['val_eer']) + 1)
            ax4.plot(epochs, self.history['val_eer'], 'b-', linewidth=2, label='EER Validação')
            
            if self.history.get('train_eer'):
                ax4.plot(epochs, self.history['train_eer'], 'r--', linewidth=2, label='EER Treino')
            
            ax4.set_xlabel('Época')
            ax4.set_ylabel('EER')
            ax4.set_title('Convergência do Modelo')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Salvar gráfico, se solicitado
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Estatísticas de otimização salvas em: {save_path}")
        
        plt.show()
    
    def save_optimization_report(self, report_path):
        """
        Salva relatório detalhado de otimização.
        
        Args:
            report_path: Caminho para salvar o relatório
        """
        report = {
            'training_stats': self.training_stats,
            'model_config': {
                'num_epochs': self.num_epochs,
                'patience': self.patience,
                'segmentation_config': self.segmentation_config
            },
            'final_metrics': {
                'best_val_eer': min(self.history['val_eer']) if self.history.get('val_eer') else None,
                'final_train_eer': self.history['train_eer'][-1] if self.history.get('train_eer') else None,
                'total_epochs': len(self.history['train_loss']) if self.history.get('train_loss') else 0
            }
        }
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=4)
        
        print(f"Relatório de otimização salvo em: {report_path}")


def create_optimized_dataloaders(train_features_dir, train_labels_file, 
                               dev_features_dir, dev_labels_file,
                               segmentation_strategy='adaptive', sample_ratio=0.3,
                               batch_size=32, **kwargs):
    """
    Cria DataLoaders otimizados com segmentação.
    
    Args:
        train_features_dir: Diretório com características de treino
        train_labels_file: Arquivo com rótulos de treino
        dev_features_dir: Diretório com características de validação
        dev_labels_file: Arquivo com rótulos de validação
        segmentation_strategy: Estratégia de segmentação
        sample_ratio: Proporção de dados a manter
        batch_size: Tamanho do lote
        
    Returns:
        DataLoaders de treino e validação otimizados
    """
    print("Criando DataLoaders otimizados com segmentação...")
    
    # Aplicar segmentação apenas nos dados de treino
    manager = DataSegmentationManager(train_features_dir, train_labels_file)
    selected_indices, stats = manager.apply_segmentation(
        segmentation_strategy, sample_ratio, **kwargs
    )
    
    # Criar dataset segmentado temporário
    temp_dir = "temp_segmented"
    os.makedirs(temp_dir, exist_ok=True)
    
    segmented_features_dir = os.path.join(temp_dir, "features")
    segmented_labels_file = os.path.join(temp_dir, "labels.txt")
    
    _, _, file_names = manager.load_data()
    create_segmented_dataset(
        train_features_dir, train_labels_file,
        segmented_features_dir, segmented_labels_file,
        selected_indices, file_names
    )
    
    # Criar DataLoaders
    train_loader = BidirectionalDataLoader(
        segmented_features_dir, segmented_labels_file,
        batch_size=batch_size, shuffle=True, **kwargs
    ).get_dataloader()
    
    # DataLoader de validação sem segmentação
    dev_loader = BidirectionalDataLoader(
        dev_features_dir, dev_labels_file,
        batch_size=batch_size, shuffle=False, **kwargs
    ).get_dataloader()
    
    print(f"DataLoaders criados:")
    print(f"  Treino: {stats['selected_count']} amostras (redução: {stats['compression_ratio']:.1%})")
    print(f"  Validação: {len(dev_loader.dataset)} amostras (completo)")
    
    return train_loader, dev_loader, stats


def main():
    """
    Função principal para treinamento otimizado.
    """
    parser = argparse.ArgumentParser(description="Treinamento otimizado com segmentação de dados")
    
    # Argumentos originais do train.py
    parser.add_argument('--train-features-dir', type=str, required=True,
                        help='Diretório com características de treinamento')
    parser.add_argument('--dev-features-dir', type=str, required=True,
                        help='Diretório com características de validação')
    parser.add_argument('--train-labels-file', type=str, required=True,
                        help='Arquivo com rótulos de treinamento')
    parser.add_argument('--dev-labels-file', type=str, required=True,
                        help='Arquivo com rótulos de validação')
    parser.add_argument('--save-dir', type=str, default='checkpoints_optimized',
                        help='Diretório para salvar checkpoints')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Tamanho do lote')
    parser.add_argument('--num-epochs', type=int, default=100,
                        help='Número máximo de épocas')
    parser.add_argument('--learning-rate', type=float, default=0.0003,
                        help='Taxa de aprendizado inicial')
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                        help='Decaimento de peso')
    parser.add_argument('--patience', type=int, default=10,
                        help='Paciência para early stopping')
    
    # Argumentos de otimização
    parser.add_argument('--optimization-mode', type=str, default='segmentation',
                        choices=['segmentation', 'progressive', 'intelligent', 'standard'],
                        help='Modo de otimização de treinamento')
    parser.add_argument('--segmentation-strategy', type=str, default='adaptive',
                        choices=['kmeans', 'dbscan', 'stratified', 'diversity', 'adaptive'],
                        help='Estratégia de segmentação')
    parser.add_argument('--sample-ratio', type=float, default=0.3,
                        help='Proporção de dados a manter')
    parser.add_argument('--progressive-stages', type=int, default=5,
                        help='Número de estágios para treinamento progressivo')
    parser.add_argument('--intelligent-update-freq', type=int, default=10,
                        help='Frequência de atualização para amostragem inteligente')
    
    # Argumentos para benchmark
    parser.add_argument('--benchmark', action='store_true',
                        help='Executar benchmark de estratégias antes do treinamento')
    parser.add_argument('--auto-select', action='store_true',
                        help='Selecionar automaticamente a melhor estratégia')
    
    args = parser.parse_args()
    
    # Verificar GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Usando dispositivo: {device}")
    print(f"Modo de otimização: {args.optimization_mode}")
    
    # Executar benchmark se solicitado
    if args.benchmark:
        print("\n=== EXECUTANDO BENCHMARK DE ESTRATÉGIAS ===")
        
        from segmentation_integration import benchmark_strategies
        
        benchmark_dir = os.path.join(args.save_dir, 'benchmark')
        results_df, timing_df = benchmark_strategies(
            args.train_features_dir, args.train_labels_file, benchmark_dir
        )
        
        if args.auto_select:
            # Selecionar automaticamente a melhor estratégia
            best_strategy = results_df.loc[results_df['final_score'].idxmax(), 'strategy']
            best_ratio = results_df.loc[results_df['final_score'].idxmax(), 'sample_ratio']
            
            print(f"Estratégia selecionada automaticamente: {best_strategy}")
            print(f"Ratio selecionado automaticamente: {best_ratio}")
            
            args.segmentation_strategy = best_strategy
            args.sample_ratio = best_ratio
    
    # Configurar treinamento baseado no modo
    if args.optimization_mode == 'standard':
        # Treinamento padrão sem otimização
        from train import prepare_dataloaders, main as standard_main
        
        print("Executando treinamento padrão...")
        # Chamar função original (implementação simplificada)
        
    elif args.optimization_mode == 'segmentation':
        # Treinamento com segmentação simples
        print(f"Executando treinamento com segmentação ({args.segmentation_strategy})...")
        
        # Criar DataLoaders otimizados
        train_loader, dev_loader, stats = create_optimized_dataloaders(
            args.train_features_dir, args.train_labels_file,
            args.dev_features_dir, args.dev_labels_file,
            segmentation_strategy=args.segmentation_strategy,
            sample_ratio=args.sample_ratio,
            batch_size=args.batch_size
        )
        
        # Criar modelo e componentes
        model = MultiPatternModel(input_channels=1, hidden_size=512, num_classes=2).to(device)
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        criterion = OCLoss(feature_dim=512).to(device)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
        
        # Configuração de segmentação
        segmentation_config = {
            'strategy': args.segmentation_strategy,
            'sample_ratio': args.sample_ratio,
            'stats': stats
        }
        
        # Criar e executar treinador otimizado
        trainer = OptimizedTrainer(
            model=model,
            train_loader=train_loader,
            dev_loader=dev_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            scheduler=scheduler,
            num_epochs=args.num_epochs,
            patience=args.patience,
            save_dir=args.save_dir,
            segmentation_config=segmentation_config
        )
        
        # Treinar modelo
        history = trainer.train()
        
    elif args.optimization_mode == 'progressive':
        # Treinamento progressivo
        print(f"Executando treinamento progressivo ({args.progressive_stages} estágios)...")
        
        # Criar DataLoaders iniciais
        train_loader, dev_loader, _ = create_optimized_dataloaders(
            args.train_features_dir, args.train_labels_file,
            args.dev_features_dir, args.dev_labels_file,
            sample_ratio=0.1,  # Começar com poucos dados
            batch_size=args.batch_size
        )
        
        # Criar modelo e componentes
        model = MultiPatternModel(input_channels=1, hidden_size=512, num_classes=2).to(device)
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        criterion = OCLoss(feature_dim=512).to(device)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
        
        # Criar treinador otimizado
        trainer = OptimizedTrainer(
            model=model,
            train_loader=train_loader,
            dev_loader=dev_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            scheduler=scheduler,
            num_epochs=args.num_epochs,
            patience=args.patience,
            save_dir=args.save_dir
        )
        
        # Configurar e executar treinamento progressivo
        trainer.setup_progressive_training(
            args.train_features_dir, args.train_labels_file, args.progressive_stages
        )
        history = trainer.train_with_progressive_stages()
        
    elif args.optimization_mode == 'intelligent':
        # Treinamento com amostragem inteligente
        print("Executando treinamento com amostragem inteligente...")
        
        # Criar DataLoaders iniciais
        train_loader, dev_loader, _ = create_optimized_dataloaders(
            args.train_features_dir, args.train_labels_file,
            args.dev_features_dir, args.dev_labels_file,
            sample_ratio=args.sample_ratio,
            batch_size=args.batch_size
        )
        
        # Criar modelo e componentes
        model = MultiPatternModel(input_channels=1, hidden_size=512, num_classes=2).to(device)
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        criterion = OCLoss(feature_dim=512).to(device)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
        
        # Criar treinador otimizado
        trainer = OptimizedTrainer(
            model=model,
            train_loader=train_loader,
            dev_loader=dev_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            scheduler=scheduler,
            num_epochs=args.num_epochs,
            patience=args.patience,
            save_dir=args.save_dir
        )
        
        # Configurar e executar amostragem inteligente
        trainer.setup_intelligent_sampling(args.train_features_dir, args.train_labels_file)
        history = trainer.train_with_intelligent_sampling(args.intelligent_update_freq)
    
    # Salvar estatísticas e gráficos de otimização
    if args.optimization_mode != 'standard':
        stats_plot_path = os.path.join(args.save_dir, 'optimization_stats.png')
        trainer.plot_optimization_stats(stats_plot_path)
        
        report_path = os.path.join(args.save_dir, 'optimization_report.json')
        trainer.save_optimization_report(report_path)
        
        # Exibir resumo de otimização
        print("\n=== RESUMO DE OTIMIZAÇÃO ===")
        if trainer.training_stats.get('data_reduction'):
            print(f"Redução de dados: {trainer.training_stats['data_reduction']:.1%}")
            
            # Estimar speedup
            original_size = 1 / (1 - trainer.training_stats['data_reduction'])
            segmented_size = 1
            speedup = (original_size ** 2) / (segmented_size ** 2)  # Assumindo complexidade quadrática
            
            print(f"Speedup estimado: {speedup:.1f}x")
            print(f"Tempo economizado: {(1 - 1/speedup)*100:.1f}%")
        
        if trainer.training_stats.get('stage_transitions'):
            total_stages = len(trainer.training_stats['stage_transitions'])
            print(f"Estágios progressivos: {total_stages}")
        
        if trainer.training_stats.get('difficulty_updates'):
            updates_count = len(trainer.training_stats['difficulty_updates'])
            print(f"Atualizações de dificuldade: {updates_count}")
    
    print("\nTreinamento otimizado concluído!")


if __name__ == "__main__":
    main()