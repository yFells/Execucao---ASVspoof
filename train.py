#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script para treinamento do modelo de detecção de ataques de replay.

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
from feature_extraction import FeatureExtractor
from bidirectional_segmentation import BidirectionalDataset, BidirectionalDataLoader
from model import MultiPatternModel, OCLoss
from evaluation import ReplayAttackEvaluator


class Trainer:
    """
    Classe para treinamento do modelo de detecção de ataques de replay.
    """
    
    def __init__(self, model, train_loader, dev_loader, optimizer, criterion, 
                 device, scheduler=None, num_epochs=100, patience=10,
                 save_dir='checkpoints'):
        """
        Inicializa o treinador.
        
        Args:
            model: Modelo a ser treinado
            train_loader: DataLoader para dados de treinamento
            dev_loader: DataLoader para dados de validação
            optimizer: Otimizador
            criterion: Função de perda
            device: Dispositivo para treinamento (CPU ou GPU)
            scheduler: Agendador de taxa de aprendizado (opcional)
            num_epochs: Número máximo de épocas
            patience: Paciência para early stopping
            save_dir: Diretório para salvar checkpoints
        """
        self.model = model
        self.train_loader = train_loader
        self.dev_loader = dev_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.scheduler = scheduler
        self.num_epochs = num_epochs
        self.patience = patience
        self.save_dir = save_dir
        
        # Criar diretório para salvar checkpoints
        os.makedirs(save_dir, exist_ok=True)
        
        # Inicializar histórico de treinamento
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_eer': [],
            'val_eer': [],
            'train_auc': [],
            'val_auc': [],
            'learning_rate': []
        }
        
        # Inicializar avaliador
        self.evaluator = ReplayAttackEvaluator(model, device)
    
    def train_epoch(self):
        """
        Treina o modelo por uma época.
        
        Returns:
            Perda média da época
        """
        self.model.train()
        epoch_loss = 0
        
        # Lista para armazenar pontuações e rótulos para cálculo de EER e AUC
        all_scores = []
        all_labels = []
        
        # Iterar sobre os lotes de treinamento
        for lbp, glcm, lpq, labels in tqdm(self.train_loader, desc="Treinando"):
            # Mover dados para o dispositivo
            lbp = lbp.to(self.device)
            glcm = glcm.to(self.device)
            lpq = lpq.to(self.device)
            labels = labels.to(self.device)
            
            # Zerar gradientes
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(lbp, glcm, lpq, labels)
            loss = outputs['loss']
            
            # Backward pass e otimização
            loss.backward()
            self.optimizer.step()
            
            # Acumular perda
            epoch_loss += loss.item()
            
            # Coletar pontuações para cálculo de métricas
            scores = self.model.get_scores(lbp, glcm, lpq)
            fused_scores = (
                self.evaluator.score_weights['lbp'] * scores['lbp_score'] +
                self.evaluator.score_weights['glcm'] * scores['glcm_score'] +
                self.evaluator.score_weights['lpq'] * scores['lpq_score']
            )
            
            all_scores.extend(fused_scores.detach().cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
        
        # Calcular perda média
        epoch_loss /= len(self.train_loader)
        
        # Calcular EER e AUC
        all_scores = np.array(all_scores)
        all_labels = np.array(all_labels)
        
        eer, _ = self.evaluator.compute_eer(all_scores, all_labels)
        auc = roc_auc_score(all_labels, all_scores)
        
        return epoch_loss, eer, auc
    
    def validate(self):
        """
        Valida o modelo no conjunto de validação.
        
        Returns:
            Perda média, EER e AUC no conjunto de validação
        """
        self.model.eval()
        val_loss = 0
        
        # Lista para armazenar pontuações e rótulos para cálculo de EER e AUC
        all_scores = []
        all_labels = []
        
        with torch.no_grad():
            for lbp, glcm, lpq, labels in tqdm(self.dev_loader, desc="Validando"):
                # Mover dados para o dispositivo
                lbp = lbp.to(self.device)
                glcm = glcm.to(self.device)
                lpq = lpq.to(self.device)
                labels = labels.to(self.device)
                
                # Forward pass
                outputs = self.model(lbp, glcm, lpq, labels)
                loss = outputs['loss']
                
                # Acumular perda
                val_loss += loss.item()
                
                # Coletar pontuações para cálculo de métricas
                scores = self.model.get_scores(lbp, glcm, lpq)
                fused_scores = (
                    self.evaluator.score_weights['lbp'] * scores['lbp_score'] +
                    self.evaluator.score_weights['glcm'] * scores['glcm_score'] +
                    self.evaluator.score_weights['lpq'] * scores['lpq_score']
                )
                
                all_scores.extend(fused_scores.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Calcular perda média
        val_loss /= len(self.dev_loader)
        
        # Calcular EER e AUC
        all_scores = np.array(all_scores)
        all_labels = np.array(all_labels)
        
        eer, _ = self.evaluator.compute_eer(all_scores, all_labels)
        auc = roc_auc_score(all_labels, all_scores)
        
        return val_loss, eer, auc
    
    def train(self):
        """
        Treina o modelo por várias épocas com early stopping.
        
        Returns:
            Histórico de treinamento
        """
        print(f"Iniciando treinamento por {self.num_epochs} épocas...")
        
        # Inicializar métricas para early stopping
        best_val_eer = float('inf')
        best_epoch = 0
        epochs_without_improvement = 0
        
        # Timer para medir tempo total de treinamento
        start_time = time.time()
        
        # Loop de épocas
        for epoch in range(1, self.num_epochs + 1):
            print(f"\nÉpoca {epoch}/{self.num_epochs}")
            
            # Treinar por uma época
            train_loss, train_eer, train_auc = self.train_epoch()
            
            # Validar o modelo
            val_loss, val_eer, val_auc = self.validate()
            
            # Atualizar o agendador, se fornecido
            current_lr = self.optimizer.param_groups[0]['lr']
            if self.scheduler:
                self.scheduler.step(val_eer)  # Considerando EER como métrica para ajustar LR
            
            # Registrar métricas
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_eer'].append(train_eer)
            self.history['val_eer'].append(val_eer)
            self.history['train_auc'].append(train_auc)
            self.history['val_auc'].append(val_auc)
            self.history['learning_rate'].append(current_lr)
            
            # Exibir métricas
            print(f"Perda Treino: {train_loss:.4f}, EER Treino: {train_eer:.4f}, AUC Treino: {train_auc:.4f}")
            print(f"Perda Val: {val_loss:.4f}, EER Val: {val_eer:.4f}, AUC Val: {val_auc:.4f}")
            print(f"Taxa de Aprendizado: {current_lr}")
            
            # Verificar se é o melhor modelo até agora
            if val_eer < best_val_eer:
                best_val_eer = val_eer
                best_epoch = epoch
                epochs_without_improvement = 0
                
                # Salvar o melhor modelo
                self.save_checkpoint(epoch, val_eer, is_best=True)
                print(f"Novo melhor modelo (EER: {val_eer:.4f})")
            else:
                epochs_without_improvement += 1
                
                # Salvar checkpoint regular
                self.save_checkpoint(epoch, val_eer)
                
                # Verificar se deve parar o treinamento (early stopping)
                if epochs_without_improvement >= self.patience:
                    print(f"Early stopping após {epoch} épocas sem melhoria.")
                    break
        
        # Calcular tempo total de treinamento
        total_time = time.time() - start_time
        hours, remainder = divmod(total_time, 3600)
        minutes, seconds = divmod(remainder, 60)
        
        print(f"\nTreinamento concluído em {int(hours)}h {int(minutes)}m {int(seconds)}s")
        print(f"Melhor modelo na época {best_epoch} com EER de validação: {best_val_eer:.4f}")
        
        # Plotar histórico de treinamento
        self.plot_history()
        
        return self.history
    
    def save_checkpoint(self, epoch, val_eer, is_best=False):
        """
        Salva checkpoint do modelo.
        
        Args:
            epoch: Época atual
            val_eer: EER de validação
            is_best: Se é o melhor modelo até agora
        """
        # Criar checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_eer': val_eer,
            'score_weights': self.evaluator.score_weights
        }
        
        # Adicionar estado do agendador, se existir
        if self.scheduler:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        # Salvar checkpoint
        if is_best:
            torch.save(checkpoint, os.path.join(self.save_dir, 'best_model.pth'))
        
        # Salvar checkpoint da época atual
        torch.save(checkpoint, os.path.join(self.save_dir, f'checkpoint_epoch_{epoch}.pth'))
    
    def plot_history(self, save_path=None):
        """
        Plota o histórico de treinamento.
        
        Args:
            save_path: Caminho para salvar o gráfico (opcional)
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Plotar perda
        axes[0, 0].plot(self.history['train_loss'], label='Treino')
        axes[0, 0].plot(self.history['val_loss'], label='Validação')
        axes[0, 0].set_title('Perda')
        axes[0, 0].set_xlabel('Época')
        axes[0, 0].set_ylabel('Perda')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Plotar EER
        axes[0, 1].plot(self.history['train_eer'], label='Treino')
        axes[0, 1].plot(self.history['val_eer'], label='Validação')
        axes[0, 1].set_title('Equal Error Rate (EER)')
        axes[0, 1].set_xlabel('Época')
        axes[0, 1].set_ylabel('EER')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Plotar AUC
        axes[1, 0].plot(self.history['train_auc'], label='Treino')
        axes[1, 0].plot(self.history['val_auc'], label='Validação')
        axes[1, 0].set_title('Area Under Curve (AUC)')
        axes[1, 0].set_xlabel('Época')
        axes[1, 0].set_ylabel('AUC')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Plotar taxa de aprendizado
        axes[1, 1].plot(self.history['learning_rate'])
        axes[1, 1].set_title('Taxa de Aprendizado')
        axes[1, 1].set_xlabel('Época')
        axes[1, 1].set_ylabel('Taxa')
        axes[1, 1].set_yscale('log')
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        # Salvar gráfico, se solicitado
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        plt.show()
    
    @staticmethod
    def load_model(model, checkpoint_path, device):
        """
        Carrega modelo a partir de checkpoint.
        
        Args:
            model: Modelo a ser carregado
            checkpoint_path: Caminho para o checkpoint
            device: Dispositivo para carregar o modelo
            
        Returns:
            Modelo carregado e pesos para fusão de pontuações
        """
        # Carregar checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Carregar estado do modelo
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        
        # Extrair pesos para fusão de pontuações
        score_weights = checkpoint.get('score_weights', {'lbp': 0.33, 'glcm': 0.33, 'lpq': 0.34})
        
        return model, score_weights


def prepare_dataloaders(train_features_dir, train_labels_file, dev_features_dir, dev_labels_file, 
                     batch_size=32, segment_length=400, stride=200, num_workers=4):
    """
    Prepara DataLoaders para treinamento e validação.
    
    Args:
        train_features_dir: Diretório com características de treinamento
        train_labels_file: Arquivo com rótulos de treinamento
        dev_features_dir: Diretório com características de validação
        dev_labels_file: Arquivo com rótulos de validação
        batch_size: Tamanho do lote
        segment_length: Comprimento do segmento em frames
        stride: Tamanho do salto entre segmentos consecutivos
        num_workers: Número de processos para carregamento paralelo
        
    Returns:
        DataLoaders para treinamento e validação
    """
    # Criar DataLoaders
    train_loader = BidirectionalDataLoader(
        train_features_dir, train_labels_file, batch_size, segment_length, stride, num_workers, shuffle=True
    ).get_dataloader()
    
    dev_loader = BidirectionalDataLoader(
        dev_features_dir, dev_labels_file, batch_size, segment_length, stride, num_workers, shuffle=False
    ).get_dataloader()
    
    return train_loader, dev_loader


def main(args):
    """
    Função principal para treinamento do modelo.
    
    Args:
        args: Argumentos de linha de comando
    """
    # Verificar GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Usando dispositivo: {device}")
    
    # Extrair características, se solicitado
    if args.extract_features:
        print("Extraindo características...")
        extractor = FeatureExtractor(
            sample_rate=16000, n_mfcc=30, n_cqcc=30, n_mels=257,
            window_size=0.025, hop_size=0.010, pre_emphasis=0.97
        )
        
        # Criar diretórios para salvar características
        os.makedirs(args.train_features_dir, exist_ok=True)
        os.makedirs(args.dev_features_dir, exist_ok=True)
        
        # Extrair características para treinamento e validação
        train_features = extractor.batch_feature_extraction(
            args.train_audio_dir, args.train_features_dir, args.audio_ext
        )
        
        dev_features = extractor.batch_feature_extraction(
            args.dev_audio_dir, args.dev_features_dir, args.audio_ext
        )
        
        print(f"Características de treinamento extraídas para {len(train_features)} arquivos")
        print(f"Características de validação extraídas para {len(dev_features)} arquivos")
    
    # Preparar DataLoaders
    print("Preparando DataLoaders...")
    train_loader, dev_loader = prepare_dataloaders(
        args.train_features_dir, args.train_labels_file,
        args.dev_features_dir, args.dev_labels_file,
        args.batch_size, args.segment_length, args.stride, args.num_workers
    )
    
    # Criar modelo
    print("Criando modelo...")
    model = MultiPatternModel(input_channels=1, hidden_size=512, num_classes=2).to(device)
    
    # Definir otimizador
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    
    # Definir função de perda
    criterion = OCLoss(feature_dim=512, alpha=20.0, m0=0.9, m1=0.2).to(device)
    
    # Definir agendador de taxa de aprendizado
    scheduler = ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6, verbose=True
    )
    
    # Inicializar pesos para fusão de pontuações
    score_weights = {'lbp': 0.33, 'glcm': 0.33, 'lpq': 0.34}
    
    # Carregar checkpoint, se fornecido
    if args.checkpoint:
        print(f"Carregando checkpoint: {args.checkpoint}")
        model, score_weights = Trainer.load_model(model, args.checkpoint, device)
    
    # Inicializar treinador
    trainer = Trainer(
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
    
    # Atualizar pesos para fusão de pontuações
    trainer.evaluator.score_weights = score_weights
    
    # Treinar modelo
    history = trainer.train()
    
    # Salvar histórico de treinamento
    history_path = os.path.join(args.save_dir, 'training_history.json')
    with open(history_path, 'w') as f:
        # Converter arrays numpy para listas
        history_json = {k: v if not isinstance(v, np.ndarray) else v.tolist() for k, v in history.items()}
        json.dump(history_json, f, indent=4)
    
    print(f"Histórico de treinamento salvo em: {history_path}")


if __name__ == "__main__":
    # Configurar parser de argumentos
    parser = argparse.ArgumentParser(description="Treinamento de modelo para detecção de ataques de replay")
    
    # Argumentos para entrada de dados
    parser.add_argument('--train-audio-dir', type=str, default='data/ASVspoof2019/train/audio',
                        help='Diretório com arquivos de áudio de treinamento')
    parser.add_argument('--dev-audio-dir', type=str, default='data/ASVspoof2019/dev/audio',
                        help='Diretório com arquivos de áudio de validação')
    parser.add_argument('--train-features-dir', type=str, default='data/ASVspoof2019/train/features',
                        help='Diretório para salvar características de treinamento')
    parser.add_argument('--dev-features-dir', type=str, default='data/ASVspoof2019/dev/features',
                        help='Diretório para salvar características de validação')
    parser.add_argument('--train-labels-file', type=str, default='data/ASVspoof2019/train/labels.txt',
                        help='Arquivo com rótulos de treinamento')
    parser.add_argument('--dev-labels-file', type=str, default='data/ASVspoof2019/dev/labels.txt',
                        help='Arquivo com rótulos de validação')
    parser.add_argument('--audio-ext', type=str, default='.flac',
                        help='Extensão dos arquivos de áudio')
    
    # Argumentos para extração de características
    parser.add_argument('--extract-features', action='store_true',
                        help='Extrair características antes do treinamento')
    
    # Argumentos para segmentação
    parser.add_argument('--segment-length', type=int, default=400,
                        help='Comprimento do segmento em frames')
    parser.add_argument('--stride', type=int, default=200,
                        help='Tamanho do salto entre segmentos consecutivos')
    
    # Argumentos para treinamento
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Tamanho do lote')
    parser.add_argument('--num-epochs', type=int, default=100,
                        help='Número máximo de épocas')
    parser.add_argument('--learning-rate', type=float, default=0.0003,
                        help='Taxa de aprendizado inicial')
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                        help='Decaimento de peso para regularização L2')
    parser.add_argument('--patience', type=int, default=10,
                        help='Paciência para early stopping')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Caminho para checkpoint para continuar treinamento')
    parser.add_argument('--save-dir', type=str, default='checkpoints',
                        help='Diretório para salvar checkpoints')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Número de processos para carregamento paralelo')
    
    # Analisar argumentos
    args = parser.parse_args()
    
    # Executar função principal
    main(args)
