#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Módulo para implementação da segmentação bidirecional para entrada de múltiplos pontos
em redes neurais convolucionais para detecção de ataques de replay.

Esta implementação é baseada na abordagem proposta por Yoon & Yu (2020).

Autores: André Thiago de Souza, Felipe de Lima dos Santos, 
         Juliano Gaona Proença, Matheus Henrique Reich Favarin Zagonel
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import os
import pickle
import random # Importar random para amostragem

class BidirectionalSegmentation:
    """
    Classe para implementação de segmentação bidirecional em características de áudio,
    permitindo a análise de múltiplos pontos da elocução simultaneamente.
    """
    
    def __init__(self, segment_length=400, stride=200):
        """
        Inicializa o módulo de segmentação bidirecional.
        
        Args:
            segment_length: Comprimento do segmento em frames (padrão: 400)
            stride: Tamanho do salto entre segmentos consecutivos (padrão: 200)
        """
        self.segment_length = segment_length
        self.stride = stride
    
    def segment_features(self, features):
        """
        Aplica segmentação bidirecional às características.
        
        Args:
            features: Matriz de características (tempo x dimensão)
            
        Returns:
            Lista de pares de segmentos (forward, backward)
        """
        T, D = features.shape
        
        # Se o comprimento for menor que segment_length, padding
        if T < self.segment_length:
            # Repetir os últimos frames para completar segment_length
            padding = self.segment_length - T
            padded_features = np.pad(features, ((0, padding), (0, 0)), mode='wrap')
            
            # Criar um único par de segmentos (forward, backward)
            forward_segment = padded_features
            backward_segment = np.flip(padded_features, axis=0)
            
            return [(forward_segment, backward_segment)]
        
        # Caso contrário, segmentar com stride
        else:
            # Crie a versão invertida das características para segmentação backward
            flipped_features = np.flip(features, axis=0)
            
            # Calcular o número de segmentos completos
            n_complete_segments = (T - self.segment_length) // self.stride + 1
            
            # Inicializar a lista de pares de segmentos
            segment_pairs = []
            
            # Gerar segmentos completos
            for i in range(n_complete_segments):
                start = i * self.stride
                end = start + self.segment_length
                
                forward_segment = features[start:end]
                backward_segment = flipped_features[start:end]
                
                segment_pairs.append((forward_segment, backward_segment))
            
            # Lidar com o restante dos frames
            remainder = T % self.segment_length
            if remainder > 0:
                # Pegar os últimos segment_length frames
                forward_segment = features[-self.segment_length:]
                backward_segment = flipped_features[-self.segment_length:]
                
                segment_pairs.append((forward_segment, backward_segment))
            
            return segment_pairs

    def segment_batch(self, batch_features):
        """
        Aplica segmentação bidirecional a um lote de características.
        
        Args:
            batch_features: Lista de matrizes de características
            
        Returns:
            Lista de listas de pares de segmentos
        """
        return [self.segment_features(features) for features in batch_features]


class BidirectionalDataset(Dataset):
    """
    Dataset que implementa a segmentação bidirecional para uso com PyTorch DataLoader.
    """
    
    def __init__(self, features_dir, labels_file, segment_length=400, stride=200, transform=None, sample_proportion=1.0):
        """
        Inicializa o dataset.
        
        Args:
            features_dir: Diretório contendo os arquivos de características (.npz)
            labels_file: Arquivo contendo os rótulos dos arquivos
            segment_length: Comprimento do segmento em frames
            stride: Tamanho do salto entre segmentos consecutivos
            transform: Transformações a serem aplicadas às características
            sample_proportion: Proporção do dataset a ser usado (0.0 a 1.0).
                               Amostragem proporcional por classe é aplicada.
        """
        self.features_dir = features_dir
        self.transform = transform
        self.bidirectional_segmentation = BidirectionalSegmentation(segment_length, stride)
        
        # Carregar lista de arquivos e rótulos
        all_files_labels = []
        with open(labels_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    file_id = parts[0]
                    label = 1 if parts[1] == 'spoof' else 0  # 1 para spoof, 0 para genuine
                    
                    npz_path = os.path.join(features_dir, f"{file_id}.npz")
                    if os.path.exists(npz_path):
                        all_files_labels.append({'file_id': file_id, 'label': label, 'npz_path': npz_path})
        
        # Aplicar amostragem se a proporção for menor que 1.0
        if 0 < sample_proportion < 1.0:
            genuine_samples = [item for item in all_files_labels if item['label'] == 0]
            spoof_samples = [item for item in all_files_labels if item['label'] == 1]

            num_genuine_to_sample = int(len(genuine_samples) * sample_proportion)
            num_spoof_to_sample = int(len(spoof_samples) * sample_proportion)

            # Garantir que não tentamos amostrar mais do que o disponível
            num_genuine_to_sample = min(num_genuine_to_sample, len(genuine_samples))
            num_spoof_to_sample = min(num_spoof_to_sample, len(spoof_samples))
            
            sampled_genuine = random.sample(genuine_samples, num_genuine_to_sample)
            sampled_spoof = random.sample(spoof_samples, num_spoof_to_sample)
            
            sampled_files_labels = sampled_genuine + sampled_spoof
            random.shuffle(sampled_files_labels) # Embaralhar para misturar genuínos e spoof
            
            self.files = [item['npz_path'] for item in sampled_files_labels]
            self.labels = [item['label'] for item in sampled_files_labels]
            print(f"Dataset reduzido para {len(self.files)} arquivos ({num_genuine_to_sample} genuínos, {num_spoof_to_sample} spoof) com proporção {sample_proportion*100:.2f}%")
        else:
            self.files = [item['npz_path'] for item in all_files_labels]
            self.labels = [item['label'] for item in all_files_labels]
            print(f"Usando o dataset completo com {len(self.files)} arquivos.")

        # Pré-processar dados para criar índices de segmentos
        self.segment_indices = []
        self.file_segments = []
        
        for file_idx, npz_path in enumerate(self.files):
            # Carregar características
            features_dict = np.load(npz_path)
            
            # Combinar MFCC e CQCC para características híbridas
            # Assumindo que 'mfcc' e 'cqcc' são as chaves corretas no .npz
            # Verificar se as chaves existem para evitar KeyError
            mfcc = features_dict['mfcc'].T if 'mfcc' in features_dict else np.array([])
            cqcc = features_dict['cqcc'].T if 'cqcc' in features_dict else np.array([])

            # Lidar com casos onde uma das características pode estar vazia após a transposição
            if mfcc.size == 0 and cqcc.size == 0:
                print(f"Aviso: Nenhuma característica MFCC ou CQCC encontrada para {npz_path}. Pulando.")
                continue
            elif mfcc.size == 0:
                hybrid_features = cqcc
            elif cqcc.size == 0:
                hybrid_features = mfcc
            else:
                # Garantir que as dimensões correspondam para concatenação (número de frames)
                if mfcc.shape[0] != cqcc.shape[0]:
                    min_frames = min(mfcc.shape[0], cqcc.shape[0])
                    mfcc = mfcc[:min_frames, :]
                    cqcc = cqcc[:min_frames, :]
                hybrid_features = np.concatenate([mfcc, cqcc], axis=1)
            
            if hybrid_features.shape[0] == 0:
                print(f"Aviso: Características híbridas vazias para {npz_path}. Pulando.")
                continue

            # Segmentar características
            segments = self.bidirectional_segmentation.segment_features(hybrid_features)
            self.file_segments.append(segments)
            
            # Criar índices para todos os segmentos deste arquivo
            for seg_idx in range(len(segments)):
                self.segment_indices.append((file_idx, seg_idx))
    
    def __len__(self):
        """
        Retorna o número total de segmentos.
        """
        return len(self.segment_indices)
    
    def __getitem__(self, idx):
        """
        Retorna um par de segmentos (forward, backward) e seu rótulo.
        
        Args:
            idx: Índice do segmento
            
        Returns:
            Tupla (forward_segment, backward_segment, label)
        """
        file_idx, seg_idx = self.segment_indices[idx]
        segments = self.file_segments[file_idx][seg_idx]
        label = self.labels[file_idx]
        
        forward_segment, backward_segment = segments
        
        # Aplicar transformações, se houver
        if self.transform:
            forward_segment = self.transform(forward_segment)
            backward_segment = self.transform(backward_segment)
        
        # Converter para tensor do PyTorch
        forward_segment = torch.FloatTensor(forward_segment)
        backward_segment = torch.FloatTensor(backward_segment)
        
        return forward_segment, backward_segment, label


class BidirectionalDataLoader:
    """
    Wrapper para gerenciar DataLoaders com segmentação bidirecional.
    """
    
    def __init__(self, features_dir, labels_file, batch_size=32, segment_length=400, 
                 stride=200, num_workers=4, shuffle=True, transform=None, sample_proportion=1.0):
        """
        Inicializa o gerenciador de DataLoader com segmentação bidirecional.
        
        Args:
            features_dir: Diretório contendo os arquivos de características (.npz)
            labels_file: Arquivo contendo os rótulos dos arquivos
            batch_size: Tamanho do lote para o DataLoader
            segment_length: Comprimento do segmento em frames
            stride: Tamanho do salto entre segmentos consecutivos
            num_workers: Número de processos para carregamento paralelo
            shuffle: Se True, embaralha os dados a cada época
            transform: Transformações a serem aplicadas às características
            sample_proportion: Proporção do dataset a ser usado (0.0 a 1.0).
        """
        # Criar dataset
        self.dataset = BidirectionalDataset(
            features_dir, 
            labels_file, 
            segment_length, 
            stride, 
            transform,
            sample_proportion # Passar sample_proportion para o Dataset
        )
        
        # Criar dataloader
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True
        )
    
    def get_dataloader(self):
        """Retorna o DataLoader configurado."""
        return self.dataloader
    
    def get_dataset(self):
        """Retorna o Dataset configurado."""
        return self.dataset


# Exemplo de uso
if __name__ == "__main__":
    import numpy as np
    
    # Criar dados de exemplo
    features = np.random.randn(600, 90)  # 600 frames, 90 dimensões
    
    # Inicializar o segmentador bidirecional
    segmenter = BidirectionalSegmentation(segment_length=400, stride=200)
    
    # Segmentar características
    segment_pairs = segmenter.segment_features(features)
    
    # Exibir informações sobre os segmentos
    print(f"Número de pares de segmentos: {len(segment_pairs)}")
    
    for i, (forward, backward) in enumerate(segment_pairs):
        print(f"Par {i+1}:")
        print(f"  Forward: shape={forward.shape}")
        print(f"  Backward: shape={backward.shape}")
        
        # Verificar se backward é realmente o inverso de forward para o primeiro par
        if i == 0:
            first_frame_forward = forward[0]
            last_frame_backward = backward[-1]
            print(f"  O primeiro frame de forward é igual ao último de backward? {np.array_equal(first_frame_forward, last_frame_backward)}")

