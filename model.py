#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Implementação de arquitetura da rede neural modificada com ResNet18 e mecanismo de atenção
para detecção de ataques de replay em sistemas de verificação automática de locutor.

Autores: André Thiago de Souza, Felipe de Lima dos Santos, 
         Juliano Gaona Proença, Matheus Henrique Reich Favarin Zagonel
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttention(nn.Module):
    """
    Mecanismo de autoatenção temporal para processar entradas de comprimento variável
    e atribuir coeficientes mais altos às partes relevantes da entrada.
    """
    
    def __init__(self, hidden_size):
        """
        Inicializa a camada de autoatenção.
        
        Args:
            hidden_size: Dimensão do espaço de características
        """
        super(SelfAttention, self).__init__()
        
        # Projeção para cálculo de pontuações de atenção
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        
        # Fator de escala para evitar gradientes muito pequenos com softmax
        self.scale = torch.sqrt(torch.FloatTensor([hidden_size])).to('cuda' if torch.cuda.is_available() else 'cpu')
    
    def forward(self, x):
        """
        Calcula a atenção e aplica ao input.
        
        Args:
            x: Tensor de entrada [batch_size, seq_len, hidden_size]
            
        Returns:
            Tensor de saída ponderado pela atenção
        """
        # Calcular query e key
        Q = self.query(x)  # [batch_size, seq_len, hidden_size]
        K = self.key(x)    # [batch_size, seq_len, hidden_size]
        
        # Calcular pontuações de atenção
        attention = torch.matmul(Q, K.permute(0, 2, 1)) / self.scale  # [batch_size, seq_len, seq_len]
        
        # Aplicar softmax para obter pesos de atenção
        attention = F.softmax(attention, dim=-1)  # [batch_size, seq_len, seq_len]
        
        # Aplicar atenção ao input
        weighted = torch.matmul(attention, x)  # [batch_size, seq_len, hidden_size]
        
        return weighted


class BasicBlock(nn.Module):
    """
    Bloco básico residual para ResNet.
    """
    
    expansion = 1
    
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        """
        Inicializa o bloco básico residual.
        
        Args:
            in_channels: Número de canais de entrada
            out_channels: Número de canais de saída
            stride: Stride para a primeira convolução
            downsample: Camada opcional para reduzir a dimensão espacial
        """
        super(BasicBlock, self).__init__()
        
        # Primeira convolução
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        # Segunda convolução
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Camada para reduzir a dimensão espacial (se necessário)
        self.downsample = downsample
        
        # Ativação
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        """
        Propagação para frente do bloco residual.
        
        Args:
            x: Tensor de entrada
            
        Returns:
            Tensor de saída após o bloco residual
        """
        # Salvar entrada original para conexão residual
        identity = x
        
        # Primeira camada convolucional
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        # Segunda camada convolucional
        out = self.conv2(out)
        out = self.bn2(out)
        
        # Aplicar downsample, se fornecido
        if self.downsample is not None:
            identity = self.downsample(x)
        
        # Adicionar conexão residual
        out += identity
        out = self.relu(out)
        
        return out


class ModifiedResNet(nn.Module):
    """
    ResNet18 modificada com camada de autoatenção para detecção de ataques de replay.
    """
    
    def __init__(self, input_channels=1, num_classes=2):
        """
        Inicializa a rede ResNet18 modificada.
        
        Args:
            input_channels: Número de canais de entrada (padrão: 1 para espectrograma)
            num_classes: Número de classes para classificação (padrão: 2 para detecção binária)
        """
        super(ModifiedResNet, self).__init__()
        
        # Primeira camada convolucional
        self.conv1 = nn.Conv2d(
            input_channels, 64, kernel_size=(9, 3), stride=1, padding=(4, 1), bias=False
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        
        # Blocos residuais
        self.layer1 = self._make_layer(64, 64, 2)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, 512, 2, stride=2)
        
        # Camada de autoatenção
        self.attention = SelfAttention(512)
        
        # Camadas totalmente conectadas
        self.fc1 = nn.Linear(512, 512)
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(512, num_classes)
        
        # Inicialização dos pesos
        self._initialize_weights()
    
    def _make_layer(self, in_channels, out_channels, blocks, stride=1):
        """
        Cria uma sequência de blocos residuais.
        
        Args:
            in_channels: Número de canais de entrada
            out_channels: Número de canais de saída
            blocks: Número de blocos na camada
            stride: Stride para o primeiro bloco
            
        Returns:
            Sequência de blocos residuais
        """
        downsample = None
        
        # Criar camada de downsample se necessário
        if stride != 1 or in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels, kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(out_channels)
            )
        
        # Construir a camada
        layers = []
        
        # Primeiro bloco (potencialmente com downsample)
        layers.append(BasicBlock(in_channels, out_channels, stride, downsample))
        
        # Blocos restantes
        for _ in range(1, blocks):
            layers.append(BasicBlock(out_channels, out_channels))
        
        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
        """
        Inicializa os pesos da rede usando a inicialização Kaiming.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        Propagação para frente da rede.
        
        Args:
            x: Tensor de entrada [batch_size, channels, height, width]
            
        Returns:
            Tensor de saída com pontuações de classe [batch_size, num_classes]
        """
        # Primeira camada convolucional
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        # Blocos residuais
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        # Global Average Pooling
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        
        # Primeira camada totalmente conectada
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        # Segunda camada totalmente conectada (classificação)
        x = self.fc2(x)
        
        return x


class BidirectionalResNet(nn.Module):
    """
    Modelo para processamento bidirecional de segmentos de áudio usando ResNet modificada.
    """
    
    def __init__(self, input_channels=1, hidden_size=512, num_classes=2, fusion_method='concat'):
        """
        Inicializa o modelo bidirecional.
        
        Args:
            input_channels: Número de canais de entrada
            hidden_size: Dimensão do espaço de características
            num_classes: Número de classes para classificação
            fusion_method: Método de fusão dos vetores de características ('concat', 'max', ou 'mean')
        """
        super(BidirectionalResNet, self).__init__()
        
        # Rede ResNet compartilhada para ambas as direções
        self.resnet = ModifiedResNet(input_channels=input_channels, num_classes=hidden_size)
        
        # Método de fusão
        self.fusion_method = fusion_method
        
        # Camada de classificação final
        if fusion_method == 'concat':
            self.classifier = nn.Linear(hidden_size * 2, num_classes)
        else:  # 'max' ou 'mean'
            self.classifier = nn.Linear(hidden_size, num_classes)
    
    def forward(self, forward_segment, backward_segment):
        """
        Propagação para frente do modelo bidirecional.
        
        Args:
            forward_segment: Tensor do segmento na direção para frente
            backward_segment: Tensor do segmento na direção para trás
            
        Returns:
            Pontuações de classe para a classificação do áudio
        """
        # Processar segmentos forward e backward
        forward_features = self.resnet(forward_segment)
        backward_features = self.resnet(backward_segment)
        
        # Fusão de características
        if self.fusion_method == 'concat':
            # Concatenar features
            combined_features = torch.cat([forward_features, backward_features], dim=1)
        elif self.fusion_method == 'max':
            # Máximo elemento a elemento
            combined_features = torch.max(forward_features, backward_features)
        else:  # 'mean'
            # Média elemento a elemento
            combined_features = (forward_features + backward_features) / 2.0
        
        # Classificação final
        output = self.classifier(combined_features)
        
        return output


class OCLoss(nn.Module):
    """
    Implementação da função de perda One-Class Softmax (OC-Softmax)
    para compressão de fala genuína e separação de ataques de spoofing.
    """
    
    def __init__(self, feature_dim=512, alpha=20.0, m0=0.9, m1=0.2):
        """
        Inicializa a função de perda OC-Softmax.
        
        Args:
            feature_dim: Dimensão do espaço de características
            alpha: Fator de escala (padrão: 20.0)
            m0: Margem para fala genuína (padrão: 0.9)
            m1: Margem para ataques de spoofing (padrão: 0.2)
        """
        super(OCLoss, self).__init__()
        
        # Parâmetros da função de perda
        self.alpha = alpha
        self.m0 = m0
        self.m1 = m1
        
        # Vetor de peso para OC-Softmax
        # CORREÇÃO: Inicializar self.weight como um tensor 2D para xavier_normal_
        self.weight = nn.Parameter(torch.Tensor(feature_dim, 1)) # Alterado de (feature_dim) para (feature_dim, 1)
        nn.init.xavier_normal_(self.weight)
    
    def forward(self, x, labels):
        """
        Calcula a perda OC-Softmax.
        
        Args:
            x: Tensor de entrada (features) [batch_size, feature_dim]
            labels: Tensor de rótulos [batch_size]
            
        Returns:
            Valor da perda
        """
        # Normalizar o vetor de peso e as features
        w = F.normalize(self.weight, p=2, dim=0)
        x = F.normalize(x, p=2, dim=1)
        
        # Calcular o produto escalar
        # CORREÇÃO: Ajustar a operação matmul para o novo formato de self.weight
        cos_theta = torch.matmul(x, w).squeeze(-1) # .squeeze(-1) para remover a dimensão extra de 1
        
        # Aplicar as margens de acordo com as classes
        m_y = torch.ones_like(cos_theta)
        m_y[labels == 0] = self.m0  # Fala genuína
        m_y[labels == 1] = self.m1  # Ataque de spoofing
        
        # Calcular a perda
        loss = torch.log(1 + torch.exp(self.alpha * ((m_y - cos_theta) * ((-1) ** labels))))
        
        return loss.mean()


class PatternBranch(nn.Module):
    """
    Ramo da rede para processar padrões específicos (LBP, GLCM, LPQ).
    """
    
    def __init__(self, input_channels=1, pattern_type='lbp'):
        """
        Inicializa o ramo de padrão.
        
        Args:
            input_channels: Número de canais de entrada
            pattern_type: Tipo de padrão a ser processado ('lbp', 'glcm', ou 'lpq')
        """
        super(PatternBranch, self).__init__()
        
        self.pattern_type = pattern_type
        
        # Rede ResNet modificada para processar o padrão
        self.resnet = ModifiedResNet(input_channels=input_channels)
    
    def forward(self, x):
        """
        Propagação para frente do ramo de padrão.
        
        Args:
            x: Tensor de entrada do padrão
            
        Returns:
            Saída do ramo de padrão
        """
        return self.resnet(x)


class MultiPatternModel(nn.Module):
    """
    Modelo completo que combina múltiplos ramos de padrões para detecção de ataques de replay.
    """
    
    def __init__(self, input_channels=1, hidden_size=512, num_classes=2):
        """
        Inicializa o modelo multi-padrão.
        
        Args:
            input_channels: Número de canais de entrada
            hidden_size: Dimensão do espaço de características
            num_classes: Número de classes para classificação
        """
        super(MultiPatternModel, self).__init__()
        
        # Ramos para cada tipo de padrão
        self.lbp_branch = PatternBranch(input_channels, 'lbp')
        self.glcm_branch = PatternBranch(input_channels, 'glcm')
        self.lpq_branch = PatternBranch(input_channels, 'lpq')
        
        # Camada de OC-Softmax para cada ramo
        # Passar hidden_size para OCLoss, pois é a dimensão das features de saída da ResNet
        self.lbp_oc_softmax = OCLoss(feature_dim=hidden_size)
        self.glcm_oc_softmax = OCLoss(feature_dim=hidden_size)
        self.lpq_oc_softmax = OCLoss(feature_dim=hidden_size)
    
    def forward(self, lbp, glcm, lpq, labels=None):
        """
        Propagação para frente do modelo multi-padrão.
        
        Args:
            lbp: Tensor de padrão LBP
            glcm: Tensor de padrão GLCM
            lpq: Tensor de padrão LPQ
            labels: Tensor de rótulos (necessário para treinamento, opcional para inferência)
            
        Returns:
            Pontuações de saída para cada ramo e perda (se labels fornecido)
        """
        # Processar cada ramo
        lbp_output = self.lbp_branch(lbp)
        glcm_output = self.glcm_branch(glcm)
        lpq_output = self.lpq_branch(lpq)
        
        # Se em modo de treinamento e labels fornecidos, calcular perda
        if self.training and labels is not None:
            lbp_loss = self.lbp_oc_softmax(lbp_output, labels)
            glcm_loss = self.glcm_oc_softmax(glcm_output, labels)
            lpq_loss = self.lpq_oc_softmax(lpq_output, labels)
            
            # Perda total
            total_loss = lbp_loss + glcm_loss + lpq_loss
            
            return {
                'lbp_output': lbp_output,
                'glcm_output': glcm_output,
                'lpq_output': lpq_output,
                'loss': total_loss
            }
        
        # Se em modo de inferência, retornar apenas saídas
        return {
            'lbp_output': lbp_output,
            'glcm_output': glcm_output,
            'lpq_output': lpq_output
        }
    
    def get_scores(self, lbp, glcm, lpq):
        """
        Calcula as pontuações para classificação.
        
        Args:
            lbp: Tensor de padrão LBP
            glcm: Tensor de padrão GLCM
            lpq: Tensor de padrão LPQ
            
        Returns:
            Pontuações para cada amostra
        """
        outputs = self.forward(lbp, glcm, lpq)
        
        # Normalizar as saídas
        # Para OCLoss, a "pontuação" é geralmente a distância ao centro ou algo similar.
        # Se a saída da OCLoss for uma medida de "anomalia" (maior para spoof),
        # então F.softmax não é o ideal.
        # Assumindo que a saída da OCLoss (cos_theta) é a similaridade ao "genuíno"
        # e que a perda é construída para que pontuações mais baixas indiquem spoof,
        # então 1 - cos_theta ou simplesmente a saída da OC-Softmax pode ser usada como score.
        # Se a saída da OCLoss já é a pontuação de "spoofness", podemos usá-la diretamente.
        # Pelo código da OCLoss, a perda é `log(1 + exp(alpha * ((m_y - cos_theta) * ((-1) ** labels))))`
        # Para inferência, queremos a pontuação de spoof. `cos_theta` é similaridade com o centro genuíno.
        # Então, `1 - cos_theta` seria uma pontuação de "spoofness".
        # No entanto, o modelo está retornando `lbp_output`, `glcm_output`, `lpq_output`
        # que são as features antes da OCLoss.
        # A OCLoss calcula a perda com base nessas features e nos labels.
        # Para `get_scores`, precisamos usar a lógica de pontuação da OCLoss.
        # O `cos_theta` da OCLoss é a similaridade com o vetor de peso `w`.
        # Se `labels=0` (genuine), `m0` é usado. Se `labels=1` (spoof), `m1` é usado.
        # Uma pontuação alta de `cos_theta` significa alta similaridade com o centro genuíno.
        # Portanto, para uma pontuação de "spoof", podemos usar `1 - cos_theta`.

        # Para obter as pontuações de spoof, precisamos re-calcular o cos_theta
        # usando as saídas do modelo e os pesos da OCLoss de cada ramo.
        
        # Obter os pesos normalizados de cada OCLoss
        w_lbp = F.normalize(self.lbp_oc_softmax.weight, p=2, dim=0)
        w_glcm = F.normalize(self.glcm_oc_softmax.weight, p=2, dim=0)
        w_lpq = F.normalize(self.lpq_oc_softmax.weight, p=2, dim=0)

        # Normalizar as features de saída do modelo
        lbp_features_norm = F.normalize(outputs['lbp_output'], p=2, dim=1)
        glcm_features_norm = F.normalize(outputs['glcm_output'], p=2, dim=1)
        lpq_features_norm = F.normalize(outputs['lpq_output'], p=2, dim=1)

        # Calcular cos_theta (similaridade com o centro genuíno)
        # E então 1 - cos_theta para pontuação de spoof.
        # .squeeze(-1) é necessário porque w_lbp é (feature_dim, 1)
        lbp_scores = 1 - torch.matmul(lbp_features_norm, w_lbp).squeeze(-1)
        glcm_scores = 1 - torch.matmul(glcm_features_norm, w_glcm).squeeze(-1)
        lpq_scores = 1 - torch.matmul(lpq_features_norm, w_lpq).squeeze(-1)

        return {
            'lbp_score': lbp_scores,
            'glcm_score': glcm_scores,
            'lpq_score': lpq_scores
        }


# Exemplo de uso
if __name__ == "__main__":
    # Verificar GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Usando dispositivo: {device}")
    
    # Criar dados de exemplo
    batch_size = 4
    input_channels = 1
    height = 113
    width = 390
    hidden_size = 512 # Definir hidden_size para o exemplo

    # Tensores de exemplo para teste
    lbp = torch.randn(batch_size, input_channels, height, width).to(device)
    glcm = torch.randn(batch_size, input_channels, height, width).to(device)
    lpq = torch.randn(batch_size, input_channels, height, width).to(device)
    labels = torch.randint(0, 2, (batch_size,)).to(device)
    
    # Criar modelo
    model = MultiPatternModel(input_channels=input_channels, hidden_size=hidden_size).to(device)
    
    # Modo de treinamento
    model.train()
    outputs = model(lbp, glcm, lpq, labels)
    print(f"Perda de treinamento: {outputs['loss'].item():.4f}")
    
    # Modo de inferência
    model.eval()
    with torch.no_grad():
        scores = model.get_scores(lbp, glcm, lpq)
        
        # Exibir pontuações
        for key, value in scores.items():
            print(f"{key}: {value}")

