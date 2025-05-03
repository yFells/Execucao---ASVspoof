#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script para converter os arquivos de protocolo ASVspoof 2019 para o formato de labels
usado pelo sistema de detecção de ataques spoofing de replay.

Autores: André Thiago de Souza, Felipe de Lima dos Santos, 
         Juliano Gaona Proença, Matheus Henrique Reich Favarin Zagonel
"""

import os
import argparse


def convert_protocol_file(protocol_file, output_file):
    """
    Converte um arquivo de protocolo ASVspoof para o formato de labels.
    
    O formato de protocolo ASVspoof:
    <formato_específico> <arquivo> <sistema> <bonafide/spoof>
    
    O formato de labels:
    <arquivo> <genuine/spoof>
    
    Args:
        protocol_file: Caminho para o arquivo de protocolo
        output_file: Caminho para o arquivo de saída
    """
    # Verificar se o arquivo de protocolo existe
    if not os.path.exists(protocol_file):
        print(f"Arquivo de protocolo não encontrado: {protocol_file}")
        return False
    
    # Criar diretório de saída se não existir
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Contar amostras genuínas e falsificadas
    count_genuine = 0
    count_spoof = 0
    
    # Processar o arquivo de protocolo
    with open(protocol_file, 'r') as f_in, open(output_file, 'w') as f_out:
        for line in f_in:
            parts = line.strip().split()
            
            # Verificar o formato do protocolo
            if len(parts) < 4:
                print(f"Formato desconhecido: {line.strip()}")
                continue
            
            # Extrair informações
            file_name = parts[1]
            
            # Verificar o tipo de amostra (genuína ou falsificada)
            if parts[-1].lower() in ['bonafide', 'genuine']:
                label = "genuine"
                count_genuine += 1
            else:
                label = "spoof"
                count_spoof += 1
            
            # Escrever no formato esperado
            f_out.write(f"{file_name} {label}\n")
    
    print(f"Arquivo de labels criado: {output_file}")
    print(f"  Amostras genuínas: {count_genuine}")
    print(f"  Amostras falsificadas: {count_spoof}")
    
    return True


def main():
    """
    Função principal.
    """
    # Configurar parser de argumentos
    parser = argparse.ArgumentParser(description="Conversor de protocolos ASVspoof para o formato de labels")
    
    parser.add_argument('--train-protocol', type=str, default="E:/ASV 2019 DATA/PA/ASVspoof2019_PA_cm_protocols/ASVspoof2019.PA.cm.train.trl.txt",
                        help='Arquivo de protocolo para conjunto de treinamento')
    parser.add_argument('--dev-protocol', type=str, default="E:/ASV 2019 DATA/PA/ASVspoof2019_PA_cm_protocols/ASVspoof2019.PA.cm.dev.trl.txt",
                        help='Arquivo de protocolo para conjunto de desenvolvimento')
    parser.add_argument('--eval-protocol', type=str, default="E:/ASV 2019 DATA/PA/ASVspoof2019_PA_cm_protocols/ASVspoof2019.PA.cm.eval.trl.txt",
                        help='Arquivo de protocolo para conjunto de avaliação')
    
    parser.add_argument('--output-dir', type=str, default="E:/ASV 2019 DATA/PA/labels",
                        help='Diretório para salvar arquivos de labels')
    
    # Analisar argumentos
    args = parser.parse_args()
    
    # Criar diretório de saída se não existir
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    # Definir caminhos de saída
    train_labels = os.path.join(args.output_dir, "train_labels.txt")
    dev_labels = os.path.join(args.output_dir, "dev_labels.txt")
    eval_labels = os.path.join(args.output_dir, "eval_labels.txt")
    
    # Converter arquivos de protocolo
    success_train = convert_protocol_file(args.train_protocol, train_labels)
    success_dev = convert_protocol_file(args.dev_protocol, dev_labels)
    success_eval = convert_protocol_file(args.eval_protocol, eval_labels)
    
    # Verificar resultados
    if success_train and success_dev and success_eval:
        print("Conversão concluída com sucesso!")
        return 0
    else:
        print("Erro na conversão de alguns arquivos.")
        return 1


if __name__ == "__main__":
    exit(main())
