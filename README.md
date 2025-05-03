# Detecção de Ataques Spoofing de Replay

Este repositório contém a implementação de um sistema de detecção de ataques de replay para verificação automática de locutor (ASV), desenvolvido como um projeto de pesquisa para a disciplina "Experiência Criativa: Projeto Transformador I" do curso de Bacharelado em Ciência da Computação da PUCPR.

## Autores

- André Thiago de Souza
- Felipe de Lima dos Santos
- Juliano Gaona Proença
- Matheus Henrique Reich Favarin Zagonel

## Descrição do Projeto

O projeto implementa um sistema avançado para detecção de ataques de replay em sistemas de verificação automática de locutor (ASV). Ataques de replay consistem na reprodução de gravações previamente capturadas da voz de um usuário legítimo, representando uma ameaça significativa devido à sua simplicidade de execução.

O sistema proposto integra:

1. Extração de características híbridas (MFCC, CQCC) e padrões (LBP, GLCM, LPQ)
2. Segmentação bidirecional para análise de múltiplos pontos da elocução
3. Redes neurais profundas com arquitetura ResNet modificada e mecanismo de atenção
4. Função de perda OC-Softmax otimizada para detecção de ataques

## Estrutura do Projeto

```
├── config/                  # Arquivos de configuração
│   └── default.json         # Configuração padrão
├── data/                    # Dados (não incluídos no repositório)
├── feature_extraction.py    # Módulo para extração de características
├── bidirectional_segmentation.py  # Implementação de segmentação bidirecional
├── model.py                 # Implementação do modelo neural
├── evaluation.py            # Funções para avaliação do modelo
├── generalization_analysis.py  # Análise de generalização
├── train.py                 # Script para treinamento
├── test.py                  # Script para teste e avaliação
├── main.py                  # Script principal
└── README.md                # Este arquivo
```

## Requisitos

Os seguintes pacotes são necessários para executar o código:

```
torch>=1.9.0
torchaudio>=0.9.0
numpy>=1.19.0
librosa>=0.8.0
scikit-learn>=0.24.0
matplotlib>=3.3.0
seaborn>=0.11.0
tqdm>=4.50.0
pandas>=1.2.0
```

Você pode instalar todos os requisitos usando:

```bash
pip install -r requirements.txt
```

## Datasets

O sistema foi desenvolvido e avaliado nos seguintes conjuntos de dados:

- **ASVspoof 2019**: Inclui dois cenários de acesso: Logical Access (LA) com ataques sintéticos e Physical Access (PA) com ataques de replay.
- **ASVspoof 2021**: Extensão do dataset de 2019 com condições de transmissão telefônica (PSTN, VoIP) usando vários codecs.

Os conjuntos de dados não estão incluídos neste repositório e devem ser baixados separadamente dos sites oficiais.

## Como Usar

### Configuração

Edite o arquivo `config/default.json` para configurar os parâmetros do sistema, como caminhos para os datasets, hiperparâmetros de treinamento, etc.

### Extração de Características

```bash
python main.py --task extract --dataset ASVspoof2019 --output-dir output/experiment1
```

### Treinamento

```bash
python main.py --task train --dataset ASVspoof2019 --output-dir output/experiment1
```

### Teste e Avaliação

```bash
python main.py --task test --dataset ASVspoof2019 --output-dir output/experiment1
```

### Fluxo Completo (Extração, Treinamento e Teste)

```bash
python main.py --task all --dataset ASVspoof2019 --output-dir output/experiment1
```

### Análise Cruzada entre Conjuntos de Dados

```bash
python main.py --task test --dataset ASVspoof2019 --cross-dataset --cross-dataset-name ASVspoof2021 --output-dir output/experiment1
```

## Resultados

A metodologia proposta atinge resultados competitivos com o estado da arte:

- **ASVspoof 2019 LA**: EER de 2.14% e t-DCF de 0.06
- **ASVspoof 2019 PA**: EER de 2.2%
- **ASVspoof 2021 LA**: EER varianado entre 4.72-7.58% dependendo da condição do codec

Os resultados completos estão disponíveis no diretório de saída após a execução.

## Referências

1. Chakravarty, N.; Dua, M. Audio Spoof Detection using Deep Residual Networks based Feature Extraction: Unveiling Synthetic, Replay and Mimicry Threats. In: 2024 International Conference on Recent Innovation in Smart and Sustainable Technology (ICRISST), Bengaluru, India, 2024.

2. Neamtu, C.-T.; Mihalache, S.; Burileanu, D. Liveness Detection – Automatic Classification of Spontaneous and Pre-recorded Speech for Biometric Applications. In: 2023 International Conference on Speech Technology and Human-Computer Dialogue (SpeD), Bucharest, Romania, 2023.

3. Ustubioglu, B.; Tahaoglu, G.; Ustubioglu, A.; Ulutas, G.; Amerini, I.; Kilic, M. Multi Pattern Features-Based Spoofing Detection Mechanism Using One Class Learning. IEEE Access, v. 12, 2024.

4. Yoon, S.-H.; Yu, H.-J. Multiple Points Input For Convolutional Neural Networks in Replay Attack Detection. In: ICASSP 2020, Barcelona, Spain, 2020.

5. Duraibi, S.; Alhamdani, W.; Sheldon, F. T. Replay Spoof Attack Detection using Deep Neural Networks for Classification. In: 2020 International Conference on Computational Science and Computational Intelligence (CSCI), Las Vegas, NV, USA, 2020.

## Licença

Este projeto está licenciado sob a Licença MIT - veja o arquivo LICENSE para detalhes.