# Instruções de Instalação e Execução - Sistema de Detecção de Ataques Spoofing de Replay

Este documento contém instruções detalhadas para instalar e executar o sistema de detecção de ataques spoofing de replay, adaptado para a sua estrutura de diretórios específica.

## Estrutura de Diretórios

O sistema espera a seguinte estrutura de diretórios para os datasets:

- **ASVspoof 2019 PA**:
  ```
  E:\ASV 2019 DATA\PA\
  ├── ASVspoof2019_PA_train\flac\
  ├── ASVspoof2019_PA_dev\flac\
  ├── ASVspoof2019_PA_eval\flac\
  └── ASVspoof2019_PA_cm_protocols\
      ├── ASVspoof2019.PA.cm.train.trl.txt
      ├── ASVspoof2019.PA.cm.dev.trl.txt
      └── ASVspoof2019.PA.cm.eval.trl.txt
  ```

- **ASVspoof 2019 LA**:
  ```
  E:\ASV 2019 DATA\LA\
  ├── ASVspoof2019_LA_train\flac\
  ├── ASVspoof2019_LA_dev\flac\
  ├── ASVspoof2019_LA_eval\flac\
  └── ASVspoof2019_LA_cm_protocols\
      ├── ASVspoof2019.LA.cm.train.trl.txt
      ├── ASVspoof2019.LA.cm.dev.trl.txt
      └── ASVspoof2019.LA.cm.eval.trl.txt
  ```

- **ASVspoof 2021 PA** (opcional):
  ```
  E:\ASV 2021 DATA\PA\
  └── flac\
  ```

## Requisitos

O sistema requer:

1. Python 3.8 ou superior
2. PyTorch e TorchAudio
3. Bibliotecas para processamento de áudio e aprendizado de máquina
4. FFmpeg (opcional, mas altamente recomendado)

## Instalação

### 1. Instalação das Dependências

Execute o script `setup.py` para verificar e instalar as dependências:

```bash
python setup.py --install-deps
```

Este script:
- Verifica se todas as bibliotecas necessárias estão instaladas
- Instala as dependências faltantes
- Verifica se o FFmpeg está instalado
- Cria a estrutura de diretórios para experimentos

### 2. Instalação Manual de Dependências (Alternativa)

Se preferir instalar manualmente, use:

```bash
pip install torch torchaudio numpy scipy librosa scikit-learn scikit-image matplotlib pandas tqdm pydub soundfile
```

### 3. Instalação do FFmpeg

O FFmpeg é usado como alternativa para carregar arquivos de áudio quando o librosa falha.

**No Windows:**
1. Baixe o FFmpeg de https://ffmpeg.org/download.html
2. Extraia os arquivos para uma pasta (ex: C:\ffmpeg)
3. Adicione a pasta bin ao PATH do sistema

## Execução

### 1. Preparação dos Dados

Primeiro, é necessário converter os arquivos de protocolo para o formato de labels:

```bash
python convert_protocols.py
```

### 2. Extração de Características

Use o script modificado `feature_extraction_fix.py` para extrair características dos arquivos de áudio:

```bash
python feature_extraction_fix.py --train-audio-dir "E:\ASV 2019 DATA\LA\ASVspoof2019_LA_train\flac" --dev-audio-dir "E:\ASV 2019 DATA\LA\ASVspoof2019_LA_dev\flac" --eval-audio-dir "E:\ASV 2019 DATA\LA\ASVspoof2019_LA_eval\flac" --output-dir "output\features\LA" --audio-ext ".flac"
```

### 3. Treinamento do Modelo

Após extrair as características, treine o modelo:

```bash
python train.py --train-features-dir "output\features\LA\train" --dev-features-dir "output\features\LA\dev" --train-labels-file "E:\ASV 2019 DATA\LA\labels\train_labels.txt" --dev-labels-file "E:\ASV 2019 DATA\LA\labels\dev_labels.txt" --save-dir "output\checkpoints\LA" --num-epochs 50
```

### 4. Avaliação do Modelo

Avalie o modelo treinado:

```bash
python test.py --test-features-dir "output\features\LA\eval" --test-labels-file "E:\ASV 2019 DATA\LA\labels\eval_labels.txt" --checkpoint "output\checkpoints\LA\best_model.pth" --results-dir "output\results\LA" --analyze-generalization
```

### 5. Teste de uma Única Amostra

Para testar uma única amostra de áudio:

```bash
python test_single_sample.py --audio "E:\ASV 2019 DATA\LA\ASVspoof2019_LA_eval\flac\LA_E_0000001.flac" --model "output\checkpoints\LA\best_model.pth" --plot --output-dir "output\single_test"
```

### 6. Execução Interativa

Alternativamente, use o script em lote `run_asvspoof.bat` para executar todas as etapas de forma interativa:

```bash
run_asvspoof.bat
```

## Solução de Problemas

### Erro ao Carregar Arquivos FLAC

Se encontrar erros como `flac decoder lost sync` ou `NoBackendError` ao carregar arquivos FLAC, tente:

1. Verificar se o FFmpeg está instalado corretamente e disponível no PATH
2. Instalar a biblioteca pydub: `pip install pydub`
3. Verificar a integridade dos arquivos FLAC

O script `feature_extraction_fix.py` foi modificado para tentar vários métodos de carregamento de áudio, o que deve resolver a maioria dos problemas.

### Problemas de Memória

Se encontrar erros de memória durante o processamento:

1. Reduza o tamanho do lote (parâmetro `--batch-size`)
2. Processe os conjuntos de dados em partes menores

## Customização

Edite o arquivo `config/custom.json` para adaptar os caminhos de diretório e outros parâmetros conforme necessário.

## Contato

Em caso de dúvidas ou problemas, entre em contato com os autores:
- André Thiago de Souza, Felipe de Lima dos Santos, Juliano Gaona Proença, Matheus Henrique Reich Favarin Zagonel