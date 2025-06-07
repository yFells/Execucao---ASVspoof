#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Módulo modificado para extração de características acústicas híbridas para detecção de ataques de replay.
Inclui suporte alternativo para carregar arquivos FLAC.

Autores: André Thiago de Souza, Felipe de Lima dos Santos, 
         Juliano Gaona Proença, Matheus Henrique Reich Favarin Zagonel
"""

import numpy as np
import os
import librosa
import soundfile as sf
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops
from scipy.signal import lfilter
import warnings
from tqdm import tqdm
import argparse
import traceback
import random # Importar random para amostragem

# Para tentar alternativas quando o librosa falhar
try:
    from pydub import AudioSegment
    PYDUB_AVAILABLE = True
except ImportError:
    PYDUB_AVAILABLE = False
    print("Pydub não está instalado. Será usada como segunda opção caso o librosa falhe.")
    print("Para instalar: pip install pydub")


class FeatureExtractor:
    """Classe para extração de características acústicas para detecção de ataques de replay."""
    
    def __init__(self, sample_rate=16000, n_mfcc=30, n_cqcc=30, n_mels=257,
                window_size=0.025, hop_size=0.010, pre_emphasis=0.97):
        """
        Inicializa o extrator de características.
        
        Args:
            sample_rate: Taxa de amostragem do áudio (padrão: 16000 Hz)
            n_mfcc: Número de coeficientes MFCC (padrão: 30)
            n_cqcc: Número de coeficientes CQCC (padrão: 30)
            n_mels: Número de bandas Mel para o espectrograma (padrão: 257)
            window_size: Tamanho da janela em segundos (padrão: 0.025s - 25ms)
            hop_size: Tamanho do salto em segundos (padrão: 0.010s - 10ms)
            pre_emphasis: Coeficiente de pré-ênfase (padrão: 0.97)
        """
        self.sample_rate = sample_rate
        self.n_mfcc = n_mfcc
        self.n_cqcc = n_cqcc
        self.n_mels = n_mels
        self.window_size = window_size
        self.hop_size = hop_size
        self.pre_emphasis = pre_emphasis
        
        # Parâmetros para FFT
        self.n_fft = int(2 ** np.ceil(np.log2(self.window_size * self.sample_rate)))
        self.hop_length = int(self.hop_size * self.sample_rate)
        self.win_length = int(self.window_size * self.sample_rate)
        
        # Parâmetros para LBP
        self.lbp_radius = 1
        self.lbp_n_points = 8
        
        # Parâmetros para GLCM
        self.glcm_distances = [1]
        self.glcm_angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
        
        # Parâmetros para detecção de silêncio
        self.silence_threshold = 0.0001
        self.min_silence_duration = 0.1

    def load_audio(self, audio_path):
        """
        Carrega arquivo de áudio com suporte a múltiplos backends.
        Tenta vários métodos para carregar o áudio em caso de falha.
        
        Args:
            audio_path: Caminho para o arquivo de áudio
            
        Returns:
            Sinal de áudio carregado e taxa de amostragem
        """
        # Método 1: Tentar com soundfile diretamente
        try:
            audio, sr = sf.read(audio_path)
            # Converter para mono se estéreo
            if len(audio.shape) > 1:
                audio = np.mean(audio, axis=1)
            # Resample se necessário
            if sr != self.sample_rate:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=self.sample_rate)
            return audio, self.sample_rate
        except Exception as e:
            # print(f"Falha ao abrir com soundfile: {str(e)}") # Comentado para evitar spam no console
            pass
        
        # Método 2: Tentar com librosa
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                audio, sr = librosa.load(audio_path, sr=self.sample_rate, mono=True)
                return audio, sr
        except Exception as e:
            # print(f"Falha ao abrir com librosa: {str(e)}") # Comentado para evitar spam no console
            pass
        
        # Método 3: Tentar com pydub se disponível
        if PYDUB_AVAILABLE:
            try:
                # Determinar o formato a partir da extensão
                ext = os.path.splitext(audio_path)[1].lower().replace('.', '')
                
                # Carregar o áudio com pydub
                audio_segment = AudioSegment.from_file(audio_path, format=ext)
                
                # Converter para mono se estéreo
                if audio_segment.channels > 1:
                    audio_segment = audio_segment.set_channels(1)
                
                # Resample se necessário
                if audio_segment.frame_rate != self.sample_rate:
                    audio_segment = audio_segment.set_frame_rate(self.sample_rate)
                
                # Extrair dados de áudio como array numpy
                samples = np.array(audio_segment.get_array_of_samples())
                
                # Normalizar para o intervalo [-1, 1] (pydub retorna inteiros)
                samples = samples.astype(np.float32) / (1 << (8 * audio_segment.sample_width - 1))
                
                return samples, self.sample_rate
            except Exception as e:
                # print(f"Falha ao abrir com pydub: {str(e)}") # Comentado para evitar spam no console
                pass
        
        # Se todas as tentativas falharem
        raise ValueError(f"Não foi possível carregar o arquivo: {audio_path}. Tente instalar ffmpeg ou verifique a integridade do arquivo.")
        
    def preprocess_audio(self, audio):
        """
        Realiza pré-processamento no sinal de áudio.
        
        Args:
            audio: Sinal de áudio
            
        Returns:
            Sinal de áudio pré-processado
        """
        # Normalização de amplitude para [-1, 1]
        audio = audio / (np.max(np.abs(audio)) + 1e-10)
        
        # Aplicação de filtro de pré-ênfase
        return lfilter([1, -self.pre_emphasis], [1], audio)
        
    def extract_mfcc(self, audio):
        """
        Extrai características MFCC do áudio.
        
        Args:
            audio: Sinal de áudio pré-processado
            
        Returns:
            Matriz de características MFCC com delta e delta-delta
        """
        # Extração de MFCCs
        mfcc = librosa.feature.mfcc(
            y=audio, 
            sr=self.sample_rate,
            n_mfcc=self.n_mfcc,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window='hamming'
        )
        
        # Cálculo de delta e delta-delta (primeira e segunda derivadas)
        delta_mfcc = librosa.feature.delta(mfcc)
        delta2_mfcc = librosa.feature.delta(mfcc, order=2)
        
        # Concatenação de MFCC, delta e delta-delta
        return np.vstack([mfcc, delta_mfcc, delta2_mfcc])
    
    def extract_cqcc(self, audio):
        """
        Extrai características CQCC do áudio.
        
        Args:
            audio: Sinal de áudio pré-processado
            
        Returns:
            Matriz de características CQCC com delta e delta-delta
        """
        try:
            from scipy import fftpack
            
            # Transformada Q Constante
            CQT = np.abs(librosa.cqt(
                y=audio, 
                sr=self.sample_rate,
                hop_length=self.hop_length,
                n_bins=96,  # Número de bins por oitava * número de oitavas
                bins_per_octave=24
            ))
            
            # Aplicação de log na CQT
            log_CQT = np.log(CQT + 1e-6)
            
            # Transformada de cosseno discreta (DCT)
            cqcc = fftpack.dct(log_CQT, axis=0, type=2, norm='ortho')
            
            # Manter apenas os n_cqcc primeiros coeficientes
            cqcc = cqcc[:self.n_cqcc, :]
            
            # Cálculo de delta e delta-delta
            delta_cqcc = librosa.feature.delta(cqcc)
            delta2_cqcc = librosa.feature.delta(cqcc, order=2)
            
            # Concatenação de CQCC, delta e delta-delta
            return np.vstack([cqcc, delta_cqcc, delta2_cqcc])
        except Exception as e:
            print(f"Erro ao extrair CQCC: {str(e)}")
            # Retornar matriz de zeros em caso de erro
            frames = 1 + int((len(audio) - self.win_length) / self.hop_length)
            return np.zeros((self.n_cqcc * 3, frames))
    
    def extract_mel_spectrogram(self, audio):
        """
        Extrai espectrograma Mel do áudio.
        
        Args:
            audio: Sinal de áudio pré-processado
            
        Returns:
            Espectrograma Mel em dB
        """
        # Cálculo do espectrograma Mel
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window='hamming',
            n_mels=self.n_mels
        )
        
        # Converter para dB
        log_mel_spec = librosa.power_to_db(mel_spec)
        
        return log_mel_spec
    
    def extract_lbp(self, mel_spectrogram):
        """
        Extrai padrões binários locais (LBP) do espectrograma Mel.
        
        Args:
            mel_spectrogram: Espectrograma Mel
            
        Returns:
            Imagem LBP
        """
        # Normalizar o espectrograma para [0, 1]
        normalized_mel = (mel_spectrogram - np.min(mel_spectrogram)) / (np.max(mel_spectrogram) - np.min(mel_spectrogram) + 1e-10)
        
        # Aplicar LBP
        lbp_image = local_binary_pattern(
            normalized_mel, 
            self.lbp_n_points, 
            self.lbp_radius,
            method='uniform'
        )
        
        return lbp_image
    
    def extract_glcm(self, mel_spectrogram):
        """
        Extrai matriz de co-ocorrência de níveis de cinza (GLCM) do espectrograma Mel.
        
        Args:
            mel_spectrogram: Espectrograma Mel
            
        Returns:
            Características GLCM
        """
        # Normalizar o espectrograma para valores entre 0 e 255
        temp_mel = (mel_spectrogram - np.min(mel_spectrogram)) / (np.max(mel_spectrogram) - np.min(mel_spectrogram) + 1e-10)
        image_uint8 = (temp_mel * 255).astype(np.uint8)
        
        # Quantizar a imagem para reduzir o número de níveis de cinza (para eficiência)
        n_levels = 16
        image_quant = (image_uint8 // (256 // n_levels)).astype(np.uint8)
        
        # Calcular GLCM
        try:
            glcm = graycomatrix(
                image_quant, 
                distances=self.glcm_distances, 
                angles=self.glcm_angles,
                levels=n_levels,
                symmetric=True,
                normed=True
            )
            
            # Extrair características de GLCM
            contrast = graycoprops(glcm, 'contrast')
            dissimilarity = graycoprops(glcm, 'dissimilarity')
            homogeneity = graycoprops(glcm, 'homogeneity')
            energy = graycoprops(glcm, 'energy')
            correlation = graycoprops(glcm, 'correlation')
            
            # Flatten e concatenar as características
            features = np.hstack([
                contrast.ravel(), dissimilarity.ravel(),
                homogeneity.ravel(), energy.ravel(),
                correlation.ravel()
            ])
            
            return features
        except Exception as e:
            print(f"Erro ao extrair GLCM: {str(e)}")
            # Retornar um vetor de zeros em caso de erro
            return np.zeros(len(self.glcm_distances) * len(self.glcm_angles) * 5)
        
    def extract_lpq(self, mel_spectrogram):
        """
        Extrai quantização de fase local (LPQ) do espectrograma Mel.
        
        Args:
            mel_spectrogram: Espectrograma Mel
            
        Returns:
            Imagem LPQ
        """
        try:
            from scipy import signal
            
            # Normalizar o espectrograma para [0, 1]
            normalized_mel = (mel_spectrogram - np.min(mel_spectrogram)) / (np.max(mel_spectrogram) - np.min(mel_spectrogram) + 1e-10)
            
            # Parâmetros para LPQ
            winSize = 5  # Tamanho da janela para LPQ
            
            # Implementação simplificada de LPQ
            # (Nota: Para uma implementação completa, recomenda-se usar bibliotecas específicas)
            STFTalpha = 1/winSize  # Parâmetro alfa para STFT
            
            # Criar filtros para STFT
            x = np.arange(-(winSize-1)/2, (winSize+1)/2, 1)
            y = x.copy()
            X, Y = np.meshgrid(x, y)
            
            # Filtros para frequências [a,0], [0,a], [a,a], e [a,-a] onde a=1/winSize
            w1 = np.exp(-2j * np.pi * STFTalpha * X)
            w2 = np.exp(-2j * np.pi * STFTalpha * Y)
            w3 = np.exp(-2j * np.pi * STFTalpha * (X + Y))
            w4 = np.exp(-2j * np.pi * STFTalpha * (X - Y))
            
            # Aplicar convolução 2D com os filtros
            f1 = signal.convolve2d(normalized_mel, w1, mode='same')
            f2 = signal.convolve2d(normalized_mel, w2, mode='same')
            f3 = signal.convolve2d(normalized_mel, w3, mode='same')
            f4 = signal.convolve2d(normalized_mel, w4, mode='same')
            
            # Obter os sinais de fase
            phase1 = np.sign(np.real(f1))
            phase2 = np.sign(np.imag(f1))
            phase3 = np.sign(np.real(f2))
            phase4 = np.sign(np.imag(f2))
            phase5 = np.sign(np.real(f3))
            phase6 = np.sign(np.imag(f3))
            phase7 = np.sign(np.real(f4))
            phase8 = np.sign(np.imag(f4))
            
            # Codificar as fases para formar o descritor LPQ
            phase1[phase1 < 0] = 0
            phase2[phase2 < 0] = 0
            phase3[phase3 < 0] = 0
            phase4[phase4 < 0] = 0
            phase5[phase5 < 0] = 0
            phase6[phase6 < 0] = 0
            phase7[phase7 < 0] = 0
            phase8[phase8 < 0] = 0
            
            # Combinar as fases para formar o descritor LPQ
            lpq = (phase1*1 + phase2*2 + phase3*4 + phase4*8 +
                phase5*16 + phase6*32 + phase7*64 + phase8*128)
            
            return lpq
        
        except Exception as e:
            print(f"Erro ao extrair LPQ: {str(e)}")
            # Retornar matriz de zeros em caso de erro
            return np.zeros_like(mel_spectrogram)
        
    def remove_silence(self, audio, threshold=None, min_silence_duration=None):
        """
        Remove segmentos de silêncio do áudio.
        
        Args:
            audio: Sinal de áudio
            threshold: Limiar de energia para detecção de silêncio
            min_silence_duration: Duração mínima de silêncio em segundos
            
        Returns:
            Áudio com silêncio removido
        """
        if threshold is None:
            threshold = self.silence_threshold
            
        if min_silence_duration is None:
            min_silence_duration = self.min_silence_duration
            
        # Calcular energia do sinal
        energy = np.abs(audio)
        
        # Definir frames de silêncio
        silence_frames = energy < threshold
        
        # Converter frames em segundos
        silence_samples = np.where(silence_frames)[0]
        
        if len(silence_samples) == 0:
            return audio
            
        # Agrupar frames consecutivos
        silence_regions = []
        start = silence_samples[0]
        
        for i in range(1, len(silence_samples)):
            if silence_samples[i] == silence_samples[i-1] + 1:
                continue
            else:
                end = silence_samples[i-1]
                # Verificar se o silêncio é longo o suficiente
                if (end - start) / self.sample_rate >= min_silence_duration:
                    silence_regions.append((start, end))
                start = silence_samples[i]
                
        # Adicionar última região
        if (silence_samples[-1] - start) / self.sample_rate >= min_silence_duration:
            silence_regions.append((start, silence_samples[-1]))
            
        # Remover regiões de silêncio (de trás para frente para evitar deslocamento de índices)
        audio_without_silence = audio.copy()
        for start, end in reversed(silence_regions):
            audio_without_silence = np.delete(audio_without_silence, np.arange(start, end + 1))
            
        return audio_without_silence
    
    def segment_audio(self, audio, segment_duration=1.0, overlap=0.5):
        """
        Segmenta o áudio em janelas com sobreposição.
        
        Args:
            audio: Sinal de áudio
            segment_duration: Duração de cada segmento em segundos
            overlap: Proporção de sobreposição entre segmentos consecutivos
            
        Returns:
            Lista de segmentos de áudio
        """
        segment_samples = int(segment_duration * self.sample_rate)
        hop_samples = int(segment_samples * (1 - overlap))
        
        # Calcular número de segmentos
        n_segments = 1 + int((len(audio) - segment_samples) / hop_samples)
        
        segments = []
        for i in range(n_segments):
            start = i * hop_samples
            end = start + segment_samples
            
            if end <= len(audio):
                segments.append(audio[start:end])
        
        # Se o último segmento for mais curto que segment_samples
        if len(audio) > (n_segments - 1) * hop_samples + segment_samples:
            last_segment = audio[(n_segments - 1) * hop_samples:]
            # Preencher com zeros para atingir segment_samples
            if len(last_segment) < segment_samples:
                pad_length = segment_samples - len(last_segment)
                last_segment = np.pad(last_segment, (0, pad_length), 'constant')
            segments.append(last_segment)
        
        return segments
    
    def extract_hybrid_features(self, audio_path, normalize=True):
        """
        Extrai características híbridas de um arquivo de áudio.
        
        Args:
            audio_path: Caminho para o arquivo de áudio
            normalize: Se True, aplica normalização às características
            
        Returns:
            Dicionário com todas as características extraídas
        """
        try:
            # Carregar o áudio
            audio, sr = self.load_audio(audio_path)
            
            # Verificar se o áudio foi carregado corretamente
            if audio is None or len(audio) == 0:
                raise ValueError(f"Áudio vazio ou não carregado: {audio_path}")
            
            # Pré-processar o áudio
            audio = self.preprocess_audio(audio)
            
            # Remover silêncio
            try:
                audio_no_silence = self.remove_silence(audio)
                # Se remoção de silêncio resultar em áudio muito curto, usar o original
                if len(audio_no_silence) < 0.1 * self.sample_rate:
                    audio_no_silence = audio
            except Exception as e:
                print(f"Erro ao remover silêncio: {str(e)}. Usando áudio original.")
                audio_no_silence = audio
            
            # Extrair características
            mfcc_features = self.extract_mfcc(audio_no_silence)
            cqcc_features = self.extract_cqcc(audio_no_silence)
            mel_spec = self.extract_mel_spectrogram(audio_no_silence)
            
            # Extrair características de padrões do espectrograma Mel
            lbp_features = self.extract_lbp(mel_spec)
            glcm_features = self.extract_glcm(mel_spec)
            lpq_features = self.extract_lpq(mel_spec)
            
            # Normalização (se solicitado)
            if normalize:
                # Para MFCC e CQCC (normalização por característica)
                mfcc_features = (mfcc_features - np.mean(mfcc_features, axis=1, keepdims=True)) / (np.std(mfcc_features, axis=1, keepdims=True) + 1e-10)
                cqcc_features = (cqcc_features - np.mean(cqcc_features, axis=1, keepdims=True)) / (np.std(cqcc_features, axis=1, keepdims=True) + 1e-10)
                
                # Para os padrões extraídos
                lbp_features = (lbp_features - np.mean(lbp_features)) / (np.std(lbp_features) + 1e-10)
                lpq_features = (lpq_features - np.mean(lpq_features)) / (np.std(lpq_features) + 1e-10)
                
                # Para GLCM (já normalizado durante a extração)
            
            # Retornar dicionário com todas as características
            return {
                'mfcc': mfcc_features,
                'cqcc': cqcc_features,
                'mel_spectrogram': mel_spec,
                'lbp': lbp_features,
                'glcm': glcm_features,
                'lpq': lpq_features
            }
        except Exception as e:
            print(f"Erro ao processar {audio_path}: {str(e)}")
            print(traceback.format_exc())
            return None
    
    def batch_feature_extraction(self, audio_dir, output_dir=None, file_ext='.flac', labels_file=None, sample_proportion=0.5):
        """
        Extrai características de todos os arquivos de áudio em um diretório.
        
        Args:
            audio_dir: Diretório contendo arquivos de áudio
            output_dir: Diretório para salvar as características (opcional)
            file_ext: Extensão dos arquivos de áudio
            labels_file: Caminho para o arquivo de rótulos (para amostragem proporcional)
            sample_proportion: Proporção do dataset a ser usado (0.0 a 1.0).
            
        Returns:
            Dicionário com características de todos os arquivos
        """
        all_features = {}
        
        # Criar diretório de saída se não existir
        if output_dir is not None and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Encontrar todos os arquivos de áudio
        all_audio_files = []
        for root, dirs, files in os.walk(audio_dir):
            for file in files:
                if file.endswith(file_ext):
                    all_audio_files.append(os.path.join(root, file))
        
        audio_files_to_process = []

        # Aplicar amostragem se a proporção for menor que 1.0 e labels_file for fornecido
        if 0 < sample_proportion < 1.0 and labels_file:
            print(f"Amostrando {sample_proportion*100:.2f}% do dataset para extração de características...")
            genuine_samples = []
            spoof_samples = []

            # Ler os rótulos para mapear file_id para label
            file_labels = {}
            try:
                with open(labels_file, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 2:
                            file_id = parts[0]
                            label = parts[1] # 'bonafide'/'genuine' ou 'spoof'
                            file_labels[file_id] = label
            except FileNotFoundError:
                print(f"Aviso: Arquivo de rótulos '{labels_file}' não encontrado. A amostragem proporcional não será aplicada.")
                # Se o arquivo de rótulos não for encontrado, processar todos os arquivos
                audio_files_to_process = all_audio_files
            
            if file_labels: # Se os rótulos foram carregados com sucesso
                for audio_path in all_audio_files:
                    file_id = os.path.splitext(os.path.basename(audio_path))[0]
                    label = file_labels.get(file_id)
                    if label:
                        if label.lower() in ['bonafide', 'genuine']:
                            genuine_samples.append(audio_path)
                        elif label.lower() == 'spoof':
                            spoof_samples.append(audio_path)
                    else:
                        # Se o arquivo de áudio não tiver um rótulo correspondente, adicioná-lo a uma categoria "desconhecida"
                        # ou simplesmente ignorá-lo para a amostragem proporcional.
                        # Por simplicidade, vamos ignorá-lo para a amostragem proporcional
                        # e ele não será incluído nos 30% se não tiver rótulo.
                        pass
                
                num_genuine_to_sample = int(len(genuine_samples) * sample_proportion)
                num_spoof_to_sample = int(len(spoof_samples) * sample_proportion)

                # Garantir que não tentamos amostrar mais do que o disponível
                num_genuine_to_sample = min(num_genuine_to_sample, len(genuine_samples))
                num_spoof_to_sample = min(num_spoof_to_sample, len(spoof_samples))
                
                sampled_genuine = random.sample(genuine_samples, num_genuine_to_sample)
                sampled_spoof = random.sample(spoof_samples, num_spoof_to_sample)
                
                audio_files_to_process = sampled_genuine + sampled_spoof
                random.shuffle(audio_files_to_process) # Embaralhar para misturar genuínos e spoof
                print(f"Extraindo características para {len(audio_files_to_process)} arquivos ({num_genuine_to_sample} genuínos, {num_spoof_to_sample} spoof) após amostragem.")
            else:
                # Se não conseguiu carregar os rótulos, processa todos os arquivos
                audio_files_to_process = all_audio_files
                print(f"Amostragem proporcional não aplicada. Processando todos os {len(audio_files_to_process)} arquivos.")
        else:
            audio_files_to_process = all_audio_files
            print(f"Processando todos os {len(audio_files_to_process)} arquivos (sample_proportion={sample_proportion}).")

        # Iterar sobre os arquivos de áudio com barra de progresso
        for audio_path in tqdm(audio_files_to_process, desc=f"Extraindo características de {audio_dir}"):
            try:
                # Obter nome de arquivo relativo
                rel_path = os.path.relpath(audio_path, audio_dir)
                filename = os.path.basename(audio_path)
                
                # Extrair características
                features = self.extract_hybrid_features(audio_path)
                
                if features is not None:
                    all_features[filename] = features
                    
                    # Salvar características em disco (se solicitado)
                    if output_dir is not None:
                        # Preservar hierarquia de diretórios
                        output_path = os.path.join(output_dir, os.path.splitext(rel_path)[0] + '.npz')
                        output_subdir = os.path.dirname(output_path)
                        
                        # Criar subdiretório se não existir
                        if not os.path.exists(output_subdir):
                            os.makedirs(output_subdir)
                        
                        np.savez(
                            output_path, 
                            mfcc=features['mfcc'],
                            cqcc=features['cqcc'],
                            mel_spectrogram=features['mel_spectrogram'],
                            lbp=features['lbp'],
                            glcm=features['glcm'],
                            lpq=features['lpq']
                        )
            except Exception as e:
                print(f"Erro ao processar {audio_path}: {str(e)}")
                print(traceback.format_exc())
        
        print(f"Extração concluída. Processados {len(all_features)}/{len(audio_files_to_process)} arquivos.")
        return all_features


def parse_args():
    """
    Analisa argumentos de linha de comando.
    
    Returns:
        Argumentos analisados
    """
    parser = argparse.ArgumentParser(description="Extração de características para detecção de ataques de replay")
    
    parser.add_argument('--train-audio-dir', type=str, default=None,
                        help='Diretório com arquivos de áudio de treinamento')
    parser.add_argument('--dev-audio-dir', type=str, default=None,
                        help='Diretório com arquivos de áudio de validação')
    parser.add_argument('--eval-audio-dir', type=str, required=True, # Eval audio dir é sempre necessário
                        help='Diretório com arquivos de áudio de avaliação')
    parser.add_argument('--output-dir', type=str, required=True,
                        help='Diretório para salvar características extraídas')
    parser.add_argument('--audio-ext', type=str, default='.flac',
                        help='Extensão dos arquivos de áudio')
    
    parser.add_argument('--train-labels-file', type=str, default=None,
                        help='Arquivo com rótulos de treinamento (para amostragem proporcional)')
    parser.add_argument('--dev-labels-file', type=str, default=None,
                        help='Arquivo com rótulos de validação (para amostragem proporcional)')
    parser.add_argument('--eval-labels-file', type=str, required=True, # Eval labels file é sempre necessário
                        help='Arquivo com rótulos de avaliação (para amostragem proporcional)')
    
    parser.add_argument('--sample-proportion', type=float, default=1.0,
                        help='Proporção do dataset a ser usado para extração de características (0.0 a 1.0)')

    parser.add_argument('--sample-rate', type=int, default=16000,
                        help='Taxa de amostragem do áudio')
    parser.add_argument('--n-mfcc', type=int, default=30,
                        help='Número de coeficientes MFCC')
    parser.add_argument('--n-cqcc', type=int, default=30,
                        help='Número de coeficientes CQCC')
    parser.add_argument('--n-mels', type=int, default=257,
                        help='Número de bandas Mel para o espectrograma')
    
    return parser.parse_args()


def main():
    """
    Função principal.
    """
    # Analisar argumentos
    args = parse_args()
    
    # Criar extrator de características
    extractor = FeatureExtractor(
        sample_rate=args.sample_rate,
        n_mfcc=args.n_mfcc,
        n_cqcc=args.n_cqcc,
        n_mels=args.n_mels
    )
    
    # Extrair características para o conjunto de treinamento
    if args.train_audio_dir and args.train_labels_file:
        train_output_dir = os.path.join(args.output_dir, 'train')
        print("Extraindo características do conjunto de treinamento...")
        train_features = extractor.batch_feature_extraction(
            args.train_audio_dir, train_output_dir, args.audio_ext,
            labels_file=args.train_labels_file, sample_proportion=args.sample_proportion
        )
    else:
        print("Diretório de áudio ou arquivo de rótulos de treinamento não fornecidos. Pulando extração de treinamento.")
        train_features = {}

    # Extrair características para o conjunto de validação
    if args.dev_audio_dir and args.dev_labels_file:
        dev_output_dir = os.path.join(args.output_dir, 'dev')
        print("Extraindo características do conjunto de validação...")
        dev_features = extractor.batch_feature_extraction(
            args.dev_audio_dir, dev_output_dir, args.audio_ext,
            labels_file=args.dev_labels_file, sample_proportion=args.sample_proportion
        )
    else:
        print("Diretório de áudio ou arquivo de rótulos de validação não fornecidos. Pulando extração de validação.")
        dev_features = {}

    # Extrair características para o conjunto de avaliação (TESTE COMPLETO)
    eval_output_dir = os.path.join(args.output_dir, 'eval')
    print("Extraindo características do conjunto de avaliação (completo para teste)...")
    eval_features = extractor.batch_feature_extraction(
        args.eval_audio_dir, eval_output_dir, args.audio_ext,
        labels_file=args.eval_labels_file, sample_proportion=0.5 # Sempre 1.0 para avaliação
    )
    
    print(f"Características extraídas e salvas em: {args.output_dir}")
    
    return 0


# Exemplo de uso individual
def test_single_file(audio_path):
    """
    Testa a extração de características para um único arquivo.
    
    Args:
        audio_path: Caminho para o arquivo de áudio
    """
    # Verificar se o arquivo existe
    if not os.path.exists(audio_path):
        print(f"Arquivo {audio_path} não encontrado.")
        return
    
    # Inicializar o extrator de características
    extractor = FeatureExtractor()
    
    # Extrair características
    features = extractor.extract_hybrid_features(audio_path)
    
    # Exibir informações sobre as características extraídas
    if features is not None:
        for feature_name, feature_data in features.items():
            if isinstance(feature_data, np.ndarray):
                print(f"{feature_name}: shape={feature_data.shape}, dtype={feature_data.dtype}")
            else:
                print(f"{feature_name}: type={type(feature_data)}")
    else:
        print("Falha na extração de características.")


if __name__ == "__main__":
    # Verificar se é para testar um único arquivo
    import sys
    if len(sys.argv) == 2 and os.path.isfile(sys.argv[1]):
        # Modo de teste para um único arquivo
        test_single_file(sys.argv[1])
    else:
        # Modo normal com argumentos
        sys.exit(main())
