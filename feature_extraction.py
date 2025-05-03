#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Módulo para extração de características acústicas híbridas para detecção de ataques de replay.
Implementa a extração de MFCC, CQCC, e espectrogramas Mel, além de características de padrões (LBP, GLCM, LPQ).

Autores: André Thiago de Souza, Felipe de Lima dos Santos, 
         Juliano Gaona Proença, Matheus Henrique Reich Favarin Zagonel
"""

import numpy as np
import librosa
import librosa.display
import scipy
import os
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops
from scipy.signal import lfilter

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
        cqcc = scipy.fftpack.dct(log_CQT, axis=0, type=2, norm='ortho')
        
        # Manter apenas os n_cqcc primeiros coeficientes
        cqcc = cqcc[:self.n_cqcc, :]
        
        # Cálculo de delta e delta-delta
        delta_cqcc = librosa.feature.delta(cqcc)
        delta2_cqcc = librosa.feature.delta(cqcc, order=2)
        
        # Concatenação de CQCC, delta e delta-delta
        return np.vstack([cqcc, delta_cqcc, delta2_cqcc])
    
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
        normalized_mel = (mel_spectrogram - np.min(mel_spectrogram)) / (np.max(mel_spectrogram) - np.min(mel_spectrogram))
        
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
        temp_mel = (mel_spectrogram - np.min(mel_spectrogram)) / (np.max(mel_spectrogram) - np.min(mel_spectrogram))
        image_uint8 = (temp_mel * 255).astype(np.uint8)
        
        # Quantizar a imagem para reduzir o número de níveis de cinza (para eficiência)
        n_levels = 16
        image_quant = (image_uint8 // (256 // n_levels)).astype(np.uint8)
        
        # Calcular GLCM
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
        
    def extract_lpq(self, mel_spectrogram):
        """
        Extrai quantização de fase local (LPQ) do espectrograma Mel.
        
        Args:
            mel_spectrogram: Espectrograma Mel
            
        Returns:
            Imagem LPQ
        """
        # Normalizar o espectrograma para [0, 1]
        normalized_mel = (mel_spectrogram - np.min(mel_spectrogram)) / (np.max(mel_spectrogram) - np.min(mel_spectrogram))
        
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
        f1 = scipy.signal.convolve2d(normalized_mel, w1, mode='same')
        f2 = scipy.signal.convolve2d(normalized_mel, w2, mode='same')
        f3 = scipy.signal.convolve2d(normalized_mel, w3, mode='same')
        f4 = scipy.signal.convolve2d(normalized_mel, w4, mode='same')
        
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
        # Carregar o áudio
        audio, _ = librosa.load(audio_path, sr=self.sample_rate, mono=True)
        
        # Pré-processar o áudio
        audio = self.preprocess_audio(audio)
        
        # Remover silêncio
        audio_no_silence = self.remove_silence(audio)
        
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
    
    def batch_feature_extraction(self, audio_dir, output_dir=None, file_ext='.flac'):
        """
        Extrai características de todos os arquivos de áudio em um diretório.
        
        Args:
            audio_dir: Diretório contendo arquivos de áudio
            output_dir: Diretório para salvar as características (opcional)
            file_ext: Extensão dos arquivos de áudio
            
        Returns:
            Dicionário com características de todos os arquivos
        """
        all_features = {}
        
        # Criar diretório de saída se não existir
        if output_dir is not None and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Iterar sobre os arquivos de áudio
        for filename in os.listdir(audio_dir):
            if filename.endswith(file_ext):
                audio_path = os.path.join(audio_dir, filename)
                features = self.extract_hybrid_features(audio_path)
                all_features[filename] = features
                
                # Salvar características em disco (se solicitado)
                if output_dir is not None:
                    output_path = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}.npz")
                    np.savez(
                        output_path, 
                        mfcc=features['mfcc'],
                        cqcc=features['cqcc'],
                        mel_spectrogram=features['mel_spectrogram'],
                        lbp=features['lbp'],
                        glcm=features['glcm'],
                        lpq=features['lpq']
                    )
        
        return all_features


# Exemplo de uso
if __name__ == "__main__":
    # Inicializar o extrator de características
    extractor = FeatureExtractor()
    
    # Extrair características de um arquivo de áudio
    audio_path = "caminho/para/arquivo_de_audio.wav"
    
    # Verificar se o arquivo existe
    if os.path.exists(audio_path):
        features = extractor.extract_hybrid_features(audio_path)
        
        # Exibir informações sobre as características extraídas
        for feature_name, feature_data in features.items():
            if isinstance(feature_data, np.ndarray):
                print(f"{feature_name}: shape={feature_data.shape}, dtype={feature_data.dtype}")
            else:
                print(f"{feature_name}: type={type(feature_data)}")
    else:
        print(f"Arquivo {audio_path} não encontrado. Substitua pelo caminho correto para testar.")
