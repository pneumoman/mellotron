import random
import os
import re
import numpy as np
import torch
import torch.utils.data
import librosa
import h5py

import layers
from utils import load_wav_to_torch, load_filepaths_and_text
from text import text_to_sequence, cmudict, sequence_to_ctc_sequence
# from yin import compute_yin
from crepe import predict
from math import floor
import time
from datetime import datetime


class TextMelLoader(torch.utils.data.Dataset):
    """
        1) loads audio, text and speaker ids
        2) normalizes text and converts them to sequences of one-hot vectors
        3) computes mel-spectrograms and f0s from audio files.
    """
    def __init__(self, audiopaths_and_text, hparams, speaker_ids=None):
        self.audiopaths_and_text = load_filepaths_and_text(audiopaths_and_text)
        self.text_cleaners = hparams["text_cleaners"]
        self.max_wav_value = hparams["max_wav_value"]
        self.sampling_rate = hparams["sampling_rate"]
        self.stft = layers.TacotronSTFT(
            hparams["filter_length"], hparams["hop_length"], hparams["win_length"],
            hparams["n_mel_channels"], hparams["sampling_rate"], hparams["mel_fmin"],
            hparams["mel_fmax"])
        self.sampling_rate = hparams["sampling_rate"]
        self.filter_length = hparams["filter_length"]
        self.win_length = hparams["win_length"]
        self.hop_length = hparams["hop_length"]
        self.f0_min = hparams["f0_min"]
        self.f0_max = hparams["f0_max"]
        self.harm_thresh = hparams["harm_thresh"]
        self.p_arpabet = hparams["p_arpabet"]
        self. verbose = hparams['verbose']
        self.load_mel_f0_from_disk = hparams["load_mel_f0_from_disk"]
        self.crepe_size = hparams["crepe_size"]
        self.shuffle = hparams["randomize_samples"]
        self.viterbi_smooth = hparams["viterbi_smooth"]

        self.cmudict = None
        if hparams["cmudict_path"] is not None:
            self.cmudict = cmudict.CMUDict(hparams["cmudict_path"])

        self.speaker_ids = speaker_ids
        if speaker_ids is None:
            self.speaker_ids = self.create_speaker_lookup_table(self.audiopaths_and_text)
        if self.shuffle :
            random.seed(1234)
            random.shuffle(self.audiopaths_and_text)

    def create_speaker_lookup_table(self, audiopaths_and_text):
        speaker_ids = np.sort(np.unique([x[2] for x in audiopaths_and_text]))
        d = {int(speaker_ids[i]): i for i in range(len(speaker_ids))}
        return d

    def get_f0(self, audio, sampling_rate=22050, frame_length=1024,
               hop_length=256, f0_min=100, f0_max=300, harm_thresh=0.1):
        #  f0, harmonic_rates, argmins, times = compute_yin(
        #      audio, sampling_rate, frame_length, hop_length, f0_min, f0_max,
        #      harm_thresh)
        step = hop_length*1000/sampling_rate
        tim, f0, conf, act = predict(audio, sampling_rate, model_capacity=self.crepe_size,
                                     step_size=step, verbose=0,viterbi=self.viterbi_smooth)
        pad = int((frame_length / hop_length) / 2)
        #  f0 = [0.0] * pad + f0 + [0.0] * pad
        f0 = np.concatenate([[0.0] * pad, f0, [0.0] * pad])
        f0 = np.array(f0, dtype=np.float32)
        return f0

    def get_data(self, audiopath_and_text):
        audiopath, text, speaker = audiopath_and_text
        seq, ctc_text = self.get_text(text)
        mel, f0 = self.get_mel_and_f0(audiopath)
        speaker_id = self.get_speaker_id(speaker)
        return seq, mel, speaker_id, f0, ctc_text, text

    def get_speaker_id(self, speaker_id):
        return torch.IntTensor([self.speaker_ids[int(speaker_id)]])

    def get_mel_and_f0(self, filepath):
        start = time.time()
        if self.load_mel_f0_from_disk:
            melspec, f0 = self.mel_f0_from_disk(filepath)
        else:
            audio, sampling_rate = load_wav_to_torch(filepath)
            if sampling_rate != self.stft.sampling_rate:
                raise ValueError("{} SR doesn't match target {} SR".format(
                    sampling_rate, self.stft.sampling_rate))
            audio_norm = audio / self.max_wav_value
            audio_norm = audio_norm.unsqueeze(0)
            melspec = self.stft.mel_spectrogram(audio_norm)
            melspec = torch.squeeze(melspec, 0)

            f0 = self.get_f0(audio.cpu().numpy(), self.sampling_rate,
                             self.win_length, self.hop_length, self.f0_min,
                             #  self.filter_length, self.hop_length, self.f0_min,
                             self.f0_max, self.harm_thresh)
            f0 = torch.from_numpy(f0)[None]
            f0 = f0[:, :melspec.size(1)]
        if self.verbose > 1:
            print("loaded f0 and mel in ", datetime.fromtimestamp(time.time()-start).strftime('%M:%S.%f'))
        elif self.verbose > 0:
            print(".", end="")
        return melspec, f0

    def get_text(self, text):
        sequence = text_to_sequence(text, self.text_cleaners, self.cmudict, self.p_arpabet)
        text_norm = torch.IntTensor(sequence)
        ctc_text_norm = torch.IntTensor(sequence_to_ctc_sequence(sequence))

        return text_norm, ctc_text_norm

    def mel_f0_to_disk(self, filepath):
        out_name = filepath[:filepath.rfind('.')] + ".hdf5"
        mel, f0 = self.get_mel_and_f0(filepath)
        with h5py.File(out_name, "w") as f:
            f.create_dataset("mel", data=mel.numpy())
            f.create_dataset("f0", data=f0.numpy())

    def mel_f0_from_disk(self, filepath):
        in_name = filepath[:filepath.rfind('.')] + ".hdf5"
        with h5py.File(in_name, "r") as data:
            mel = torch.from_numpy(np.array(data.get("mel")))
            f0 = torch.from_numpy(np.array(data.get("f0")))
        return mel, f0

    def __getitem__(self, index):
        return self.get_data(self.audiopaths_and_text[index])

    def __len__(self):
        return len(self.audiopaths_and_text)




class TextMelCollate():
    """ Zero-pads model inputs and targets based on number of frames per setep
    """
    def __init__(self, n_frames_per_step):
        self.n_frames_per_step = n_frames_per_step

    def __call__(self, batch):
        """Collate's training batch from normalized text and mel-spectrogram
        PARAMS
        ------
        batch: [text_normalized, mel_normalized]
        """
        # Right zero-pad all one-hot text sequences to max input length
        input_lengths, ids_sorted_decreasing = torch.sort(
            torch.LongTensor([len(x[0]) for x in batch]),
            dim=0, descending=True)
        max_input_len = input_lengths[0]

        text_padded = torch.LongTensor(len(batch), max_input_len)
        text_padded.zero_()
        for i in range(len(ids_sorted_decreasing)):
            text = batch[ids_sorted_decreasing[i]][0]
            text_padded[i, :text.size(0)] = text

        max_ctc_txt_len = max([len(x[4]) for x in batch])
        ctc_text_padded = torch.LongTensor(len(batch), max_ctc_txt_len)
        ctc_text_padded.zero_()
        ctc_text_lengths = torch.LongTensor(len(batch))
        for i in range(len(ids_sorted_decreasing)):
            ctc_text = batch[ids_sorted_decreasing[i]][4]
            ctc_text_padded[i, :ctc_text.size(0)] = ctc_text
            ctc_text_lengths[i] = ctc_text.size(0)

        # Right zero-pad mel-spec
        num_mels = batch[0][1].size(0)
        max_target_len = max([x[1].size(1) for x in batch])
        if max_target_len % self.n_frames_per_step != 0:
            max_target_len += self.n_frames_per_step - max_target_len % self.n_frames_per_step
            assert max_target_len % self.n_frames_per_step == 0

        # include mel padded, gate padded and speaker ids
        mel_padded = torch.FloatTensor(len(batch), num_mels, max_target_len)
        mel_padded.zero_()
        gate_padded = torch.FloatTensor(len(batch), max_target_len)
        gate_padded.zero_()
        output_lengths = torch.LongTensor(len(batch)+1)
        speaker_ids = torch.LongTensor(len(batch))
        f0_padded = torch.FloatTensor(len(batch), 1, max_target_len)
        f0_padded.zero_()
        phrase = []
        for i in range(len(ids_sorted_decreasing)):
            item = batch[ids_sorted_decreasing[i]]
            mel = item[1]
            mel_padded[i, :, :mel.size(1)] = mel
            gate_padded[i, mel.size(1)-1:] = 1
            output_lengths[i] = mel.size(1)
            speaker_ids[i] = item[2]
            f0 = item[3]
            f0_padded[i, :, :f0.size(1)] = f0
            if len(item) > 5:
                phrase.append(item[5])
            else:
                print("got empty phrase item {}".format(i))
                phrase.append('')

        output_lengths[i+1] = max_target_len

        model_inputs = (text_padded, input_lengths, mel_padded, gate_padded,
                        output_lengths, speaker_ids, f0_padded, phrase, ctc_text_padded, ctc_text_lengths)

        return model_inputs
