#import tensorflow as tf
from tensorflow.compat.v1 import logging
from hparams import HParams
from text.symbols import symbols
import random




def create_hparams(hparams_string=None, verbose=False):
    """Create model hyperparameters. Parse nondefault from given string."""

    hparams = HParams(
        ################################
        # Experiment Parameters        #
        ################################
        verbose=-1,
        epochs=80000,
        iters_per_checkpoint=500,
        seed=random.randint(1000,9999),  # 1234,
        dynamic_loss_scaling=True,
        fp16_run=False,
        distributed_run=False,
        dist_backend="nccl",
        dist_url="tcp://localhost:54321",
        cudnn_enabled=True,
        cudnn_benchmark=False,
        load_mel_f0_from_disk=True,
        randomize_samples=True,
        ignore_layers='',
        #  ignore_layers=['speaker_embedding.weight'],

        ################################
        # Data Parameters             #
        ################################
        #  training_files='filelists/ljs_audiopaths_text_sid_train_filelist.txt',
        #  validation_files='filelists/ljs_audiopaths_text_sid_val_filelist.txt',
        validation_files='filelists/libritts_train_clean_100_audiopath_text_sid_atleast5min_val_filelist.txt',
        training_files='filelists/libritts_train_clean_100_audiopath_text_sid_shorterthan10s_atleast5min_train_filelist.txt',

        text_cleaners=['english_cleaners'],
        p_arpabet=1.0,
        cmudict_path="data/cmu_dictionary",

        ################################
        # Audio Parameters             #
        ################################
        max_wav_value=32768.0,
        sampling_rate=24000,  # 22050,
        filter_length=1024,
        hop_length=256,
        win_length=1024,
        n_mel_channels=80,
        mel_fmin=0.0,
        mel_fmax=8000.0,
        f0_min=80,
        f0_max=880,
        harm_thresh=0.25,
        crepe_size='full',  #    'tiny', 'small', 'medium', 'large', 'full'
        viterbi_smooth=False,

        ################################
        # Model Parameters             #
        ################################
        init_bias=False,
        n_symbols=len(symbols),
        symbols_embedding_dim=512,

        # Encoder parameters
        encoder_kernel_size=5,
        encoder_n_convolutions=3,
        encoder_embedding_dim=512,

        # Decoder parameters
        n_frames_per_step=1,  # currently only 1 is supported
        decoder_rnn_dim=1024,
        prenet_dim=256,
        prenet_f0_n_layers=1,
        prenet_f0_dim=1,
        prenet_f0_kernel_size=1,
        prenet_rms_dim=0,
        prenet_rms_kernel_size=1,
        max_decoder_steps=1000,
        gate_threshold=0.5,
        p_attention_dropout=0.3,
        p_decoder_dropout=0.3,
        p_teacher_forcing=0.8,

        # Attention parameters
        attention_rnn_dim=1024,
        attention_dim=128,

        # Location Layer parameters
        attention_location_n_filters=32,
        attention_location_kernel_size=31,

        # Mel-post processing network parameters
        postnet_embedding_dim=512,
        postnet_kernel_size=5,
        postnet_n_convolutions=5,

        # Speaker embedding
        n_speakers=123,
        speaker_embedding_dim=128,

        # Reference encoder
        with_gst=True,
        ref_enc_filters=[32, 32, 64, 64, 128, 128],
        ref_enc_size=[3, 3],
        ref_enc_strides=[2, 2],
        ref_enc_pad=[1, 1],
        ref_enc_gru_size=128,

        # Style Token Layer
        token_embedding_size=256,
        token_num=10,
        num_heads=8,

        ################################
        # Optimization Hyperparameters #
        ################################
        mmi_size=8192,
        use_mmi=True,
        use_saved_learning_rate=False,
        learning_rate=2e-4,
        learning_rate_min=1e-6,
        learning_rate_anneal=70000,
        weight_decay=1e-6,
        grad_clip_thresh=1.0,
        batch_size=16,  # 16,  # 32,
        mask_padding=True,  # set model's padded outputs to padded values

    )

    if hparams_string:
        logging.info('Parsing command line hparams: %s', hparams_string)
        hparams.parse_hparam_args(hparams_string)

    if verbose > 1:
        logging.info('Final parsed hparams: %s', hparams.values())

    return hparams
