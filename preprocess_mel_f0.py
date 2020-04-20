from data_utils import TextMelLoader
from hparam_setup import create_hparams
import tqdm


if __name__ == '__main__':

    hparams = create_hparams()
    hparams['load_mel_f0_from_disk'] = False
    hparams['verbose']=0
    train_loader = TextMelLoader(hparams["training_files"], hparams)
    eval_loader = TextMelLoader(hparams["validation_files"], hparams)

    for i in train_loader.audiopaths_and_text:
        train_loader.mel_f0_to_disk(i[0])

    for i in eval_loader.audiopaths_and_text:
        eval_loader.mel_f0_to_disk(i[0])

