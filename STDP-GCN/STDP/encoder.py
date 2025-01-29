import spikingjelly
import scipy.io as scio
import numpy as np
import torch
from scipy import signal
from matplotlib import pyplot as plt
from scipy.signal import firwin
from sklearn import preprocessing
from tqdm import tqdm

from STDP.stdp_weight import STDP


class BSA:
    def __init__(self, threshold=0.955, filter_length=20, cutoff=0.8):
        self.threshold = threshold
        self.filter_length = filter_length

        filter_values = firwin(filter_length, cutoff=cutoff)
        self.filter = filter_values

    def encode(self, input):
        spike = [0]
        for i in range(1, len(input)):
            error1 = 0
            error2 = 0
            for j in range(1, self.filter_length):
                if i + j - 1 < len(input):
                    # print(f'i:{i} j:{j} i + j - 1:{i + j - 1} len filter {len(self.filter)}')
                    error1 += abs(input[i + j - 1] - self.filter[j])
                    error2 += abs(input[i + j - 1])

            if error1 <= error2 - self.threshold:
                # print(1)  # fire!!!
                spike.append(1)
                for j in range(1, self.filter_length):
                    if i + j - 1 < len(input):  # 这几个i+j-1可能不太对
                        input[i + j - 1] -= self.filter[j]
            else:
                # print(0)  # no fire
                spike.append(0)
        return np.array(spike)

    def multi_epoch_encode(self, input):
        """
        :param input: [epoch, channel, eeg]
        :return: [epoch, channel, spikes]
        """
        result = np.zeros((input.shape))
        epoch_num, channel_num, eeg_length = input.shape
        pbar = tqdm(range(epoch_num))
        for epoch in pbar:
            pbar.set_description('encoding: ')
            for channel in range(channel_num):
                result[epoch][channel] = self.encode(input[epoch][channel])

        print(result.shape)
        return result

    def multi_channel_encode(self, input):
        pass

    def reconstruct(self, input):
        pass


def norm_eeg(eeg_epoch_channel):
    Min = np.min(eeg_epoch_channel)
    Max = np.max(eeg_epoch_channel)
    after_norm = (eeg_epoch_channel - Min) / (Max - Min)
    return after_norm


def load_psg(path_extracted, subject_id, channels, resample=3000):
    psg = scio.loadmat('{}\\subject{}.mat'.format(path_extracted, subject_id))
    psg_resample = []
    for c in channels:
        psg_resample.append(
            np.expand_dims(psg[c], 1)  
        )
    psg_resample = np.concatenate(psg_resample, axis=1)  
    return psg_resample


def stdp_test():
    path_extracted = ''
    channels = ['C3_A2', 'C4_A1', 'F3_A2', 'F4_A1', 'O1_A2', 'O2_A1',
                'LOC_A2', 'ROC_A1', 'X1', 'X2']
    psg = load_psg(path_extracted, 1, channels)

    eeg_channel_0 = psg[521][0][:250]
    eeg_channel_1 = psg[521][3][:250]
    print(eeg_channel_0)

    after_norm_0 = norm_eeg(eeg_channel_0)
    after_norm_1 = norm_eeg(eeg_channel_1)

    BSA_encoder = BSA()
    after_encode0 = torch.from_numpy(BSA_encoder.encode(after_norm_0)).view(-1, 1)
    after_encode1 = torch.from_numpy(BSA_encoder.encode(after_norm_1)).view(-1, 1)
    # print(after_encode0)

    stdp = STDP(pre_tau=100., post_tau=100.)
    stdp.get_trace_stdp_weight(pre_spikes=after_encode0, post_spikes=after_encode1)
    w = stdp.w

    fig = plt.figure(figsize=(11, 6))
    x = np.array(list(range(len(after_encode0))))
    T = len(after_encode0)
    after_encode0 = after_encode0[:, 0].numpy()
    after_encode1 = after_encode1[:, 0].numpy()

    # pre eeg and spike
    plt.subplot(3, 1, 1)
    plt.plot(x, after_norm_0)
    plt.eventplot((x * after_encode0)[after_encode0 == 1], lineoffsets=0.5, colors='r')
    plt.yticks([])
    plt.ylabel('$spike_{pre}$', rotation=0, labelpad=60, fontsize=18)
    plt.xticks([])
    plt.xlim(0, T)

    # post eeg and spike
    plt.subplot(3, 1, 2)
    plt.plot(x, after_norm_1)
    plt.eventplot((x * after_encode1)[after_encode1 == 1], lineoffsets=0.5, linelengths=2, colors='r')
    plt.yticks([])
    plt.ylabel('$spike_{post}$', rotation=0, labelpad=60, fontsize=18)
    plt.xticks([])
    plt.xlim(0, T)

    # w change
    plt.subplot(3, 1, 3)
    plt.plot(x, w)
    plt.ylabel('$w$', rotation=0, labelpad=30, fontsize=18)

    plt.show()
    fig.savefig('trace_stdp.pdf', format='pdf', bbox_inches='tight')



if __name__ == '__main__':
    
    stdp_test()


