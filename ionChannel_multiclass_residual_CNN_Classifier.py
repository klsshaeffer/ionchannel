#! /usr/bin/env /usr/bin/python3

# /srv/projects/tf2/ionch/ionChannel_multiclass_residual_CNN_Classifier.py
# created: 081820)
# modified: (082620),(082720),(082820),(090120),(090220),(090320),(090420),(090520),
#           (090620),(090720),(090920),(091820),(091920,(092020),(092120),(092220),
#           (092320),

# The train and test data set drift issue has been resolved. See this email discussion:
# https://www.kaggle.com/c/liverpool-ion-switching/discussion/137537
#
# I will implement a simple preprocessing signal conditioning process to remove the drift from both
# the 5E6 training sample points and the 2E6 test samples points.

from absl import app
from absl import flags
from typing import Callable

import signal

import numpy as np
import pandas as pd
pd.set_option("display.precision",3)
pd.set_option('display.max_columns',1000)
pd.set_option('display.max_rows',500)

from typing import Iterable,Tuple
import itertools
import copy as cp
import gc

from collections import Counter
import operator

import datetime as dt
import os

#from sklearn.model_selection import KFold, train_test_split, StratifiedKFold, GridSearchCV, cross_validate
import sklearn.metrics as metrics

import tensorflow as tf


# https://stackoverflow.com/questions/46421258/limit-number-of-cores-used-in-keras?noredirect=1&lq=1
#from tensorflow.keras import backend as K
#K.set_session(K.tf.Session(config=K.tf.ConfigProto(intra_op_parallelism_threads=4,
    #inter_op_parallelism_threads=4)))


FLAGS = flags.FLAGS

flags.DEFINE_string("train_file","data/train.csv", "File path to the training data.")
flags.DEFINE_string("test_file","data/test.csv", "File path to the test data.")
flags.DEFINE_string("solution_template","data/sample_submission.csv", "kaggle solution template file path.")
flags.DEFINE_string("solution_file","data/pred_ionch_resid_6CNN_092320_vacc_0.95_split_0.2.csv", "kaggle solution file path.")
flags.DEFINE_string("logs_path","data/logs_resid_6CNN_092320_vacc_0.95_split_0.2","tensorboard logs path")
flags.DEFINE_string("checkpoint_file", "ckptdata/checkpoint", "File path for checkpointing model parameters.")

# Batch_size 400, 320, 250, 200, 100, 50, 25, 20, 10
flags.DEFINE_integer("batch_size", 440, "Train and evaluation batch size")
flags.DEFINE_integer("epochs", 100, "Maximimum number of epochs during the training process.")
flags.DEFINE_integer("window_size", 15, "number of features per processed data point")
flags.DEFINE_integer("kfold_splits", 5, "Do not change: Number of KFold splits for generating validation data.")
flags.DEFINE_integer("current_split",1, "1 <= curr_spl <= 5 The specific split to train on for this process.")
flags.DEFINE_integer("nmbr_classes", 11, "Number of classes mapping to number of ion channels open")
flags.DEFINE_integer("max_index", 5000000, "Maximum training data index.")
flags.DEFINE_float("valid_split", 0.2, "Percent of training data split off for validation.")
flags.DEFINE_integer("tr_patience", 12, "Number of epochs to wait for improved training metric.")

flags.DEFINE_boolean("load_model_params", False, "Conditionally load pretrained model parameters")
flags.DEFINE_boolean("remove_50Hz_fft", True, "Remove 50 Hz power line noise from signal with FFT")

flags.DEFINE_integer("conv1D_nmbr_layers", 6, "Number of Conv1D layers, with 2 per residual block")
flags.DEFINE_integer("conv1D_filters", 48, "Number of Conv1D filters for each layer. Constant for this model.")
flags.DEFINE_integer("conv1D_kernel_size", 3, "The node witdh of each kernel. Constant for this model.")
flags.DEFINE_integer("conv1D_strides", 1, "The layer's kernel stride. Constant for this model.")
flags.DEFINE_string("conv1D_padding", "same", "The layer's padding string parameter. Constant for this model.")
flags.DEFINE_integer("conv1D_dilation_rate", 1, "The layer's dilation rate. Constant for this model.")
flags.DEFINE_integer("lr_decay_modulo", 4, "The learning rate callback learning rate decay parameter.")
flags.DEFINE_float("initial_learning_rate", 0.008, "Initial learning rate of the optimization process.")


# Six conv1D layers, each with kernel and bias l2 regularization parameters
# conv1D_regularization['layer'][0] == kernel   conv1D_regularization['layer'][1] == bias
conv1D_regularization = [(0.75e-2,0.75e-2),
                         (1.00e-2,1.00e-2),
                         (1.50e-2,1.50e-2),
                         (3.25e-2,3.25e-2),
                         (3.75e-2,3.75e-2),
                         (4.50e-2,4.50e-2),
                         (5.50e-2,5.50e-2),     # These last two are for the Dense layer.
                         ]

def get_drift_signal():
    amplitude = 5
    sample_rate = 10000     # samples/second
    sine_wave_period = 100  # 1e6 samples / 10000 samples/second = 100 seconds

    sine_wave_samples = sample_rate * sine_wave_period  # 1E6 samples per period

    signal_sample_times = np.arange(sine_wave_samples)/sample_rate        # we only use 1/2 the sine wave

    return amplitude * np.sin((2 * np.pi / sine_wave_period) * signal_sample_times)


class IonChannelData():
    def __init__(self, config):
        self.data_processed = False
        self.train_file = config.train_file
        self.test_file = config.test_file
        self.solution_file = config.solution_file
        self.solution_template_file = config.solution_template
        self.window_size = config.window_size
        self.nmbr_classes = config.nmbr_classes
        self.remove_50Hz_fft = config.remove_50Hz_fft

        print(f"{self.train_file}    {self.test_file}")
        print(f"{self.solution_file}    {self.solution_template_file}")

        self.trainDf = pd.read_csv(self.train_file)
        self.testDf = pd.read_csv(self.test_file)
        self.solutDf = pd.read_csv(self.solution_template_file)

        #for u in [x for x in dir(self) if not callable(getattr(self,x))]:
            #print(f"{u}\n")

        # This would need investigated.
        # Outlier mitigation
        # https://www.kaggle.com/c/liverpool-ion-switching/discussion/154011
        #outlier_samples = (47.8580,47.8629)
        #noise = np.random.normal(loc=-2.732151,scale=0.19,size=50)
        #self.trainDf.loc[(self.trainDf.time >= outlier_samples[0]) & (
            #self.trainDf.time <= outlier_samples[1]), "signal"] = noise

        # This would need investigated.
        # FIX THIS. Removing those would require refactoring of the feature windows.
        # Maybe examine the data and impute reasonable values. Maybe use imputation function.
        #outlier_samples2 = (364.290,382.2899)
        #self.trainDf.loc[(self.trainDf.time < outlier_samples2[0]) or (
            #self.trainDf.time > outlier_samples2[1]), :]


        print(f"self.testDf['signal'].mean() = {self.testDf['signal'].mean()}\n")
        print(f"self.testDf['signal'].max() = {self.testDf['signal'].max()}\n")
        print(f"self.testDf['signal'].min() = {self.testDf['signal'].min()}\n\n")

        # This does not work as implemented.
        # Here we remove the leaked data corruption.
        # https://www.kaggle.com/c/liverpool-ion-switching/discussion/153824
        #self.testDf['signal'][700000:800000] -= self.trainDf['signal'][4000000:4100000]

        #for i in range(600000,810000,5000):
             #print(f"i = {i} self.testDf.iloc[i:i+5000,1].max() = {self.testDf.iloc[i:i+5000,1].max()}\n")
             #print(f"i = {i} self.testDf.iloc[i:i+5000,1].min() = {self.testDf.iloc[i:i+5000,1].min()}\n")
             #print(f"i = {i} self.testDf.iloc[i:i+5000,1].mean() = {self.testDf.iloc[i:i+5000,1].mean()}\n\n",flush=True)

        print(f"self.trainDf.shape = {self.trainDf.shape}")
        print(f"self.testDf.shape = {self.testDf.shape}")
        print(f"self.solutDf.shape = {self.solutDf.shape}")

        for u in self.trainDf.columns:
            print(f"{u}")
        print(f"\n")
        for u in self.testDf.columns:
            print(f"{u}")
        print(f"\n")

        print(f"{self.trainDf['open_channels'].value_counts(bins=11)}\n\n")

        print(f"0 == {len(self.trainDf[self.trainDf.open_channels == 0])}")
        print(f"1 == {len(self.trainDf[self.trainDf.open_channels == 1])}")
        print(f"2 == {len(self.trainDf[self.trainDf.open_channels == 2])}")
        print(f"3 == {len(self.trainDf[self.trainDf.open_channels == 3])}")
        print(f"4 == {len(self.trainDf[self.trainDf.open_channels == 4])}")
        print(f"5 == {len(self.trainDf[self.trainDf.open_channels == 5])}")
        print(f"6 == {len(self.trainDf[self.trainDf.open_channels == 6])}")
        print(f"7 == {len(self.trainDf[self.trainDf.open_channels == 7])}")
        print(f"8 == {len(self.trainDf[self.trainDf.open_channels == 8])}")
        print(f"9 == {len(self.trainDf[self.trainDf.open_channels == 9])}")
        print(f"10 == {len(self.trainDf[self.trainDf.open_channels == 10])}\n\n",flush=True)

        return None


    def preprocess_50Hz_line_noise(self,) -> None:
        for s in range(10):
            low = s * 500000
            high = 500000 + s * 500000

            if self.remove_50Hz_fft == True:
                # 101 tap FIR hanning window filter to remove 50 Hz noise with notch filter
                # 50 Hz is at afft[2500]
                hanning_window = np.hanning(101)
                hanning_fft = np.fft.rfft(hanning_window,n=101,axis=0,norm=None)

                afft = np.fft.rfft(self.trainDf.iloc[low:high,1].values,n=500000,axis=0,norm=None)
                afft[2475:2526] = afft[2475:2526] - afft[2475:2526] * hanning_fft

                self.trainDf.iloc[low:high,1] = np.fft.irfft(afft,n=500000,axis=0,norm=None)

                if s < 4:
                    aefft = np.fft.rfft(self.testDf.iloc[low:high,1].values,n=500000,axis=0,norm=None)
                    aefft[2475:2526] = aefft[2475:2526] - aefft[2475:2526] * hanning_fft

                    self.testDf.iloc[low:high,1] = np.fft.irfft(aefft,n=500000,axis=0,norm=None)

                    del aefft
                del afft
                gc.collect()
        return None

    def channel_alignments(self,):
        """
        (0,11) mean of 0 thru 499999, sequence 0 for each open channel class
        (1,11) mean of 500000 thru 999999, sequence 1 for each open channel class
        (2,11) mean of 1000000 thru 1499999, sequence 2 for each open channel class
        (3,11) mean of 1500000 thru 1999999, sequence 3 for each open channel class
        (4,11) mean of 2000000 thru 2499999, sequence 4 for each open channel class
        (5,11) mean of 2500000 thru 2999999, sequence 5 for each open channel class
        (6,11) mean of 3000000 thru 3499999, sequence 6 for each open channel class
        (7,11) mean of 3500000 thru 3999999, sequence 7 for each open channel class
        (8,11) mean of 4000000 thru 4499999, sequence 8 for each open channel class
        (9,11) mean of 4500000 thru 4999999, sequence 9 for each open channel class
        (10,11) mean of 0 thru 4999999, concat sequence for each open channel class
        """
        self.trchan_means = np.zeros((11,11),dtype=np.float)
        self.trchan_counts = np.zeros((11,11),dtype=np.float) + 1e-4
        self.trchan_shifts = np.zeros((10,11), dtype=np.float)

        self.trchan_shifts[0] += np.array([0.0,0.0,0.,0.,0.,0.,0.,0.,0.,0.,0.,])
        self.trchan_shifts[1] += np.array([0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,])
        self.trchan_shifts[2] += np.array([0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,])
        self.trchan_shifts[3] += np.array([0.,0.,0.,-0.03,0.,0.,0.,0.,0.,0.,0.,])
        self.trchan_shifts[4] += np.array([2.6,2.88,2.75,2.73,2.52,1.81,0.,0.,0.,0.,0.])
        self.trchan_shifts[5] += np.array([0.,0.,0.,-0.03,-0.2,-0.92,0.,0.,0.,0.,0.,])
        self.trchan_shifts[6] += np.array([0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,])
        self.trchan_shifts[7] += np.array([0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,])
        self.trchan_shifts[8] += np.array([-0.16,-0.04,-0.01,-0.02,-0.20,-0.87,0.,0.,0.,0.,0.,])
        self.trchan_shifts[9] += np.array([0.0,2.66,2.69,2.69,2.51,1.81,0.,0.,0.,0.,0.,])

        total_mean_idx = 10
        for s in range(10):
            low = s * 500000
            high = low + 499999
            for u in range(low, high+1):
                self.nptr[u] += self.trchan_shifts[s,self.open_chann[u]]
                self.trchan_means[s,(self.open_chann[u])] += self.nptr[u]
                self.trchan_counts[s,(self.open_chann[u])] += 1
                self.trchan_means[total_mean_idx,(self.open_chann[u])] += self.nptr[u]
                self.trchan_counts[total_mean_idx,(self.open_chann[u])] += 1

        trch_means = np.divide(self.trchan_means,self.trchan_counts)
        for u in range(11):
            print(f"\n{u}  {trch_means[u,:]}")
            print(f"\n{u}  {self.trchan_counts[u,:].astype(np.int32)}\n")


        self.tesig_dist = np.zeros((5,11),dtype=np.float)
        self.tesig_knns = np.zeros((5,11),dtype=np.float) + 1e-4
        self.tesig_means = np.zeros((5,11),dtype=np.float)

        te_mean_idx = 4
        for s in range(4):
            low = s * 500000
            high = low + 500000
            for u in range(low,high):
                dist = np.abs(trch_means[total_mean_idx] - self.npte[u])
                self.tesig_dist[s] = np.abs(trch_means[total_mean_idx] - self.npte[u])
                self.tesig_knns[s,np.argmin(dist)] += 1
                self.tesig_means[s,np.argmin(dist)] += self.npte[u]
                self.tesig_means[te_mean_idx,np.argmin(dist)] += self.npte[u]
                self.tesig_knns[te_mean_idx,np.argmin(dist)] += 1

        tesig_means = np.divide(self.tesig_means,self.tesig_knns)
        for u in range(5):
            print(f"\n{u}  {tesig_means[u,:]}\n")
            print(f"\n{u}  {self.tesig_knns[u,:].astype(np.int32)}\n")

        return None


    def remove_signal_drift(self,):
        drift = get_drift_signal()
        print(f"type(drift) = {type(drift)}  drift.shape = {drift.shape}\n",flush=True)

        for s in range(10):
            low = s * 500000
            high = 500000 + low
            dct = 0
            for u in range(low,high):
                if low == 500000:
                    if u == 500000:
                        dct = 0
                    if u < 600000:
                        self.nptr[u] = self.nptr[u] - drift[dct]
                        dct += 1
                elif low in [3000000,3500000,4000000,4500000]:
                    if u % 500000 == 0:
                        dct = 0
                    self.nptr[u] = self.nptr[u] - drift[dct]
                    dct += 1
                if low == 0:
                    if u in [0,100000,400000]:
                        dct = 0
                    if u < 100000:
                        self.npte[u] = self.npte[u] - drift[dct]
                        dct += 1
                    elif u > 99999 and u < 200000:
                        self.npte[u] = self.npte[u] - drift[dct]
                        dct += 1
                    elif u > 399999 and u < 500000:
                        self.npte[u] = self.npte[u] - drift[dct]
                        dct += 1
                elif low == 500000:
                    if u in [100000,200000,300000]:
                        dct = 0
                    if u > 99999 and u < 200000:
                        self.npte[u] = self.npte[u] - drift[dct]
                        dct += 1
                    elif u > 199999 and u < 300000:
                        self.npte[u] = self.npte[u] - drift[dct]
                        dct += 1
                    elif u > 299999 and u < 400000:
                        self.npte[u] = self.npte[u] - drift[dct]
                        dct += 1
                elif low == 1000000:
                    if u == 1000000:
                        dct = 0
                    self.npte[u] = self.npte[u] - drift[dct]
                    dct += 1
        return None

    def tr_te_normalize(self,) -> None:
        xtr_mean = np.mean(self.nptr)
        xtr_min = np.min(self.nptr)
        xtr_max = np.max(self.nptr)
        self.nptr[:] = (self.nptr[:] - xtr_mean - xtr_min)/(xtr_max - xtr_min)

        xtr_min = np.min(self.nptr)
        self.nptr[:] = self.nptr[:] - xtr_min

        xte_mean = np.mean(self.npte)
        xte_min = np.min(self.npte)
        xte_max = np.max(self.npte)

        self.npte = (self.npte - xte_mean - xte_min)/(xte_max - xte_min)
        xte_min = np.min(self.npte)
        self.npte = self.npte - xte_min

        print(f"\nnp.mean(self.nptr) = {np.mean(self.nptr)}\n")
        print(f"np.min(self.nptr) = {np.min(self.nptr)}\n")
        print(f"np.max(self.nptr) = {np.max(self.nptr)}\n\n")

        print(f"np.mean(self.npte) = {np.mean(self.npte)}\n")
        print(f"np.min(self.npte) = {np.min(self.npte)}\n")
        print(f"np.max(self.npte) = {np.max(self.npte)}\n\n")

        return None


    def preprocess_data(self,):
        if self.remove_50Hz_fft == True:
            self.preprocess_50Hz_line_noise()

        self.npte = cp.deepcopy(self.testDf['signal'].values)
        self.nptr = cp.deepcopy(self.trainDf['signal'].values)

        print(f"np.mean(self.npte[:]) = {np.mean(self.npte[:])}\n")
        print(f"np.max(self.npte[:]) = {np.max(self.npte[:])}\n")
        print(f"np.min(self.npte[:]) = {np.min(self.npte[:])}\n",flush=True)

        print(f"np.mean(self.nptr[:]) = {np.mean(self.nptr[:])}\n")
        print(f"np.max(self.nptr[:]) = {np.max(self.nptr[:])}\n")
        print(f"np.min(self.nptr[:]) = {np.min(self.nptr[:])}\n",flush=True)

        self.open_chann = cp.deepcopy(self.trainDf['open_channels'].values.astype(np.int32))
        self.open_ch = np.zeros((self.open_chann.shape[0],self.nmbr_classes),dtype=np.int32)

        self.remove_signal_drift()
        self.channel_alignments()
        self.tr_te_normalize()

        for u in range(self.open_chann.shape[0]):
            self.open_ch[u][self.open_chann[u]] += 1

        del self.open_chann
        del self.trainDf
        del self.testDf
        gc.collect()

        print(f"self.open_ch.shape = {self.open_ch.shape}\n")
        print(f"self.open_ch =\n{self.open_ch}\n")

        window = self.window_size    # 11
        ctr = window // 2

        self.xtr = np.zeros((int(5e6),window),np.float32)
        self.xte = np.zeros((int(2e6),window),np.float32)

        for u in range(0, 5000000):
            self.xtr[u][ctr] = self.nptr[u]
            if u < 2000000:
                self.xte[u][ctr] = self.npte[u] 

        # Addition of np.abs(xtr_min2) results in 0.0 <= values <= 1.0
        # Not implemented here, because the Convolution batch normalization uses mean == 0.0,
        # And so we maintain that consistency here, by not shifting the values.
        #xtr_min2 = np.min(self.xtr[:,ctr])
        #self.xtr[:,ctr] = self.xtr[:,ctr] + np.abs(xtr_min2)

        nptrCnt = 0
        for s in range(10):
            #print(f"Loop Invariant 2: s = {s}\n",flush=True)
            low = s * 500000
            high = 499999 + s * 500000

            for c in range(500000):
                low_tail = self.xtr[low][ctr]
                high_tail = self.xtr[high][ctr]
                if s < 4:
                    low_test_tail = self.xte[low][ctr]
                    high_test_tail = self.xte[high][ctr]
                if c <= ctr:
                    if s < 4:
                        for u in range(1,c):
                            self.xtr[nptrCnt][ctr-u] = self.xtr[nptrCnt - u][ctr]
                            self.xte[nptrCnt][ctr-u] = self.xte[nptrCnt-u][ctr]
                        for u in range(ctr - c):
                            self.xtr[nptrCnt][u] = low_tail + np.random.normal(0.0,0.1)
                            self.xte[nptrCnt][u] = low_test_tail + np.random.normal(0.0,0.1)
                        for u in range(1,ctr+1):
                            self.xtr[nptrCnt][ctr+u] = self.xtr[nptrCnt + u][ctr]
                            self.xte[nptrCnt][ctr+u] = self.xte[nptrCnt + u][ctr]
                    else:
                        for u in range(1,c):
                            self.xtr[nptrCnt][ctr-u] = self.xtr[nptrCnt - u][ctr]
                        for u in range(ctr - c):
                            self.xtr[nptrCnt][u] = low_tail + np.random.normal(0.0,0.1)
                        for u in range(1,ctr+1):
                            self.xtr[nptrCnt][ctr+u] = self.xtr[nptrCnt + u][ctr]
                elif c <= 499999 - ctr:
                    if s < 4:
                        for u in range(1, ctr+1):
                            self.xtr[nptrCnt][ctr-u] = self.xtr[nptrCnt-u][ctr]
                            self.xtr[nptrCnt][ctr+u] = self.xtr[nptrCnt+u][ctr]
                            self.xte[nptrCnt][ctr-u] = self.xte[nptrCnt-u][ctr]
                            self.xte[nptrCnt][ctr+u] = self.xte[nptrCnt+u][ctr]
                    else:
                        for u in range(1,ctr+1):
                            self.xtr[nptrCnt][ctr-u] = self.xtr[nptrCnt-u][ctr]
                            self.xtr[nptrCnt][ctr+u] = self.xtr[nptrCnt+u][ctr]
                else:
                    if s < 4:
                        for u in range(1,(499999 - c + 1)):  # 1,2,3,4; 1,2,3; 1,2; 1
                            self.xtr[nptrCnt][ctr+u] = self.xtr[nptrCnt+u][ctr]
                            self.xte[nptrCnt][ctr+u] = self.xte[nptrCnt+u][ctr]

                        for u in range(c - 499994):     # 0; 0,1; 0,1,2; 0,1,2,3; 0,1,2,3,4
                            self.xtr[nptrCnt][2 * ctr - u] = high_tail + np.random.normal(0.0,0.1)
                            self.xte[nptrCnt][2 * ctr - u] = high_test_tail + np.random.normal(0.0,0.1)

                        for u in range(1,ctr+1):
                            self.xtr[nptrCnt][ctr-u] = self.xtr[nptrCnt-u][ctr]
                            self.xte[nptrCnt][ctr-u] = self.xte[nptrCnt-u][ctr]
                    else:
                        for u in range(1,(499999 - c + 1)):  # 1,2,3,4; 1,2,3; 1,2; 1
                            self.xtr[nptrCnt][ctr+u] = self.xtr[nptrCnt+u][ctr]

                        for u in range(c - 499994):     # 0; 0,1; 0,1,2; 0,1,2,3; 0,1,2,3,4
                            self.xtr[nptrCnt][2 * ctr - u] = high_tail + np.random.normal(0.0,0.1)

                        for u in range(1,ctr+1):
                            self.xtr[nptrCnt][ctr-u] = self.xtr[nptrCnt-u][ctr]

                nptrCnt += 1

        del self.nptr
        del self.npte
        gc.collect()

        print(f"\nmethod: preprocess_data() completed successfully.\n")
        self.data_processed = True
        return None

    @property
    def xtrain(self):
        if self.data_processed == False:
            preprocess_data(self,)
        return self.xtr

    @property
    def xtest(self,):
        if self.data_processed == False:
            preprocess_data(self,)
        return self.xte

    @property
    def open_channels(self,):
        return self.open_ch


def random_shuffled_indexes(max_index:int, valid_split:float, seed:int) -> Tuple[Iterable,Iterable]:
    np.random.seed(seed)
    valid_idx_cnt = np.int32(valid_split * max_index)
    train_idx_cnt = max_index - valid_idx_cnt

    index_pool = np.arange(max_index).astype(np.int32)
    np.random.shuffle(index_pool)

    xtr_idx = cp.deepcopy(index_pool[0:train_idx_cnt].tolist())
    xval_idx = cp.deepcopy(index_pool[train_idx_cnt:max_index].tolist())

    print(f"type(xtr_idx) = {type(xtr_idx)}   type(xval_idx) = {type(xval_idx)}\n")
    print(f"len(xtr_idx) = {len(xtr_idx)}  len(xval_idx) = {len(xval_idx)}\n")

    del index_pool
    gc.collect()
    return xtr_idx,xval_idx


class IonChannelModel():                      #tf.keras.Model): super(IonChannelModel,self).__init__()
    class_caught_SIGTERM = False
    def __init__(self,ionChData,ionCh_config,conv1D_regularization,):
        self.ionChanData = ionChData
        self.xtr = ionChData.xtr
        self.xte = ionChData.xte
        self.open_ch = ionChData.open_ch

        self.solutDf = ionChData.solutDf
        self.solution_file = ionChData.solution_file

        self.batch_size = ionCh_config.batch_size
        self.epochs = ionCh_config.epochs
        self.window_size = ionCh_config.window_size
        self.kfold_splits = ionCh_config.kfold_splits
        self.current_split = ionCh_config.current_split
        self.nmbr_classes = ionCh_config.nmbr_classes
        self.max_index = ionCh_config.max_index
        self.valid_split = ionCh_config.valid_split
        self.tr_patience = ionCh_config.tr_patience
        self.logs_path = ionCh_config.logs_path
        self.load_model_params = ionCh_config.load_model_params
        self.conv1D_nmbr_layers = ionCh_config.conv1D_nmbr_layers
        self.conv1D_filters = ionCh_config.conv1D_filters
        self.conv1D_kernel_size = ionCh_config.conv1D_kernel_size
        self.conv1D_strides = ionCh_config.conv1D_strides
        self.conv1D_padding = ionCh_config.conv1D_padding
        self.conv1D_dilation_rate = ionCh_config.conv1D_dilation_rate
        self.lr_decay_modulo = ionCh_config.lr_decay_modulo
        self.initial_learning_rate = ionCh_config.initial_learning_rate
        self.checkpoint_file = ionCh_config.checkpoint_file

        self.first_epoch = True
        self.current_lr = self.initial_learning_rate

        self.regularization = conv1D_regularization

        self.rseed = np.random.randint(low=0,high=10000,size=3)
        print(f"\nrandom seeds: {self.rseed[0]},  {self.rseed[1]}  {self.rseed[2]}\n"\
              f"self.rseed.shape = {self.rseed.shape}\n\n",flush=True)

        return None

    def get_Conv1D_layer(self,filters,kernel_size,strides,
                         padding,dilation_rate,regularization_tuple):

        return tf.keras.layers.Conv1D(filters=filters,kernel_size=(kernel_size),
                strides=strides,padding=padding,data_format="channels_last",
                dilation_rate=dilation_rate,activation=None,use_bias=True,
                kernel_initializer='glorot_uniform',bias_initializer='zeros',
                kernel_regularizer=tf.keras.regularizers.l2(regularization_tuple[0]),
                bias_regularizer=tf.keras.regularizers.l2(regularization_tuple[1]),
                activity_regularizer=None,kernel_constraint=None,
                bias_constraint=None,)


    def get_BatchNorm_layer(self,):

        return tf.keras.layers.BatchNormalization(axis=-1, momentum=0.99,
            epsilon=0.001,center=True, scale=True, beta_initializer='zeros',
            gamma_initializer='ones',moving_mean_initializer='zeros',
            moving_variance_initializer='ones',beta_regularizer=None,
            gamma_regularizer=None, beta_constraint=None,gamma_constraint=None,
            renorm=False, renorm_clipping=None, renorm_momentum=0.99,
            fused=None, trainable=True, virtual_batch_size=None,
            adjustment=None, name=None)


    def get_ReLU_activation_layer(self,):

        return tf.keras.layers.ReLU(max_value=None, negative_slope=0,
            threshold=0)


    def instantiate_model(self,):
        self.Inputs = tf.keras.Input(shape=(self.window_size,1), batch_size=self.batch_size,
            dtype="float32",)

        print(f"self.Inputs.shape = {self.Inputs.shape}\n")

        # Padding may be one of {"valid", "causal", "same"}
        # "valid" does not add zero-padding
        # "same" adds zero-padding to maintain the same dimensions for the output.
        # "causal" is compatible with dilated convolution.
        self.Conv1D_1 = self.get_Conv1D_layer(self.conv1D_filters,
            self.conv1D_kernel_size,self.conv1D_strides,
            self.conv1D_padding,self.conv1D_dilation_rate,
            self.regularization[0])(self.Inputs)

        self.BatchNorm_1 = self.get_BatchNorm_layer()(self.Conv1D_1)

        self.ReLU_1 = self.get_ReLU_activation_layer()(self.BatchNorm_1)

        self.Conv1D_2 = self.get_Conv1D_layer(self.conv1D_filters,
            self.conv1D_kernel_size,self.conv1D_strides,
            self.conv1D_padding,self.conv1D_dilation_rate,
            self.regularization[1])(self.ReLU_1)

        self.BatchNorm_2 = self.get_BatchNorm_layer()(self.Conv1D_2)

        self.Elementwise_Add_1 = tf.keras.layers.add([self.Inputs,self.BatchNorm_2])

        self.ReLU_2 = self.get_ReLU_activation_layer()(self.Elementwise_Add_1)

        #self.ResBlk_1_Drop = tf.keras.layers.Dropout(rate=0.25)(self.ReLU_2)

        self.Conv1D_3 = self.get_Conv1D_layer(self.conv1D_filters,
            self.conv1D_kernel_size,self.conv1D_strides,
            self.conv1D_padding,self.conv1D_dilation_rate,
            self.regularization[2])(self.ReLU_2)

        self.BatchNorm_3 = self.get_BatchNorm_layer()(self.Conv1D_3)

        self.ReLU_3 = self.get_ReLU_activation_layer()(self.BatchNorm_3)

        self.Conv1D_4 = self.get_Conv1D_layer(self.conv1D_filters,
            self.conv1D_kernel_size,self.conv1D_strides,
            self.conv1D_padding,self.conv1D_dilation_rate,
            self.regularization[3])(self.ReLU_3)

        self.BatchNorm_4 = self.get_BatchNorm_layer()(self.Conv1D_4)

        self.Elementwise_Add_2 = tf.keras.layers.add([self.ReLU_2,self.BatchNorm_4])

        self.ReLU_4 = self.get_ReLU_activation_layer()(self.Elementwise_Add_2)

        #self.ResBlk_2_Drop = tf.keras.layers.Dropout(rate=0.25)(self.ReLU_4)

        self.Conv1D_5 = self.get_Conv1D_layer(self.conv1D_filters,
            self.conv1D_kernel_size,self.conv1D_strides,
            self.conv1D_padding,self.conv1D_dilation_rate,
            self.regularization[4])(self.ReLU_4)

        self.BatchNorm_5 = self.get_BatchNorm_layer()(self.Conv1D_5)

        self.ReLU_5 = self.get_ReLU_activation_layer()(self.BatchNorm_5)

        self.Conv1D_6 = self.get_Conv1D_layer(self.conv1D_filters,
            self.conv1D_kernel_size,self.conv1D_strides,
            self.conv1D_padding,self.conv1D_dilation_rate,
            self.regularization[5])(self.ReLU_5)

        self.BatchNorm_6 = self.get_BatchNorm_layer()(self.Conv1D_6)

        self.Elementwise_Add_3 = tf.keras.layers.add([self.ReLU_4,self.BatchNorm_6])

        self.ReLU_6 = tf.keras.layers.ReLU(max_value=None, negative_slope=0,
            threshold=0)(self.Elementwise_Add_3)

        #self.ResBlk_3_Drop = tf.keras.layers.Dropout(rate=0.25)(self.ReLU_6)

        self.GlobalMaxPool1D_1 = tf.keras.layers.GlobalMaxPool1D(
            data_format='channels_last')(self.ReLU_6)

        # Note this Dense layer may not be helpful with the Global{Ave|Max}Pool1D() layer
        self.class_nodes = tf.keras.layers.Dense(
            units = 11, activation=None, use_bias=True, kernel_initializer='glorot_uniform',
            bias_initializer='zeros',
            kernel_regularizer=tf.keras.regularizers.l2(self.regularization[6][0]),
            bias_regularizer=tf.keras.regularizers.l2(self.regularization[6][1]),
            activity_regularizer=None,
            kernel_constraint=None,
            bias_constraint=None)(self.GlobalMaxPool1D_1)

        self.raw_outputs = tf.keras.layers.ReLU(max_value=None, negative_slope=0,
            threshold=0)(self.class_nodes)

        self.predicts = tf.keras.layers.Softmax(axis=-1)(self.raw_outputs)

        self.model = tf.keras.Model(inputs=self.Inputs,outputs=self.predicts)

        print(f"\n",flush=True)
        self.model.summary()
        print(f"\n",flush=True)

        self.model.compile(optimizer=tf.keras.optimizers.Nadam(
            learning_rate = self.initial_learning_rate,
            beta_1=0.9,beta_2=0.999,epsilon=1e-7,name='Nadam',),
            loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
            metrics=["accuracy"],)

        if self.load_model_params == True:
            self.model.load_weights(self.checkpoint_file,by_name=False,skip_mismatch=False)
            self.load_model_params = False
            print(f"model.load_weights successfully completed.\n",flush=True)

        return None

    def train_model(self, training=False):

        class TrainingMgmt_CB(tf.keras.callbacks.Callback):
            def __init__(self,tr_patience):
                super(TrainingMgmt_CB,self).__init__()
                self.patience = tr_patience
                self.best_weights = None
                self.best_weights_OK = False
                return None

            def on_train_begin(self, logs=None):
                self.wait = 0
                self.stopped_epoch = 0
                self.best_loss = np.Inf
                self.best_val_acc = 0.0
                return None

            def on_epoch_end(self, epoch, logs=None):
                # keys == ['loss', 'accuracy', 'val_loss', 'val_accuracy']
                # keys = list(logs.keys())

                wait_flag_incr = False
                current_loss = logs.get("loss")
                current_val_acc = logs.get("val_accuracy")

                if np.less_equal(current_loss,self.best_loss):
                    self.best_loss = current_loss
                    if np.greater_equal(current_val_acc,self.best_val_acc):
                        self.best_val_acc = current_val_acc
                        self.wait = 0
                        self.best_weights = self.model.get_weights()
                        print(f"\nSaved best weights: tr_loss = {current_loss} val_acc = {current_val_acc}",end="")
                    else:
                        self.wait += 1
                        wait_flag_incr = True
                else:
                    self.wait += 1
                    wait_flag_incr = True

                if wait_flag_incr == True:
                    if self.wait >= self.patience:
                        self.stopped_epoch = epoch
                        print(f"Restoring model with the best weights.\n")
                        self.model.set_weights(self.best_weights)
                        self.model.stop_training = True

                if IonChannelModel.class_caught_SIGTERM == True:
                    if self.model.stop_training == False:
                        print(f"Restoring model with the best weights.\n")
                        self.model.set_weights(self.best_weights)
                        self.model.stop_training = True

                gc.collect()
                return None

            def on_train_end(self, logs=None):
                if self.stopped_epoch > 0:
                    print(f"Epoch {(self.stopped_epoch + 1):05d} early stopping\n")
                return None


        # self.checkpoint_file = "ckptdata/checkpoint"
        check_pt_CB = tf.keras.callbacks.ModelCheckpoint(
            filepath=self.checkpoint_file, monitor="val_accuracy", verbose=1,
            save_best_only=True,mode="max", save_weights_only=True,
            save_freq="epoch")


        def scheduler(epoch):
            if self.first_epoch == True:
                self.current_lr = self.initial_learning_rate
                self.first_epoch = False
            elif epoch % self.lr_decay_modulo == 0:
                    self.current_lr /= 2.0

            print(f"scheduled learning rate = {self.current_lr}\n")
            return self.current_lr


        learn_rate_CB = tf.keras.callbacks.LearningRateScheduler(scheduler, verbose=1)

        logdir = f"{self.logs_path}/scalars/" + dt.datetime.now().strftime("%Y%m%d-%H%M%S")
        tboard_CB = tf.keras.callbacks.TensorBoard(log_dir=logdir)


        raw_preds = np.zeros((self.xte.shape[0],self.nmbr_classes)) # 11 classes: 0 thru 10 open channels

        print(f"raw_preds.shape = {raw_preds.shape}\n")
        print(f"self.xte.shape = {self.xte.shape}\n")
        print(f"self.xtr.shape = {self.xtr.shape}\n")
        print(f"self.open_ch.shape = {self.open_ch.shape}\n",flush=True)

        self.instantiate_model()

        xtr_idx,xval_idx = random_shuffled_indexes(self.max_index,self.valid_split,self.rseed[2])

        _xtr = cp.deepcopy(np.take(self.xtr,xtr_idx,axis=0))
        _ytr = cp.deepcopy(np.take(self.open_ch,xtr_idx,axis=0))
        _xval = cp.deepcopy(np.take(self.xtr,xval_idx,axis=0))
        _yval = cp.deepcopy(np.take(self.open_ch,xval_idx,axis=0))

        del self.xtr
        del self.open_ch
        del xtr_idx
        del xval_idx
        gc.collect()
        self.xtr = None
        self.open_ch = None

        history = self.model.fit(_xtr,_ytr,self.batch_size,self.epochs,verbose=2,
                callbacks=[TrainingMgmt_CB(self.tr_patience),tboard_CB,
                learn_rate_CB,check_pt_CB],validation_split=0.0,
                validation_data=(_xval,_yval),shuffle=True,class_weight=None,
                sample_weight=None,steps_per_epoch=None)

        self.model.load_weights(self.checkpoint_file,by_name=False,skip_mismatch=False)

        test_scores = self.model.predict(self.xte,self.batch_size,verbose=1,steps=None)

        print(f"\ntest_scores.shape = {test_scores.shape}\n")
        print(f"test_scores =\n{test_scores}\n")

        raw_preds += test_scores
        print(f"xte test scores predicted.\n")
        print(f"raw_preds =\n{raw_preds}\n")

        tf.keras.backend.clear_session()
        gc.collect()

        ionCh_preds = np.argmax(raw_preds,axis=1)
        print(f"raw_preds.shape = {raw_preds.shape}\n")
        print(f"ionCh_preds.shape = {ionCh_preds.shape}\n")
        print(f"raw_preds =\n{raw_preds}\n")
        print(f"ionCh_preds =\n{ionCh_preds}\n")

        #vec = self.solutDf.as_matrix(columns=[self.solutDf.columns[0]])
        vec = self.solutDf['time'].values
        vec_ions = np.column_stack((vec,ionCh_preds))
        print(f"vec.shape = {vec.shape}   vec_ions.shape = {vec_ions.shape}\n")

        binCount = 11
        histbins = np.zeros(binCount).astype(np.int32)
        with open(self.solution_file,'w') as fd:
            fd.write(f"time,open_channels\n")
            for idx in range(vec.shape[0]):
                histbins[vec_ions[idx,1].astype(np.int32)] += 1

                # {var:8.4f} denotes field width 8, including decimal point, with decimal precision 4
                fd.write(f"{vec_ions[idx,0]:8.4f},{vec_ions[idx,1].astype(np.int32)}\n")
        self.print_histbins(binCount,histbins,f"\nxtest Histbin distribution:\n")

        return None


    # Print out the histogram bins counts
    def print_histbins(self,binCnt,histbins,title):
        print(title)
        for v in range(binCnt):
            print(f"{v}   {histbins[v]}")

        print(f"\n\n")
        return None

def process_termination(signal_number, frame) -> None:
    print(f"process_termination signal handler caught: SIGTERM.\n",flush=True)
    IonChannelModel.class_caught_SIGTERM = True
    return None


def get_ionCh_config() -> Callable[[],None]:
    """ wrapper function returns the Data and Model Configuration parameters"""
    class IonChannelConfig():
        """ Class variables for data and model configuration """
        train_file = FLAGS.train_file
        test_file = FLAGS.test_file
        solution_template = FLAGS.solution_template
        solution_file = FLAGS.solution_file
        logs_path = FLAGS.logs_path
        checkpoint_file = FLAGS.checkpoint_file
        batch_size = FLAGS.batch_size
        epochs = FLAGS.epochs
        window_size = FLAGS.window_size
        kfold_splits = FLAGS.kfold_splits
        current_split = FLAGS.current_split
        nmbr_classes = FLAGS.nmbr_classes
        max_index = FLAGS.max_index
        valid_split = FLAGS.valid_split
        tr_patience = FLAGS.tr_patience
        load_model_params = FLAGS.load_model_params
        remove_50Hz_fft = FLAGS.remove_50Hz_fft
        conv1D_nmbr_layers = FLAGS.conv1D_nmbr_layers
        conv1D_filters = FLAGS.conv1D_filters
        conv1D_kernel_size = FLAGS.conv1D_kernel_size
        conv1D_strides = FLAGS.conv1D_strides
        conv1D_padding = FLAGS.conv1D_padding
        conv1D_dilation_rate = FLAGS.conv1D_dilation_rate
        lr_decay_modulo = FLAGS.lr_decay_modulo
        initial_learning_rate = FLAGS.initial_learning_rate

    return IonChannelConfig()



def main(argv):
    del argv
    ionCh_config = get_ionCh_config()
    print(f"ionCh_config.train_file = {ionCh_config.train_file}\n"
          f"ionCh_config.test_file = {ionCh_config.test_file}\n"
          f"ionCh_config.solution_template = {ionCh_config.solution_template}\n"
          f"ionCh_config.solution_file = {ionCh_config.solution_file}\n"
          f"ionCh_config.logs_path = {ionCh_config.logs_path}\n"
          f"ionCh_config.checkpoint_file = {ionCh_config.checkpoint_file}\n"
          f"ionCh_config.batch_size = {ionCh_config.batch_size}\n"
          f"ionCh_config.epochs = {ionCh_config.epochs}\n"
          f"ionCh_config.window_size = {ionCh_config.window_size}\n"
          f"ionCh_config.kfold_splits = {ionCh_config.kfold_splits}\n"
          f"ionCh_config.current_split = {ionCh_config.current_split}\n"
          f"ionCh_config.nmbr_classes = {ionCh_config.nmbr_classes}\n"
          f"ionCh_config.max_index = {ionCh_config.max_index}\n"
          f"ionCh_config.valid_split = {ionCh_config.valid_split}\n"
          f"ionCh_config.tr_patience = {ionCh_config.tr_patience}\n"
          f"ionCh_config.load_model_params = {ionCh_config.load_model_params}\n"
          f"ionCh_config.remove_50Hz_fft = {ionCh_config.remove_50Hz_fft}\n"
          f"ionCh_config.conv1D_nmbr_layers = {ionCh_config.conv1D_nmbr_layers}\n"
          f"ionCh_config.conv1D_filters = {ionCh_config.conv1D_filters}\n"
          f"ionCh_config.conv1D_kernel_size = {ionCh_config.conv1D_kernel_size}\n"
          f"ionCh_config.conv1D_strides = {ionCh_config.conv1D_strides}\n"
          f"ionCh_config.conv1D_padding = {ionCh_config.conv1D_padding}\n"
          f"ionCh_config.conv1D_dilation_rate = {ionCh_config.conv1D_dilation_rate}\n"
          f"ionCh_config.lr_decay_modulo = {ionCh_config.lr_decay_modulo}\n"
          f"ionCh_config.initial_learning_rate = {ionCh_config.initial_learning_rate}\n"
          )

    ionChannelData = IonChannelData(ionCh_config)

    ionChannelData.preprocess_data()
    ionChannelModel = IonChannelModel(ionChannelData,ionCh_config,
                         conv1D_regularization)

    ionChannelModel.train_model(training=True)




if __name__ == '__main__':
    signal.signal(signal.SIGTERM, process_termination)
    app.run(main)
