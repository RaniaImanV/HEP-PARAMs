"""
HEP computation methods
1. Generates Epochs
2. Computes HEPs
3. Does both of the above in batched, parameterised ways
"""

import mne
import os
import numpy as np
import random
import sys
import pandas as pd
from tqdm import tqdm
import concurrent.futures
from numba import jit
import csv
import itertools


class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


def GetECGEpochs(raw, ECGEvents=None, eegPicks=None, maxEpochAmp=100, MinRR=0.88, ecg_lead = "EKG1"):
    """

    """
    # standardise channel names
    if "P7" in raw.info["ch_names"]:
        raw.rename_channels({"P7": "T5"})
    if "P8" in raw.info["ch_names"]:
        raw.rename_channels({"P8": "T6"})

    print(raw.ch_names)

    if eegPicks is None:
        eegPicks = ["Fp1", "Fp2", "Fz", "Cz", "Pz", "F3", "F4", "F7", "F8", "C3", "C4", "P3", "P4", "T3", "T4", "T5",
                    "T6", "O1", "O2"]

    # Define a dictionary mapping channel names to types
    channel_types = {ch:"eeg" for ch in eegPicks}

    # Set the channel types
    raw.set_channel_types(channel_types)

    sampling_freq = raw.info["sfreq"]
    # get ECG events
    if ECGEvents is None:
        ECGEvents = mne.preprocessing.find_ecg_events(raw, ch_name=ecg_lead)
    RTimes = ECGEvents[0][:, 0]
    OriginalEpochN = len(RTimes)  # initial number of Epochs

    # identify bad ECG epochs
    minRR_samples = sampling_freq * MinRR
    diffs = np.diff(RTimes)
    threshdiffs = np.append((diffs < minRR_samples), [False])
    threshtimes = RTimes[threshdiffs]
    threshtimes += 1
    threshtimes = threshtimes / sampling_freq  # gives us the start of the R-R periods under minRR in seconds

    # label bad epochs to exclude
    raw.set_annotations(mne.Annotations(onset=list(threshtimes), duration=0.1, description="bad_short_RR"))
    print("Epochs thresholded to be more than ", MinRR, " seconds.")
    annotations = raw.annotations
    ShortRRN = len(annotations[annotations.description == "bad_short_RR"])  # number of epochs rejected due to short RR

    # compile epochs object and reject high amplitude epochs
    maxEpochAmp *= 10 ** -6  # convert this to volts
    try:
        ecg_epochs = mne.Epochs(raw, ECGEvents[0], tmin=-0.8, tmax=0.8,
                                picks=eegPicks,
                                reject=dict(eeg=maxEpochAmp),
                                reject_by_annotation=True,
                                verbose=False, preload=True)
    except ValueError as e:
        print(e)
        return None
    ArtefactRejectN = (OriginalEpochN - len(ecg_epochs)) - ShortRRN  # number of epochs rejected due to high amp

    return ecg_epochs, OriginalEpochN, ShortRRN, ArtefactRejectN


def getDirFIFs(fifDir, type="_eeg"):
    fifs = []
    for file in os.scandir(fifDir):
        if file.name.endswith(type + ".fif"):
            fifs.append(file.path)
        elif file.is_dir():
            subd_files = getDirFIFs(file.path)
            fifs = fifs + subd_files

    return fifs


def BatchEpochGeneration(fifDir, epochDir=None, maxEpochAmps=[100], MinRRs=[0.88], highPasses=[0.05], lowPasses=[70],
                         ResultsFile="EpochData.csv"
                         ):
    if type(fifDir) == list:
        fifs = fifDir

    else:
        fifs = getDirFIFs(fifDir)

    os.chdir(epochDir)

    resultDF = pd.DataFrame(columns=["Filename",
                                     "High pass",
                                     "Low pass",
                                     "Max Epoch Amplitude",
                                     "Minimum RR interval",
                                     "Original Epoch Number",
                                     "Epochs dropped due to length",
                                     "Epochs with too much artefact"])

    if epochDir is None:
        epochDir = fifDir


    for filename in tqdm(fifs):

        sys.stdout = open(os.devnull, 'w')

        # load raw
        raw = mne.io.read_raw_fif(filename, preload=True)

        print(f"Working on :{filename}")
        for lowPass in lowPasses:
            print(f"Low pass filer at: {lowPass}")
            for highPass in highPasses:
                print(f"High pass filer at: {highPass}")
                for maxEpochAmp in maxEpochAmps:
                    print(f"Excluding epochs over {maxEpochAmp}µV")
                    for MinRR in MinRRs:
                        print(f"Minimum R-R interval = {MinRR}")
                        FiltRaw = raw.filter(l_freq=highPass, h_freq=lowPass, n_jobs="cuda")

                        ecg_epochs, OriginalEpochN, ShortRRN, ArtefactRejectN = GetECGEpochs(FiltRaw,
                                                                                             maxEpochAmp=maxEpochAmp,
                                                                                             MinRR=MinRR)

                        epochSaveName = filename.split("\\")[-1].split("_eeg.")[0] + "Bandpass=" + str(
                            lowPass) + "-" + str(highPass) + "_MinRR=" + str(MinRR) + "_maxEpochAmp=" + str(
                            maxEpochAmp) + "_epo.fif"

                        print(f"Saving {epochSaveName}, which had {len(ecg_epochs)} R-R epochs, to {epochDir}")
                        print(f"{OriginalEpochN} epochs to start with, {ShortRRN} too short, {ArtefactRejectN} too artefactual")
                        ecg_epochs.save(epochSaveName, overwrite=True)

                        resultDF.loc[len(resultDF)] = [filename, highPass, lowPass, maxEpochAmp, MinRR, OriginalEpochN,
                                                       ShortRRN, ArtefactRejectN]

                        # Save the intermediate results after each loop iteration
                        resultDF.to_csv(ResultsFile, index=False)
                        print(f"Saved {ResultsFile} after processing {filename} - in the directory {epochDir}")

    resultDF.to_csv(ResultsFile, index=False)

    sys.stdout = sys.__stdout__
    print(f"{len(resultDF)} EPOCH FILES ALL PROCESSED AND SAVED!")


def LoadEpochs(fileName):
    """
    Function to load a saved Epochs .fif file
    :param fileName: filename (should end with epo.fif)
    :return: epoch object
    """
    try:
        return mne.epochs.read_epochs(fileName, preload=True)
    except FileNotFoundError as e:
        print(e)
        return None


def HEP_process(epochFile, baseStarts, baseEnds, windowStarts, windowEnds, repeats, fraction_samples):
    if len(baseStarts) > 1:
        variableName = "baseStarts"
    elif len(baseEnds) > 1:
        variableName = "baseEnds"
    elif len(windowStarts) > 1:
        variableName = "windowStarts"
    elif len(windowEnds) > 1:
        variableName = "windowEnds"
    else:
        variableName = "preprocessing"
    print("list_before:", baseStarts, baseEnds, windowStarts, windowEnds)
    sub_result_save_name = r"path_to\ModalHEPResults\\" + \
                           epochFile.split("_epo")[0].split("\\")[-1] + str(variableName) + "_HEP_results.csv"
    if os.path.exists(sub_result_save_name):
        return sub_result_save_name
    else:
        def extract_info(path):
            parts = path.split('\\')
            filename = parts[-1]

            return {
                'Patient': filename.split("K")[0],
                'ASR/ICA': filename.split("K")[1].split("_")[1],
                'Artefact Settings': filename.split("Bandpass")[0].split("_")[-1],
                'Lowpass': filename.split("Bandpass=")[1].split("-")[0],
                'Highpass': filename.split("_MinRR")[0].split("-")[-1],
                'Minimum RR': filename.split("MinRR=")[1].split("_")[0],
                'Max Epoch Amplitude': filename.split("maxEpochAmp=")[1].split("_")[0]
            }

        FileMetaData = extract_info(epochFile)
        with HiddenPrints():
            try:
                epochs = mne.epochs.read_epochs(epochFile, preload=True)
                sfreq = epochs.info["sfreq"]
            except AttributeError as e:
                return "No Epochs"

        all_results = HEP_SubProcess(epochs, FileMetaData, sfreq, baseStarts, baseEnds, windowStarts, windowEnds, repeats,
                                     fraction_samples)

        with open(sub_result_save_name, 'w', newline='') as csvfile:
            fieldnames = all_results[0].keys()
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_results)

        return sub_result_save_name


def HEP_SubProcess(ecg_epochs_original, FileMetaData, sfreq, baseStarts, baseEnds, windowStarts, windowEnds, repeats,
                   fraction_samples, EvokedFolder = r"path_to\ModalEvoked"):
    all_results = []

    if True:
        #with HiddenPrints():
        for repeat in range(repeats):
            ecg_epochs = ecg_epochs_original.copy()
            print("lists:", baseStarts, baseEnds, windowStarts, windowEnds)

            if fraction_samples < 1:
                n_epochs = len(ecg_epochs) - 1
                N = int(n_epochs - fraction_samples * n_epochs)
                numbers = set()
                while len(numbers) < N:
                    numbers.add(random.randint(1, n_epochs))
                droppers = list(numbers)
                ecg_epochs.drop(droppers)
            else:
                n_epochs = len(ecg_epochs)
            for baseStart in baseStarts:
                for baseEnd in baseEnds:
                    if baseEnd > baseStart:
                        print("Computing HEPs for baseStart = ", baseStart, " and baseEnd = ", baseEnd)
                        ecg_evoked = ecg_epochs.apply_baseline((baseStart, baseEnd)).average()
                        os.chdir(EvokedFolder)
                        chars_to_remove = "/\\ "
                        translation_table = str.maketrans('', '', chars_to_remove)
                        evokedSaveName = "".join([f"{k.translate(translation_table)}={v.translate(translation_table)}" for k, v in FileMetaData.items()]) + "baseStart" + "=" + str(baseStart) + "baseEnd" + "=" + str(baseEnd) + "-ave.fif"
                        ecg_evoked.save(evokedSaveName, overwrite=True)
                        for windowStart in windowStarts:
                            for windowEnd in windowEnds:
                                if windowEnd > windowStart:
                                    print("Computing HEPs for windowStart = ", windowStart, " and windowEnd = ", windowEnd)
                                    avg_ecg_epochs_mean_HEP = ecg_evoked.copy().crop(tmin=windowStart, tmax=windowEnd)
                                    MeanHEP = np.mean(avg_ecg_epochs_mean_HEP.data, axis=1)
                                    SDHEP = np.std(avg_ecg_epochs_mean_HEP.data, axis=1)

                                    ch_names = ecg_epochs.ch_names

                                    HEPdict = {}

                                    for i, lead in enumerate(ch_names):
                                        HEPdict[lead] = MeanHEP[i-1]
                                        HEPdict[lead + "SD"] = SDHEP[i-1]

                                    result_dict = {
                                        **FileMetaData,
                                        'Baseline Start': baseStart,
                                        'Baseline End': baseEnd,
                                        'Start of HEP window': windowStart,
                                        'End of HEP window': windowEnd,
                                        'Repeat': repeat,
                                        'Number of epochs': n_epochs,
                                        **HEPdict
                                    }

                                    all_results.append(result_dict)

        return all_results


def BatchHEPComputation(epochDir, fraction_samples=0.5, repeats=4, baseStarts=[-0.5], baseEnds=[-0.1],
                        windowStarts=[0.455], windowEnds=[0.595], resultsfile="HEP_results.csv"):
    if type(epochDir) == list:
        epochFiles = epochDir

    else:
        epochFiles = getDirFIFs(epochDir, type="_epo")


    fileList = []
    print(baseStarts, baseEnds, windowStarts, windowEnds)

    for epochFile in tqdm(epochFiles, desc="Processing files", unit="file"):
        try:
            OutputFileName = HEP_process(epochFile, baseStarts, baseEnds, windowStarts, windowEnds,
                                       repeats, fraction_samples)
        except RuntimeError as e:
            print(e)
            OutputFileName = "No Epochs"
        if OutputFileName != "No Epochs":
            fileList.append(OutputFileName)

    if True:
        fileDataframe = pd.DataFrame(fileList)
        fileDataframe.to_csv(resultsfile)

    return fileList


def HEP_Computation(ecg_epochs, fraction_samples=1.0, baseStart=-0.5, baseEnd=-0.1, windowStart=0.455, windowEnd=0.595):
    HEPdict = {}

    # 1 - Drop objects if fraction_samples < 1
    if fraction_samples < 1:
        M = len(ecg_epochs) - 1
        N = int(M - fraction_samples * M)
        numbers = set()
        while len(numbers) < N:
            numbers.add(random.randint(1, M))
        droppers = list(numbers)
        ecg_epochs.drop(droppers)

    # 2 - Generate Evoked object
    ecg_evoked = ecg_epochs.apply_baseline((baseStart, baseEnd)).average()
    avg_ecg_epochs_mean_HEP = ecg_evoked.copy().crop(tmin=windowStart, tmax=windowEnd)
    MeanHEP = avg_ecg_epochs_mean_HEP.data.mean(axis=1)

    ch_names = ecg_epochs.info['ch_names']

    for i, lead in enumerate(ch_names):
        HEPdict[lead] = MeanHEP[i]

    # 3 - perform further statistics
    IndividualEvokedWindows = [y.apply_baseline((baseStart, baseEnd)) for y in list(ecg_epochs.iter_evoked(copy=True))]

    IndividualMeanHEPs = [x.crop(tmin=windowStart, tmax=windowEnd).data.mean(axis=1) for x in IndividualEvokedWindows]

    SDs = [hep.mean() for hep in IndividualMeanHEPs]
    Means = [hep.std() for hep in IndividualMeanHEPs]

    for i, lead in enumerate(ch_names):
        HEPdict[lead + "SD"] = SDs[i]
        HEPdict[lead + "Mean"] = Means[i]

    return HEPdict


if __name__ == "__main__":
    l = BatchHEPComputation([r"E:\EEGs\HEP-Params\ModalEpochs\00010417_s001_t000K_ICA_CFA_removedBandpass=15-0.1_MinRR=0.7_maxEpochAmp=100_epo.fif"],
                        fraction_samples=1, repeats=1, baseStarts=[-0.5], baseEnds=[-0.1],
                        windowStarts=[0.1, 0.2, 0.3, 0.455], windowEnds=[0.595], resultsfile="E:\EEGs\HEP-Params\ModalEvoked\Test_HEP_results.csv")
    print(l)

