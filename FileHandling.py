"""
Functions to load files, either one by one or according to a csv of data.

Should be able to handle TUEG files and Micromed EDFs (and sort out all the problems with them;
there is an error in the format of EDFs exported by Micromed - it is not compliant with EDF standards)

Throughout this file, there will be problems when switching between users and using the default directories.
When running, consider which directory you are using.
"""

import os
from tkinter import *
from tkinter import filedialog as FD
import mne
import re
import pandas as pd
import csv
from EEG_ICA import Auto_ICA
import numpy as np
import asrpy

def get_info_with_mne(file_path):
    """ read info from the edf file without loading the data. loading data is done in multiprocessing since it takes
    some time. getting info is done before because some files had corrupted headers or weird sampling frequencies
    that caused the multiprocessing workers to crash. therefore get and check e.g. sampling frequency and duration
    beforehand
    :param file_path: path of the recording file
    :return: file name, sampling frequency, number of samples, number of signals, signal names, duration of the rec
    """
    try:
        raw = mne.io.read_raw_edf(file_path, verbose='error')
    except ValueError:
        return None, None, None, None, None, None

    # some recordings have a very weird sampling frequency. check twice before skipping the file
    sampling_frequency = int(raw.info['sfreq'])
    if sampling_frequency < 10:
        sampling_frequency = 1 / (raw.times[1] - raw.times[0])
        if sampling_frequency < 10:
            return None, sampling_frequency, None, None, None, None

    n_samples = raw.n_times
    signal_names = raw.ch_names
    n_signals = len(signal_names)
    # some weird sampling frequencies are at 1 hz or below, which results in division by zero
    duration = n_samples / max(sampling_frequency, 1)

    return raw, sampling_frequency, n_samples, n_signals, signal_names, duration

def eegFixer(file):
    with open(file, 'r+', encoding="UTF-8", errors='ignore') as f:
        f.seek(0)
        string = "MCH-0234567 F 02-MAY-1951 Haagse_Harry"
        f.seek(8)
        f.write(string)
        f.seek(88)
        recstring = "Startdate 02-MAR-2002 PSG-1234/2002 NN Telemetry03"
        f.write(recstring)
        f.seek(8+80+80)
        datestring = "02.03.02"
        f.write(datestring)

def rename_eeg_channels(ch_name):
    """Rename TUH channels and ignore non-EEG and custom channels.
    Rules:
    - drop '-REF' from the end of the name (if present)
    - drop 'EEG ' from the start of the name (if present)
    - 'Z' should always be lowercase.
    - 'P' following a 'F' should be lowercase.
    - 'EKG1' not 'ECG1'
    """

    match = re.findall(r"(?<= )(.*)(?=-)", ch_name)
    if len(match) == 1:
        out = match[0]
        out = out.replace('FP', 'Fp').replace('Z', 'z')  # Apply rules
    else:
        out = ch_name
    if out.startswith("ECG"):
        out = "EKG1"
    if out.startswith("EKG"):
        out = "EKG1"
    if out in ['POL T1', 'POL T2', 'POL X1', 'POL X2', 'POL X3', 'POL X4', 'POL X5', 'POL X6', 'POL X7', 'POL SpO2', 'POL EtCO2', 'POL DC03', 'POL DC04', 'POL DC05', 'POL DC06', 'POL Pulse', 'POL CO2Wave']:
        out = out
    return out


def mappingMaker(rawChannelNames):
    # function that builds a mapping dict
    global mapping
    mapping = {}
    notFound = []
    for chan in rawChannelNames:
        if chan in mapping.keys():
            pass
        else:
            mapping[chan] = rename_eeg_channels(chan)

def OpenEEG(fname, start=None, stop=None, dir=None):

    if dir != None:
        os.chdir(dir)

    #load info
    raw, sfreq, n_samples, n_channels, chan_names, n_sec = get_info_with_mne(
        fname)

    #irrespective of the original units, mne will always convert amplitudes to Volts (not uV)

    raw.load_data()

    montage = mne.channels.make_standard_montage('standard_1020')
    mappingMaker(raw.ch_names)
    submapping = {k:v for k,v in mapping.items() if k in raw.ch_names}
    raw.rename_channels(submapping)

    raw.set_channel_types({"EKG1": "ecg"}, verbose=None)

    rawCopy = raw.copy()
    rawCopy.pick_channels(["EKG1"])
    availableChans = montage.ch_names

    exclude = [x for x in raw.ch_names if x not in availableChans]
    if len(exclude) > 0:
        raw.drop_channels(exclude)  # removing all channels not in montage

    raw.set_montage(montage)

    raw.add_channels([rawCopy])  # re-adds ECG lead to bypass the montage

    chan_names = raw.ch_names #updates this with new info
    n_channels = len(chan_names)

    if type(start) == list and type(stop) == list:
        segments = []
        annotStarts = []
        annotDurs = []
        onsetsum = 0
        for starti, stopi in zip(start, stop):

            segments.append(raw.copy().crop(starti, stopi))

            annotStarts.append(onsetsum)
            annotDurs.append((stopi-starti))
            onsetsum+=(stopi-starti)

        if len(segments) > 1:
            cleanRaw = mne.concatenate_raws(segments, on_mismatch="warn")
        else:
            cleanRaw = segments[0]

        # Add annotations
        annotations = mne.Annotations(onset=annotStarts, duration=annotDurs,
                                     description=['Clean_section'] * len(start))

        cleanRaw.set_annotations(annotations)

    else:
        cleanRaw = raw

    return cleanRaw, fname, sfreq, n_samples, n_channels, chan_names, n_sec

def ChooseEEG():
    """
    Function for selecting an EDF using dialog
    Suggest using tkinter filedialog
    :return: eeg object
    """
    root = Tk()
    root.withdraw()

    file = FD.askopenfile(parent=root).name
    #file is the whole path

    global sampling_freq

    #try to open the file directly, if it fails then use eegFixer

    raw, fname, sfreq, n_samples, n_channels, chan_names, n_sec = OpenEEG(file)

    return raw, fname, sfreq, n_samples, n_channels, chan_names, n_sec

def Trim_From_File(EEG_CSV, suffix="K", saveDir = r"C:\Users\Admin\OneDrive - NHS\EEG-Share\TUAB\Clipped", save = True):
    EEG_DF = LoadEEGsfromList(EEG_CSV) #this includes trimming

    #save the files
    if save:
        EEG_DF["ExportFileName"] = EEG_DF["fname"].apply(lambda x: saveDir + "/" + x.split(".")[0].split("/")[-1] + suffix + ".edf")
        EEG_DF["saved"] = EEG_DF.apply(lambda x: SaveRaw(raw = x["raw"], fname = x["ExportFileName"]), axis = 1)

    return EEG_DF

def SaveRaw(raw, fname):
    try:
        mne.export.export_raw(fname, raw, fmt="edf", overwrite=False)
        return True
    except FileExistsError as e:
        print(e)
        return False
    except RuntimeError as e:
        print(e)
        print("Attempting to change date")
        raw.info["meas_date"] = "2000-12-30 00:00:00+00:00"
        mne.export.export_raw(fname, raw, fmt="edf", overwrite=False)

def LoadEEGsfromList(EEG_CSV):
    """
    load EEGs from a csv, and clip them (also copes with xls and xlsx)
    NOTE: Due to the csv having locations on sharepoint, and this running on local oneDrive, there's a DirAdjust function
    :param EEG_CSV: csv containing file paths
    :return: List object containing the EEG objects
    """
    if EEG_CSV.endswith(".csv"):
        EEGPD = pd.read_csv(EEG_CSV, encoding="cp1252")
    elif EEG_CSV.endswith(".xls") or EEG_CSV.endswith(".xlsx"):
        EEGPD = pd.read_excel(EEG_CSV)
    EEGPD.dropna(how='all', inplace = True)
    EEGPD["fname"] = EEGPD["dir"].apply(DirAdjust)
    EEGPD["start"] = EEGPD["start"].apply(secondConvert)
    EEGPD["stop"] = EEGPD["stop"].apply(secondConvert)
    EEGPD = EEGPD[EEGPD["start"]<EEGPD["stop"]]
    EEG_DF_2 = (EEGPD.groupby('fname').agg(lambda x: list(x)))[["start","stop"]].reset_index()

    EEG_DF_2['raw'], EEG_DF_2['fname'], _, _, _, _, _ = zip(*EEG_DF_2.apply(lambda x: OpenEEG(x['fname'], x['start'], x['stop']), axis=1))


    return EEG_DF_2




def secondConvert(string):
    segs = string.split(":")
    if len(segs) == 1:
        return int(segs[0])
    elif len(segs) == 2:
        return int(segs[0])*60 + int(segs[1])
    elif len(segs) == 3:
        return int(segs[0])*60*60 + int(segs[1])*60 + int(segs[2])

def DirAdjust(string, user = "Admin"):
    """
    Function to adjust the file locations from sharepoint to my local onedrive - user specific
    :param string: Directory as saved in csv
    :return: Directory as it is in my onedrive
    """
    if string.startswith(r"https://nhs-my.sharepoint.com"):
        end = string.split("EEG-Share")[1]
        userstring = r"C:/Users/" + user + r"/OneDrive - NHS/EEG-Share/"
        return userstring + end
    else:
        return string

def getDirEDFs(edfDir):
    edfs=[]
    for file in os.scandir(edfDir):
        if file.name.endswith(".edf"):
            edfs.append(file.path)
            eegFixer(file.path)
        elif file.is_dir():
            subd_files = getDirEDFs(file.path)
            edfs = edfs + subd_files

    return edfs

def getDirFIFs(fifDir):
    fifs=[]
    for file in os.scandir(fifDir):
        if file.name.endswith(".fif"):
            fifs.append(file.path)
            eegFixer(file.path)
        elif file.is_dir():
            subd_files = getDirFIFs(file.path)
            fifs = fifs + subd_files

    return fifs


def convert_excel_to_csv(input_file, output_file):
    # Read the Excel file
    df = pd.read_excel(input_file)

    # Write the DataFrame to a CSV file
    df.to_csv(output_file, index=False)

def BulkSortEEGs(edfDir=None, saveDir=None, ASRThresholds = [20], removeCFA = False, doASR=False, artefactCorrect=True):
    """
    Loads edfs from a directory, and then applies ica twice, saving once with CFA removed, and once without CFA being removed.
    :param edfDir: dir of EEGs
    :param saveDir: dir to save EEGs to
    :return: None
    """

    edfs = getDirEDFs(edfDir)
    if saveDir == None:
        saveDir = edfDir

    for index, e in enumerate(edfs):
        # List files in the folder
        files_in_folder = os.listdir(saveDir)

        # Check if any file begins with the specified substring
        matching_files = [file for file in files_in_folder if file.startswith(e.split("\\")[-1].split(".")[0])]

        #if matching_files:
        #    print(f"Already processed {e}")
        #    continue

        print(f"Processing number {index}")
        # load info
        try:
            raw = mne.io.read_raw_edf(e, preload=True)
        except ValueError as VE:
            print(e)
            print(VE)
            continue

        montage = mne.channels.make_standard_montage('standard_1020')

        # function that builds a mapping dict
        mapping = {}
        for chan in raw.ch_names:
            if chan in mapping.keys():
                pass
            else:
                mapping[chan] = rename_eeg_channels(chan)

        submapping = {k: v for k, v in mapping.items() if k in raw.ch_names}
        raw.rename_channels(submapping)

        raw.set_channel_types({"EKG1": "ecg"}, verbose=None)

        ecg_data = raw.get_data(picks=['EKG1'])
        ecg_info = mne.create_info(['EKG1'], raw.info['sfreq'], ch_types=['ecg'])
        ecg_raw = mne.io.RawArray(ecg_data, ecg_info)

        availableChans = montage.ch_names
        exclude = [x for x in raw.ch_names if x not in availableChans]

        if len(exclude) > 0:
            raw.drop_channels(exclude)  # removing all channels not in montage

        raw.set_montage(montage)

        if not artefactCorrect:
            os.chdir(saveDir)
            baseName = e.split("\\")[-1].split(".")[0] + "_"
            uncorrectedName = baseName + "Uncorrected_eeg.fif"

            for ch_name, ch_data in zip(ecg_raw.ch_names, ecg_raw.get_data()):
                raw.add_channels([mne.io.RawArray(ch_data[np.newaxis, :], ecg_raw.info)],
                                 force_update_info=True)

            raw.save(uncorrectedName, overwrite=True)
            print(f"Saved {uncorrectedName}")
            continue

        #cleanAnnots = [annot for annot in raw.annotations if annot["description"] == "Clean_section"]

        if doASR:
            asrDict = {}
            for ASR_level in ASRThresholds:
                try:
                    asrDict[ASR_level] = asrpy.ASR(sfreq=raw.info["sfreq"], cutoff=ASR_level)
                    asrDict[ASR_level].fit(raw)
                except ValueError as v:
                    print(e)
                    print(v)
                    continue

        # Apply ICA
        try:
            EEG_with_CFA_removed, EEG_without_CFA_removed = Auto_ICA(raw, n_comp=(raw.info["nchan"] - 1))
        except IndexError as j:
            print(j)
            print(e)
            print(raw.ch_names)
            print(raw.n_times)
            continue

        baseName = e.split("\\")[-1].split(".")[0] + "_"
        CF_removed_name = baseName + "ICA_CFA_removed_eeg.fif"
        CF_not_removed_name = baseName + "ICA_CFA_not_removed_eeg.fif"

        if removeCFA:
            # WITH CFA REMOVED
            # Create a new Raw object
            EEG_with_CFA_removed = mne.io.RawArray(EEG_with_CFA_removed.get_data(), EEG_with_CFA_removed.info)

            # Add channels from ecg to raw_combined
            for ch_name, ch_data in zip(ecg_raw.ch_names, ecg_raw.get_data()):
                EEG_with_CFA_removed.add_channels([mne.io.RawArray(ch_data[np.newaxis, :], ecg_raw.info)],
                                                  force_update_info=True)

        # WITHOUT CFA REMOVED
        # Create a new Raw object
        EEG_without_CFA_removed = mne.io.RawArray(EEG_without_CFA_removed.get_data(),
                                                  EEG_without_CFA_removed.info)

        # Add channels from ecg to raw_combined
        for ch_name, ch_data in zip(ecg_raw.ch_names, ecg_raw.get_data()):
            EEG_without_CFA_removed.add_channels([mne.io.RawArray(ch_data[np.newaxis, :], ecg_raw.info)],
                                                 force_update_info=True)

        os.chdir(saveDir)

        if removeCFA:
            EEG_with_CFA_removed.save(CF_removed_name, overwrite=True)
        EEG_without_CFA_removed.save(CF_not_removed_name, overwrite=True)

        for ch_name, ch_data in zip(ecg_raw.ch_names, ecg_raw.get_data()):
            raw.add_channels([mne.io.RawArray(ch_data[np.newaxis, :], ecg_raw.info)],
                                     force_update_info=True)

        if doASR:
            # Apply ASR
            for ASR_level in ASRThresholds:
                try:
                    ASRraw = asrDict[ASR_level].transform(raw)
                    ASR_savename = baseName + "ASR_" + str(ASR_level) + "_eeg.fif"
                    ASRraw.save(ASR_savename, overwrite=True)
                    print(f"Saved {ASR_savename}")
                except ValueError as jkl:
                    print(jkl)
                    print(e)
                    continue

        uncorrectedName = baseName + "Uncorrected_eeg.fif"

        raw.save(uncorrectedName, overwrite=True)
        print(f"Saved {uncorrectedName}")


    print("COMPLETED PROCESSING FILES")

def getBases(fifDir=r"path_to\Processed"):
    fifs = getDirFIFs(fifDir=fifDir)
    starts = ([s.split("ICA")[0] for s in fifs])
    with open('bases.csv', 'w', newline='') as file:
        writer = csv.writer(file)

        # Iterate over each item in the list
        for item in starts:
            # Write the item as a single-row CSV
            writer.writerow([item])
    return starts

def getFifs(fifDir=r"path_to\Processed"):
    fifs = getDirFIFs(fifDir=fifDir)
    with open('fifs.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        # Iterate over each item in the list
        for item in fifs:
            # Write the item as a single-row CSV
            writer.writerow([item])
    return fifs

if __name__ == "__main__":
    #Trim_From_File(r"path_to\EEG_clean_sections.csv")
    BulkSortEEGs(edfDir=r"path_to\Clipped", saveDir=r"path_to\Processed", ASRThresholds=[5,10,20])
