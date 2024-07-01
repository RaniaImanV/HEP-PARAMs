"""
This goes from a csv of configs and a csv of EEG clips to two csvs of results.
Review directories!
"""
import os
from FileHandling import BulkSortEEGs
import HEP_computation as HC
import FileHandling as FH
import pandas as pd
from tqdm import tqdm
import csv


def IterateFromConfigs(ClippingCSV=r"path_to\EEG_clean_sections.csv",
                       ClippedDir=r"path_to\Clipped",
                       ProcessedDir=r"path_to\Processed",
                       EpochDir=r"path_to\Epochs",
                       EpochDataFile=r"path_to\NewEpochData.csv",
                       ClipEEG=False,
                       preProcess=True,
                       artefactCorrect=True,
                       generateEpochs=False,
                       computeHEPData=False,
                       repeats=1,
                       resultsFile=r"path_to\HEP_results_data.csv",
                       Test=False):
    """
        Variables we can modify, by function:
        1. BulkSortEEGs: ASRThresholds (default 20)
        2. BatchEpochGeneration: maxEpochAmps (default 100), MinRRs (default 0.88)
        3. BatchHEPComputation: baseStarts=[-0.5], baseEnds=[-0.1], windowStarts=[0.455], windowEnds=[0.595]
    """

    if Test:
        ClippedDir = r"C:\Users\rohan\PycharmProjects\HEP-Params\testEDFs\clipped"
        EpochDir = r"C:\Users\rohan\PycharmProjects\HEP-Params\testEDFs\epochs"
        ProcessedDir = r"C:\Users\rohan\PycharmProjects\HEP-Params\testEDFs\preprocessed"
        resultsFile = r"C:\Users\rohan\PycharmProjects\HEP-Params\testEDFs\test_HEP_Results_File.csv"
        EpochDataFile = r"C:\Users\rohan\PycharmProjects\HEP-Params\testEDFs\test_EpochData.csv"
        ConfigDict = {
            "ASRThresholds": [5, 10, 20],
            "maxEpochAmps": [80, 100, 150],
            "MinRRs": [0.7, 0.8, 1],
            "lowPasses": [30, 50],
            "highPasses": [0.1, 0.05],
            "maxEpochAmps": [0.050, 0.100],
            "windowStarts": [0.100, 0.150],
            "windowEnds": [0.150, 0.200],
            "baseStarts": [-0.200, -0.100],
            "baseEnds": [-0.050, 0],
            "MinRRs": [0.7, 1]
        }
        configVariables = list(ConfigDict.keys())
    else:
        ConfigDict = {
            "ASRThresholds": [5, 10, 20],
            "maxEpochAmps": [80, 100, 150],
            "MinRRs": [0.7, 0.8, 1],
            "lowPasses": [30, 50, 70, 100],
            "highPasses": [0.1, 0.05, 0.01],
            "maxEpochAmps": [0.050, 0.100, 0.150],
            "windowStarts": [0.100, 0.150, 0.200, 0.250, 0.300, 0.350, 0.400, 0.450, 0.500],
            "windowEnds": [0.150, 0.200, 0.250, 0.300, 0.350, 0.400, 0.500, 0.550, 0.600],
            "baseStarts": [-0.200, -0.100, -0.075],
            "baseEnds": [-0.050, 0],
            "MinRRs": [0.7, 0.75, 0.8, 1]

        }
        configVariables = list(ConfigDict.keys())

    if ClipEEG:
        # Clip EEGs
        FH.Trim_From_File(EEG_CSV=ClippingCSV, saveDir=ClippedDir)
        print("EEGs clipped")

    if preProcess:
        # Preprocess EEGs
        FH.BulkSortEEGs(edfDir=ClippedDir, saveDir=ProcessedDir, ASRThresholds=ConfigDict["ASRThresholds"],
                        artefactCorrect=artefactCorrect)
        print("EEGS preprocessed")

    if generateEpochs:
        # Generate Epochs
        HC.BatchEpochGeneration(fifDir=ProcessedDir, epochDir=EpochDir, maxEpochAmps=ConfigDict["maxEpochAmps"], MinRRs=
        ConfigDict["MinRRs"], highPasses=ConfigDict["highPasses"], lowPasses=ConfigDict["lowPasses"],
                                ResultsFile=EpochDataFile)

    # Compute HEP data
    if computeHEPData:
        HC.BatchHEPComputation(epochDir=EpochDir, fraction_samples=0.5, repeats=repeats,
                               baseStarts=ConfigDict["baseStarts"],
                               baseEnds=ConfigDict["baseEnds"], windowStarts=ConfigDict["windowStarts"],
                               windowEnds=ConfigDict["windowEnds"], resultsfile=resultsFile)


def IterateFromModalValues(ProcessedDir=r"path_to\Processed",
                           ModalEpochDir=r"path_to\ModalEpochs",
                           EpochDataFile=r"path_to\NewEpochData.csv",
                           HEPResultsDir=r"path_to\ModalHEPResults",
                           EpochResultsFile=r"path_to\ModalEpochResults.csv",
                           BaseEpochResultsFile = r"path_to\EpochResults",
                           GenerateEpochs=True, computeHEPData=False):
    ModalValueDict = {
        "lowPasses": [30],
        "highPasses": [0.1],
        "maxEpochAmps": [100],
        "windowStarts": [0.200],
        "windowEnds": [0.600],
        "baseStarts": [-0.200],
        "baseEnds": [0],
        "minRRs": [0.88]}

    VariationDict = {
        "lowPasses": [15, 20, 24, 25, 30, 35, 40, 45, 50, 62.5, 80, 100],
        "highPasses": [0.01, 0.05, 0.1, 0.3, 0.5, 0.7, 1],
        "maxEpochAmps": [30, 50, 70, 75, 80, 100, 120, 150, 250, 300],
        "windowStarts": [0.050, 0.100, 0.150, 0.200, 0.250, 0.300, 0.350, 0.400, 0.450, 0.500],
        "windowEnds": [0.150, 0.200, 0.250, 0.300, 0.350, 0.400, 0.450, 0.500, 0.550, 0.600, 0.650, 0.700],
        "baseStarts": [-0.3, -0.2, -0.15, -0.1, -0.075, -0.05],
        "baseEnds": [-0.2, -0.05, 0],
        "minRRs": [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.88, 0.9]}

    keywords = ['ASR_5', 'ASR_10', 'ASR_20', 'ICA_CFA_removed', 'Uncorrected']

    processedfifs = [fif.path for fif in os.scandir(ProcessedDir) if any(fif.name.endswith(f"{keyword}_eeg.fif") for keyword in keywords)]
    print("number of epoch files will be:", len(processedfifs) * 29)

    # Now we iterate over variables, changing the selected one only

    EpochVariables = ["lowPasses", "highPasses", "maxEpochAmps", "minRRs"]
    HEPVariables = ["windowStarts", "windowEnds", "baseStarts", "baseEnds"]

    for variable in EpochVariables:
        print(f"Iterating with different values of {variable}")
        # make a dict
        parameters = {x: ModalValueDict[x] if x != variable else VariationDict[x] for x in EpochVariables}

        if GenerateEpochs:
            # Generate Epochs
            HC.BatchEpochGeneration(fifDir=processedfifs, epochDir=ModalEpochDir,
                                    maxEpochAmps=parameters["maxEpochAmps"], MinRRs=parameters["minRRs"],
                                    highPasses=parameters["highPasses"], lowPasses=parameters["lowPasses"],
                                    ResultsFile=f"{BaseEpochResultsFile}_{variable}.csv")

        lp = parameters["lowPasses"]
        hp = parameters["highPasses"]
        ea = parameters["maxEpochAmps"]
        mr = parameters["minRRs"]

        if variable == "lowPasses":
            epochSelectionSubString = f"-{hp[0]}_MinRR={mr[0]}_maxEpochAmp={ea[0]}"

            modalEpochFiles = [file.path for file in os.scandir(ModalEpochDir) if
                               epochSelectionSubString in file.name]
        elif variable == "highPasses":
            epochSelectionSubString = f"Bandpass={lp[0]}-..._MinRR={mr[0]}_maxEpochAmp={ea[0]}_epo.fif"
            lpstring = f"Bandpass={lp[0]}"
            eastring = f"MinRR={mr[0]}_maxEpochAmp={ea[0]}"

            modalEpochFiles = [file.path for file in os.scandir(ModalEpochDir) if
                               lpstring in file.name and eastring in file.name]

        elif variable == "maxEpochAmps":
            epochSelectionSubString = f"Bandpass={lp[0]}-{hp[0]}_MinRR={mr[0]}"

            modalEpochFiles = [file.path for file in os.scandir(ModalEpochDir) if
                               epochSelectionSubString in file.name]

        elif variable == "minRRs":
            epochSelectionSubString = f"Bandpass={lp[0]}-{hp[0]}..._maxEpochAmp={ea[0]}_epo.fif"
            bpstring = f"Bandpass={lp[0]}-{hp[0]}"
            eastring = f"maxEpochAmp={ea[0]}_epo.fif"

            modalEpochFiles = [file.path for file in os.scandir(ModalEpochDir) if
                               bpstring in file.name and eastring in file.name]

        print(f"Selecting files with {epochSelectionSubString}")
        if computeHEPData:
            HC.BatchHEPComputation(epochDir=modalEpochFiles, fraction_samples=1, repeats=1,
                                   baseStarts=ModalValueDict["baseStarts"],
                                   baseEnds=ModalValueDict["baseEnds"], windowStarts=ModalValueDict["windowStarts"],
                                   windowEnds=ModalValueDict["windowEnds"], resultsfile=f"HEP_results_{variable}.csv")

    if not computeHEPData:
        return
    modalLowPass = ModalValueDict['lowPasses']
    modalHighPass = ModalValueDict["highPasses"]
    modalEpochAmp = ModalValueDict["maxEpochAmps"]
    modalMinRR = ModalValueDict["minRRs"]

    epochSelectionSubString = f"Bandpass={modalLowPass[0]}-{modalHighPass[0]}_MinRR={modalMinRR[0]}_maxEpochAmp={modalEpochAmp[0]}_epo.fif"
    print(epochSelectionSubString)
    modalEpochFiles = [file.path for file in os.scandir(ModalEpochDir) if file.name.endswith(epochSelectionSubString)]

    print(f"Selecting files with {epochSelectionSubString}")

    for variable in tqdm(HEPVariables):
        print(f"Iterating with different values of {variable}")
        parameters = {x: ModalValueDict[x] if x != variable else VariationDict[x] for x in HEPVariables}
        print(parameters)

        HC.BatchHEPComputation(epochDir=modalEpochFiles, fraction_samples=1, repeats=1,
                               baseStarts=parameters["baseStarts"],
                               baseEnds=parameters["baseEnds"], windowStarts=parameters["windowStarts"],
                               windowEnds=parameters["windowEnds"], resultsfile=f"HEP_results_{variable}.csv")

    HEP_results_folder = HEPResultsDir

    # Specify the folder containing CSV files and the merged CSV path
    folder_path = HEP_results_folder
    merged_csv_path = r'C:\HEP_params\ModalIterationHEPData.csv'

    # List to store data rows
    data_rows = []

    # Loop through each CSV file in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith('.csv'):
            file_path = os.path.join(folder_path, filename)

            with open(file_path, 'r') as csv_file:
                csv_reader = csv.reader(csv_file)

                # Skip the header row
                header_row=next(csv_reader)

                # Process each data row
                for data_row in csv_reader:
                    data_rows.append(data_row)

    # Write the data to the merged CSV file
    with open(merged_csv_path, 'w', newline='') as merged_csv_file:
        csv_writer = csv.writer(merged_csv_file)

        # Write the header row
        csv_writer.writerow(header_row)

        # Write all data rows
        csv_writer.writerows(data_rows)

    print(f'Merged CSV file saved as "{merged_csv_path}"')


if __name__ == "__main__":
    # Replace the edfDir and the saveDir with ClippedDir and ProcessedDir locations
    ClippedDir = r"path_to\Clipped"
    ProcessedDir = r"path_to\processed"
    ModalEpochDir = r"path_to\ModalEpochs2"
    EpochDataFile = r"path_to\EpochData2.csv"
    HEPResultsDir = r"path_to\ModalHEPResults2"
    EpochResultsFile = r"path_to\ModalEpochResults2.csv"
    BaseEpochResultsFile = r"path_to\EpochResults"

    #BulkSortEEGs(edfDir=ClippedDir, saveDir=ProcessedDir, artefactCorrect=False, doASR=False)

    #files = os.listdir(r"path_to\Processed")
    #for file in files:
    #    if "Uncorrected" in file or ("CFA" in file and "not_removed" in file):
    #       continue
    #    os.system(f'copy "path_to\Processed\{file}" "path_to\processed"')

    IterateFromModalValues(ProcessedDir=ProcessedDir,
                           ModalEpochDir=ModalEpochDir,
                           EpochDataFile=EpochDataFile,
                           HEPResultsDir=HEPResultsDir,
                           EpochResultsFile=EpochResultsFile,
                           BaseEpochResultsFile=BaseEpochResultsFile,
                           GenerateEpochs=True)
