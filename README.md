# HEP-PARAMs

HEP-PARAMs is a repository for parametric exploration of HEP derivation.
Please note that the path for each file needs to be replaced with the appropriate path before running the code.

# Outline

## Variables investigated:

1. Low-pass and high-pass cut-off frequencies in Hz
2. epoch timeframes in ms
3. epoch amplitude rejection in uV
4. HEP window in ms
5. minimum RR interval in ms
6. ICA vs ASR
7. ASR threshold
8. ICA with or without CFA removed 
9. electrodes used 

Default values (lists of different values can be entered into the functions as parameters in lists)
Variables that can be modified, by function:
    1. BulkSortEEGs: ASRThresholds (default 20)
    2. BatchEpochGeneration: maxEpochAmps (default 100), MinRRs (default 0.88)
    3. BatchHEPComputation:  maxEpochAmps=[100], MinRRs=[0.88], highPasses=[0.05], lowPasses=[70]
    


# Workflow

### Clean sections from EEG edf files.
Selecting clean sections allows us to avoid obvious artefacts. Artefact correction is a "fine tuning" of this. Use CSV of edfs: format: FileLocation, start of clean section, end of clean section
This is loaded by HEP-Params, and the edf is loaded as a raw object, which is trimmed to the clean section. All clean sections from a given edf are concatenated. These are then exported as a new EDF with a suffix "K" (for klean).

### Artefact correction
Method one: ICA Fully automated. Run once, but then two reconstructions are performed, with and without CFA. (These are saved as two separate .fifs)
Method two: ASR This is done automatically and runs unsupervised. This is written parametrically, so along with other HEP computation variables the threshold can be pulled from a config csv.

### Epoch derivation
First, R waves are identified in the "EKG1" lead. Then, epochs are created - the epochs are saved as an "-epo.fif" file. The filename also contains details about the epoch processing:

* ICA vs ASR
* if ASR then what threshold
* if ICA then whether or not CFA is removed
* Minimum RR interval
* Maximum amplitude in an epoch

### HEP computation
The epoch object is then processed to generate the HEP values, and an "Evoked" object is saved. Again the filename will contain details about the processing. Here there are two modes:

#### All permutations

iterateFromConfigs()
Every combination of variables for The HEP computation returns a csv containing all the data, including repeats of the same parameters with the randomised K-folding applied.

#### Use modal values for all but one variable

IterateFromModalValues()
Using provided modal values (or whatever means chosen to select the "usual" value for each variable), and a list of values, it will iterate changing only one variable at a time. This assumes that the chosen modal values are sensible.
