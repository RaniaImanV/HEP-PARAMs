# HEP-Tech

HEP-Technique is a repository for parametric exploration of HEP derivation

# Outline

## Variables investigated:

1. Low-pass and high-pass cut-off frequencies in Hz
2. epoch timeframes in ms
3. epoch amplitude rejection in uV
4. HEP window in ms
5. minimum RR interval in ms
6. ICA vs ASR
7. ASR threshold
8. electrodes used 

Default values (lists of different values can be entered into the functions as parameters in lists)
Variables that can be modified, by function:
    1. BulkSortEEGs: ASRThresholds (default 20)
    2. BatchEpochGeneration: maxEpochAmps (default 100), MinRRs (default 0.88)
    3. BatchHEPComputation: baseStarts (default -0.5, baseEnds= (default -0.1), windowStarts (default 0.455), windowEnds (default 0.595)
