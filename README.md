# The Readiness Potential reflects internal source of actions, not decision uncertainty

`Eoin Travers & Patrick Haggard`

 `[Data and Code]`


### Abstract

> Voluntary actions are preceded by the Readiness Potential (RP), a
> slow EEG component generated in supplementary motor area. The RP is
> usually thought to be specific to internally-driven decisions to
> act, and reflect post-decision motor preparation. Recent work
> suggests instead that it may reflect noise or conflict during the
> decision itself, with internally-driven decisions tending to be more
> random, more conflictual and thus more uncertain than
> externally-driven actions. To contrast accounts based on
> endogenicity with accounts based on uncertainty, we recorded EEG in
> a task where participants decided to act or withhold action to
> accept or reject gambles. We found no difference in an RP-related
> motor component when comparing actions driven by strong versus weak
> evidence, indicating that the RP does not reflect uncertainty. In
> contrast, the same RP-related component showed higher amplitudes
> actions performed without external evidence (guesses) than for those
> performed in response to equivocal, conflicting evidence. This
> supports the view that the RP reflects the internal generation of
> action, rather than decision uncertainty.

This repository contains the data, experiment, and analysis scripts
for this manuscript.

It is hosted on GitHub (<https://github.com/EoinTravers/ezmc>) and the
Open Science Framework (https://osf.io/m834c/).


## Contents


- `Procedure/*` Python/PsychoPy scripts used to run the experiment
- `Results/*` Data and anlalysis scripts
  - `src/*`: Shared python and R functions
  - `data/*`: Data files. See notes below if downloading from GitHub.
  - `Behaviour.Rmd`: Behavioural analyses
  - `Behavioural_Supplementary.Rmd`: Supplementary behavioural analyses
  - `0.Preprocess.ipynb`: EEG preprocessing script. Note that raw EEG isn't included, so this can't be run
  - `1.Merge.ipynb`: Merge preprocessed EEG. Also can't be run
  - `2.ERPs.ipynb`: Plot simple ERPs. This can be run
  - `3a.PCA_Rotation.ipynb`: PCA-CSD stuff
  - `3b.PCA_Supplementary.ipynb`: Some extra information about the PCA-CSD approach
  - `6.EasyVsDifficult.Rmd`: Comparison of Easy and Difficult decisions
  - `7.GuessVsDifficult.Rmd`: Comparison of Guesses and Difficult decisions
  - `TraversHaggard2019.Rproj`: RStudio project file


## Data

We provide the combined, preprocessed EEG data for this experiment.
If you're accessing these files on GitHub, you'll need to run the `download.sh` script in the data folder.

## Requirements

Python, R, various packages.

Confusingly, the experiment script requires python 2, but the EEG analysis scripts use python 3.
