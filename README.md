# Super Restriction Factor Hypermutation Analysis

#### Adapted by Dr. Mollie McDonnell
#### Original code authors: Kate H. D. Crawford, Adam S. Dingens

The original analysis code was developed by Adam Dingens and Kate Crawford in the [Bloom lab](http://research.fhcrc.org/bloom/en.html) in early 2019 at the repository SuperRestrictionFactor_Hypermutation available [here](https://github.com/molliemcdonnell/SuperRestrictionFactor_Hypermutation). Results were published in "APOBEC3C Tandem Domain Proteins Create Super Restriction Factors Against HIV-1."


This repository adapts the original code to produce analysis of 2 new sets of similar experiments. These results are for publication in "Highly-potent, synthetic APOBEC3s restrict HIB-1 through deamination-independent mechanisms."



## Organization
The analysis is performed by the iPython notebook [`analysis_notebook.ipynb`](analysis_notebook.ipynb).

Subdirectories:

   * `./results/` iss generated by [`analysis_notebook.ipynb`](analysis_notebook.ipynb).

   * `./results/FASTQ_files/` contains the input Illumina deep sequencing data. This file is generated by [`analysis_notebook.ipynb`](analysis_notebook.ipynb), which downloads the sequencing files from the [Sequence Read Archive](http://www.ncbi.nlm.nih.gov/sra).

   * `./data/` contains all input data needed to run [`analysis_notebook.ipynb`](analysis_notebook.ipynb)). Files are described in the iPython notebok when used.