# BrainAgeNeXt: Advancing Brain Age Modeling for Individuals with Multiple Sclerosis

Welcome to the code repository of [BrainAgeNeXt](https://doi.org/10.1101/2024.08.10.24311686), a novel deep learning approach to predict brain age from T1-weighted MRI scans acquired at *any* magnetic field strength.

## Table of Contents
- [Introduction](#introduction)
- [Installation](#installation)
- [Usage](#usage)
- [References](#references)
- [License](#license)

## Introduction
BrainAgeNeXt is a deep learning model designed to predict brain age with high accuracy across different MRI scanning conditions. The model builds on the MedNeXt framework [2], inspired by the ConvNeXt blocks [3].

## Installation
To get started install all requirements of the [MedNeXt repository](https://github.com/MIC-DKFZ/MedNeXt).
Next install our [customized MedNeXt repository](https://github.com/FrancescoLR/MedNeXt). 

## Usage
First, preprocess all images by performing skull stripping on the T1-weighted MRI scans (SynthSeg from Freesurfer is the preferred tool), followed by an affine registration to the MNI 152 standard space and an N4 bias field correction using ANTs.
Next, run the Brain_estimation.py script to predict brain age on your data.


## References
Please cite the following papers if using any code from this project:

1. **La Rosa, F. et al. (2024).** *BrainAgeNeXt: Advancing Brain Age Modeling for Individuals with Multiple Sclerosis.* Imaging Neuroscience (2025). [https://doi.org/10.1101/2024.08.10.24311686](https://doi.org/10.1162/imag_a_00487)

2. **Roy, S. et, al (2023).** *Mednext: transformer-driven scaling of convnets for medical image segmentation.* MICCAI. [https://rdcu.be/dRt53](https://rdcu.be/dRt53)

3. **Liu, Z. et al. (2022).** *A convnet for the 2020s.* arXiv. [https://doi.org/10.48550/arXiv.2201.03545](
https://doi.org/10.48550/arXiv.2201.03545)

## License
This repository, FrancescoLR/BrainAgeNeXt, is licensed under the Apache License 2.0. This means you are free to use, modify, and distribute the code, provided that you include a copy of the license in any distributed version of the project and comply with its terms. For more details, please refer to the [LICENSE](LICENSE) file in this repository.

