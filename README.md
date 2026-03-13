# K-TempCast v1

Deep-learning–based subseasonal-to-seasonal (S2S) prediction framework for Korean summer temperature variability.

This repository contains the official research implementation of **K-TempCast v1**, developed for subseasonal-to-seasonal prediction of **monthly mean summer (June–August) surface air temperature anomalies over the Korean Peninsula**.

The model is based on an **episode-based meta-learning framework** with a CNN encoder and a gated global–regional residual architecture. The framework is designed to address two key challenges in seasonal climate prediction:

- Limited sample size in observational climate datasets
- Nonstationary relationships between large-scale circulation and regional temperature variability

The code in this repository corresponds to the methodology described in the associated research manuscript.


## Model Overview

K-TempCast consists of three main components:

1. **CNN trunk (feature extractor)**  
   Extracts large-scale circulation features from climate input fields.

2. **Dual prediction heads**
   - **Global branch**: captures global circulation signals  
   - **Regional branch**: focuses on dynamically relevant regional domains

3. **Gated residual aggregation**

The final prediction is computed as:

p = p_g + w (p_r − p_g)

where  

- p_g : global prediction  
- p_r : regional prediction  
- w : learned gating weight

Meta-learning enables the model to **adapt quickly to each prediction year using a small support set**.


## Meta-Learning Framework

Each prediction year is treated as an individual **task (episode)**.

Training proceeds as:

1. **Support set**  
   Used for inner adaptation (3 gradient descent steps)

2. **Query set**  
   Used to compute prediction error

3. **Outer update**  
   Query loss updates all trainable parameters of the model.

To account for climate nonstationarity, **support samples are drawn from a rolling window of recent years**.


## Repository Structure

```
K-TempCast/
│
├── ae/
│   Autoencoder-based pretraining code for the CNN trunk
│
├── grad/
│   Grad-CAM analysis tools for model interpretation
│
├── model/
│   Core K-TempCast model architecture
│
├── scripts/
│   ktemp.py
│       Main training and prediction pipeline for the K-TempCast model
│
│   run_ktemp.sh
│       Example script to run the model
│
├── utils/
│   Data loading and preprocessing utilities
│
└── README.md
```
The ae/ directory contains the autoencoder-based pretraining code used to initialize the CNN trunk before the meta-learning stage. The pretrained encoder provides a stable representation of large-scale circulation patterns that is later used by the K-TempCast model.

The grad/ directory includes Grad-CAM analysis scripts used for model interpretation. These tools are used to identify the spatial regions that contribute most strongly to the model’s predictions and to diagnose the circulation patterns captured by the network.

The model/ directory contains the core implementation of the K-TempCast architecture. This includes the CNN trunk, the global and regional prediction heads, and the gated residual aggregation module that combines the two predictions.

The scripts/ directory provides the main scripts for training and running the model. In particular, ktemp.py contains the main training and prediction pipeline of the K-TempCast model, including meta-training, meta-testing, and ensemble prediction procedures. The run_ktemp.sh file provides an example shell script for executing the model and running experiments.

The utils/ directory contains supporting utilities such as data loading and preprocessing functions used to prepare the climate datasets for training and prediction.

Finally, this README.md file provides an overview of the repository, including model description, usage notes, references, and citation information.

## Related Work

This work builds upon the meta-learning framework introduced in:

Oh, S. H., & Ham, Y. G. (2024).  
Few-shot learning for Korean winter temperature forecasts.  
npj Climate and Atmospheric Science, 7(1), 279.

The present implementation also integrates ideas from **ExtremeCast**, which improves prediction of extreme values through specialized loss functions and training strategies:

Xu, W., Chen, K., Han, T., Chen, H., Ouyang, W., & Bai, L. (2024).  
ExtremeCast: Boosting extreme value prediction for global weather forecast.  
arXiv:2402.01295

ExtremeCast code:  
https://github.com/black-yt/ExtremeCast


## Authors

Research Institute of Basic Sciences, Seoul National University, Seoul, Republic of Korea
Seohee H. Yang, Chang-Hyun Park, Seok-Geun Oh
Interdisciplinary Program in AI, Seoul National University, Seoul, Republic of Korea 
Yelim Kim
Information & Electronics Research Institute, Korea Advanced Institute of Science and Technology
Seol-Hee Oh
Department of Environmental Managements, Graduate School of Environmental Studies, Seoul National University, Seoul, South Korea
Yoo-Geun Ham 
School of Earth and Environmental Sciences, Seoul National University, Republic of Korea Seohee H. Yang, Chang-Hyun Park, Seok-Geun Oh, Seok-Woo Son


## Funding

This research was supported by the  
**Korea Meteorological Administration Research and Development Program**  
under Grant **RS-2025-02307979**.


## Citation

If you use this code in academic work, please cite:

Yang et al. (2026)
Subseasonal-to-Seasonal Summer Temperature Prediction over Korea Using K-TempCast model


## Legal Notice

This repository contains original research code developed for academic and scientific purposes.

Any use, reproduction, modification, distribution, or derivative work based on this code is strictly prohibited without prior written permission from the authors.

If this code is used in any form of academic, research, or commercial work, proper citation and explicit acknowledgment of all original authors and affiliated institutions is mandatory.

Unauthorized use or redistribution without permission may result in legal consequences.
