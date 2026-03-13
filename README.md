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
