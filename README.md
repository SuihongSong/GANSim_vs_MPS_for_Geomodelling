# Codes, Data, and Trained Models for  
**"Generative Geomodelling: Deep Learning vs. Geostatistics"**  
*Suihong Song, Jiayuan Huang, and Tapan Mukerji (Stanford University)*

## 📄 Paper Abstract

Generative geomodelling aims to simulate subsurface facies distributions while honoring multiple types of conditioning data and geological knowledge. This study compares three representative multiple-point statistics (MPS) approaches—Direct Sampling (DS), Quick Sampling (QS), and SNESIM—with two Generative Adversarial Network (GAN) workflows: Post-GANs perturbation and GANSim. The comparison is conducted using two sedimentary reservoir scenarios.

In addition to recent GANSim enhancements—such as the local discriminator and facies-indicator output design—this paper introduces a new strategy: injecting global feature information directly into intermediate layers of the generator (instead of concatenating it with latent vectors). This significantly improves conditioning effectiveness.

The results show that GANs, especially GANSim, consistently produce geologically realistic and diverse facies models that honor well facies data, global features, and facies probability maps. In contrast, MPS approaches, while capable of conditioning to well and probability maps, often produce less realistic models and are less reliable for conditioning to global features. Furthermore, GANSim is hundreds of times faster than MPS methods. Flow simulation results indicate that GANSim achieves more accurate and less uncertain predictions. Despite being trained on stationary conceptual geomodels, GANSim generalizes well to model large, nonstationary reservoirs by spatially varying input global feature maps and designing conditioning probability maps.

**Key findings:**
- **Geological realism**: GANSim consistently generates realistic and diverse facies models.
- **Conditioning accuracy**: Effectively honors global features, well facies data, and probability maps.
- **Speed**: Hundreds of times faster than MPS approaches.
- **Generalization**: Performs well on large, nonstationary reservoirs.
- **Flow prediction**: Produces more accurate and less uncertain simulation outcomes.

📎 **[Read the full paper on ResearchGate](https://www.researchgate.net/publication/392870185_Generative_geomodelling_Deep_Learning_vs_Geostatistics)**

---

## 📁 Repository Structure

This repository contains code, datasets, and trained models used in the paper. Content is organized by dataset:
- `2DPointBar`
- `2DChannel`

### 📂 Inside `2DPointBar`:

#### 🔹 MPS Baseline Algorithms
- Direct Sampling and Quick Sampling are implemented in `3_5_MPS_Codes.ipynb`.
- SNESIM is run using **Petrel** (not included here).

#### 🔹 GANSim Workflows and Comparisons

##### 1. **Data Preparation and Visualization**
- `1_1_Training_Test_Data_Preparation.ipynb` — Prepares training/test datasets for GANSim.
- `1_2_Visualize_Training_Test_Data.ipynb` — Visualizes training/test data.

##### 2. **GANSim Training**
Located in the `2_GANSimTraining` folder.

1. **Set hyperparameters in `config.py`**:
   - Facies codes
   - Directory paths for training/results
   - Conditioning types: `cond_label`, `cond_well`, `cond_prob`
   - Loss weights and other settings

2. **Train the model**:
   - Run `%run train.py` in a notebook, or
   - Run `python train.py` from terminal

##### 3. **Conditional Geomodelling Workflows**

###### 📌 *Conditioning to Global Features Only*
- `3_1_PostGANs_MCMC_condition_to_global_features.ipynb` — Post-GANs MCMC workflow.
- `3_2_GANSim_condition_to_global_features.ipynb` — GANSim workflow and comparison with MPS methods.

###### 📌 *Conditioning to Global Features + Well Facies + Probability Maps*
- `3_3_GANSim_condition_to_wellfacies_faciesprob_globalfeatures.ipynb` — GANSim workflows and comparison with SNESIM.
- `3_4_Codes_for_Flow_Simulation_by_Calling_Eclipse.ipynb` — Flow simulation with Eclipse on generated facies models.

##### 4. **Hands-on Demo with Pretrained GANSim (Google Colab)**
- `3_6_HandsOn_with_Trained_Conditional_GANSim_in_Colab.ipynb` — Easy-to-use Colab notebook to test a pretrained GANSim model with global features, well facies, and facies probability maps.  
  No deep learning experience required!

---

## 🚀 Getting Started

1. **Clone this repository**:
   ```bash
   git clone https://github.com/SuihongSong/GANSim_vs_MPS_for_Geomodelling.git
   cd GANSim_vs_MPS_for_Geomodelling
