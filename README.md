# Codes, Data, and Trained Models for  
**"Generative Geomodelling: Deep Learning vs. Geostatistics"**  
*Suihong Song, Jiayuan Huang, and Tapan Mukerji (Stanford University)*

## 📄 Paper Abstract

Generative geomodelling aims to simulate subsurface facies distributions while honoring multiple types of conditioning data and geological knowledge. This study selects three typical multiple-point statistics (MPS) approaches—Direct Sampling (DS), Quick Sampling (QS), and SNESIM—and two Generative Adversarial Network (GAN) workflows—post-GANs perturbation and GANSim—as representatives to compare traditional geostatistics-based and deep learning-based generative geomodelling methods, based on two sedimentary reservoir scenarios. In addition to the latest GANSim enhancements—namely, the local discriminator and facies-indicator output designs—this paper further proposes injecting global feature information into intermediate layers of the generator, instead of concatenating global features with latent vectors, to improve constraint effectiveness. The geomodelling results demonstrate that GANs, especially GANSim, consistently produce geologically realistic and diverse facies models that are accurately conditioned to well facies data, global features, and facies probability maps. In comparison, MPS approaches perform well in honoring well facies and probability maps but produce facies models with significantly lower geological realism. Their conditioning effectiveness on global features is also less reliable. GANSim achieves geomodelling speeds hundreds of times faster than MPS methods. Flow simulations show that GANSim results yield more accurate and less uncertain predictions than MPS outputs. Moreover, although trained on stationary conceptual geomodels, the trained GANSim generalizes well to model large, nonstationary reservoirs by spatially varying the input global feature maps and carefully designing the conditioning probability maps, making it a powerful and flexible tool for high-fidelity conditional geomodelling.

**Key findings:**
- **Geological realism**: GANSim consistently generates realistic, diverse facies models.
- **Conditioning accuracy**: GANSim effectively honors well facies data, global features, and probability maps.
- **Speed**: GANSim runs hundreds of times faster than MPS approaches.
- **Generalization**: GANSim generalizes well to nonstationary reservoirs using spatially varying global feature and probability maps.
- **Flow prediction**: GANSim produces more accurate and less uncertain flow simulation results.

📎 **[Read the full paper on ResearchGate](https://www.researchgate.net/publication/392870185_Generative_geomodelling_Deep_Learning_vs_Geostatistics)**

---

## 📁 Repository Structure

This repository contains code, datasets, and trained models used in the paper. Materials are organized into two dataset folders:
- `2DPointBar`
- `2DChannel`

### 🔧 1. Data Preparation and Visualization (`2DPointBar`)
- `1_1_Training_Test_Data_Preparation.ipynb` — Prepare training/test datasets for GANs.
- `1_2_Visualize_Training_Test_Data.ipynb` — Visualize training/test datasets.

### 🧠 2. GANSim Training
- Folder: `2_GANSimTraining`
- Configure `config.py`:
  - Facies codes
  - Directory paths for training/results
  - Conditioning types: `cond_label`, `cond_well`, `cond_prob`
  - Loss weights and other hyperparameters
- Train the model via:
  - `%run train.py` inside a notebook, or
  - `python train.py` from a terminal.

### 🧪 3. Conditional Geomodelling Workflows

#### 🔹 Conditioning to Global Features Only
- `3_1_PostGANs_MCMC_condition_to_global_features.ipynb` — Post-GANs MCMC workflow.
- `3_2_GANSim_condition_to_global_features.ipynb` — GANSim vs. MPS results.
- `3_5_MPS_Codes.ipynb` — Direct Sampling and Quick Sampling code.
- SNESIM runs in **Petrel** (not included in repo).

#### 🔹 Conditioning to Global Features + Well Facies + Facies Probability Maps
- `3_3_GANSim_condition_to_wellfacies_faciesprob_globalfeatures.ipynb` — GANSim and SNESIM comparison, with flow simulation.
- `3_4_Codes_for_Flow_Simulation_by_Calling_Eclipse.ipynb` — Scripts to call Eclipse for flow simulation.

### 🤖 4. Try Pretrained GANSim in Colab
- `3_6_HandsOn_with_Trained_Conditional_GANSim_in_Colab.ipynb` — A beginner-friendly Google Colab notebook to try the pretrained GANSim model with conditioning. No deep learning experience needed.

---

## 🚀 Getting Started

1. Clone this repository.
2. Choose a dataset folder: `2DPointBar` or `2DChannel`.
3. Follow the notebooks in sequence:
   - Data preparation → model training → conditional geomodelling → (optional) flow simulation.
4. Try the pretrained model using the Colab notebook (`3_6_...`).

---

If you find this repository useful or have questions, feel free to reach out or cite the original publication.  
We hope this resource helps advance your understanding and application of generative geomodelling!

