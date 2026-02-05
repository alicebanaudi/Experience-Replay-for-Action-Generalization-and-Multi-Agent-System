# Efficient Online RL for Traffic Light Control with Generative Modeling

This project studies how **generative models** can improve **sample efficiency in Reinforcement Learning**, especially in settings where data collection is expensive or limited.

We focus on **Synthetic Experience Replay (SYNTHER)** and **Prioritized Generative Replay (PGR)**, using diffusion models to generate realistic transitions that augment the agent’s replay buffer. The approach is evaluated on both continuous control and multi-agent environments.

---

## Idea

Standard off-policy RL methods often overfit when trained with high update-to-data ratios and limited real experience.  
To address this, we train a **diffusion model** to approximate environment dynamics and rewards, and use it to generate synthetic transitions that complement real data.

The goal is to:
- Improve sample efficiency
- Stabilize learning at high update rates
- Reduce reliance on large critic ensembles

---

## Method

- A diffusion model is trained on real transitions `(s, a, r, s', done)`  
- Synthetic transitions are periodically generated and stored in a separate replay buffer  
- Training batches mix real and synthetic data  
- **Prioritized Generative Replay (PGR)** guides generation toward high-relevance transitions using a curiosity-based signal  

The RL backbone is based on **REDQ + Soft Actor-Critic**, allowing very high update-to-data ratios while controlling value overestimation.

---

## Environments

- **SUMO (Traffic Signal Control)**  
  Continuous control task where the agent learns to manage traffic phases under varying vehicle densities.
  <img width="1352" height="773" alt="image" src="https://github.com/user-attachments/assets/26344b75-d24b-45d9-8dc0-9210786dcc30" />


- **Overcooked-AI**  
  Cooperative multi-agent environment used to test the limits of generative replay in discrete, coordination-heavy tasks.
  <img width="512" height="411" alt="image" src="https://github.com/user-attachments/assets/00a85801-48e3-4637-be11-85bcf8beacd9" />

---

## Results (Summary)

- In **SUMO**, generative replay significantly improves stability and convergence speed compared to real-data-only baselines.
- Synthetic transitions act as a strong regularizer in continuous state spaces.
- In **Overcooked-AI**, diffusion-based replay struggles to model precise discrete coordination, highlighting limitations of standard diffusion models in multi-agent settings.

---

## Tech Stack

- Python
- PyTorch
- Soft Actor-Critic (SAC)
- REDQ
- Diffusion Models (MLP-based)
- SUMO-RL
- Overcooked-AI

---


This project was developed for research purposes and focuses on understanding when and why generative experience replay helps reinforcement learning.
