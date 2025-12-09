# 🎓 Experience Replay for Action Generalization and Multi-Agent Systems

## 📘 Overview

This project explores **generative experience replay** as a mechanism to improve **action generalization** and **multi-agent learning** in reinforcement learning (RL).  
Traditional experience replay allows agents to store and reuse past interactions, but real experience is often limited and biased, leading to overfitting and poor adaptability.

Building on recent advances such as **SYNTHER (Synthetic Experience Replay)** [1] and **PGR (Prioritized Generative Replay)** [2], we investigate how **diffusion-based generative modeling** can produce synthetic, high-quality transitions that enrich the replay buffer.  
Our goal is to evaluate whether generative replay can scale from **single-agent control** to **multi-agent coordination**, enhancing both sample efficiency and generalization.

---

## 🚀 Objectives

- Extend **generative replay** to diverse environments beyond standard locomotion tasks.
- Evaluate its effectiveness in **action generalization** (continuous control) and **multi-agent coordination** (cooperative learning).
- Measure whether synthetic experience improves learning stability, zero-shot generalization, and coordination.

---

## 🧠 Methodology

1. **Baseline Frameworks**
   - Start from **PGR** and **SYNTHER**, integrating diffusion-based generative replay.
   - Replace or augment the replay buffer with synthetic transitions.

2. **Single-Agent Environment**
   - **Traffic Light Control (SUMO-RL)**
     - The agent learns to optimize signal timing to minimize congestion and waiting time.
     - The model interpolates between explored action parameters (e.g., signal durations) to generate new transitions.
     - Expected outcome: smoother value landscape and improved decision-making in unseen conditions.

3. **Multi-Agent Environment**
   - **Overcooked-AI**
     - Two cooperative agents must coordinate to prepare and deliver dishes.
     - We hypothesize that our model can act as a _virtual partner generator_, producing diverse joint experiences that foster policy generalization and **zero-shot coordination**.

---

## 🧩 Environments

| Setting          | Environment                         | Description                                                                  |
| ---------------- | ----------------------------------- | ---------------------------------------------------------------------------- |
| **Single-Agent** | **Traffic Light Control (SUMO-RL)** | Optimize signal phases to reduce congestion and waiting times.               |
| **Multi-Agent**  | **Overcooked-AI**                   | Cooperative cooking environment to test coordination and zero-shot learning. |

## Checkpoints and Models

- SAC Actor: [Download Link](https://drive.google.com/file/d/19gw3RGVBaDCeZm6WQ-_aXapJcMbvVhgg/view?usp=drive_link)
