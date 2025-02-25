# NDHU Machine Learning 2021 - Othello AI Agent  

📌 **Final Project - Designing an AI Agent for Othello**  

This repository contains the **final project** for the **NDHU Machine Learning (2021)** course. The project focuses on designing an **AI agent** to play **Othello (Reversi)** by training a model that makes strategic decisions.  

## 🎯 Project Overview  
The AI agent utilizes a combination of **Dual Networks and Monte Carlo Search** to predict the future board state and make optimal moves. The **MiniMax algorithm** is used for comparison to evaluate the model's performance.  

## 🏗️ Components  
### 1️⃣ **Model Training Script**  
- Implements a **Dual Network** combined with **Monte Carlo Tree Search (MCTS)** to evaluate board positions.  
- The AI **plays the first 6 moves randomly** before using **MCTS for decision-making**.  
- Each move is determined after **5 Monte Carlo simulations**, where the **rollout process** is guided by the evaluation network.  

### 2️⃣ **Battle Program**  
- Compares the **MCTS-based model** with the **MiniMax-trained model**.  
- Uses a **Policy Network** to assess move quality.  
- The predicted **best move is selected** based on **weighted scores** from the Policy Network.  
- Experimental results indicate **MiniMax performs better** than MCTS in the current setting.  

### 3️⃣ **Model Processing Script**  
- **ResNet** is modified into a **Dual Network** with two outputs:  
  1. **Policy Function** - Predicts the best move.  
  2. **Value Function** - Evaluates the overall board state.  
- The **Value Function helps MCTS** during the **rollout phase**, providing better board state evaluation.  
