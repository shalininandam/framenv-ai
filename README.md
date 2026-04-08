# FarmEnv AI 🌾
## Real-World OpenEnv Environment for AI Agent Learning

### Problem Statement
Build a complete, real-world OpenEnv environment 
that an AI agent can learn from through the 
standard step()/reset()/state() API.

### Our Solution
FarmEnv — A farming simulation environment where
an AI agent learns optimal crop management strategies!

### API Endpoints
- POST /reset → Reset environment
- POST /step → Take action, get state/reward/done
- GET /state → Get current farm state
- POST /agent/run → Run one agent episode

### Actions
- 0: Water Crops
- 1: Apply Fertilizer
- 2: Apply Pesticide
- 3: Do Nothing
- 4: Harvest

### Tech Stack
- Python + Flask
- NumPy (Q-Learning)
- HTML + CSS + JavaScript

### Team
TechMinds