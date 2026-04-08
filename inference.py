import os
import requests

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:5000")
MODEL_NAME = os.getenv("MODEL_NAME", "farm-agent-v1")

def reset():
    res = requests.post(f"{API_BASE_URL}/reset")
    return res.json()["state"]

def step(action):
    res = requests.post(
        f"{API_BASE_URL}/step",
        json={"action": action}
    )
    data = res.json()
    return data["state"], data["reward"], data["done"], data["info"]

def get_state():
    res = requests.get(f"{API_BASE_URL}/state")
    return res.json()["state"]

if __name__ == "__main__":
    print(f"Model: {MODEL_NAME}")
    print(f"API: {API_BASE_URL}")
    
    print("\nResetting environment...")
    state = reset()
    print(f"Initial state: {state}")
    
    total_reward = 0
    done = False
    step_count = 0
    
    print("\nRunning baseline agent...")
    while not done and step_count < 30:
        action = step_count % 5
        state, reward, done, info = step(action)
        total_reward += reward
        step_count += 1
        print(f"Step {step_count}: action={action} reward={reward} info={info}")
    
    print(f"\nTotal Reward: {total_reward}")
    print(f"Final State: {state}")