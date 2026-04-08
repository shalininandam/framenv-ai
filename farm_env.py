import numpy as np
import random

class FarmEnv:
    def __init__(self):
        self.day = 0
        self.max_days = 30
        self.state = {}
        self.done = False
        self.total_reward = 0
        
        # Actions
        # 0 = Water crops
        # 1 = Apply fertilizer
        # 2 = Apply pesticide
        # 3 = Do nothing
        # 4 = Harvest
        self.action_space = 5
        self.action_names = [
            "Water Crops",
            "Apply Fertilizer", 
            "Apply Pesticide",
            "Do Nothing",
            "Harvest"
        ]
        self.reset()

    def reset(self):
        self.day = 0
        self.done = False
        self.total_reward = 0
        self.state = {
            "day": 0,
            "water_level": random.randint(40, 60),
            "soil_health": random.randint(60, 80),
            "pest_level": random.randint(0, 20),
            "growth_stage": 0,
            "weather": self._get_weather(),
            "yield_score": 0
        }
        return self.state

    def _get_weather(self):
        weathers = ["sunny", "cloudy", "rainy", "stormy"]
        weights = [0.5, 0.3, 0.15, 0.05]
        return random.choices(weathers, weights=weights)[0]

    def step(self, action):
        if self.done:
            return self.state, 0, True, "Episode already done"

        reward = 0
        info = ""
        self.day += 1
        weather = self._get_weather()
        self.state["weather"] = weather
        self.state["day"] = self.day

        # Weather effects
        if weather == "rainy":
            self.state["water_level"] = min(100,
                self.state["water_level"] + 20)
        elif weather == "stormy":
            self.state["water_level"] = min(100,
                self.state["water_level"] + 30)
            self.state["soil_health"] = max(0,
                self.state["soil_health"] - 10)
        elif weather == "sunny":
            self.state["water_level"] = max(0,
                self.state["water_level"] - 10)

        # Natural pest increase
        self.state["pest_level"] = min(100,
            self.state["pest_level"] + random.randint(0, 5))

        # Process action
        if action == 0:  # Water crops
            if self.state["water_level"] < 40:
                self.state["water_level"] = min(100,
                    self.state["water_level"] + 30)
                reward += 15
                info = "Good watering!"
            elif self.state["water_level"] > 70:
                self.state["water_level"] = min(100,
                    self.state["water_level"] + 10)
                reward -= 5
                info = "Overwatering!"
            else:
                self.state["water_level"] = min(100,
                    self.state["water_level"] + 20)
                reward += 5
                info = "Watered crops"

        elif action == 1:  # Fertilizer
            if self.state["soil_health"] < 50:
                self.state["soil_health"] = min(100,
                    self.state["soil_health"] + 25)
                reward += 20
                info = "Great fertilizing!"
            else:
                self.state["soil_health"] = min(100,
                    self.state["soil_health"] + 10)
                reward += 5
                info = "Fertilized soil"

        elif action == 2:  # Pesticide
            if self.state["pest_level"] > 40:
                self.state["pest_level"] = max(0,
                    self.state["pest_level"] - 40)
                reward += 20
                info = "Pest controlled!"
            else:
                self.state["pest_level"] = max(0,
                    self.state["pest_level"] - 10)
                reward -= 5
                info = "Unnecessary pesticide"

        elif action == 3:  # Do nothing
            reward -= 2
            info = "Did nothing"

        elif action == 4:  # Harvest
            if self.state["growth_stage"] >= 3:
                harvest_reward = (
                    self.state["soil_health"] +
                    self.state["water_level"] -
                    self.state["pest_level"]
                )
                reward += harvest_reward
                self.state["yield_score"] = harvest_reward
                self.done = True
                info = f"Harvested! Yield: {harvest_reward}"
            else:
                reward -= 20
                info = "Too early to harvest!"

        # Growth stage
        if (self.state["water_level"] > 30 and
            self.state["soil_health"] > 40 and
            self.state["pest_level"] < 60):
            self.state["growth_stage"] = min(5,
                self.state["growth_stage"] + 0.2)

        # Critical conditions
        if self.state["water_level"] < 10:
            reward -= 20
            info = "Crop dying - no water!"
        if self.state["pest_level"] > 80:
            reward -= 20
            info = "Crop dying - pests!"
        if self.state["soil_health"] < 10:
            reward -= 20
            info = "Soil critically unhealthy!"

        # Max days reached
        if self.day >= self.max_days:
            self.done = True
            info = "Season ended!"

        self.total_reward += reward
        self.state["yield_score"] = round(self.total_reward, 2)

        return self.state, reward, self.done, info

    def get_state(self):
        return self.state