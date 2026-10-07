import json
import random
from pytag import PyTAG

env = PyTAG(game_id="Pandemic")
rng = random.Random(23)
observation, info = env.reset(seed=23)

while True:
    state = json.loads(observation)
    actions = info["legal_actions"]

    # Simple first strategy: treat disease if possible.
    treat_actions = [
        a for a in actions
        if a["details"].get("type") == "TreatDisease"
    ]
    choice = rng.choice(treat_actions or actions)

    print(state["phase"], "→", choice["label"])
    observation, reward, done, info = env.step(choice["id"])

    if done:
        print("Game over. Reward:", reward)
        break