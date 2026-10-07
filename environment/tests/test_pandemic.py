"""Run from the TAG checkout so its data/pandemic assets resolve."""
import json
import os
import random
import subprocess
import sys
import unittest

import numpy as np

from pytag import PyTAG, MultiAgentPyTAG, list_supported_games


class PandemicIntegrationTest(unittest.TestCase):
    def test_default_constructor(self):
        env = PyTAG(game_id="Pandemic")
        obs, info = env.reset(seed=23)
        self.assertIsInstance(obs, str)
        self.assertEqual(len(json.loads(obs)["players"]), 2)
        self.assertTrue(info["legal_actions"])
        _, _, _, _ = env.step(info["legal_actions"][0]["id"])

    def check_game(self, python_players, seed):
        agents = ["python"] * python_players + ["random"] * (2 - python_players)
        cls = MultiAgentPyTAG if python_players == 2 else PyTAG
        env = cls(agents, game_id="Pandemic", seed=seed, obs_type="json")
        rng = random.Random(seed)
        try:
            obs, info = env.reset()
            phases = set()
            seen_ids = {}
            for turn in range(2000):
                player = env.getPlayerID()
                data = json.loads(obs[player] if python_players == 2 else obs)
                mask = env.get_action_mask()
                legal = np.flatnonzero(mask)
                actions = env.get_legal_actions()
                self.assertEqual(mask.shape, (env.action_space,))
                self.assertGreater(len(legal), 0)
                self.assertEqual(len(legal), len(env._java_env.getActions()))
                self.assertEqual(len(legal), len(actions))
                self.assertEqual(set(legal), {item["mask_index"] for item in actions})
                self.assertEqual([item["id"] for item in actions], sorted(item["id"] for item in actions))
                self.assertEqual(len(actions), len({item["id"] for item in actions}))
                self.assertEqual(data["player_id"], player)
                self.assertEqual(len(data["cities"]), 48)
                self.assertEqual(len(data["players"]), 2)
                self.assertTrue(all(len(city["cubes"]) == 4 for city in data["cities"].values()))
                self.assertTrue(all(all(0 <= n <= 3 for n in city["cubes"])
                                    for city in data["cities"].values()))
                self.assertTrue(all(0 <= n <= 24 for n in data["tracks"]["cubes_available"].values()))
                self.assertGreaterEqual(data["player_deck_count"], 0)
                self.assertGreaterEqual(data["infection_deck_count"], 0)
                self.assertNotIn("player_deck", data)
                self.assertNotIn("infection_deck", data)
                if data["phase"] != "Forecast":
                    self.assertNotIn("forecast_cards", data)
                phases.add(data["phase"])
                for item in actions:
                    self.assertTrue(item["label"])
                    self.assertEqual(item["label"], env.get_action_label(item["id"]))
                    self.assertEqual(seen_ids.setdefault(item["id"], item["mask_index"]), item["mask_index"])
                with self.assertRaises(ValueError):
                    env.step(env.action_space)
                chosen = rng.choice(actions)
                try:
                    obs, reward, done, info = env.step(chosen)
                except Exception as exc:
                    print(f"FAILED players={python_players} seed={seed} turn={turn} phase={data['phase']} action={chosen['label']}")
                    if hasattr(exc, "stacktrace"):
                        print(exc.stacktrace())
                    raise
                if done:
                    self.assertEqual(int(np.count_nonzero(env.get_action_mask())), 0)
                    self.assertEqual(env.get_legal_actions(), [])
                    self.assertTrue(all(result in (1.0, -1.0, 0.5)
                                        for result in env.terminal_rewards()))
                    return turn + 1, phases
            self.fail("Game did not finish within 2000 Python decisions")
        finally:
            # JPype cannot restart its JVM; do not call env.close().
            pass

    def test_seeded_complete_games(self):
        self.assertTrue(list_supported_games(as_json=True)["Pandemic"]["json"])
        for players in (1, 2):
            for seed in (7, 31, 103):
                with self.subTest(players=players, seed=seed):
                    decisions, phases = self.check_game(players, seed)
                    print(f"players={players} seed={seed} decisions={decisions} phases={sorted(phases)}")

    def test_gymnasium_registration(self):
        import gymnasium as gym
        import pytag.gym_wrapper  # registers TAG/Pandemic-v0

        env = gym.make("TAG/Pandemic-v0", seed=19)
        obs, info = env.reset()
        self.assertTrue(env.observation_space.contains(obs))
        self.assertEqual(info["action_mask"].shape, (env.action_space.n,))
        obs, reward, terminated, truncated, info = env.step(env.unwrapped.sample_rnd_action())
        self.assertTrue(env.observation_space.contains(obs))
        self.assertFalse(truncated)

    def test_forced_discard(self):
        env = MultiAgentPyTAG(["python", "python"], game_id="Pandemic", seed=11, obs_type="json")
        obs, info = env.reset()
        for _ in range(160):
            data = json.loads(obs[env.getPlayerID()])
            legal = np.flatnonzero(env.get_action_mask())
            self.assertEqual(len(legal), len(env._java_env.getActions()))
            if data["phase"] == "DiscardReaction":
                self.assertTrue(all("DrawCard" in env.get_action_label(int(a)) for a in legal))
                return
            drives = [int(a) for a in legal if "DriveFerry" in env.get_action_label(int(a))]
            chosen = drives[0] if drives else int(legal[0])
            obs, _, done, _ = env.step(chosen)
            if done:
                break
        self.fail("Did not reach a forced discard")

    def test_reseed_repeats_initial_observation(self):
        env = MultiAgentPyTAG(["python", "python"], game_id="Pandemic", seed=5, obs_type="json")
        first, _ = env.reset(seed=23)
        first_actions = [(item["id"], item["details"]) for item in env.get_legal_actions()]
        second, _ = env.reset(seed=23)
        self.assertEqual(first, second)
        self.assertEqual(first_actions, [(item["id"], item["details"])
                                         for item in env.get_legal_actions()])

    def test_ids_match_across_fresh_processes(self):
        code = """
import json
from pytag import MultiAgentPyTAG
e = MultiAgentPyTAG(['python', 'python'], game_id='Pandemic', obs_type='json')
results = []
for seed in (0, 5, 23, 47, 101):
    e.reset(seed=seed)
    results.append((seed, [(a['id'], a['details']) for a in e.get_legal_actions()]))
print(json.dumps(results, sort_keys=True))
"""
        outputs = [subprocess.check_output([sys.executable, "-c", code],
                                           cwd=os.getcwd(), text=True).strip()
                   for _ in range(2)]
        self.assertEqual(outputs[0], outputs[1])
        self.assertGreater(len(json.loads(outputs[0])), 0)

    def test_action_traces_match_across_fresh_processes(self):
        code = """
import json, random
from pytag import MultiAgentPyTAG, PyTAG
results = []
for player_count in (1, 2):
    agents = ['python'] * player_count + ['random'] * (2 - player_count)
    cls = MultiAgentPyTAG if player_count == 2 else PyTAG
    env = cls(agents, game_id='Pandemic', obs_type='json')
    for seed in (0, 1, 7):
        _, info = env.reset(seed=seed)
        rng = random.Random(seed)
        for _ in range(2000):
            player = env.getPlayerID()
            actions = info[player]['legal_actions'] if player_count == 2 else info['legal_actions']
            chosen = rng.choice(actions)
            _, _, done, info = env.step(chosen['id'])
            if done:
                results.append((player_count, seed, env.get_action_trace(), env.terminal_rewards()))
                break
        else:
            raise AssertionError('game did not finish')
print(json.dumps(results))
"""
        outputs = [subprocess.check_output([sys.executable, "-c", code],
                                           cwd=os.getcwd(), text=True).strip()
                   for _ in range(2)]
        self.assertEqual(outputs[0], outputs[1])
        self.assertEqual(len(json.loads(outputs[0])), 6)

    def test_selected_move_executes_its_tag_action(self):
        env = MultiAgentPyTAG(["python", "python"], game_id="Pandemic", obs_type="json")
        obs, info = env.reset(seed=43)
        player = env.getPlayerID()
        item = next(item for item in info[player]["legal_actions"]
                    if item["details"].get("move_type") == "DriveFerry"
                    and item["details"].get("target_player") == player)
        city = item["details"]["city"]
        self.assertTrue(info[player]["action_mask"][item["mask_index"]])
        obs, _, done, info = env.step(item)
        self.assertEqual(json.loads(obs[env.getPlayerID()])["players"][player]["location"], city)
        other = MultiAgentPyTAG(["python", "python"], game_id="Pandemic", obs_type="json")
        other.reset(seed=43)
        self.assertIn(item["id"], {action["id"] for action in other.get_legal_actions()})
        other_obs, _, _, _ = other.step(item["id"])
        self.assertEqual(json.loads(other_obs[other.getPlayerID()])["players"][player]["location"], city)
        with self.assertRaises(ValueError):
            env.step(item["id"])

    def test_role_and_event_action_details(self):
        env = MultiAgentPyTAG(["python", "python"], game_id="Pandemic", obs_type="json")
        roles = set()
        events = set()
        for seed in range(180):
            obs, info = env.reset(seed=seed)
            player = env.getPlayerID()
            state = json.loads(obs[player])
            actions = info[player]["legal_actions"]
            role = state["players"][player]["role"]
            if role == "Dispatcher" and any(a["details"].get("move_type") == "Dispatcher" for a in actions):
                roles.add(role)
            if role == "Operations Expert" and any(a["details"].get("move_type") == "OperationsExpert" for a in actions):
                roles.add(role)
            for item in actions:
                details = item["details"]
                if details["type"] in ("Forecast", "QuietNight"):
                    events.add(details["type"])
                    self.assertIn("card", details)
                if details.get("move_type") == "Airlift":
                    events.add("Airlift")
                    self.assertEqual(details["card"], "Airlift")
                    self.assertIn("target_player", details)
                if details.get("card") == "Government Grant" and details["type"].startswith("AddResearchStation"):
                    events.add("Government Grant")
                    self.assertIn("city", details)
            if len(roles) == 2 and len(events) == 4:
                break
        self.assertEqual(roles, {"Dispatcher", "Operations Expert"})
        self.assertEqual(events, {"Forecast", "QuietNight", "Airlift", "Government Grant"})

    def test_selected_role_and_event_actions_execute(self):
        env = MultiAgentPyTAG(["python", "python"], game_id="Pandemic", obs_type="json")
        checked = set()
        for seed in range(180):
            if len(checked) == 3:
                break
            obs, _ = env.reset(seed=seed)
            player = env.getPlayerID()
            actions = env.get_legal_actions()
            for item in actions:
                details = item["details"]
                kind = ("OperationsExpert" if details.get("move_type") == "OperationsExpert"
                        else "Airlift" if details.get("move_type") == "Airlift"
                        else "Government Grant" if details.get("card") == "Government Grant"
                        and details["type"].startswith("AddResearchStation") else None)
                if kind is None or kind in checked:
                    continue
                if kind == "OperationsExpert" and details.get("target_player") != player:
                    continue
                city = details["city"]
                target = details.get("target_player", player)
                obs, _, _, _ = env.step(item["id"])
                after = json.loads(obs[env.getPlayerID()])
                if kind == "Government Grant":
                    self.assertTrue(after["cities"][city]["research_station"])
                    self.assertIn("Government Grant", after["player_discard"])
                else:
                    self.assertEqual(after["players"][target]["location"], city)
                    self.assertIn(details["card"], after["player_discard"])
                checked.add(kind)
                break
        self.assertEqual(checked, {"OperationsExpert", "Airlift", "Government Grant"})

    def test_planner_airlift_discards_planner_card(self):
        import jpype
        from jpype.types import JLong

        # Start JPype through the normal interface, then inspect the TAG action directly.
        PyTAG(["python", "random"], game_id="Pandemic", seed=2, obs_type="json")
        game = jpype.JClass("games.GameType").Pandemic.createGameInstance(2, JLong(49))
        players = jpype.java.util.ArrayList()
        players.add(jpype.JClass("players.simple.RandomPlayer")())
        players.add(jpype.JClass("players.simple.RandomPlayer")())
        game.reset(players)
        state = game.getGameState()
        constants = jpype.JClass("games.pandemic.PandemicConstants")
        planner = state.getComponent(constants.plannerDeckHash)
        hand = state.getComponent(jpype.JClass("core.CoreConstants").playerHandHash, 0)
        discard = state.getComponent(constants.playerDeckDiscardHash)
        card = jpype.JClass("core.components.Card")("Airlift")
        card.setProperty(jpype.JClass("core.properties.PropertyString")("name", "Airlift"))
        planner.add(card)
        hand_size = hand.getSize()
        action_class = jpype.JClass("games.pandemic.actions.MovePlayerWithCard")
        move_type = jpype.JClass("games.pandemic.actions.MovePlayer.MoveType").Airlift
        action = action_class(move_type, 0, "Chicago", 0, 0, planner.getComponentID())
        self.assertTrue(action.execute(state))
        self.assertEqual(planner.getSize(), 0)
        self.assertEqual(hand.getSize(), hand_size)
        self.assertEqual(discard.get(0).getComponentID(), card.getComponentID())

    def test_dispatcher_flight_uses_dispatcher_hand(self):
        env = MultiAgentPyTAG(["python", "python"], game_id="Pandemic", obs_type="json")
        for seed in range(180):
            obs, info = env.reset(seed=seed)
            actor = env.getPlayerID()
            state = json.loads(obs[actor])
            if state["players"][actor]["role"] != "Dispatcher":
                continue
            candidate = next((action for action in info[actor]["legal_actions"]
                              if action["details"].get("move_type") in ("CharterFlight", "DirectFlight")
                              and action["details"].get("target_player") != actor), None)
            if candidate is None:
                continue
            card = candidate["details"]["card"]
            target = candidate["details"]["target_player"]
            self.assertEqual(candidate["details"]["card_owner"], actor)
            self.assertIn(card, state["players"][actor]["hand"])
            obs, _, _, _ = env.step(candidate["id"])
            after = json.loads(obs[env.getPlayerID()])
            self.assertEqual(after["players"][target]["location"], candidate["details"]["city"])
            self.assertNotIn(card, after["players"][actor]["hand"])
            self.assertIn(card, after["player_discard"])
            return
        self.fail("No Dispatcher flight for another player found in 180 seeds")

    def test_controlled_final_cure_wins(self):
        import jpype

        env = PyTAG(["python", "random"], game_id="Pandemic", seed=29, obs_type="json")
        env.reset(seed=29)
        java_class = env._java_env.getClass()
        def private_field(name):
            field = java_class.getDeclaredField(name)
            field.setAccessible(True)
            return field.get(env._java_env)

        state = private_field("gameState")
        model = private_field("forwardModel")
        action_space = private_field("pandemicActions")
        constants = jpype.JClass("games.pandemic.PandemicConstants")
        core = jpype.JClass("core.CoreConstants")
        hash_class = jpype.JClass("utilities.Hash")
        for color in ("yellow", "red", "black"):
            state.getComponent(hash_class.GetInstance().hash("Disease " + color)).setValue(1)

        actor = state.getCurrentPlayer()
        hand = state.getComponent(core.playerHandHash, actor)
        deck = state.getComponent(constants.playerDeckHash)
        hand.clear()
        blue_cards = [card for card in deck.getComponents()
                      if card.getProperty(core.colorHash) is not None
                      and str(card.getProperty(core.colorHash).valueStr) == "blue"]
        self.assertGreaterEqual(len(blue_cards), 5)
        for card in blue_cards[:5]:
            deck.remove(card)
            hand.add(card)

        observation = state.copy(actor)
        action_space.update(model.computeAvailableActions(observation), observation)
        env._update_data()
        final_cure = next(action for action in env.get_legal_actions()
                          if action["details"]["type"] == "CureDisease"
                          and action["details"]["color"] == "blue")
        _, reward, done, _ = env.step(final_cure["id"])
        self.assertTrue(done)
        self.assertEqual(reward, 1.0)
        self.assertEqual(str(state.getGameStatus()), "WIN_GAME")
        self.assertEqual(env.terminal_rewards(), [1.0, 1.0])


if __name__ == "__main__":
    unittest.main()
