"""Legacy direct-JPype Pandemic example. Prefer ``PyTAG(game_id='Pandemic')``."""

from pathlib import Path
import os
import random
import jpype
from jpype.types import JLong

root = Path(__file__).resolve().parent
tag = root / "pytag" / "TabletopGames"
jar = Path(os.environ.get("PYTAG_JAR_PATH", tag / "target" / "TAG-pytag.jar"))
os.chdir(tag)  # TAG reads data/pandemic from here

jpype.startJVM(
    "--add-opens=java.base/java.lang=ALL-UNNAMED",
    classpath=[str(jar)],
)

GameType = jpype.JClass("games.GameType")
RandomPlayer = jpype.JClass("players.simple.RandomPlayer")
ArrayList = jpype.JClass("java.util.ArrayList")

players = ArrayList()
players.add(RandomPlayer())
players.add(RandomPlayer())

game = GameType.Pandemic.createGameInstance(2, JLong(12345))
game.reset(players)
state = game.getGameState()
model = game.getForwardModel()
rng = random.Random(12345)

for step in range(500):
    if not state.isNotTerminal():
        break

    player = state.getCurrentPlayer()
    actions = model.computeAvailableActions(state.copy(player))
    action = actions.get(rng.randrange(actions.size()))

    if step < 20:
        print(f"{step}: player {player} chose {action}")

    model.next(state, action)

print("Game status:", state.getGameStatus())
print("Player results:", [str(result) for result in state.getPlayerResults()])
