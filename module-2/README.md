# Module 2: Using Existing Environments

## The Environment Hub

OpenEnv environments live on Hugging Face Spaces. The [Environment Hub collection](https://huggingface.co/collections/openenv/openenv-environment-hub) has ready-to-use environments you can connect to immediately.

Every Space gives you three things:

| Component | What it provides | How to access |
|-----------|------------------|---------------|
| **Server** | Running environment endpoint | `https://<username>-<space-name>.hf.space` |
| **Repository** | Installable Python package | `pip install git+https://huggingface.co/spaces/<space>` |
| **Registry** | Docker container image | `docker pull registry.hf.space/<space>:latest` |

You don't build environments to use them. Install the client, point it at a server, and go.

## Type-Safe Models

Every OpenEnv environment defines typed models for actions, observations, and state. These aren't just documentation — they're real Python dataclasses that your IDE can autocomplete and your type checker can validate.

For OpenSpiel environments:

```python
@dataclass
class OpenSpielAction(Action):
    action_id: int           # Which action to take
    game_name: str           # Which game
    game_params: Dict        # Game configuration

@dataclass
class OpenSpielObservation(Observation):
    done: bool               # Is the episode over?
    reward: Optional[float]  # Reward signal
    info_state: List[float]  # Game state as a vector
    legal_actions: List[int] # Valid actions this step
    game_phase: str          # Current phase
    current_player_id: int   # Whose turn
```

No more guessing what `obs[0][3]` means.

## OpenSpiel Integration

OpenEnv wraps 6 games from DeepMind's OpenSpiel library:

| Single-Player | Multi-Player |
|---------------|-------------|
| Catch — catch falling ball | Tic-Tac-Toe — classic 3×3 |
| Cliff Walking — navigate grid | Kuhn Poker — imperfect info |
| 2048 — tile puzzle | |
| Blackjack — card game | |

All six use the same `OpenSpielEnv` client and the same `OpenSpielAction`/`OpenSpielObservation` types. The only difference is the `game_name` parameter.

## Writing Policies

A policy is just a function that takes an observation and returns an action. Here are four approaches for Catch:

**Random** — baseline, ~20% success:
```python
def random_policy(obs):
    return random.choice(obs.legal_actions)
```

**Always Stay** — terrible, ~20% success:
```python
def stay_policy(obs):
    return 1  # STAY
```

**Smart Heuristic** — optimal, 100% success:
```python
def smart_policy(obs):
    ball_col = find_ball(obs.info_state)
    paddle_col = find_paddle(obs.info_state)
    if paddle_col < ball_col: return 2  # RIGHT
    if paddle_col > ball_col: return 0  # LEFT
    return 1  # STAY
```

**Epsilon-Greedy** — learns over time, ~85% success:
```python
def learning_policy(obs, step):
    epsilon = max(0.1, 1.0 - step / 100)
    if random.random() < epsilon:
        return random.choice(obs.legal_actions)
    return smart_policy(obs)
```

The key insight: all four policies work with the same `OpenSpielObservation` type. Swap the game from Catch to Tic-Tac-Toe and the observation format stays the same — only the game logic changes.

## Switching Games

Because all OpenSpiel games share the same client interface, switching games is trivial:

```python
# Catch
with OpenSpielEnv(base_url="https://openenv-openspiel-catch.hf.space").sync() as env:
    result = env.reset()

# Tic-Tac-Toe — same client, different URL
with OpenSpielEnv(base_url="https://openenv-openspiel-tictactoe.hf.space").sync() as env:
    result = env.reset()
```

Your policy code doesn't change. The observation has the same fields. You just need a new strategy for the new game.

## What's Next

In the [notebook](notebook.ipynb), you'll build and compare 4 policies on Catch, run a competition, then switch to another game with the same client code.

**Key takeaway:** You don't build environments to use them. Same client interface across all games.
