# Hangman Agent with LSTM, Proximal Policy Optimization, and Curriculum Learning

## Training Method

### 1. Environment Overview
The Hangman environment is a custom-built gymnasium environment designed to train a reinforcement learning agent to play the game of Hangman. The environment incorporates dynamic state representations, curriculum learning, and probabilistic reasoning to provide a rich training ground for advanced RL algorithms such as LSTM-PPO. Below is a detailed explanation of the components and functionalities of the environment:

#### 1.1 Curriculum Learning Framework
Curriculum learning is implemented to gradually increase the difficulty of the environment, enabling the agent to learn progressively. The curriculum consists of multiple phases, with each phase defining:
- **Word Length Range**: Minimum and maximum word lengths to select words for that phase.
- **Max Attempts**: The number of incorrect guesses allowed for that phase.

The phases are managed through a `Curriculum` class that tracks the current phase, provides configurations, and advances to the next phase upon successful completion.

#### 1.2 Observation Space
The observation space is a concatenated vector of multiple features representing the game's current state. These features include:
- **Revealed Word**: A partially revealed representation of the target word. Unknown letters are encoded as `-1`, while revealed letters are represented as integers (`0-25`).
- **Guessed Letters**: A binary vector indicating whether each alphabet letter has been guessed.
- **Remaining Attempts**: The normalized number of incorrect guesses allowed before the game ends.
- **Word Length**: The normalized length of the target word.
- **Letter Frequencies**: A precomputed vector of letter frequencies based on the training word list.
- **Last Action**: A one-hot encoded representation of the agent's previous action.
- **Unique Letters Remaining**: The normalized count of unique, unrevealed letters in the target word.
- **Letter Probabilities**: A vector of probabilities for each letter, computed using regex-based filtering of possible matches.

The observation space is defined as a continuous vector using the `Box` space from gymnasium.

#### 1.3 Action Space
The action space is discrete and represents the 26 English alphabet letters. The agent selects an action by choosing an index corresponding to a letter.

#### 1.4 Reset Mechanism
The `reset` method initializes the environment's state:
- A word is selected randomly from the training word list based on the current curriculum phase's word length range.
- Revealed word representation and guessed letters are reset.
- The maximum number of allowed incorrect guesses (`max_attempts`) is set based on the current phase.
- Precomputed letter probabilities and other normalized state variables are updated.

The environment's initial state is returned as a concatenated observation vector.

#### 1.5 State Updates
The `_update_state` method normalizes and updates:
- Remaining attempts.
- Word length.
- Unique letters remaining.
- Last action taken (one-hot encoded).

This ensures that the observation vector reflects the most current state of the game.

#### 1.6 Step Function
The `step` method processes the agent’s actions and returns:
- **Observation**: Updated state representation.
- **Reward**:
  - Positive rewards for correctly guessing letters or winning.
  - Negative rewards for incorrect guesses, repeated guesses, or losing.
- **Terminated**: Whether the game ends due to a win or exhaustion of attempts.
- **Truncated**: Set to `False` as truncation is not implemented.
- **Info**: Additional debugging or performance metrics (empty by default).

The agent's chosen action is validated, and its impact on the environment is computed based on:
- Whether the guessed letter is correct or incorrect.
- How many new letters are revealed.
- The probabilistic heuristic associated with the guessed letter.

#### 1.7 Reward System
The reward system encourages optimal behavior by the agent and discourages poor actions. It combines heuristic and probabilistic approaches to compute rewards and penalties based on the agent's guesses.

##### Reward Structure:
- **Correct Guesses**:
  - **Base Reward**: 5 points per newly revealed letter.
  - **Heuristic Reward**: `+10 × letter probability` for guessing probable letters.
- **Incorrect Guesses**:
  - **Penalty**: `-2 points` for an incorrect guess.
  - **Heuristic Penalty**: `-5 × (1 - letter probability)` for improbable guesses.
- **Repeated Guesses**:
  - **Heavy Penalty**: `-10 points` for guessing a letter already guessed.
- **Winning the Game**:
  - **Bonus Reward**: `+50 points` for winning.
  - **Efficiency Bonus**: `+10 × remaining_attempts / max_attempts` for winning efficiently.
- **Losing the Game**:
  - **Penalty**: `-20 points` for running out of attempts.

This reward system balances immediate incentives (e.g., rewarding correct guesses) with long-term strategy (e.g., penalizing inefficient or redundant guesses). It promotes strategic thinking and efficient gameplay by scaling rewards based on letter probabilities and remaining attempts.

#### 1.8 Rendering
The `render` method provides a human-readable view of the environment’s current state, including:
- The partially revealed word.
- Letters guessed so far.
- Remaining attempts.

This is useful for debugging and visualizing the training process.

#### 1.9 Letter Probabilities
The `_compute_letter_probabilities` method uses regex and vectorized operations to compute the probability of each letter appearing in the target word. This method ensures that:
- Probabilities reflect the current state of revealed and guessed letters.
- Words eliminated by incorrect guesses are excluded from consideration.

#### 1.10 Curriculum Progression
The `advance_phase` method allows the curriculum to progress to higher difficulty levels as the agent improves. Each phase introduces longer words and fewer attempts, ensuring the agent adapts to increasingly challenging tasks.

---

### 2. Training Setup and Implementation

#### 2.1 Word List Preprocessing
Preprocessing ensures the agent operates within a consistent and manageable word space, enabling efficient learning and avoiding edge cases caused by overly long or invalid words.

#### 2.2 Curriculum and Environment Integration
The `Curriculum` class dynamically adjusts difficulty, ensuring stable training progression across five increasingly challenging phases.

#### 2.3 PPO with LSTM Architecture
The Hangman Agent leverages Proximal Policy Optimization (PPO) with an LSTM-based recurrent policy, designed to handle the partially observable nature of the Hangman game. It uses:
- **Policy and Value Networks**: Shared LSTM layer with fully connected layers ([256, 128]).
- **Temporal Dependencies**: LSTM retains game context for better decision-making.

#### 2.4 Callbacks for Curriculum and Monitoring
Custom callbacks ensure:
- Dynamic curriculum advancement.
- Reward tracking and model checkpointing.

#### 2.5 Vectorized Environment and Training Setup
The environment setup includes various optimizations for efficient training and compatibility with Stable-Baselines3.

##### Environment Wrapping:
- **DummyVecEnv**: Wraps the environment to support vectorized operations, enabling compatibility with RL frameworks.
- **VecMonitor**: Tracks and logs rewards and performance, providing insights into the agent's progress during training.

##### Device Allocation:
- Training is conducted on an **NVIDIA RTX 4090 GPU**, significantly speeding up computations, particularly for recurrent policies like LSTM.

##### Training Hyperparameters:
- **Learning Rate**: \( 1 \times 10^{-4} \)
- **Steps per Update (n\_steps)**: 256
- **Batch Size**: 128
- **Epochs per Update (n\_epochs)**: 4
- **Discount Factor (\( \gamma \))**: 0.99
- **GAE Lambda**: 0.95
- **Clip Range**: 0.2
- **Entropy Coefficient**: 0.01
- **Value Function Coefficient**: 0.5
- **Max Gradient Norm**: 0.5

These hyperparameters are carefully chosen to ensure stable and efficient training.

#### 2.6 Training Execution
##### Training Process:
- The model is initialized using `RecurrentPPO` with the `MlpLstmPolicy`.
- Training spans over 10,000,000 timesteps with:
  - Dynamic curriculum advancement at predefined intervals.
  - Logging of average rewards.
  - Periodic model checkpointing to prevent progress loss.

##### TensorBoard Integration:
- The training process is logged to **TensorBoard**, providing real-time visualization of performance metrics, such as rewards and loss.

##### Progress Bar:
- A `ProgressBarCallback` offers a user-friendly visual indicator of the training's progress.

#### 2.7 Final Notes on PPO and LSTMs
##### Advantages of PPO:
- **Stable Training**: PPO’s clipped objective prevents destabilizing changes to the policy.
- **Scalability**: Efficient for large-scale environments and parallelization.

##### Advantages of LSTMs:
- **Memory Retention**: LSTMs retain information about past states, which is critical in partially observable environments like Hangman.
- **Sequential Processing**: Well-suited for time-series and sequential decision-making tasks.

By combining **PPO** with LSTM-based policies, the agent achieves robust learning in the Hangman environment, adeptly managing its challenges of partial observability, dynamic state updates, and curriculum-based complexity.

---

This concludes the detailed writeup in Markdown format. You can copy and paste this into any Markdown-supported platform for documentation or sharing.

