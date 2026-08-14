import random


class OriginalOrderPolicy:
    """
    Dynamic version of the original Stage-1 order.

    Among the currently legal actions, always choose
    the smallest original loop ID.
    """

    def reset(self):
        pass

    def select(self, state, actions):
        if not actions:
            raise RuntimeError(
                "OriginalOrderPolicy received "
                "an empty action set"
            )

        return min(actions)


class RandomPolicy:
    """
    Randomly choose from the CURRENT legal action set.

    This is different from generating one random
    permutation before the episode starts.
    """

    def __init__(self, seed=None):
        self.seed = seed
        self.rng = random.Random(seed)

    def reset(self):
        self.rng = random.Random(self.seed)

    def select(self, state, actions):
        if not actions:
            raise RuntimeError(
                "RandomPolicy received "
                "an empty action set"
            )

        return self.rng.choice(actions)


class ReplayPolicy:
    """
    Replay a predefined sequence.

    Useful for regression tests such as random_seed3.
    """

    def __init__(self, sequence):
        self.sequence = list(sequence)
        self.position = 0

    def reset(self):
        self.position = 0

    def select(self, state, actions):
        if self.position >= len(self.sequence):
            raise RuntimeError(
                "Replay sequence ended "
                "before the episode became terminal"
            )

        action = self.sequence[self.position]
        self.position += 1

        if action not in actions:
            raise RuntimeError(
                f"Replay action {action} is not legal. "
                f"Current legal actions: {actions}"
            )

        return action