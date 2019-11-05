"""This model contains objects and functions related to hypotheses."""

from typing import Optional
import mesa
from .constants import (
    HYPOTHESIS_AUTHOR,
    HYPOTHESIS_INITIAL_OUTCOME,
    HYPOTHESIS_TALLY,
    HYPOTHESIS_TRUTH,
)


def create_hypothesis(
    initial_outcome: str,
    truth_value: bool,
    author_id: Optional[int] = None,
    tally: Optional[int] = 0,
) -> dict:
    """Return a hypothesis with specified attributes.

    Args:
        initial_outcome: The outcome of the initial investigation of
            the hypothesis.
        truth_value: The truth value of the hypothesis.
        author_id: The ID of the agent which first
            published an investigation of this hypthesis.
        tally: The tally of the hypothesis
    """
    return {
        HYPOTHESIS_AUTHOR: author_id,
        HYPOTHESIS_INITIAL_OUTCOME: initial_outcome,
        HYPOTHESIS_TALLY: tally,
        HYPOTHESIS_TRUTH: truth_value,
    }


class HypothesisManager:
    """A class to manage hypotheses.

    Each hypothesis' attributes are tracked by four arrays.

    Attributes:
        hypotheses: An array consisting of hypotheses.
        hypothesis_map: A dictionary which has tallies as keys and
            hypothesis indices as values.
    """

    def __init__(self) -> None:
        """Initialize the hypothesis manager."""
        self.hypotheses = []
        self.hypothesis_map = {}

    def compute_hypothesis_map(self) -> None:
        """Re-process the hypothesis map.

        As this operation is expensive, it should be run only once per
        time step.
        """
        self.hypothesis_map = {}

        for hyp, idx in enumerate(self.hypotheses):
            # Get the tally of the hypothesis
            s = hyp[HYPOTHESIS_TALLY]

            # Mark this hypothesis in the map
            if s in self.hypothesis_map:
                self.hypothesis_map[s].append(idx)
            else:
                self.hypothesis_map[s] = [idx]

    def find_hypothesis_closest_to_target_tally(
        self, s: float, agent: mesa.Agent
    ) -> int:
        """Select a random hypothesis with tally closest to a target tally.

        The point of passing the agent in is so that we can use its
        random seed, which is seeded by the model. This makes sure
        that for a given seed this simulation is deterministic.

        Args:
            s: The tally to target.
            agent: The agent calling this function.

        Returns:
            The index of a hypothesis closest to the target tally.
        """
        # Get all available tallies
        tallies = list(self.hypothesis_map.keys())

        # Find the closest tally
        closest_tally = min(tallies, key=lambda x: abs(x - s))

        # Select a random hypothesis at this tally and return its index
        return agent.random.choice(self.hypothesis_map[closest_tally])
