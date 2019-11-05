"""This model contains objects and functions related to hypotheses."""

import uuid
from typing import Optional
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
):
    """Return a hypothesis map with specified attributes.

    Args:
        truth_value: The truth value of the hypothesis.
        tally: The tally of the hypothesis
        author_id: The ID of the agent which first
            published an investigation of this hypthesis.
        initial_outcome: The outcome of the initial investigation of
            the hypothesis.
    """
    return {
        HYPOTHESIS_AUTHOR: author_id,
        HYPOTHESIS_INITIAL_OUTCOME: initial_outcome,
        HYPOTHESIS_TALLY: tally,
        HYPOTHESIS_TRUTH: truth_value,
    }


class HypothesisManager:
    """A class to manage hypotheses.

    Attributes:
        hypotheses: An array consisting of hypotheses.
    """

    def __init__(self):
        """Initialize the hypothesis manager."""
        self.hypotheses = []
