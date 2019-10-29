from typing import List, Optional
from mesa import Agent, Model
from mesa.time import BaseScheduler, RandomActivation
from schema import Schema

# Constants for result of experiment
POSITIVE = "positive"
NEGATIVE = "negative"

# Constant for hypotheses
TRUE = "true"
TALLY = "tally"
TESTED_HYPOTHESIS_SCHEMA = Schema({TRUE: bool, TALLY: int})


class ScienceAgent(Agent):
    """An agent representing a research entity.

    Attributes:
        staged_hypothesis: A hypothesis to publish to the field after
            all three stages have completed in a time step.
    """

    def __init__(self, unique_id: int, model: Model) -> None:
        """Initializes an agent."""
        super().__init__(unique_id, model)

        self.staged_hypothesis = None

    def pop_staged_hypothesis(self) -> Optional[dict]:
        """Pop and return staged hypothesis provided one exists."""
        hypothesis = self.staged_hypothesis
        self.staged_hypothesis = None

        return hypothesis

    def step(self) -> None:
        """Perform agent actions."""
        # Decide whether to investigate a novel hypothesis or perform a
        # replication
        will_replicate = self.random.choices(
            population=[True, False], weights=[self.model.r, 1 - self.model.r]
        )[0]

        if will_replicate:
            self.replication_action()
        else:
            self.novel_hypothesis_action()

    def novel_hypothesis_action(self) -> None:
        """Investigate a novel hypothesis."""
        # Determine if hypothesis is true
        is_true = self.random.choices(
            population=[True, False], weights=[self.model.b, 1 - self.model.b]
        )[0]

        # Determine the result of the experiment
        if is_true:
            is_positive = self.random.choices(
                population=[True, False],
                weights=[1 - self.model.beta, self.model.beta],
            )[0]
        else:
            is_positive = self.random.choices(
                population=[True, False],
                weights=[self.model.alpha, 1 - self.model.alpha],
            )[0]

        result = POSITIVE if is_positive else NEGATIVE

        # Determine whether to publish the experiment
        if result == POSITIVE:
            will_publish = self.random.choices(
                population=[True, False],
                weights=[self.model.c_n_pos, 1 - self.model.c_n_pos],
            )[0]
        else:
            will_publish = self.random.choices(
                population=[True, False],
                weights=[self.model.c_n_neg, 1 - self.model.c_n_neg],
            )[0]

        # Publish the result
        if will_publish:
            self.staged_hypothesis = {
                TRUE: is_true,
                TALLY: 1 if result == POSITIVE else -1,
            }

    def replication_action(self) -> None:
        """Conduct a replication."""
        # Pick a hypothesis to replicate. Note that any changes to the
        # hypothesis will be made in place in the list of hypothesis in
        # the model.
        hypothesis = self.random.choice(self.model.published_hypotheses)

        # Determine the result of the experiment
        if hypothesis[TRUE]:
            is_positive = self.random.choices(
                population=[True, False],
                weights=[1 - self.model.beta, self.model.beta],
            )[0]
        else:
            is_positive = self.random.choices(
                population=[True, False],
                weights=[self.model.alpha, 1 - self.model.alpha],
            )[0]

        result = POSITIVE if is_positive else NEGATIVE

        # Determine whether to publish the replication
        if result == POSITIVE:
            will_publish = self.random.choices(
                population=[True, False],
                weights=[self.model.c_r_pos, 1 - self.model.c_r_pos],
            )[0]
        else:
            will_publish = self.random.choices(
                population=[True, False],
                weights=[self.model.c_r_neg, 1 - self.model.c_r_neg],
            )[0]

        # Publish the result
        if will_publish:
            hypothesis[TALLY] += 1 if result == POSITIVE else -1


class ScienceModel(Model):
    """The simulation model.

    Attributes:
        n: The number of agents.
        r: The probability that an agent chooses to replicate a
            hypothesis.
        b: The probability that a novel hypothesis is true.
        alpha: Type I error rate.
        beta: Type II error rate.
        c_n_pos: The probability that a positive novel result is
            published.
        c_n_neg: The probability that a negative novel result is
            published.
        c_r_pos: The probability that a positive replication result is
            published.
        c_r_neg: The probability that a negative replication result is
            published.
        initial_data: A list of published hypotheses to start the data
            with.
        scheduler: A scheduler instance that determines in which order
            agents act.
    """

    def __init__(
        self,
        n: int,
        r: float,
        b: float,
        alpha: float,
        beta: float,
        c_n_pos: float,
        c_n_neg: float,
        c_r_pos: float,
        c_r_neg: float,
        initial_data: List[dict],
        scheduler: Optional[BaseScheduler] = RandomActivation,
    ) -> None:
        """Initializes a model."""
        # Model parameters. Validate that probabilities are satisfied
        # and then assign.
        for p in (r, b, alpha, beta, c_n_pos, c_n_neg, c_r_pos, c_r_neg):
            Schema(lambda x: 0 < x < 1).validate(p)

        self.n = n
        self.r = r
        self.b = b
        self.alpha = alpha
        self.beta = beta
        self.c_n_pos = c_n_pos
        self.c_n_neg = c_n_neg
        self.c_r_pos = c_r_pos
        self.c_r_neg = c_r_neg

        # Model scheduler
        self.scheduler = scheduler(self)

        # Validate initial published hypothesis data and store it
        for hypothesis in initial_data:
            TESTED_HYPOTHESIS_SCHEMA.validate(hypothesis)

        self.published_hypotheses = initial_data

        # Initialize agents
        for i in range(self.n):
            self.scheduler.add(ScienceAgent(i, self))

    def step(self) -> None:
        """Iterate through all agent actions for one time step."""
        # Run through all agent actions
        self.scheduler.step()

        # Add new novel hypotheses that have been published
        new_hypotheses = [
            hypothesis
            for hypothesis in [
                agent.pop_staged_hypothesis()
                for agent in self.scheduler.agents
            ]
            if hypothesis is not None
        ]
        self.published_hypotheses += new_hypotheses
