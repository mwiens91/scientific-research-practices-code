"""Contains main simulation code for the SRS model."""

from typing import Optional
from mesa import Agent as MesaAgent, Model
from mesa.time import BaseScheduler, RandomActivation
import life_cycle_helpers
from .constants import (
    HYPOTHESIS_AUTHOR,
    HYPOTHESIS_INITIAL_OUTCOME,
    HYPOTHESIS_TALLY,
    HYPOTHESIS_TRUTH,
    INVESTIGATION_NOVEL,
    INVESTIGATION_REPLICATION,
    INVESTIGATION_RESULT,
    INVESTIGATION_TARGET_HYPOTHESIS,
    INVESTIGATION_TYPE,
    RESULT_NEGATIVE,
    RESULT_POSITIVE,
)
from .hypotheses import HypothesisManager


class Agent(MesaAgent):
    """An agent representing a research group.

    Attributes:
        y: The number of time steps the agent has been in the model. We
            will often refer to this attribute as "age".
        w: The accumulated sum of payoffs the agent has received. We
            will often refer to this attribute as "reputation".
        gamma: The power of the agent.
        tau: The rigour of the agent.
        r: The probability that the agent chooses a replication
            investigation when choosing a hypothesis to investigate.
        c_N_pos: The probability that the agent chooses to publish a
            positive novel result.
        c_N_neg: The probability that the agent chooses to publish a
            negative novel result.
        c_R_pos: The probability that the agent chooses to publish a
            positive replication result.
        c_R_neg: The probability that the agent chooses to publish a
            negative replication result.
        alpha: The agent's type I error.
        beta: The agent's type I error.
        staged_investigation: An (optional) investigation the agent has
            staged.
    """

    def __init__(
        self,
        unique_id: int,
        model: Model,
        gamma: float,
        tau: float,
        r: float,
        c_N_pos: float,
        c_N_neg: float,
        c_R_pos: float,
        c_R_neg: float,
    ) -> None:
        """Initialize an agent."""
        # Assign the agent its unique ID and model
        super().__init__(unique_id, model)

        # Set the age and reputation of the agent to zero
        self.y = 0
        self.w = 0

        # Set the agent's research parameters
        self.gamma = gamma
        self.tau = tau
        self.r = r
        self.c_N_pos = c_N_pos
        self.c_N_neg = c_N_neg
        self.c_R_pos = c_R_pos
        self.c_R_neg = c_R_neg

        # Determine the agent's type I and II errors
        self.alpha = life_cycle_helpers.alpha(self.gamma, self.tau)
        self.beta = life_cycle_helpers.beta(self.gamma)

        # Set the staged investigation
        self.staged_investigation = None

    def step(self) -> None:
        """Perform the Science stage."""
        # Productivity Check
        p_s = life_cycle_helpers.p_s(self.model.eta_s, self.tau)

        check_passed = self.random.choices(
            population=[True, False], weights=[p_s, 1 - p_s]
        )[0]

        if not check_passed:
            return

        # We now branch depending on whether we have an investigation
        # staged
        if self.staged_investigation is None:
            # Hypothesis Selection. First determine whether to
            # investigate a novel hypothesis or replicate an existiing
            # hypothesis.
            will_replicate = self.random.choices(
                population=[True, False], weights=[self.r, 1 - self.r]
            )[0]

            # On the first few iterations of the simulation we need to
            # account for there being no hypotheses to replicate. In
            # this case, we force agents to investigate novel
            # hypotheses.
            if not self.model.hypothesis_manager.hypotheses:
                will_replicate = False

            # Now either choose a hypothesis to replicate, or choose a
            # novel hypothesis
            if will_replicate:
                # Set the investigation type
                investigation_type = INVESTIGATION_REPLICATION

                # Select a target tally
                target_tally = self.random.gauss(0, self.model.sigma_t)

                # Select a random hypothesis closest to this tally
                target_hypothesis_idx = self.model.hypothesis_manager.find_hypothesis_closest_to_target_tally(
                    target_tally, self
                )

                # Record truth of hypothesis
                hypothesis_truth = self.model.hypotheses[
                    target_hypothesis_idx
                ][HYPOTHESIS_TRUTH]
            else:
                # Set the investigation type
                investigation_type = INVESTIGATION_NOVEL

                # Determine whether hypothesis selected is true or false
                hypothesis_truth = self.random.choices(
                    population=[True, False],
                    weights=[self.model.b, 1 - self.model.b],
                )[0]

            # Investigation
            if hypothesis_truth:
                investigation_result = self.random.choices(
                    population=[RESULT_POSITIVE, RESULT_NEGATIVE],
                    weights=[1 - self.beta, self.beta],
                )[0]
            else:
                investigation_result = self.random.choices(
                    population=[RESULT_POSITIVE, RESULT_NEGATIVE],
                    weights=[self.alpha, 1 - self.alpha],
                )[0]

            # Communication Decision
            if investigation_type == INVESTIGATION_NOVEL:
                investigation_to_stage = {
                    INVESTIGATION_TYPE: INVESTIGATION_NOVEL,
                    INVESTIGATION_RESULT: investigation_result,
                }

                if investigation_result == RESULT_POSITIVE:
                    will_stage = self.random.choices(
                        population=[True, False],
                        weights=[self.c_N_pos, 1 - self.c_N_pos],
                    )[0]
                else:
                    will_stage = self.random.choices(
                        population=[True, False],
                        weights=[self.c_N_neg, 1 - self.c_N_neg],
                    )[0]

                if will_stage:
                    self.staged_investigation = investigation_to_stage
            else:
                investigation_to_stage = {
                    INVESTIGATION_TYPE: INVESTIGATION_REPLICATION,
                    INVESTIGATION_RESULT: investigation_result,
                    INVESTIGATION_TARGET_HYPOTHESIS: target_hypothesis_idx,
                }

                if investigation_result == RESULT_POSITIVE:
                    will_stage = self.random.choices(
                        population=[True, False],
                        weights=[self.c_R_pos, 1 - self.c_R_pos],
                    )[0]
                else:
                    will_stage = self.random.choices(
                        population=[True, False],
                        weights=[self.c_R_neg, 1 - self.c_R_neg],
                    )[0]

                if will_stage:
                    self.staged_investigation = investigation_to_stage
        else:
            # Communication. First unpack the investigation which is
            # staged.
            investigation_type = self.staged_investigation[INVESTIGATION_TYPE]
            investigation_result = self.staged_investigation[
                INVESTIGATION_RESULT
            ]

            if investigation_type == INVESTIGATION_REPLICATION:
                target_hypothesis_idx = self.staged_investigation[
                    INVESTIGATION_TARGET_HYPOTHESIS
                ]

            # Determine the probability that this investigation will be
            # published
            if investigation_type == INVESTIGATION_NOVEL:
                if investigation_result == RESULT_POSITIVE:
                    j_0 = self.model.j_0_N_pos
                else:
                    j_0 = self.model.j_0_N_neg
            else:
                if investigation_result == RESULT_POSITIVE:
                    j_0 = self.model.j_0_R_pos
                else:
                    j_0 = self.model.j_0_R_neg

            j = life_cycle_helpers.j(j_0, self.model.eta_j, self.w)

            # Publish (or don't)
            will_publish = self.random.choices(
                population=[True, False], weights=[j, 1 - j]
            )[0]

            if will_publish:
                self.published_investigations.append(self.staged_investigation)

            # Unstage the investigation
            self.staged_investigation = None


class SrsModel(Model):
    """The simulated SRS model.

    Attributes:
        a: Number of agents in the model.
        d: Number of agents sampled for Expansion and Retirement.
        b: Probability for novel hypotheses being true.
        j_0_N_pos: Base probability for positive novel results being
            published.
        j_0_N_neg: Base probability for negative novel results being
            published.
        j_0_R_pos: Base probability for positive replication results
            being published.
        j_0_R_neg: Base probability for negative replication results
            being published.
        v_N_pos: Payoff for publishing a novel hypothesis with a
            positive result.
        v_N_neg: Payoff for publishing a novel hypothesis with a
            negative result.
        v_0_R_pos: Base payoff for publishing a replication with a
            positive result.
        v_0_R_neg: Base payoff for publishing a replication with a
            negative result.
        v_RS: Payoff for having an original hypothesis successfully
            replicated.
        v_RF: Payoff for having an original hypothesis unsuccessfully
            replicated.
        eta_s: Influence of rigour on Productivity Checks.
        eta_r: Influence of tallies on replication payoffs.
        eta_j: Influence of reputation on publication probabilities.
        eta_N_pos: Influence of reputation on publishing positive novel
            results.
        eta_N_neg: Influence of reputation on publishing negative novel
            results.
        eta_R_pos: Influence of reputation on publishing positive
            replication results.
        eta_R_neg: Influence of reputation on publishing negative
            replication results.
        sigma_t: Standard deviation for targeted tally replication.
        sigma_c_N_pos: Standard deviation for c_N_pos mutation
            magnitude.
        sigma_c_N_neg: Standard deviation for c_N_neg mutation
            magnitude.
        sigma_c_R_pos: Standard deviation for c_R_pos mutation
            magnitude.
        sigma_c_R_neg: Standard deviation for c_R_neg mutation
            magnitude.
        agent_map: A dictionary which has agent IDs as keys and the
            corresponding Agent instance as values.
        hypothesis_manager: An instance of the hypothesis manager class.
        published_investigations: An array containing investigations
            published during a time step.
        scheduler: A scheduler instance that determines in which order
            agents act.
    """

    def __init__(
        self,
        a: int,
        d: int,
        b: float,
        j_0_N_pos: float,
        j_0_N_neg: float,
        j_0_R_pos: float,
        j_0_R_neg: float,
        v_N_pos: float,
        v_N_neg: float,
        v_0_R_pos: float,
        v_0_R_neg: float,
        v_RS: float,
        v_RF: float,
        eta_s: float,
        eta_r: float,
        eta_j: float,
        eta_N_pos: float,
        eta_N_neg: float,
        eta_R_pos: float,
        eta_R_neg: float,
        sigma_t: float,
        sigma_c_N_pos: float,
        sigma_c_N_neg: float,
        sigma_c_R_pos: float,
        sigma_c_R_neg: float,
        scheduler: Optional[BaseScheduler] = RandomActivation,
    ) -> None:
        """Initialize the SRS model."""
        super().__init__()

        # Set model parameters
        self.a = a
        self.d = d
        self.b = b
        self.j_0_N_pos = j_0_N_pos
        self.j_0_N_neg = j_0_N_neg
        self.j_0_R_pos = j_0_R_pos
        self.j_0_R_neg = j_0_R_neg
        self.v_N_pos = v_N_pos
        self.v_N_neg = v_N_neg
        self.v_0_R_pos = v_0_R_pos
        self.v_0_R_neg = v_0_R_neg
        self.v_RS = v_RS
        self.v_RF = v_RF
        self.eta_s = eta_s
        self.eta_r = eta_r
        self.eta_j = eta_j
        self.eta_N_pos = eta_N_pos
        self.eta_N_neg = eta_N_neg
        self.eta_R_pos = eta_R_pos
        self.eta_R_neg = eta_R_neg
        self.sigma_t = sigma_t
        self.sigma_c_N_pos = sigma_c_N_pos
        self.sigma_c_N_neg = sigma_c_N_neg
        self.sigma_c_R_pos = sigma_c_R_pos
        self.sigma_c_R_neg = sigma_c_R_neg

        # Set model scheduler
        self.scheduler = scheduler(self)

        # Initialize agent map
        self.agent_map = {}

        # Initialize hypothesis manager
        self.hypothesis_manager = HypothesisManager()

        # Initialize array for investigations published during a time
        # step
        self.published_investigations = []

    def initialize_agents(
        self,
        gamma: float,
        tau: float,
        r: float,
        c_N_pos: float,
        c_N_neg: float,
        c_R_pos: float,
        c_R_neg: float,
    ) -> None:
        """Initialize agents in the model.

        This approach to initializing agents constrains all initial
        agents to have the same research strategy parameters.
        Initializing agents with different research parameters isn't too
        difficult, but it's more complicated and so we currently do not
        provide functionality for this.

        Args:
            gamma: The power of the agents.
            tau: The rigour of the agents.
            r: The probability that an agent chooses a replication
                investigation when choosing a hypothesis to investigate.
            c_N_pos: The probability an agent choose to publish a
                positive novel result.
            c_N_neg: The probability an agent agent chooses to publish a
                negative novel result.
            c_R_pos: The probability an agent chooses to publish a
                positive replication result.
            c_R_neg: The probability an agent chooses to publish a
                negative replication result.
        """
        for _ in range(self.a):
            # Create an agent
            _id = self.next_id()
            agent = Agent(
                unique_id=_id,
                model=self,
                gamma=gamma,
                tau=tau,
                r=r,
                c_N_pos=c_N_pos,
                c_N_neg=c_N_neg,
                c_R_pos=c_R_pos,
                c_R_neg=c_R_neg,
            )

            # Add agent to scheduler and agent map
            self.scheduler.add(agent)
            self.agent_map[_id] = agent

    def step(self) -> None:
        """Iterate through all agent actions for one time step."""
        # Perform the Science stage
        self.scheduler.step()

        # Handle newly published investigations and payoffs

        # Perform the Expansion stage

        # Perform the Retirement stage
