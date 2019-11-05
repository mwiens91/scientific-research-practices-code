"""Contains main simulation code for the SRS model."""

from typing import List, Optional
from mesa import Agent as MesaAgent, Model
from mesa.time import BaseScheduler, RandomActivation


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

        # Set the staged investigation
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
            self.scheduler.add(
                Agent(
                    unique_id=self.next_id(),
                    model=self,
                    gamma=gamma,
                    tau=tau,
                    r=r,
                    c_N_pos=c_N_pos,
                    c_N_neg=c_N_neg,
                    c_R_pos=c_R_pos,
                    c_R_neg=c_R_neg,
                )
            )

    def step(self) -> None:
        """Iterate through all agent actions for one time step."""
        self.scheduler.step()