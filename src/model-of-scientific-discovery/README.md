# Model of Scientific Discovery

This is the model of science proposed by Richard McElreath and Paul E.
Smaldino in their paper

McElreath R, Smaldino PE (2015) Replication, Communication, and the Population Dynamics of Scientific Discovery. PLoS ONE 10(8): e0136088. https://doi.org/10.1371/journal.pone.0136088

## How the model works

The model has `n` researchers/research groups (or other appropriate
organizational unit) that conduct and publish research within some
scientific field. We will refer to these researchers/research groups as
**agents** and the scientific field as the **field**.

Within the field we have a number of **published hypotheses**, which are
published results from agents that have investigated **novel
hypotheses**. Each published hypothesis has a **tally**. When a nove
result is published, its corresponding "published hypothesis" starts at a
tally of 1 if the result was positive and -1 if the result was negative.
When agents attempt to **replicate** a published hypothesis and publish
their results, they add 1 to its tally if the result was positive or
subtract 1 from its tally if the result was negative.

The model starts with one or more hypotheses with corresponding tallies
and proceeds in discrete time steps. Each time step, every agent
proceeds through three stages simultaneously:

#### Stage 1: choosing a hypothesis

The agent chooses whether to investigate a novel hypothesis or to
attempt to replicate a random published hypothesis; these actions are
chosen with probability `r` and `1 - r`, respectively.

If the agent chooses to investigate a novel hypothesis, it is true with
probability `b`.

#### Stage 2: investigating the hypothesis

A researcher produces a true/false positive/negative result with
probabilities given in the table below:

|| true | false|
| --- | --- | --- |
|**+** | `1 - beta` | `alpha`|
|**-** | `beta` | `1 - alpha`|

#### Stage 3: communicating the results

If the agent investigated a novel hypothesis, a positive result is
published with probability `c_n_pos` and a negative result is published
with probability `c_n_neg`.

If the agent attempted to replicate a published hypothesis, a positive
result is published with probability `c_r_pos` and a negative result is
published with probability `c_r_neg`.

## Parameters

In addition to the starting set of published hypotheses (and corresponding
tallies), we have a number of parameters, described as follows:

parameter | description | constraints
--------- | ----------- | -----------
n | number of agents | integral, n > 0
r | probability that an agent chooses to replicate a published hypothesis | 0 <= r <= 1
b | probability that a novel hypothesis is true | 0 <= b <= 1
alpha | type I error rate | 0 <= alpha <= 1
beta | type II error rate | 0 <= beta <= 1
c_n_pos | probability that a positive novel result is published | 0 <= c_n_pos <= 1
c_n_neg | probability that a negative novel result is published | 0 <= c_n_neg <= 1
c_r_pos | probability that a positive replication result is published | 0 <= c_r_pos <= 1
c_r_neg | probability that a negative replication result is published | 0 <= c_r_neg <= 1
