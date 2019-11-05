"""Contains functions used in the life cycle of the model.

This module is sparse on documentation. To learn more about these
functions, consult the main paper.
"""

from math import log10

phi = lambda x: max(min(x, 1), 0)

p_s = lambda eta, tau: phi(1 - eta * log10(tau))

alpha = lambda gamma, tau: gamma / (1 + (1 - gamma) * tau)
beta = lambda gamma: 1 - gamma

v_R = lambda v_0, eta, s: max(v_0 - eta * abs(s), 0)

j = lambda j_0, eta, w: phi(j_0 + eta * w ** 3)
