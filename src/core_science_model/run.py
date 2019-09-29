from science_model import ScienceModel, TALLY, TRUTH

model = ScienceModel(
    n=100,
    r=0.5,
    b=0.5,
    alpha=0.5,
    beta=0.5,
    c_n_neg=0.5,
    c_r_pos=0.5,
    c_r_neg=0.5,
    initial_data=[{TRUTH: True, TALLY: 1}],
)

for i in range(200):
    model.step()

from pprint import pprint
pprint(model.tested_hypotheses)
