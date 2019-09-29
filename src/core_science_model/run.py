import matplotlib.pyplot as plt
from science_model import ScienceModel, TALLY, TRUE

# Run the model
# TODO use a config file or something to determine the parameters
model = ScienceModel(
    n=100,
    r=0.2,
    b=0.1,
    alpha=0.05,
    beta=0.2,
    c_n_neg=0.9,
    c_r_pos=0.9,
    c_r_neg=0.9,
    initial_data=[{TRUE: True, TALLY: 1}],
)

for i in range(200):
    model.step()


# Analyze results - first group tested hypotheses by tallies
hypothesis_summary = {}

for hypothesis in model.tested_hypotheses:
    if hypothesis[TALLY] in hypothesis_summary:
        hypothesis_summary[hypothesis[TALLY]]["total_count"] += 1

        if hypothesis[TRUE]:
            hypothesis_summary[hypothesis[TALLY]]["true_count"] += 1
    else:
        hypothesis_summary[hypothesis[TALLY]] = {
            "total_count": 1,
            "true_count": 1 if hypothesis[TRUE] else 0,
        }

tally_vals = list(sorted(hypothesis_summary.keys()))
tally_total_counts = [hypothesis_summary[t]["total_count"] for t in tally_vals]

plt.bar(tally_vals, tally_total_counts)
plt.show()

tally_percent_true_counts = [
    hypothesis_summary[t]["true_count"] / hypothesis_summary[t]["total_count"]
    for t in tally_vals
]

plt.bar(tally_vals, tally_percent_true_counts)
plt.show()
