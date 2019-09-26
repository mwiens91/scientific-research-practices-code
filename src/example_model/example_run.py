import matplotlib.pyplot as plt
from example_model import MoneyModel

model = MoneyModel(50)

for i in range(50000):
    model.step()

agent_wealths = [a.wealth for a in model.schedule.agents]
plt.hist(agent_wealths)

plt.show()
