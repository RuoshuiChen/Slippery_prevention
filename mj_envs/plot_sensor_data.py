import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("sensor.csv",header=None)
fig, axs = plt.subplots(5,sharex=True)
fig.suptitle('sensor data')
for i in range(5):
    axs[i].plot(df[i])
plt.savefig("sensor data")