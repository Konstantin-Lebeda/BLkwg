from itertools import count
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
matplotlib.use('TKAgg')

plt.style.use('fivethirtyeight')

x_vals = []
y_vals = []

index = count()

def animate(i):
    data = pd.read_csv('data.csv')
    Rex = data['Rex']
    Cf = data['Cf']
    Cf_analit = data['Cf_analit']
    Cf_analit_turb = data['Cf_analit_turb']
    plt.cla()

    plt.plot(Rex, Cf, marker = 'o', markersize = 2, linewidth = 1, color='r')
    plt.plot(Rex, Cf_analit, linewidth = 1, color='black')
    plt.plot(Rex, Cf_analit_turb, linewidth = 1, color='black')
    plt.ylabel('Cf/2')
    plt.xscale('log')
    plt.yscale('log')

    plt.tight_layout()

ani = FuncAnimation(plt.gcf(), animate, interval=1)

plt.tight_layout()
plt.show()