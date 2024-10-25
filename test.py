import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

fig, ax = plt.subplots()
line, = ax.plot([])

def init():
    line.set_data([], [])
    return line,

def update(frame):
    line.set_data(np.linspace(0, 2, 1000), np.sin(2 * np.pi * (np.linspace(0, 2, 1000) - 0.01 * frame)))
    return line,

ani = animation.FuncAnimation(fig, update, frames=100, init_func=init, blit=True)
ani.save('test.mp4', writer='ffmpeg')
