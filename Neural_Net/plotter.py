import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import linregress
import data_sampler


class Plotter:
    def __init__(self, sampler):
        self.sampler = sampler
        _, self.ax = plt.subplots()
        self.update_line, = self.ax.plot([], [], color="orange", label="current model")

    def plot_sample(self, x, y):
        """Creates a scatter plot of x, y samples and a fitted straight line."""
        # create scatter plot of points
        self.ax.scatter(x, y, s=0.3, label="data points")

        # fit points to line
        slope, intercept, *_ = linregress(x, y)

        # create plot of fitted line
        x_points, y_points = self.create_plot_points(slope, intercept)
        self.ax.plot(
            x_points,
            y_points,
            color="red",
            label=" best model: f(x) = {:.2f}*x + {:.2f}".format(slope, intercept),
        )
        self.ax.legend()
        # plt.show()

    def update_plot(self, slope, intercept):
        """Updates line plot using new slope and intercept."""
        plt.pause(1e-17)
        x_points, y_points = self.create_plot_points(slope, intercept)
        self.update_line.set_xdata(x_points)
        self.update_line.set_ydata(y_points)
        plt.draw()

    def create_plot_points(self, slope, intercept):
        """Creates points necessary for plotting a line plot of linear function
        with slope and intercept."""
        x_points = np.array([self.sampler.x_min, self.sampler.x_max])
        y_points = data_sampler.linear_func(x_points, slope, intercept)
        return x_points, y_points
