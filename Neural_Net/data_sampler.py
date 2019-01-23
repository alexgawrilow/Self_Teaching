import numpy as np


def linear_func(x, slope, intercept):
    """Linear function with slope and intercept."""
    return slope * x + intercept


class DataSampler:
    def __init__(self, n_data=10000):
        self.n_data = n_data
        """Number of data points to be drawn."""
        self.slope, self.intercept = np.random.rand(2)
        self.x_min = -20
        self.x_max = 20

    def draw_sample(self):
        """Draws x-values from uniform distribution between -10 and 10.
        Computes y-values using a linear function and adds gaussian noise.
        Returns a list of (x, y)-tuples."""
        x = np.random.uniform(self.x_min, self.x_max, size=self.n_data)
        y = linear_func(x, self.slope, self.intercept) + np.random.normal(
            size=self.n_data
        )

        return list(zip(x, y))

    def draw_train_val_test_sample(self, val_size=0.1, test_size=0.1):
        """Draws a sample and splits it into train, validation and test set."""
        sample = self.draw_sample()
        tr_index = int(len(sample) * (1 - val_size - test_size))
        val_index = tr_index + int(len(sample) * val_size)

        return sample[:tr_index], sample[tr_index:val_index], sample[val_index:]
