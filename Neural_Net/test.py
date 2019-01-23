import matplotlib.pyplot as plt
import numpy as np

import plotter
import data_sampler
import network2

sampler = data_sampler.DataSampler()
data_plotter = plotter.Plotter(sampler)
x_tr, x_val, x_test, y_tr, y_val, y_test = sampler.draw_train_val_test_sample()

data_plotter.plot_sample(x_tr, y_tr)
# plt.show()

net = network2.Network([2, 30, 1])
net.train(
    np.array([x_tr, y_tr]),
    epochs=10,
    batch_size=10,
    eta=1.0,
    test_data=(x_test, y_test),
)
