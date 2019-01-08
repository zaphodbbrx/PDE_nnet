from itertools import permutations
from random import shuffle

import tensorflow as tf
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt


class BatchGenerator:
    def __init__(self, n_samples, batch_size):
        self.n_samples = n_samples
        self.idx = list(range(n_samples))
        shuffle(self.idx)
        self.current = 0
        self.batch_size = batch_size
        self.iter = permutations(self.idx, 2)

    def __len__(self):
        return int(self.n_samples * (self.n_samples-1) / self.batch_size)

    def __iter__(self):
        return self

    def __next__(self):
        idx, idt = [], []
        for i in range(self.batch_size):
            x, t = next(self.iter)
            idx.append(x)
            idt.append(t)
        return idx, idt


class NeuralNetwork():

    def __init__(self,
                 D: float,
                 C0: float,
                 intial_condition,
                 x_range: tuple,
                 t_range: tuple,
                 grid_size: int,
                 learning_rate=0.001,
                 n_epochs=42,
                 batch_size=512,
                 n_hidden_1=1024,
                 num_input=2,
                 num_classes=1,
                 ):

        self.D = D
        self.C0 = C0
        self.L = x_range[1]
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.weights = {
            'h1': tf.Variable(tf.random_normal([num_input, n_hidden_1])),
            'h2': tf.Variable(tf.random_normal([n_hidden_1, 512])),
            'out': tf.Variable(tf.random_normal([512, num_classes]))
        }
        self.biases = {
            'b1': tf.Variable(tf.random_normal([n_hidden_1])),
            'b2': tf.Variable(tf.random_normal([512])),
            'out': tf.Variable(tf.random_normal([num_classes]))
        }
        self.initial_condition = intial_condition
        self.X = tf.placeholder("float", [None, 1])
        self.T = tf.placeholder("float", [None, 1])
        self.X_init = tf.placeholder("float", [None, 1])
        self.x = np.linspace(start=x_range[0], stop=x_range[1], num=grid_size).reshape(-1, 1)
        self.t = np.linspace(start=t_range[0], stop=t_range[1], num=grid_size).reshape(-1, 1)
        self.loss_op, self.train_op, self.net = self.build_graph()

    def build_graph(self):
        zero = self.X * 0.0
        L = self.X * 0.0+self.L
        net = self.neural_net(self.X, self.T)
        net_dt = tf.concat(tf.gradients(xs=[self.T], ys=net), 1)
        net_dx1 = tf.concat(tf.gradients(xs=[self.X], ys=net), 1)
        net_dx2 = tf.concat(tf.gradients(xs=[self.X], ys=net_dx1), 1)
        net_dx0 = tf.concat(tf.gradients(xs=[zero], ys=self.neural_net(zero, self.T)), 1)
        net_dxl = tf.concat(tf.gradients(xs=[L], ys=self.neural_net(L, self.T)), 1)

        loss_op = tf.square(self.D * net_dx2 - net_dt) + tf.square(net_dx0 * self.D - self.C0) + tf.square(
            self.neural_net(self.X, zero)-self.X_init)#+tf.square(net_dxl)

        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        train_op = optimizer.minimize(loss_op)

        return loss_op, train_op, net

    def neural_net(self, x, t, training=True):
        netin = tf.concat([x, t], 1)
        layer_1 = tf.add(tf.matmul(netin, self.weights['h1']), self.biases['b1'])
        layer_1 = tf.nn.sigmoid(layer_1)
        layer_2 = tf.add(tf.matmul(layer_1, self.weights['h2']), self.biases['b2'])
        layer_2 = tf.nn.sigmoid(layer_2)
        out_layer = tf.matmul(layer_2, self.weights['out']) + self.biases['out']
        out_layer = tf.nn.sigmoid(out_layer)
        return out_layer


    def solve_pde(self):
        init = tf.global_variables_initializer()
        losses = []
        saver = tf.train.Saver()
        with tf.Session() as sess:
            sess.run(init)
            for epoch in range(self.n_epochs):
                batch_generator = BatchGenerator(n_samples=self.x.shape[0], batch_size=self.batch_size)
                pbar = tqdm(batch_generator,desc=f'epoch: {epoch}')
                batch_losses = []
                for batch_idx, batch_idt in pbar:
                    batch_x = self.x[batch_idx]
                    batch_t = self.t[batch_idt]
                    batch_ic = self.initial_condition[batch_idx]
                    sess.run(self.train_op, feed_dict={self.X: batch_x, self.T: batch_t, self.X_init: batch_ic})
                    loss = sess.run(self.loss_op, feed_dict={self.X: batch_x, self.T: batch_t, self.X_init: batch_ic})
                    batch_losses.append(loss)
                    pbar.set_description(f'epoch: {epoch} loss: {np.mean(batch_losses):.4e}')
                losses.append(loss)
            saver.save(sess, './solution')

    def compute_profile(self, time, model_path='./solution'):
        init = tf.global_variables_initializer()
        saver = tf.train.Saver()
        with tf.Session() as sess:
            sess.run(init)
            saver.restore(sess, model_path)
            x_in = self.x.reshape(-1, 1)
            t_in = time * np.ones(self.x.shape[0]).reshape(-1, 1)
            profile = sess.run(self.net, feed_dict={self.X: x_in, self.T: t_in})
        return profile

if __name__=='__main__':
    C0 = -0.415
    D = 6.25
    L = 1000
    x_range = (0, L)
    t_range = (0, 1000)
    grid_size = 200
    net = NeuralNetwork(
        D=D,
        C0=C0,
        intial_condition=np.zeros((grid_size, 1)),
        x_range=x_range,
        t_range=t_range,
        grid_size=grid_size,
        n_epochs=100,
        learning_rate=1e-4
    )
    net.solve_pde()
    profile = net.compute_profile(500.0)
    x = np.linspace(start=x_range[0], stop=x_range[1], num=grid_size).reshape(-1, 1)
    times = np.linspace(0, 1000, 10).tolist()
    legend = []
    for time in times:
        profile = net.compute_profile(time)
        plt.plot(x, profile)
        legend.append(f"t={time}")
    plt.legend(legend)
    plt.show()


    #plt.plot(x, profile)
    #plt.show()
    C0 *= -1
    profiles = [profile]
    for i in range(0):
        net = NeuralNetwork(
            D=D,
            C0=C0,
            intial_condition=profile,
            x_range=x_range,
            t_range=t_range,
            grid_size=grid_size,
            n_epochs=42,
            learning_rate=1e-4
        )
        net.solve_pde()
        profile = net.compute_profile(5.0)
        x = np.linspace(start=x_range[0], stop=x_range[1], num=grid_size).reshape(-1, 1)
        profiles.append(profile)
        plt.plot(x, profile)
        plt.show()
        C0 *= -1
    profiles = np.hstack(profiles)
    np.save('profiles.npy',profiles)