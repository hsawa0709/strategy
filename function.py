import numpy as np
import math
import scipy.stats as stats
from matplotlib import cm
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt

class Function():
    def __init__(self, robot_num):
        self.grid_size = 10
        self.field_x_size = 12000
        self.field_y_size = 9000
        self.robot_size = 90
        self.robot_num = robot_num

        self.field_x_min = - self.field_x_size / 2
        self.field_x_max = self.field_x_size / 2
        self.field_y_min = - self.field_y_size / 2
        self.field_y_max = self.field_y_size / 2

        self.robot_mu = np.array([[0, 0] for i in range(self.robot_num)])
        self.enemy_mu = np.array([[0, 0] for i in range(self.robot_num)])

        sigma = [[self.robot_size,0],
                    [0, self.robot_size]]
        self.robot_sigma = np.array([sigma for i in range(self.robot_num)])
        self.enemy_sigma = np.array([sigma for i in range(self.robot_num)])

        x = np.arange(self.field_x_min, self.field_x_max, self.grid_size)
        y = np.arange(self.field_y_min, self.field_y_max, self.grid_size)
        self.X, self.Y = np.meshgrid(x, y)


    def gaussian_2D(self, mu, sigma):
        X, Y = self.X, self.Y
        alpha = 1/2
        f = lambda X, Y, alpha: alpha * stats.multivariate_normal(mu, sigma).pdf([X, Y])
        Z = np.vectorize(f)(X, Y)
        return X, Y, Z


    def gaussian_mixture_2D(self, num_gauss, mu, sigma, alpha):
        f = None
        for i in range(num_gauss):
            #f += math.
            None


if __name__ == "__main__":
    func = Function(robot_num=8)
    fig = plt.figure()
    ax = fig.add_subplot(111, aspect='equal')
    mu = np.array([90.5,100.5])
    sigma = np.array([[func.robot_size,0],
                  [0,func.robot_size]])
    X, Y, Z = func.gaussian_2D(mu, sigma)
    #ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.coolwarm)
    ax.contourf(X,Y,Z)
    ax.set_xlim(func.field_x_min,func.field_x_max+1)
    ax.set_ylim(func.field_y_min,func.field_y_max+1)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('pdf')
    plt.show()


