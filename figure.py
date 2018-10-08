import math
import random as rand
import numpy as np
import matplotlib.pyplot as plt

class PositionFigure():
    def __init__(self, time_steps):
        self.figure_num = time_steps
        self.iter = 0
        self.column_num = 5
        self.row_num = math.ceil(self.figure_num / self.column_num)

    def set_sub_plot(self):
        self.iter += 1
        plt.subplot(self.row_num, self.column_num, self.iter)
        plt.xlim(-4.5, 4.5)
        plt.ylim(-3., 3.)

    def set_robot_plot(self,x,y,z):
        x_ = math.cos(math.asin(z))
        y_ = math.sin(math.asin(z))
        plt.quiver(x,y,x_,y_,color="b")

    def set_enemy_plot(self,x,y,z):
        x_ = math.cos(math.asin(z))
        y_ = math.sin(math.asin(z))
        plt.quiver(x,y,x_,y_,color="r")

    def show_figure(self):
        plt.show()

if __name__ == "__main__":
    for i in range(10):
        plt.subplot(2,5,i+1)
        for j in range(10):
            x = rand.uniform(0.,1.)
            y = rand.uniform(0.,1.)
            z = rand.uniform(0.,1.)
            x_ = math.cos(math.atan(z))
            y_ = math.sin(math.atan(z))
            plt.quiver(x,y,x_,y_,color="b")

    plt.show()



