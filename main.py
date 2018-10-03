import numpy as np
import matplotlib.pyplot as plt
import entity

class set_first_position():
    def __init__(self, num_robots):
        """
        robot_odom = [[robot_0_x, robot_0_y],
                        [robot_1_x, robot_1_y],
                        ....
                        [robot_7_x, robot_7_y]]
        """
        self.num_robots = num_robots
        self.robot = [entity.robot() for i in range(num_robots)]
        print(self.robot)

    def set_init_position(self):
        for i in range(num_ro)
        self.robot[i] = [[0.5, 3.],[]]

    def set_first_position(self):
        self.robot_odom = [[-4.0, 0.],[-2.5, 0.5],[-2.5, -0.5], [-1., 2.], [-1., -2.], [1., 1.],[1., -1.], [2.5, 0.]]

    def show_position(self):
        for odom in self.robot_odom:
            print(odom[0])
            print(odom[1])
            graph = plt.plot(odom[0], odom[1],"ro")
            plt.setp(graph, markersize=10)
        plt.xlim(-4.5, 4.5)
        plt.ylim(-3., 3.)
        plt.show()


class set_position():
    def __init__(self):
        self.robot_odom = np.zeros([8,3])
        self.enemy_odom = np.zeros([8,3])
        self.ball_odom = np.zeros([3])

    def set_position(self):
        None





if __name__ == "__main__":
    a = set_first_position(num_robots=8)
    a.set_first_position()
    a.show_position()