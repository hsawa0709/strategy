import math
import numpy as np
import matplotlib.pyplot as plt
import entity
import figure
import sys

#import physics as phy

class WorldModel():
    def __init__(self, robot_num):
        self.robot_num = robot_num
        self.robot = [entity.Robot() for i in range(robot_num)]
        self.enemy = [entity.Robot() for i in range(robot_num)]
        self.ball = entity.Ball()
        self.rate = 100 #ms

    def get_information_from_vision(self, x):
            self.robot[0].set_current_position(x=-3.0, y=0., theta=0.)
            self.robot[1].set_current_position(x=-2.2, y=0.5, theta=0.)
            self.robot[2].set_current_position(x=-2.2, y=-0.5, theta=0.)
            self.robot[3].set_current_position(x=-1.5, y=2., theta=0.)
            self.robot[4].set_current_position(x=-1.5, y=-2., theta=0.)
            self.robot[5].set_current_position(x=-0.5, y=1., theta=0.)
            self.robot[6].set_current_position(x=-0.5, y=-1., theta=0.)
            self.robot[7].set_current_position(x=-0.5, y=0., theta=0.)
            self.enemy[0].set_current_position(x=3.0, y=0., theta=math.pi)
            self.enemy[1].set_current_position(x=2.2, y=0.5, theta=math.pi)
            self.enemy[2].set_current_position(x=2.2, y=-0.5, theta=math.pi)
            self.enemy[3].set_current_position(x=1.5, y=2., theta=math.pi)
            self.enemy[4].set_current_position(x=1.5, y=-2., theta=math.pi)
            self.enemy[5].set_current_position(x=0.5, y=1., theta=math.pi)
            self.enemy[6].set_current_position(x=0.5, y=-1., theta=math.pi)
            self.enemy[7].set_current_position(x=0.5, y=0., theta=math.pi)

    def set_init_positions(self, char="robot"):
        if char == "robot":
            self.robot[0].set_future_position(x=-3.0, y=0., theta=0.)
            self.robot[1].set_future_position(x=-2.2, y=0.5, theta=0.)
            self.robot[2].set_future_position(x=-2.2, y=-0.5, theta=0.)
            self.robot[3].set_future_position(x=-1.5, y=2., theta=0.)
            self.robot[4].set_future_position(x=-1.5, y=-2., theta=0.)
            self.robot[5].set_future_position(x=-0.5, y=1., theta=0.)
            self.robot[6].set_future_position(x=-0.5, y=-1., theta=0.)
            self.robot[7].set_future_position(x=-0.5, y=0., theta=0.)
        elif char == "enemy":
            self.enemy[0].set_future_position(x=3.0, y=0., theta=math.pi)
            self.enemy[1].set_future_position(x=2.2, y=0.5, theta=math.pi)
            self.enemy[2].set_future_position(x=2.2, y=-0.5, theta=math.pi)
            self.enemy[3].set_future_position(x=1.5, y=2., theta=math.pi)
            self.enemy[4].set_future_position(x=1.5, y=-2., theta=math.pi)
            self.enemy[5].set_future_position(x=0.5, y=1., theta=math.pi)
            self.enemy[6].set_future_position(x=0.5, y=-1., theta=math.pi)
            self.enemy[7].set_future_position(x=0.5, y=0., theta=math.pi)
        else:
            print("Please put robot or enemy")

    def set_first_positions(self):
        self.robot[0].set_future_position(x=-4.0, y=0., theta=0.)
        self.robot[1].set_future_position(x=-2.5, y=0.5, theta=0.)
        self.robot[2].set_future_position(x=-2.5, y=-0.5, theta=0.)
        self.robot[3].set_future_position(x=-1.0, y=2., theta=0.)
        self.robot[4].set_future_position(x=-1.0, y=-2., theta=0.)
        self.robot[5].set_future_position(x=1.0, y=1., theta=0.)
        self.robot[6].set_future_position(x=1.0, y=-1., theta=0.)
        self.robot[7].set_future_position(x=2.5, y=0., theta=0.)

    def set_target_positions(self, position):
        self.robot[0].set_future_position(x=position[0][0], y=position[0][1], theta=position[0][2])
        self.robot[1].set_future_position(x=position[1][0], y=position[1][1], theta=position[1][2])
        self.robot[2].set_future_position(x=position[2][0], y=position[2][1], theta=position[2][2])
        self.robot[3].set_future_position(x=position[3][0], y=position[3][1], theta=position[3][2])
        self.robot[4].set_future_position(x=position[4][0], y=position[4][1], theta=position[4][2])
        self.robot[5].set_future_position(x=position[5][0], y=position[5][1], theta=position[5][2])
        self.robot[6].set_future_position(x=position[6][0], y=position[6][1], theta=position[6][2])
        self.robot[7].set_future_position(x=position[7][0], y=position[7][1], theta=position[7][2])

    def show_positions_figure(self):
        #graph = figure.PositionFigure()
        for robot in self.robot:
            x, y, theta = robot.get_current_position()
            #graph.set_robot_plot(x, y, z)
            figure = plt.plot(x,y,"ro")
            print(x,y,theta)
            plt.setp(figure, markersize=10)

        for enemy in self.enemy:
            x, y, theta = enemy.get_current_position()
            #graph.set_robot_plot(x, y, z)
            figure = plt.plot(x,y,"bo")
            print(x,y,theta)
            plt.setp(figure, markersize=10)

        plt.xlim(-4.5, 4.5)
        plt.ylim(-3., 3.)
        plt.show()


    def set_positions_figure(self, plot_num=1):
        if plot_num == 1:
            for robot in self.robot:
                x, y, theta = robot.get_current_position()
                graph = plt.plot(x, y,"ro")
                plt.setp(graph, markersize=10)
            plt.xlim(-4.5, 4.5)
            plt.ylim(-3., 3.)
            plt.show()

    def robot_collision_detect(self):
        robot = self.robot + self.enemy
        for i in range(self.robot_num*2):
            position_1_x, position_1_y, z = robot[i].get_current_position()
            for j in range(self.robot_num*2-i-1):
                position_2_x, position_2_y, z = robot[j+i+1].get_current_position()
                if (position_1_x - position_2_x)**2 + (position_1_y - position_2_y)**2 < 0.18*0.18:
                    print("dangerous!!")
                    #sys.exit()
                    break
            else:
                continue
            break
        else:
            print("safety!!")

    def robot_move_moment(self):
        for i in range(self.robot_num):
            x, y, z = self.robot[i].get_future_position()
            self.robot[i].set_current_position(x, y, z)

    def robot_move_all_linear(self, time=0, time_steps=10):
        difference = np.array([[0.,0.,0.] for i in range(self.robot_num)])
        for i in range(self.robot_num):
            x_current, y_current, z_current = self.robot[i].get_current_position()
            x_future, y_future, z_future = self.robot[i].get_future_position()
            difference[i] = [x_future - x_current, y_future - y_current, z_future - z_current]

        for time_step in range(time_steps):
            for j in range(self.robot_num):
                v_x, v_y, v_z = difference[j] / time_steps
                x_current, y_current, z_current = self.robot[j].get_current_position()
                x_current += v_x
                y_current += v_y
                z_current += v_z
                self.robot[j].set_current_position(x_current, y_current, z_current)
            self.show_positions_figure()
            self.robot_collision_detect()
            # under construct

    def robot_enemy_move_all_linear(self, time=0, time_steps=10):
        difference = np.array([[0.,0.,0.] for i in range(self.robot_num)])
        difference_enemy = np.array([[0.,0.,0.] for i in range(self.robot_num)])
        for i in range(self.robot_num):
            x_current, y_current, z_current = self.robot[i].get_current_position()
            x_future, y_future, z_future = self.robot[i].get_future_position()
            difference[i] = [x_future - x_current, y_future - y_current, z_future - z_current]
            x_current_enemy, y_current_enemy, z_current_enemy = self.enemy[i].get_current_position()
            x_future_enemy, y_future_enemy, z_future_enemy = self.enemy[i].get_future_position()
            difference_enemy[i] = [x_future_enemy - x_current_enemy, y_future_enemy - y_current_enemy, z_future_enemy - z_current_enemy]

        for time_step in range(time_steps):
            for j in range(self.robot_num):
                v_x, v_y, v_z = difference[j] / time_steps
                x_current, y_current, z_current = self.robot[j].get_current_position()
                x_current += v_x
                y_current += v_y
                z_current += v_z
                self.robot[j].set_current_position(x_current, y_current, z_current)
                v_x_enemy, v_y_enemy, v_z_enemy = difference_enemy[j] / time_steps
                x_current_enemy, y_current_enemy, z_current_enemy = self.enemy[j].get_current_position()
                x_current_enemy += v_x_enemy
                y_current_enemy += v_y_enemy
                z_current_enemy += v_z_enemy
                self.enemy[j].set_current_position(x_current_enemy, y_current_enemy, z_current_enemy)
            self.robot_collision_detect()
            #self.show_positions_figure()



    def robot_move_individual_linear(self, rate, num):
        None


"""
class SetPositions():
    def __init__(self):
        self.robot_odom = np.zeros([8,3])
        self.enemy_odom = np.zeros([8,3])
        self.ball_odom = np.zeros([3])

    def set_positions(self):
        None

"""



if __name__ == "__main__":
    a = WorldModel(robot_num=8)
    a.set_init_positions(char="robot")
    a.set_init_positions(char="enemy")
    #a.show_positions_figure()
    a.robot_enemy_move_all_linear()
    a.robot_collision_detect()
    a.set_first_positions()
    a.robot_move_all_linear()
    #a.enemy_move_all_linear()
    #a.show_positions_figure()
    #b = figure.PositionFigure()
    """
    a.robot_move_moment()
    a.show_positions_figure()
    a.robot_collision_detect()
    a.set_first_positions()
    a.robot_move_moment()
    a.show_positions_figure()
    a.robot_collision_detect()
    """