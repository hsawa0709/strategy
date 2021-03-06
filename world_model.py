import math
import numpy as np
import matplotlib.pyplot as plt
import entity
import figure
import sys
import time
import threading
from sklearn.cluster import KMeans

#import physics as phy
"""
主にフィールドの情報の取得、物理的な衝突の確認をするコード
"""

class WorldModel():
    def __init__(self, robot_num, enemy_num):
        self.robot_num = robot_num
        self.enemy_num = enemy_num
        self.robot = [entity.Robot() for i in range(robot_num)]
        self.enemy = [entity.Robot() for i in range(enemy_num)]
        self.ball = entity.Ball()

        self.field_x_size = 12000.
        self.field_y_size = 9000.
        self.field_x_min = - self.field_x_size / 2.
        self.field_x_max = self.field_x_size / 2.
        self.field_y_min = - self.field_y_size / 2.
        self.field_y_max = self.field_y_size / 2.

        self.goal_y_size = 1200.
        self.goal_y_min = -self.goal_y_size / 2.
        self.goal_y_max = self.goal_y_size / 2.

        self.ball_dynamics_window = 10
        self.ball_dynamics_x = [0. for i in range(self.ball_dynamics_window)]
        self.ball_dynamics_y = [0. for i in range(self.ball_dynamics_window)]

        self.goal_keeper_x = self.field_x_min + 20

        # position
        if self.robot_num == 8:
            self.robot[0].position = "GK"
            self.robot[1].position = "LCB"
            self.robot[2].position = "RCB"
            self.robot[3].position = "LSB"
            self.robot[4].position = "RSB"
            self.robot[5].position = "LMF"
            self.robot[6].position = "RMF"
        elif self.robot_num == 4:
            self.robot[0].position = "GK"
            self.robot[1].position = "LCB"
            self.robot[2].position = "CCB"
            self.robot[3].position = "RCB"
        elif self.robot_num == 5:
            self.robot[0].position = "GK"
            self.robot[1].position = "LCB"
            self.robot[2].position = "CCB"
            self.robot[3].position = "RCB"
            self.robot[4].position = "CF"

        self.rate = 0.05 #s

        self.strategy = None

        self.referee = None

        self.which_has_a_ball = None
        self.attack_or_defence = None

        # weight for potential
        self.weight_enemy = 1.0
        self.weight_goal = 5.0
        self.delta = 0.1
        self.speed = self.robot[0].max_velocity

        self.counter = 0

    def get_information_from_vision(self, x):
        self.robot[0].set_current_position(x=-3000., y=0., theta=0.)
        self.robot[1].set_current_position(x=-2200., y=500., theta=0.)
        self.robot[2].set_current_position(x=-2200., y=-500., theta=0.)
        self.robot[3].set_current_position(x=-1500., y=2000., theta=0.)
        self.robot[4].set_current_position(x=-1500., y=-2000., theta=0.)
        self.robot[5].set_current_position(x=-500., y=1000., theta=0.)
        self.robot[6].set_current_position(x=-500., y=-1000., theta=0.)
        self.robot[7].set_current_position(x=-500., y=0., theta=0.)
        self.enemy[0].set_current_position(x=3000., y=0., theta=math.pi)
        self.enemy[1].set_current_position(x=2200., y=500., theta=math.pi)
        self.enemy[2].set_current_position(x=2200., y=-500., theta=math.pi)
        self.enemy[3].set_current_position(x=1500., y=2000., theta=math.pi)
        self.enemy[4].set_current_position(x=1500., y=-2000., theta=math.pi)
        self.enemy[5].set_current_position(x=500., y=1000., theta=math.pi)
        self.enemy[6].set_current_position(x=500., y=-1000., theta=math.pi)
        self.enemy[7].set_current_position(x=500., y=0., theta=math.pi)
        self.ball.set_current_position(x=100., y=100., theta=0.)

    def set_init_positions(self, char="robot"):
        if char == "robot":
            self.robot[0].set_future_position(x=-6000., y=0., theta=0.)
            self.robot[1].set_future_position(x=-4500., y=1000., theta=0.)
            self.robot[2].set_future_position(x=-4500., y=-1000., theta=0.)
            self.robot[3].set_future_position(x=-3000., y=3500., theta=0.)
            self.robot[4].set_future_position(x=-3000., y=-3500., theta=0.)
            self.robot[5].set_future_position(x=-1000., y=2000., theta=0.)
            self.robot[6].set_future_position(x=-1000., y=-2000., theta=0.)
            self.robot[7].set_future_position(x=-500., y=0., theta=0.)
        elif char == "enemy":
            self.enemy[0].set_future_position(x=6000., y=0., theta=math.pi)
            self.enemy[1].set_future_position(x=4500., y=1000., theta=math.pi)
            self.enemy[2].set_future_position(x=4500., y=-1000., theta=math.pi)
            self.enemy[3].set_future_position(x=3000., y=3500., theta=math.pi)
            self.enemy[4].set_future_position(x=3000., y=-3500., theta=math.pi)
            self.enemy[5].set_future_position(x=1000., y=2000., theta=math.pi)
            self.enemy[6].set_future_position(x=1000., y=-2000., theta=math.pi)
            self.enemy[7].set_future_position(x=500., y=0., theta=math.pi)
        elif char == "ball":
            self.ball.set_current_position(x=4500., y=1000., theta=0.)
        else:
            print("Please put robot, enemy or ball")

    def set_first_positions(self):
        self.robot[0].set_future_position(x=-4000., y=0., theta=0.)
        self.robot[1].set_future_position(x=-2500., y=500., theta=0.)
        self.robot[2].set_future_position(x=-2500., y=-500., theta=0.)
        self.robot[3].set_future_position(x=-1000., y=2000., theta=0.)
        self.robot[4].set_future_position(x=-1000., y=-2000., theta=0.)
        self.robot[5].set_future_position(x=1000., y=1000., theta=0.)
        self.robot[6].set_future_position(x=1000., y=-1000., theta=0.)
        self.robot[7].set_future_position(x=2500., y=0., theta=0.)

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

        plt.xlim(-self.field_x_size/2, self.field_x_size/2)
        plt.ylim(-self.field_y_size/2, self.field_y_size/2)
        plt.show()

    """
    def set_positions_figure(self, plot_num=1):
        if plot_num == 1:
            for robot in self.robot:
                x, y, theta = robot.get_current_position()
                graph = plt.plot(x, y,"ro")
                plt.setp(graph, markersize=10)
            plt.xlim(-self.field_x_size/2, self.field_x_size/2)
            plt.ylim(-self.field_y_size/2, self.field_y_size/2)
            plt.show()
    """

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
        difference_enemy = np.array([[0.,0.,0.] for i in range(self.enemy_num)])
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


    def create_mesh(self):
        self.mesh_size = 50
        self.mesh_x_num = int(self.field_x_size / (2*self.mesh_size)) + 1
        self.mesh_y_num = int(self.field_y_size / self.mesh_size) + 1
        self.mesh_x = [ i * self.mesh_size for i in range(self.mesh_x_num)]
        self.mesh_y = [- self.field_y_size / 2 + i * self.mesh_size for i in range(self.mesh_y_num)]
        #print(self.mesh_x)
        #print(self.mesh_y)
        self.mesh_xy = []
        for x in self.mesh_x:
            for y in self.mesh_y:
                self.mesh_xy.append([x, y])
        return self.mesh_xy

    def enemy_density(self):
        mesh_xy = np.array(self.create_mesh())
        distance = np.zeros([self.mesh_x_num * self.mesh_y_num])
        enemy_position = []
        for i in range(self.robot_num-1):
            x, y, _ = self.enemy[i+1].get_current_position()
            #print(x,y)
            distance_ = (mesh_xy - np.array([x,y]))**2
            distance += np.sqrt(distance_[:,0] + distance_[:,1])
        mean_distance = distance / 7
        return mean_distance.reshape([self.mesh_x_num, self.mesh_y_num])

    def space_clustering(self):
        self.threshold = 3000
        self.n_cluster = 3
        mean_distance = self.enemy_density().reshape([self.mesh_x_num * self.mesh_y_num])
        mesh_xy = np.array(self.mesh_xy)
        plots = mesh_xy[mean_distance > self.threshold]
        print(plots.shape)
        cls = KMeans(self.n_cluster)
        pred = cls.fit_predict(plots)

        #"""
        for i in range(self.n_cluster):
            labels = plots[pred == i]
            plt.scatter(labels[:, 0], labels[:, 1])
        #"""
        centers = cls.cluster_centers_

        plt.scatter(centers[:, 0], centers[:, 1], s=100,
                facecolors='none', edgecolors='black')
        plt.xlim(-self.field_x_size/2, self.field_x_size/2)
        plt.ylim(-self.field_y_size/2, self.field_y_size/2)
        plt.show()

    def who_has_a_ball(self):
        x_ball, y_ball = self.ball.get_current_parameter_xy()
        flag = 0
        for i in range(self.robot_num):
            x_robot, y_robot = self.robot[i].get_current_parameter_xy()
            if (x_ball - x_robot)**2 + (y_ball - y_robot)**2 < self.robot[i].size_r**2:
                self.robot[i].ball = True
                flag = flag + 1
            else:
                self.robot[i].ball = False
        for i in range(self.robot_num):
            x_robot, y_robot = self.enemy[i].get_current_parameter_xy()
            if (x_ball - x_robot)**2 + (y_ball - y_robot)**2 < self.enemy[i].size_r**2:
                self.enemy[i].ball = True
                flag = flag - 1
            else:
                self.enemy[i].ball = False
        if flag == 1:
            self.which_has_a_ball = "robots"
        elif flag == -1:
            self.which_has_a_ball = "enemy"
        else:
            self.which_has_a_ball = "free"

    def update_from_vison(self, x):
        self.get_information_from_vision(x)
        self.who_has_a_ball()
        # under construction

    def attack_or_defence(self):
        if self.which_has_a_ball == "robots":
            self.attack_or_defence = "attack"
        else:
            self.attack_or_defence = "defence"

    def decide_attack_strategy(self):
        if self.attack_or_defence == "attack":
            #self.space_clustering(
            print("attack!")
        else:
            print("defence!")
            #None

    def get_potential(self, x, y, x_goal, y_goal):
        U = 0.
        for i in range(self.enemy_num):
            x_enemy, y_enemy, _ = self.enemy[i].get_current_position()
            U +=  (self.weight_enemy *  1.0) / math.sqrt((x - x_enemy + 0.00001)*(x - x_enemy + 0.00001) + (y - y_enemy + 0.00001)*(y - y_enemy + 0.00001))
        U += (self.weight_goal * -1.0) / math.sqrt((x - x_goal + 0.00001)*(x - x_goal + 0.00001) + (y - y_goal + 0.00001)*(y - y_goal + 0.00001));
        return U

    def move_by_potential_method(self):
        v_all = np.zeros([self.robot_num, 2])
        for i in range(self.robot_num):
            x_goal, y_goal, _ = self.robot[i].get_future_position()
            x_current, y_current, _ = self.robot[i].get_current_position()
            vx = - (self.get_potential(x_current + self.delta, y_current, x_goal, y_goal) - self.get_potential(x_current, y_current, x_goal, y_goal)) / self.delta
            vy = - (self.get_potential(x_current, y_current + self.delta, x_goal, y_goal) - self.get_potential(x_current, y_current, x_goal, y_goal)) / self.delta
            v = math.sqrt(vx * vx + vy * vy)
            vx /= v/(self.speed * self.rate)
            vy /= v/(self.speed * self.rate)

            if vx * vx + vy * vy < (x_goal - x_current) ** 2 + (y_goal - y_current) ** 2:
                x_current += vx
                y_current += vy
                self.robot[i].set_current_position(x_current, y_current, 0.)
            else:
                vx = x_goal - x_current
                vy = y_goal - y_current
                self.robot[i].set_current_position(x_goal, y_goal, 0.)
            v_all[i][0] = vx
            v_all[i][1] = vy
        self.counter += 1
        print(self.counter, ":", v_all)


    def ball_liner_fitting(self):
        x = [0.,1.2, 2.1, 3.4, 4.6, 5.1, 6.2, 7.5, 8.54, 9.1]
        self.ball_dynamics_x = [0.,1., 2., 3., 4., 5., 6., 7., 8., 9.]
        self.ball_dynamics_y = [10. * x[i] + 12. for i in range(10)]

        array_x = np.array(self.ball_dynamics_x)
        array_y = np.array(self.ball_dynamics_y)

        random = np.array([np.random.normal(1., 5.) for i in range(10)])
        array_x + random
        random = np.array([np.random.normal(1., 5.) for i in range(10)])
        array_y + random

        n = self.ball_dynamics_window
        xy_sum = np.dot(array_x, array_y)
        x_sum = np.sum(array_x)
        y_sum = np.sum(array_y)
        x_square_sum = np.dot(array_x, array_x)
        a = (n * xy_sum - x_sum * y_sum) / (n * x_square_sum - (x_sum ** 2))
        b = (x_square_sum * y_sum - xy_sum * x_sum) / (n * x_square_sum - x_sum ** 2)
        _error = a * array_x + b - array_y
        error = np.dot(_error, _error)
        return a, b, error

    def goal_keeper_strategy(self):
        a, b, error = self.ball_liner_fitting()
        print(error)
        y = a * self.goal_keeper_x + b
        if y > self.goal_y_min and y < self.goal_y_max:
            self.robot[0].set_future_position(self.goal_keeper_x, y, 0.)

    def calculate_goal_ball_linear_distance(self):
        x_ball = self.ball_dynamics_x[self.ball_dynamics_window-1]
        y_ball = self.ball_dynamics_y[self.ball_dynamics_window-1]
        x_ball = -1125.
        y_ball = 313.
        a_max = (self.goal_y_max - y_ball) / (self.field_x_min - x_ball)
        b_max = (self.field_x_min * y_ball - x_ball * self.goal_y_min) / (self.field_x_min - x_ball)
        a_mid = (0. - y_ball) / (self.field_x_min - x_ball)
        b_mid = (self.field_x_min * y_ball - x_ball * 0.) / (self.field_x_min - x_ball)
        a_min = (self.goal_y_min - y_ball) / (self.field_x_min - x_ball)
        b_min = (self.field_x_min * y_ball - x_ball * self.goal_y_min) / (self.field_x_min - x_ball)
        print(x_ball, y_ball)
        return a_max, b_max, a_mid, b_mid, a_min, b_min

    def center_back_strategy(self):
        # x座標は常にself.field_x_min + 2000
        _, _, a_mid, b_mid, _, _ = self.calculate_goal_ball_linear_distance()
        x = self.field_x_min + 2000.
        y_1 = a_mid * x + b_mid + 200.
        y_2 = a_mid * x + b_mid - 200.
        print(y_1, y_2)
        self.robot[1].set_future_position(x, y_1, 0.)
        self.robot[2].set_future_position(x, y_2, 0.)

    def about_taking_time(self, robot_id, x_start, y_start, x_goal, y_goal):
        v_max = self.robot[robot_id].max_velocity
        a_max = self.robot[robot_id].max_acceleration
        distance = np.sqrt((x_start - x_goal)**2 + (y_start - y_goal)**2)
        if distance < v_max ** 2 / (2 * a_max):
            t = np.sqrt(2*distance / a_max)
        else:
            t = distance / v_max + v_max / (2*a_max)
        return t


    def defender_strategy(self):
        None

    def defence_defender_strategy(self):
        None

    def defence_strategy(self):
        self.goal_keeper_strategy()
        if self.who_has_a_ball == "free":
            None

    def decision_making(self):
        while True:
            self.who_has_a_ball()
            self.attack_or_defence()
            self.decide_attack_strategy()
            if self.attack_or_defence == "attack":
                None
            elif self.attack_or_defence == "defence":
                self.defence_strategy()
            print("decision_making")
            time.sleep(0.1)

    def send_data(self):
        while True:
            self.move_by_potential_method()
            print("send_data")
            time.sleep(self.rate)

    def get_vision_data(self):
        while True:
            del self.ball_dynamics[0]
            self.ball_dynamics.append([x, y])
            time.sleep(1./60.)
            print("get_data")




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
    a = WorldModel(robot_num=8, enemy_num=8)
    a.set_init_positions(char="robot")
    a.set_init_positions(char="enemy")
    a.set_init_positions(char="ball")
    #a.show_positions_figure()
    a.robot_enemy_move_all_linear()
    #a.robot_collision_detect()
    a.set_first_positions()
    a.defence_strategy()
    a.center_back_strategy()
    #a.robot_move_all_linear()
    #a.space_clustering()
    #a.who_has_a_ball()
    #a.attack_or_defence()
    #a.decide_attack_strategy()
    #a.set_first_positions()
    #a.show_positions_figure()
    #a.move_by_potential_method()
    #a.show_positions_figure()

    #thread_1 = threading.Thread(target=a.decision_making)
    #thread_2 = threading.Thread(target=a.send_data)
    #thread_1.start()
    #thread_2.start()





    ## thread化
    """
    positions_from_vision = threading.Thread(target=a.get_information_from_vision)

    """


    #a.enemy_move_all_linear()
    #a.show_positions_figure()
    #b = figure.PositionFigure()


    """
    流れ：
    現在地確認
    戦略を決定する
    robot_future_positionの決定
    移動経路の決定
    robot_current_position -> robot_future_positionに近づく

    繰り返し
    """