import numpy as np
import entity

class CollisionDetection():
    def __init__(self, robot_num):
        self.robot_num = robot_num
        #self.robot = robot
        #self.robot = [0,0,0,0,0,0,0,0]

    def robot_collision_detect(self, robots):
        for i in range(self.robot_num):
            for j in range(self.robot_num-i-1):
                position_1_x, position_1_y = robots[i].get_parameter_xy()
                position_2_x, position_2_y = robots[j+i+1].get_parameter_xy()
                if (position_1_x - position_2_x)**2 + (position_2_x - position_2_y)**2 < 0.18*0.18:
                    print("collision!!")
                    break
        print("safety!!")

if __name__ == "__main__":
    detector = CollisionDetection()
    detector.robot_collision_detect()