import numpy as np

class entity():
    def __init__(self):
        self.size_r = 0.
        self.position_x = 0.
        self.position_y = 0.
        self.orientation = 0.

    def set_position(self, x, y, theta):
        self.position_x = x
        self.position_y = y
        self.orientation = theta

    def show_parameters(self):
        return "size_r:{}, position_x:{}, position_y:{}, orientation:{}".format(
                self.size_r, self.position_x, self.position_y, self.orientation)

class robot(entity):
    def __init__(self):
        super().__init__()
        self.size_r = 10. * 1000

class ball(entity):
    def __init__(self):
        super().__init__()
        self.size_r = 21.33




if __name__ == "__main__":
    robot_0 = robot()
    print(robot_0.show_parameters())
    ball_ = ball()
    print(ball_.show_parameters())
