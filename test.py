import numpy as np
import entity

robot = entity.robot()
print(robot.show_parameters())
robots = [entity.robot(), entity.robot()]
print(robots[0].show_parameters())