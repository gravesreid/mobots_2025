# make a simple path for the robot to follow. Save as a csv file of the form:
# x,y,t (x,y in meters, t in seconds)

# The robot should move at 0.5 m/s. The path should go 1 m forward (+x), then make a 0.5 m radius turn -90 degrees, then go 4 m in -y direction.

import numpy as np
import csv

# Constants
speed = 0.5 # m/s
sample_rate = 0.05 # s
forward_distance = 1 # m
turn_radius = 0.5 # m
turn_angle = -np.pi/2 # radians
backward_distance = 4 # m

positions = [] # list of (x,y,t) tuples

# Forward
t = 0
x = 0
y = 0
while x < forward_distance:
    positions.append((x, y, t))
    t += sample_rate
    x += speed * sample_rate

# Turn
# cacluate theta per second
theta_dot = speed / turn_radius

cur_angle = 0
while cur_angle > turn_angle:
    positions.append((x, y, t))
    t += sample_rate
    x += speed * np.cos(cur_angle) * sample_rate
    y += speed * np.sin(cur_angle) * sample_rate
    cur_angle -= theta_dot * sample_rate

# Backward
while y > -backward_distance:
    positions.append((x, y, t))
    t += sample_rate
    y -= speed * sample_rate

# plot the path
import matplotlib.pyplot as plt
x = [p[0] for p in positions]
y = [p[1] for p in positions]
plt.plot(x, y)

# save the plot
plt.savefig('simple_path.png')

# save the path
with open('simple_path.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerow(['x', 'y'])
    for p in positions:
        writer.writerow(p[0:2])