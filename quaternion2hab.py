# -*- coding: utf-8 -*-
from math import pi, cos, sin, atan2, asin
import numpy as np
import quaternion

# take a unit quaternion:
q = quaternion.from_rotation_vector([1, 2, 3])

# convert it to a rotation matrix:
R = quaternion.as_rotation_matrix(q)


# https://www.euclideanspace.com/maths/geometry/rotations/euler/index.htm
def quaternion2hab(q):
    c = 180 / pi
    if q.x*q.y + q.z*q.w == 0.5: # north pole
        heading = 2 * atan2(q.x, q.w) * c
        attitude = 90
        bank = 0
    elif q.x*q.y + q.z*q.w == - 0.5: # south pole
        heading = - 2 * atan2(q.x, q.w) * c
        attitude = - 90
        bank = 0
    else:
        heading = atan2(2*(q.y*q.w - q.x*q.z) , 1 - 2*(q.y*q.y + q.z*q.z)) * c
        attitude = asin(2*(q.x*q.y + q.z*q.w)) * c
        bank = atan2(2*(q.x*q.w - q.y*q.z), 1 - 2*(q.x*q.x + q.z*q.z)) * c
    return (heading, attitude, bank)

# then the rotation is Ry(h) @ Rz(a) @ Rx(b)

# b = bank
# Rx = np.array([
#     [1,      0,       0],
#     [0, cos(b), -sin(b)],
#     [0, sin(b),  cos(b)]    
# ])

# h = heading
# Ry = np.array([
#     [ cos(h), 0, sin(h)],
#     [0      , 1,      0],
#     [-sin(h), 0, cos(h)]    
# ])

# a = attitude
# Rz = np.array([
#     [cos(a), -sin(a), 0],
#     [sin(a),  cos(a), 0],
#     [0     ,      0,  1]    
# ])

# Ry @ Rz @ Rx


