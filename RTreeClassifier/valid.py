#!/usr/bin/python2
import numpy as np

velo2cam_data = [6.927964000000e-03, -9.999722000000e-01, -2.757829000000e-03,
 -2.457729000000e-02, -1.162982000000e-03, 2.749836000000e-03,
  -9.999955000000e-01, -6.127237000000e-02, 9.999753000000e-01,
   6.931141000000e-03, -1.143899000000e-03, -3.321029000000e-01]

velo2cam = np.array(velo2cam_data)
velo2cam = velo2cam.reshape(3, -1)

inv_Tr = np.zeros_like(velo2cam) # 3x4
inv_Tr[0:3,0:3] = np.transpose(velo2cam[0:3,0:3])
inv_Tr[0:3,3] = np.dot(-np.transpose(velo2cam[0:3,0:3]), velo2cam[0:3,3])
print(velo2cam)
print(inv_Tr)

#  0.00692796 -0.00116298    0.999975    0.332194
#   -0.999972  0.00274984  0.00693114  -0.0221063
# -0.00275783   -0.999996  -0.0011439  -0.0617198

p = np.array([1.84, 1.47, 8.4])

R0_rect = np.array([9.999128000000e-01, 1.009263000000e-02, -8.511932000000e-03,
 -1.012729000000e-02, 9.999406000000e-01, -4.037671000000e-03,
  8.470675000000e-03, 4.123522000000e-03, 9.999556000000e-0])

R0_rect = R0_rect.reshape([3, 3])

p_ref = np.transpose(np.dot(np.linalg.inv(R0_rect), np.transpose(p)))
# print(np.linalg.inv(R0_rect))
# [[ 9.99977772e-01 -1.00964986e-02  8.47135259e-04]
#  [ 1.01242292e-02  9.99955517e-01  4.12385124e-04]
#  [-8.51261216e-04 -4.03799370e-04  1.00003553e-01]]

P_xx = np.array([ 2.325,  2.325, -2.325, -2.325,  2.325,  2.325, -2.325, -2.325,
     0,      0,      0,      0,  -1.48,  -1.48,  -1.48,  -1.48,
 0.855, -0.855, -0.855,  0.855,  0.855, -0.855, -0.855,  0.855])
P_xx = P_xx.reshape([3,8])
# print(np.dot(np.linalg.inv(R0_rect), P_xx))

#  8.87522  8.36902 -8.39522  -10.439 -11.5648  2.42902 -1.53522 -2.42902
#      1.5     1.54     1.89     1.93     0.83    -1.48    -1.48    -1.48
#  8.45416  9.90634  6.28584  22.5437  27.9642     -nan     -nan     -nan