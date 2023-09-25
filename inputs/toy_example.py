list_lbs = [[-1,-1], [0.8, -2, -2], [0.8, 0, 0], [0,8, 0, 0]]
list_ubs = [[1, 1], [4.8, 2, 2], [4.8, 2, 2], [4.8, 2, 2]]
# constraint Hx(L)+d <= 0, w.r.t output neurons
# y1-y2-y3 <= 0, -y2 <= 0, -y3 <= 0, 1-1.25y1 <= 0, y1-2 <= 0, y2-2 <= 0, y3-2 <= 0
H_matrix = [[1, -1, -1], [0, -1, 0], [0, 0, -1], [-1.25, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]
d_vector = [0, 0, 0, 1, -2, -2, -2]

# constraint Pxi + P_hatxi_hat - p <= 0, w.r.t intermediate unstable neurons and their respective inputs
# -x7 <= 0, -x8 <= 0, -0.5x4 +x7 -1 <= 0, -0.5x5+x8 -1 <= 0, 2x4+x5-x7-x8 <= 0, -x7-x8-2 <= 0
# xi is [x4, x5], xi_hat is [x7, x8]
P_matrix = [[0,0], [0,0], [-0.5, 0], [0, -0.5], [2, 1], [0,0]]
Phat_matrix = [[-1,0], [0,-1], [1, 0], [0, 1], [-1, -1], [-1,-1]]
p_vector = [0, 0, 1, 1, 0, 2]
