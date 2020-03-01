from engine import *
from numpy import random

if __name__ == '__main__':
      random.seed(1)
      # Setting arena
      SIZE = 9
      feasibility = np.zeros(shape=[SIZE,SIZE], dtype=np.int)
      feasibility[0][1] = 1; feasibility[0][3] = 1
      feasibility[3][0] = 1; feasibility[3][4] = 1; feasibility[3][6] = 1
      feasibility[6][3] = 1; feasibility[6][7] = 1
      feasibility[1][0] = 1; feasibility[1][4] = 1; feasibility[1][2] = 1
      feasibility[4][1] = 1; feasibility[4][3] = 1; feasibility[4][5] = 1; feasibility[4][7] = 1
      feasibility[7][4] = 1; feasibility[7][6] = 1; feasibility[7][8] = 1
      feasibility[2][1] = 1; feasibility[2][5] = 1
      feasibility[5][4] = 1; feasibility[5][2] = 1; feasibility[5][8] = 1
      feasibility[8][8] = 1

      reward = np.zeros(shape=[SIZE,SIZE], dtype=np.float32)
      for row in range(SIZE):
            for col in range(SIZE):
                  reward[row][col] = -0.1
      reward[5][8] = 10.0
      reward[7][8] = 10.0

      reward[3][4] = -10.0
      reward[1][4] = -10.0
      reward[5][4] = -10.0
      reward[7][4] = -10.0
      reward[3][0] = 5.0
      reward[1][0] = 5.0
      
      Q = np.zeros(shape=[SIZE,SIZE], dtype=np.float32)
      source = 6
      goal = 8
      lrn_rate = 0.5
      gamma = 0.5
      times = 1000
      train(feasibility, reward, Q, gamma, lrn_rate, goal, SIZE, times)

      print("------------------------------------Q----------------------------------")
      m_print(Q)
      path(source, goal, Q)
