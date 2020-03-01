from engine import *
import numpy as np

if __name__ == '__main__':
      np.random.seed(1)
      # Maze description

      # Feasibility
      feasibility = np.zeros(shape=[15,15], dtype=np.int)
      feasibility[0,1] = 1; feasibility[0,5] = 1; feasibility[1,0] = 1; feasibility[2,3] = 1; feasibility[3,2] = 1
      feasibility[3,4] = 1; feasibility[3,8] = 1; feasibility[4,3] = 1; feasibility[4,9] = 1; feasibility[5,0] = 1
      feasibility[5,6] = 1; feasibility[5,10] = 1; feasibility[6,5] = 1; feasibility[7,8] = 1; feasibility[7,12] = 1
      feasibility[8,3] = 1; feasibility[8,7] = 1; feasibility[9,4] = 1; feasibility[9,14] = 1; feasibility[10,5] = 1
      feasibility[10,11] = 1; feasibility[11,10] = 1; feasibility[11,12] = 1; feasibility[12,7] = 1
      feasibility[12,11] = 1; feasibility[12,13] = 1; feasibility[13,12] = 1; feasibility[14,14] = 1

      # Rewards
      reward = np.zeros(shape=[15,15], dtype=np.int)
      reward[0,1] = -0.1; reward[0,5] = -0.1; reward[1,0] = -0.1; reward[2,3] = -0.1
      reward[3,2] = -0.1; reward[3,4] = -0.1; reward[3,8] = -0.1; reward[4,3] = -0.1
      reward[4,9] = -0.1; reward[5,0] = -0.1; reward[5,6] = -0.1; reward[5,10] = -0.1
      reward[6,5] = -0.1; reward[7,8] = -0.1; reward[7,12] = -0.1; reward[8,3] = -0.1
      reward[8,7] = -0.1; reward[9,4] = -0.1; reward[9,14] = 10.0; reward[10,5] = -0.1
      reward[10,11] = -0.1; reward[11,10] = -0.1; reward[11,12] = -0.1
      reward[12,7] = -0.1; reward[12,11] = -0.1; reward[12,13] = -0.1
      reward[13,12] = -0.1; reward[14,14] = -0.1

      Q = np.zeros(shape=[15,15], dtype=np.float32)

      source = 0; goal = 14
      # Number of states
      number_of_states = 15
      discount_factor = 0.5
      lrn_rate = 0.5
      times = 1000
      train(feasibility, reward, Q, discount_factor, lrn_rate, goal, number_of_states, times)
      path(source, goal, Q)


