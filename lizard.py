import numpy as np 

"""
      Prints Q matrix
"""
def m_print(Q):
      rows = len(Q)
      cols = len(Q[0])
      print("       0      1      2      3      4      5\
      6      7      8")
      for row in range(rows):
            print(f"{row} ", end="")
            if row < 10:
                  print(" ", end=" ")
            for col in range(cols):
                  print(" %6.2f" % Q[row, col], end="")
            print("")
      print("")

"""
      Finds all possible states from source
"""
def next_states_from(source, feasibility, number_of_states):
      return [state for state in range(number_of_states) if feasibility[source, state] == 1]

"""
      Picks a feasible next state from source
"""
def next_feasible(source, feasibility, number_of_states):
      all = next_states_from(source, feasibility, number_of_states)
      return all[np.random.randint(0, len(all))]

"""
      Find best state from source at any given time
"""
def best(source, feasibility, number_of_states, Q):
      best = float('-inf')
      around = next_states_from(source, feasibility, number_of_states)
      for someone in around:
            q = Q[source, someone]
            if q > best:
                  best = q
      return best 

"""
      Computes Q matrix
"""
def train(feasibility, reward, Q, gamma, lrn_rate, goal, number_of_states, times):
      for time in range(times):
            # Select a random source
            current_state = np.random.randint(0, number_of_states)
            # Updates Q-values while destination is not reached
            while current_state != goal:
                  # Select a random next state
                  next = next_feasible(current_state, feasibility, number_of_states)
                  # Compute reward
                  best_choice = best(next, feasibility, number_of_states, Q)
                  # Update Q
                  Q[current_state][next] = ((1-lrn_rate) * Q[current_state][next] + (lrn_rate * (reward[current_state][next] + (gamma * best_choice))))
                  # Moving towards next
                  current_state = next

"""
      Prints path from source to goal
"""

def path(source, goal, Q):
      while source != goal:
            next = np.argmax(Q[source])
            print(f"{source} --> {next}")
            source = next

if __name__ == '__main__':
      np.random.seed(1)
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
      
      Q = np.zeros(shape=[SIZE,SIZE], dtype=np.float32)
      source = 6
      goal = 8
      lrn_rate = 0.5
      gamma = 0.5
      times = 1000
      train(feasibility, reward, Q, gamma, lrn_rate, goal, SIZE, times)

      print("------------Q--------------------")
      m_print(Q)
      path(source, goal, Q)
