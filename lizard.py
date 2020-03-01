import numpy as np
def my_print(Q):
  # hard-coded hack for this problem only
  rows = len(Q); cols = len(Q[0])
  print("       0      1      2      3      4      5\
      6      7      8     ")
  for i in range(rows):
    print("%d " % i, end="")
    if i < 10: print(" ", end="")
    for j in range(cols): print(" %6.2f" % Q[i,j], end="")
    print("")
  print("")
def path(start, destination, Q):
      print("-----Solution found-------")
      i = 1
      while start != destination:
            if i >= 10:
                  break
            intermediate = np.argmax(Q[start])
            print(f"{start} -> {intermediate}")
            start = intermediate
            i+=1
      print('-------End---------------')
      
def reachable_from(start, valid, number_of_states):
      candidates = []
      for state in range(number_of_states):
            if valid[start][state] == 1:
                  candidates.append(state)
      return candidates

def random_movement(start, valid, number_of_states):
      states = reachable_from(start, valid, number_of_states)
      return states[np.random.randint(0, len(states))]

def learn(valid, reward, discount_factor, learning_rate, destination, number_of_states, episodes):
      for episode in range(episodes):
            current_state = np.random.randint(0, number_of_states)
            # print(f"Current state: {current_state}")
            while current_state != destination:
                  other = random_movement(current_state, valid, number_of_states)
                  future_look_up = reachable_from(other, valid, number_of_states)
                  # Finding best future movement
                  best = -9999.99
                  for someone in range(len(future_look_up)):
                        action = future_look_up[someone]
                        q_value = Q[other][action]
                        best = q_value if q_value > best else best
                  Q[current_state][other] = ((1 - learning_rate) * Q[current_state] \
                        [other]) + (learning_rate * (reward[current_state][other] + \
                        (discount_factor * best)))
                  current_state = other


if __name__ == '__main__':

      np.random.seed(1)
      # Represent valid moves on the grid
      valid = np.zeros(shape=[9,9], dtype= np.int)
      valid[0][1] = 1; valid[0][3] = 1
      valid[1][0] = 1; valid[1][2] = 1; valid[1][4] = 1
      valid[2][1] = 1; valid[2][5] = 1
      valid[3][0] = 1; valid[3][4] = 1; valid[3][6] = 1
      valid[4][1] = 1; valid[4][3] = 1; valid[4][5] = 1; valid[4][7] = 1
      valid[5][4] = 1; valid[5][2] = 1; valid[5][8] = 1
      valid[6][3] = 1; valid[6][7] = 1
      valid[7][6] = 1; valid[7][4] = 1; valid[7][8] = 1
      
      # Represents reward for any combination of states
      reward = np.zeros(shape=[9,9], dtype=np.float32)
      for i in range(9):
            for j in range(9):
                  reward[i][j] = -0.1
      # reward[3][0] = 3
      # reward[1][0] = 3
      # reward[3][4] = -0.001
      # reward[1][4] = -0.001
      # reward[5][4] = -0.001
      # reward[7][4] = -0.001
      reward[7][8] = 10.0
      reward[5][8] = 10.0
      reward[8][8] = 10.0

      # Represents Q matrix
      Q = np.zeros(shape=[9,9], dtype=np.float32)
      start = 6; destination = 8
      number_of_states = 9
      discount_factor = 0.5
      learning_rate = 0.5
      episodes = 1500
      learn(valid, reward, discount_factor, learning_rate, destination, number_of_states, episodes)
      path(start, destination, Q)
      my_print(Q)


