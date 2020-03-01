import numpy as np

"""
      Prints path found
"""
def walk(source, destination, Q):
      print("-----Start-----")
      while source != destination:
            intermediate = np.argmax(Q[source])
            print(f"{source} -> {intermediate}")
            source = intermediate
      print("-----End-----")
"""
      Given a state source returns all reachable states
      from source according to 'reachable'
"""
def reachable(source, legal, number_of_states):
      found = []
      for state in range(number_of_states):
            if legal[source, state] == 1:
                  found.append(state)
      return found

def random_movement(source, legal, number_of_states):
      candidates = reachable(source, legal, number_of_states)
      return candidates[ np.random.randint(0, len(candidates)) ]

def learn(legal, reward, core, learning_rate,
      discount_factor, number_of_states, episodes, destination):

      for episode in range(episodes):         
            # Selecting a random start state
            current_state = np.random.randint(0, number_of_states)
            # Traversing through
            while current_state != destination:
                  other = random_movement(current_state, legal, number_of_states)
                  future_look_up = reachable(other, legal, number_of_states)
                  # Finding best future movement
                  best = float('-inf')
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
      # Maze description
      # Reachability
      legal = np.zeros(shape=[15,15], dtype=np.int)
      legal[0,1] = 1; legal[0,5] = 1; legal[1,0] = 1; legal[2,3] = 1; legal[3,2] = 1
      legal[3,4] = 1; legal[3,8] = 1; legal[4,3] = 1; legal[4,9] = 1; legal[5,0] = 1
      legal[5,6] = 1; legal[5,10] = 1; legal[6,5] = 1; legal[7,8] = 1; legal[7,12] = 1
      legal[8,3] = 1; legal[8,7] = 1; legal[9,4] = 1; legal[9,14] = 1; legal[10,5] = 1
      legal[10,11] = 1; legal[11,10] = 1; legal[11,12] = 1; legal[12,7] = 1
      legal[12,11] = 1; legal[12,13] = 1; legal[13,12] = 1; legal[14,14] = 1
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

      start = 0; destination = 14
      # Number of states
      number_of_states = 15
      discount_factor = 0.5
      learning_rate = 0.5
      episodes = 1000
      learn(legal, reward, Q, learning_rate, discount_factor, number_of_states, episodes, destination)
      walk(start, destination, Q)
