import numpy as np
 
def debug_array2str(a,t):
    return ''.join(['.' if i < t else '█' for i in a])

def print_debug(array2d):
    row = 0
    for a in array2d:
        print(debug_array2str(a,1),row)
        row += 1

def get_avg_pos(a,t):
    indexes = np.where(a > t)[0]
    #print(indexes)
    return np.mean(indexes) if len(indexes) > 0 else None

# TODO make abstract
class GymPlayer:
    def __init__(self):
        pass

    def process_state(self, observation, reward):
        pass



