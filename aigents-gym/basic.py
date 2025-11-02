import os, sys
import numpy as np
import pickle

INT_NONE = -sys.maxsize - 1

def debug_array2str(a,t):
    return ''.join(['.' if i < t else '█' for i in a])

def print_debug(array2d):
    row = 0
    for a in array2d:
        print(debug_array2str(a,1),row)
        row += 1

def get_avg_pos(a,t):
    indexes = np.where(a > t)[0]
    return np.mean(indexes) if len(indexes) > 0 else None

def cosine_similarity(a,b):
    return np.dot(a, b)/(np.linalg.norm(a)*np.linalg.norm(b))

def model_new():
    """
    games: games count
    steps: steps count
    states: map states to (utility,count,transtions) triple
        transitions: map states pair to (utility,count)
    """
    return {'steps':0, 'games':0, 'states':{}, 'transitions':{}}

def model_read_file(model_name):
    model_path = model_name + '.pkl'
    if os.path.exists(model_path):
        with open(model_path, 'rb') as f:
            data = pickle.load(f)
            return data
    else:
        return model_new()

def model_write_file(model_name, model):
    model_path = model_name + '.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)

#TODO optional exponentional reward deccay
def model_add_states(model,states,global_feeddback):
    """
    Add emotinally valuable episode of N of states with some global feeddback
    """
    model['steps'] += len(states)
    model_states = model['states']
    model_transitions = model['transitions']
    previous = None
    for state in states:
        if state in model_states:
            (utility, count, transitions) = model_states[state]
            model_states[state] = (utility + global_feeddback, count + 1, transitions)
        else:
            model_states[state] = (global_feeddback, 1, {})
        if not previous is None:
            (utility, count, transitions) = model_states[previous]
            if state in transitions:
                (transition_utility, transition_count) = transitions[state]
                transitions[state] = (transition_utility + global_feeddback, transition_count + 1)
            else:
                transitions[state] = (global_feeddback, 1)
            model_states[previous] = (utility, count, transitions)
        previous = state
    return model

assert(str(model_add_states(model_new(),[],0))=="{'steps': 0, 'games': 0, 'states': {}, 'transitions': {}}")
assert(str(model_add_states(model_new(),[(0,0),(0,1)],0))=="{'steps': 2, 'games': 0, 'states': {(0, 0): (0, 1, {(0, 1): (0, 1)}), (0, 1): (0, 1, {})}, 'transitions': {}}")
assert(str(model_add_states(model_new(),[(0,0),(0,1),(0,0),(0,1)],0))=="{'steps': 4, 'games': 0, 'states': {(0, 0): (0, 2, {(0, 1): (0, 2)}), (0, 1): (0, 2, {(0, 0): (0, 1)})}, 'transitions': {}}")
assert(str(model_add_states(model_new(),[(0,0),(0,1),(0,0),(0,1)],1))=="{'steps': 4, 'games': 0, 'states': {(0, 0): (2, 2, {(0, 1): (2, 2)}), (0, 1): (2, 2, {(0, 0): (1, 1)})}, 'transitions': {}}")



# TODO make abstract
class GymPlayer:
    def __init__(self):
        pass

    def process_state(self, observation, reward, previous_action):
        pass



