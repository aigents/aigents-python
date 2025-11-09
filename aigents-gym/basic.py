import os, sys
import numpy as np
import pickle
import random

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
    Add emotionally valuable episode of N of states with some global feeddback
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


def find_similar(states,state,state_count_threshold,state_similarity_threshold):
    max_sim = 0
    bests = []
    for s, utility_count in states.items():
        if utility_count[1] < state_count_threshold: # disregard rare evidence
            continue
        sim = cosine_similarity(s,state)
        if sim < state_similarity_threshold:
            continue
        if max_sim < sim:
            max_sim = sim
            bests.clear()
            bests.append(s)
        elif max_sim == sim:
            bests.append(s)
    best = bests[0] if len(bests) == 1 else random.choice(bests) if len(bests) > 1 else None
    return states[best] if not best is None else None


def find_useful(transitions,transition_utility_thereshold,transition_count_threshold):
    max_utility = -1000000000
    max_count = 0
    bests = []
    for s, utility_count in transitions.items():
        utility, count = utility_count
        if utility < transition_utility_thereshold: # disregard low utility
            continue
        if count < transition_count_threshold: # disregard rare evidence
            continue
        if max_utility < utility:
            max_utility = utility
            max_count = count
            bests.clear()
            bests.append(s)
        elif max_utility == utility:
            bests.append(s)
    best = bests[0] if len(bests) == 1 else random.choice(bests) if len(bests) > 1 else None
    if not best is None:
        #print('found',max_utility,max_count,len(transitions),best[0] if not best is None else '-')
        pass
    return best


def find_useful_action(actions,transitions,transition_utility_thereshold,transition_count_threshold,debug = False):
    actions_uc = {a:0 for a,k in enumerate(actions)} # TODO: optimize to arrray from map!?
    for s, utility_count in transitions.items():
        utility, count = utility_count
        if not transition_utility_thereshold is None and utility < transition_utility_thereshold: # disregard low utility
            continue
        if count < transition_count_threshold: # disregard rare evidence
            continue
        actions_uc[s[0]] += utility * count
        #actions_uc[s[0]] += utility
    max_uc = None # TODO configure
    acts = []
    for a in actions_uc:
        uc = actions_uc[a]
        if max_uc is None or max_uc < uc:
            max_uc = uc
            acts.clear()
            acts.append(a)
        elif max_uc == uc:
            acts.append(a)
    act = acts[0] if len(acts) == 1 else random.choice(actions)
    if False:#debug:
        print(str(actions_uc),str({a:round(actions_uc[a]) for a,k in enumerate(actions)}),act)
    return act


# TODO make abstract
class GymPlayer:
    def __init__(self,debug):
        self.debug = debug

    def process_state(self, observation, reward, previous_action):
        pass

    def debugging(self):
        if self.debug: #or os.path.exists('debug'): # avoid OS hitting HDD!?
            return True
        return False 



