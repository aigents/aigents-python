#import sys
import numpy as np
from queue import Queue, Full, Empty

from basic import *

class BreakoutHacky(GymPlayer):
    def __init__(self,model=None,debug=False):
        super().__init__(debug)
        self.model = model
        self.rocket_row = None
        self.diff_vert = None
        self.initial_array = None 
        self.prev_rocket_size = 0

    def process_state(self, raw_observation, reward, previous_action):

        observation = np.ndarray((178,144))
        for i in range(32, len(raw_observation)):
            observation[i-32] = raw_observation[i][8:152]

        if self.initial_array is None:
            self.initial_array = np.copy(observation)
            return 1 # fire ball 
        
        if self.rocket_row is None:
            max = 0
            self.diff_vert = [int(np.sum(d)) for d in observation] 
            for row in range(100, len(self.diff_vert)): # ignore rows with tiles
                if self.diff_vert[row] > max:
                    max = self.diff_vert[row]
                    self.rocket_row = row
            if self.debug:
                print(self.rocket_row) # = 157

        diff = np.maximum(np.subtract(observation,self.initial_array),0)
        diff_ball = diff[0:self.rocket_row]

        # ball size 2, rocket size starts with 16 and gets smaller
        ball_col = np.argmax(np.convolve(np.mean(diff_ball, axis=0), [1,1,1], mode='same')) + (2)/2
        rocket_size = int(np.sum([1 for d in observation[self.rocket_row] if d>0]))
        rocket_col = np.argmax(np.convolve(observation[self.rocket_row], [1,1,1], mode='same')) - 1
        if rocket_col + rocket_size <= 143: # no right wall collision
            self.prev_rocket_size = rocket_size
        rocket_col += (self.prev_rocket_size)/2

        # check if ball is in game
        if int(np.sum([int(np.sum(d)) for d in diff_ball])) == 0:
            # return rocket to the center of the screen
            if rocket_col < 72:
                return 2 # RIGHT
            elif rocket_col > 72:
                return 3 # LEFT
            else:
                #prev_action = 1
                return 1 # fire new ball

        if rocket_col < ball_col - 4:
            if rocket_col + (self.prev_rocket_size)/2 > 143:
                act = 0 # NOOP - right wall collision
            else:    
                act = 2 # RIGHT
        elif rocket_col > ball_col + 4:
            act = 3 # RIGHT
        else:
            act = 0 # NOOP

        return act




class BreakoutXXProgrammable(GymPlayer):
    def __init__(self,width,debug=False):
        super().__init__(debug)
        self.reactivity_base = 4
        self.randomness = 0

        self.ball_col_old = None
        self.ball_dir = 0 # ball direction and speed
        self.racket_col_old = None
        self.width = width

        self.started = False
        self.prev_racket_size = 0


    def process_state_complex(self, observation, reward): # Original Anton's version
        (racket_col, ball_col) = observation
        if racket_col is None:
            act = 0
        else:
            racket_dir = 0 if (self.racket_col_old is None or racket_col is None) else racket_col - self.racket_col_old # not used, for debugging 
            self.racket_col_old = racket_col

            if ball_col is None: # ball is not "visible"
                #ball_col_pred = len(observation[self.racket_row]) / 2 # HACK: assume that ball will get into middle by default
                ball_col_pred = random.choice(list(range(self.width))) # HACK: guess where the ball is randomly
            else:
                self.ball_dir = 0 if (self.ball_col_old is None) else ball_col - self.ball_col_old
                ball_col_pred = ball_col + self.ball_dir * 2 # be over-predictive, double the ball speed!?
    
            self.ball_col_old = ball_col

            reactivity = random.choice(list(range(self.reactivity_base-self.randomness,self.reactivity_base+self.randomness+1))) # HACK: randomness preventing dead cycles
            
            assert(not racket_col is None)
            assert(not ball_col_pred is None)
            if ball_col is None:
                act = 1  
            elif racket_col - ball_col_pred < -reactivity:
                act = 2 # RIGHT
            elif racket_col - ball_col_pred > reactivity:
                act = 3 # LEFT
            else: # TODO FIRE if ball is NOT visible, otherwise 0 (NOOP) !!!
                act = 1 # FIRE

        return act
    

    def process_state(self, observation, reward, racket_size): # Latest Vladimir's version adopted by Anton
        (racket_col, ball_col) = observation
        if not self.started:
            self.started = True
            return 1

        #print(self.width - 1,self.width /2)

        if racket_col + racket_size/2 <= (self.width - 1): # 143 # no right wall collision
            self.prev_racket_size = racket_size

        # check if ball is in game
        if ball_col == INT_NONE:
            if random.choice([True, False]): # HACK: to fire the ball enventually 
                return 1
            # return rocket to the center of the screen
            if racket_col < self.width / 2: # == 72
                return 2 # RIGHT
            elif racket_col > self.width /2: # == 72
                return 3 # LEFT
            else:
                return 1 # fire new ball # TODO: why do we never get here?
        if racket_col < ball_col - 4:
            if racket_col + self.prev_racket_size/2 > (self.width - 1): # 143
                act = 0 # NOOP - right wall collision
            else:    
                act = 2 # RIGHT
        elif racket_col > ball_col + 4:
            act = 3 # RIGHT
        else:
            act = 0 # NOOP
        return act


class BreakoutProgrammable(GymPlayer):

    def __init__(self,model=None,learn_mode=0,context_size=1,debug=False):
        super().__init__(debug)
        self.memory_size = 30
        self.background_refresh_rate = 10
        
        #self.observation_top = 0 # Fair
        self.observation_top = 93 # Hack for Breakout - cut ceiling off
        #self.observation_border = 0 # Fair
        self.observation_border = 8 # Hack for Breakout - cut walls off

        self.observations = Queue(maxsize=self.memory_size) 
        self.epoch = 0

        self.racket_row = 96 # None # 96 (with top already cut!?)
        self.diff_vert = None
        self.average_array = None 

        self.ball_col_old = None
        self.ball_dir = 0 # ball direction and speed
        self.racket_col_old = None

        self.diff = None # may be not used

        self.eval = None

        self.model = model
        self.learn_mode = learn_mode
        self.context_size = context_size
        self.states = []

    def process_observation(self,observation,reward,previous_action):
        if self.observation_top > 0: 
            observation = observation[self.observation_top:]
        if self.observation_border > 0: 
            observation = [o[self.observation_border:-self.observation_border] for o in observation]
        # accumulate observations in rolling window
        if self.observations.qsize() == self.memory_size:
            self.observations.get()
        self.observations.put((observation, reward))
        # update background
        self.epoch += 1
        if self.epoch % self.background_refresh_rate == 0: 
            self.observation_maps = [a[0] for a in list(self.observations.queue)] # grayscale!
            self.average_array = np.mean(self.observation_maps, axis=0)
        return observation

    def racket_ball_x(self,observation):
        if self.racket_row is None:
            if self.average_array is None:
                return (INT_NONE, INT_NONE)
            self.diff = np.maximum(np.subtract(observation,self.average_array),0)
            max = 0
            diff_vert = [int(np.sum(d)) for d in self.diff] 
            for row in range(len(diff_vert)):
                if diff_vert[row] > max:
                    max = diff_vert[row]
                    self.racket_row = row
        #racket_col = np.argmax(np.convolve(diff[racket_row], [1,1,1], mode='same'))
        racket_x = get_avg_pos(observation[self.racket_row],1)
        #ball_col = np.argmax(np.convolve(np.mean(diff_ball, axis=0), [1,1,1], mode='same'))
        ball_hor = observation[0:self.racket_row]
        ball_x = get_avg_pos(np.mean(ball_hor, axis=0),1)
        return (INT_NONE if racket_x is None else int(round(racket_x)), INT_NONE if ball_x is None else int(round(ball_x)))

    def learn_model(self,state,reward):
        if not self.model is None and self.learn_mode != 0:
            if reward != 0:
                if self.learn_mode == 1:
                    feedback = reward if reward > 0 else 0 # positive only
                else:
                    feedback = reward # positive or negative
                model_add_states(self.model,self.states,feedback)
                self.states.clear() # clear the states including the rewarded one to start over with new state and new action on it
            self.states.append(state)


    def process_state(self, observation, reward, previous_action):
        """
        Value Meaning
        0 NOOP
        1 FIRE
        2 RIGHT
        3 LEFT
        """
        observation = self.process_observation(observation,reward,previous_action)

        # find racket & ball X
        (racket_col, ball_col) = self.racket_ball_x(observation)
        current_state = (previous_action,)+(1 if reward > 0 else 0,1 if reward < 0 else 0)+(racket_col, ball_col)
        #TODO aggregate states based on context_size
        if self.context_size == 1:
            state = current_state
        else:
            pass #TODO states

        self.learn_model(state,reward)

        if self.eval is None: # Lazy init
            self.eval = BreakoutXXProgrammable(len(observation[self.racket_row]))

        #act = self.eval.process_state((racket_col, ball_col), reward)
        act = self.eval.process_state((racket_col, ball_col), reward, racket_size = np.sum([1 for d in observation[self.racket_row] if d>0]))

        if self.debug and 0:
                try:
                    input("Press enter to continue")
                except SyntaxError:
                    pass

        if self.debug:
                #print(str(np.mean(diff_ball, axis=0)))
                #print(str(diff[racket_row]))
                #print(get_avg_pos(np.mean(diff_ball, axis=0),1))
                #print(get_avg_pos(diff[racket_row],1))
                #print(np.convolve(np.mean(diff_ball, axis=0), [1,1,1], mode='same'))
                #print(np.convolve(diff[racket_row], [1,1,1], mode='same'))
                print(debug_array2str(np.mean(observation[0:self.racket_row], axis=0),1),ball_col,self.ball_dir)
                print(debug_array2str(observation[self.racket_row],1),self.racket_row,racket_col,act)

        return act


class BreakoutModelDriven(BreakoutProgrammable):

    def __init__(self,actions,model=None,learn_mode=0,context_size=1,debug=False):
        super().__init__(model,learn_mode,context_size,debug)
        self.actions = actions

    def process_state(self, observation, reward, previous_action):
        observation = self.process_observation(observation,reward,previous_action)

        # find racket & ball X
        (racket_col, ball_col) = self.racket_ball_x(observation)
        state = (previous_action,)+(1 if reward > 0 else 0,1 if reward < 0 else 0)+(racket_col, ball_col)

        self.learn_model(state,reward)

        states = self.model['states']
        try:
            found = states[state]
            match = 'exact'
        except KeyError:
            found = find_similar(states,state, state_count_threshold=2, state_similarity_threshold=0.99)
            match = 'exact'

        if not found is None:
            (utility,count,transitions) = found
            #print('found',match,state,'=>',found,'=',len(transitions))
            # new code TODO: fix and test!!!
            return find_useful_action(self.actions,transitions, transition_utility_thereshold=0, transition_count_threshold=1)
            # old code:
            best = find_useful(transitions,transition_utility_thereshold=0,transition_count_threshold=1)
            if not best is None:
                if self.debugging():
                    print('found',match,utility,count,len(transitions),best[0] if not best is None else '-')
                return best[0]
    
        #print("found none")
        return random.choice(self.actions)


## Old code from Nov 3 2025 TODO remove later!?

def find_similarNov32025(states,state,count_threshold,similarity_threshold):
    max_sim = 0
    best = None
    for s, utility_count in states.items():
        #print(s,state)
        if utility_count[1] < count_threshold: # disregard rare evidence
            continue
        sim = cosine_similarity(s,state)
        if sim < similarity_threshold:
            continue
        if max_sim < sim:
            max_sim = sim
            best = s
    #if not best is None:
    #    print('similarity',max_sim)
    return states[best] if not best is None else None


def find_usefulNov32025(transitions,utility_thereshold,count_threshold):
    max_utility = 0
    max_count = 0
    best = None
    for s, utility_count in transitions.items():
        utility, count = utility_count
        if utility < utility_thereshold: # disregard low utility
            continue
        if count < count_threshold: # disregard rare evidence
            continue
        if max_utility < utility:
            max_utility = utility
            max_count = count
            best = s
    if not best is None:
        #print('found',max_utility,max_count,len(transitions),best[0] if not best is None else '-')
        return best


# TODO remove later
class BreakoutModelDrivenNov32025(BreakoutProgrammable):

    def __init__(self,actions,model=None,learn_mode=0,debug=False):
        super().__init__(model,debug)
        self.actions = actions
        self.learn_mode = learn_mode

    def process_state(self, observation, reward, previous_action):
        observation = self.process_observation(observation,reward,previous_action)

        # find racket & ball X
        (racket_col, ball_col) = self.racket_ball_x(observation)
        state = (previous_action,)+(1 if reward > 0 else 0,1 if reward < 0 else 0)+(racket_col, ball_col)

        if not self.model is None and self.learn_mode != 0:
            if reward != 0:
                if self.learn_mode == 1:
                    feedback = reward if reward > 0 else 0 # positive only
                else:
                    feedback = reward # positive or negative
                model_add_states(self.model,self.states,feedback)
                self.states.clear() # clear the states including the rewarded one to start over with new state and new action on it
            self.states.append(state)

        states = self.model['states']
        try:
            found = states[state]
            match = 'exact'
        except KeyError:
            found = find_similarNov32025(states,state,2,0.99) # HACK: threshold!?
            match = 'exact'

        if not found is None:
            (utility,count,transitions) = found
            #print('found',match,state,'=>',found,'=',len(transitions))
            best = find_usefulNov32025(transitions,0,1)
            if not best is None:
                #print('found',match,utility,count,len(transitions),best[0] if not best is None else '-')
                return best[0]

        #print("found none")
        return random.choice(self.actions)