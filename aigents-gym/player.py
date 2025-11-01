import sys
import random
import numpy as np
from queue import Queue, Full, Empty

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


class BreakoutXXProgrammable(GymPlayer):
    def __init__(self,width):
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
        if ball_col is None:
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

    def __init__(self,debug):
        self.debug = debug
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

    def process_observation(self,observation,reward):
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
                return (None, None)
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
        return (racket_x, ball_x)
    
    def process_state(self, observation, reward):
        """
        Value Meaning
        0 NOOP
        1 FIRE
        2 RIGHT
        3 LEFT
        """

        observation = self.process_observation(observation,reward)

        # find racket & ball X
        (racket_col, ball_col) = self.racket_ball_x(observation)

        if self.eval is None:
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


