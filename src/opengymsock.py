#https://gym.openai.com/docs/#environments

#https://gym.openai.com/envs/Pong-v0/
#https://ai.stackexchange.com/questions/2449/what-do-the-different-actions-of-the-openai-gyms-environment-of-pong-v0-repre

#https://gym.openai.com/envs/Breakout-ram-v0/
#https://gym.openai.com/envs/Breakout-v0/
#https://github.com/openai/gym/issues/588

import gym

import sys
import socket
import numpy as np

HOST = '127.0.0.1'  # Standard loopback interface address (localhost)
PORT = 65432        # Port to listen on (non-privileged ports are > 1023)
sock = None

def getinput(prompt):
        if sock == None:
                val = input(prompt+'\n')
        else:
                sock.sendall((prompt+'\n').encode())
                val = sock.recv(1024).decode().strip()
                #print('got:',val)
        return val


def putout(val):
        val = str(val)
        if sock == None:
                print(val)
        else:
                sock.sendall((val+'\n').encode())

#https://stackoverflow.com/questions/32805549/ellipses-when-converting-list-of-numpy-arrays-to-string-in-python-3
def tostringa(a):
        np.set_printoptions(threshold = np.prod(a.shape))
        return str(a)


def compactRGB(a):#numpy.ndarray
        if len(a.shape) != 3: 
                return a
        new_shape = [a.shape[0],a.shape[1]]
        #print(a)
        #print(new_shape,a.dtype)
        new_a = [] #np.empty(new_shape,dtype=a.dtype)
        for row in a:
                new_r = []
                for col in row:
                        bw = 0
                        for rgb in col:
                                #print(bw,rgb)
                                bw += int(rgb)
                                #print(bw)
                        bw/=3
                        new_r.append(int(bw))
                #print('row:',new_r)
                new_a.append(new_r)#np.append(new_a,new_r)
        new_a = np.array(new_a)
        #print(new_a)
        return new_a

def compactBy2(a):#numpy.ndarray
        if len(a.shape) != 2: 
                return a
        new_shape = [int(a.shape[0]/2),int(a.shape[1]/2)]
        new_a = np.empty(new_shape,dtype=a.dtype)
        #print(a.shape,new_shape)
        r = 0
        for row in new_a:
                c = 0
                new_r = []
                for col in row:
                        #print(r,c) 
                        i = int((a[r*2,c*2] + a[r*2,c*2+1] + a[r*2+1,c*2] + a[r*2+1,c*2+1] )/ 4)
                        #print(i)
                        new_a[r,c] = i
                        c+=1
                r+=1
        return new_a

def compactBy4(a):#numpy.ndarray
        if len(a.shape) != 2: 
                return a
        new_shape = [int(a.shape[0]/4),int(a.shape[1]/4)]
        new_a = np.empty(new_shape,dtype=a.dtype)
        #print(a.shape,new_shape)
        r = 0
        for row in new_a:
                c = 0
                new_r = []
                for col in row:
                        #print(r,c) 
                        r4 = r*4
                        c4 = c*4
                        i = int((a[r4,c4] + a[r4,c4+1] + a[r4+1,c4] + a[r4+1,c4+1] +
                                 a[r4+2,c4] + a[r4+2,c4+1] + a[r4+2+1,c4] + a[r4+2+1,c4+1] +
                                 a[r4,c4+2] + a[r4,c4+2+1] + a[r4+1,c4+2] + a[r4+1,c4+2+1] +
                                 a[r4+2,c4+2] + a[r4+2,c4+2+1] + a[r4+2+1,c4+2] + a[r4+2+1,c4+2+1] 
                                )/ 16)
                        #print(i)
                        new_a[r,c] = i
                        c+=1
                r+=1
        return new_a



def serve():
        #env = gym.make('Pong-v0')
        #env = gym.make('Breakout-ram-v0')
        #env = gym.make('Breakout-v0')
        env = gym.make(getinput("env"));
        cycles = int(getinput("cycles"))
        env.reset()
        putout(str(env.action_space))
        for cycle in range(cycles):
                env.render()
                #action = env.action_space.sample()
                action = int(getinput("action"))
                observation, reward, done, info = env.step(action) # take a random action
                #print(type(observation))#<class 'numpy.ndarray'>
                #print(tostringa(observation))
                putout(reward)
                putout(done)
                observation = compactBy4(compactRGB(observation))
                #print(observation)
                #input('111')
                putout(tostringa(observation))
                if reward > 0 or done == True:
                        print(cycle,reward,done)
                if done:
                        env.reset()
                        #break
        env.close()

#testa = np.array([[[0,64,128],[0,128,255]],[[128,128,128],[255,255,255]]])
#compact(testa)
#testb = np.array([[1,3,5],[2,4,6],[7,8,9]])
#print(testb[0,0])
#print(testb[1,1])
#print(testb[2,2])
#testc = np.array([[1,1,1,1],[3,3,3,3],[5,5,5,5],[7,7,7,7]])
#print(compactBy2(testc))
#sys.exit()
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        print('Serving at', HOST,':',PORT)
        s.bind((HOST, PORT))
        s.listen()
        conn, addr = s.accept()
        with conn:
                sock = conn
                print('Connected by', addr)
                serve()

