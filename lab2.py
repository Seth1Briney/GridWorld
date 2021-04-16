'''
For the homework in week 3, you will implement the random policy for gridworld, as described in the text. The policy is not being updated, only the values.

1. For the random policy, you know the probabilities for each action, and you know the reward and next state for each state-action pair, you can use dynamic programming and Bellman's equation to update the value for each state iteratively.

2. [optional] Again for the random policy, compute the values of each state using a Monte Carlo simulation, choosing one of the random actions, performing it, and updating the value of the state. You could even imagine simulating 2 actions away from the state.
'''

import numpy as np

def argmax_rand_tie(x):
    x=x.reshape(-1,)

    shuf = np.array([n for n in range(9)]).reshape(-1,)
    np.random.shuffle(shuf)

    argmax=np.argmax(x[shuf])

    return shuf[argmax]

class Grid():
    def __init__(self, ylim=10, xlim=10, A0=(0,0), A=(3,7), B0=(9,9), B=(8,7), coords=[0,0], alpha=0.01, epsilon=0.01, gamma=0.9):
        self.xlim=xlim
        self.ylim=ylim
        self.A0,self.A,self.B0,self.B=A0,A,B0,B
        self.coords=coords
        self.Q = np.zeros((ylim,xlim,4))
        self.alpha = alpha
        self.epsilon = epsilon
        self.gamma = gamma
    def pi(self):
        # determine direction
        
        # a = np.random.choice(['north','south','east','west'])
        # return a

        if np.random.rand() < self.epsilon: # epsilon greedy random action for exploration,
            a = np.random.choice(['north','south','east','west'])
        else:
            # a = argmax_rand_tie(self.Q[self.coords[0],self.coords[1],:])
            a = np.argmax(self.Q[self.coords[0],self.coords[1],:])
            a = {0:'north',1:'south',2:'east',3:'west'}[a]
        return a

    def a_to_num(self,a):
        return {'north':0,'south':1,'east':2,'west':3}[a]

    def update(self,a,r,yx):
        self.Q[ yx[0], yx[1], self.a_to_num(a) ] = r
        # self.Q[self.coords[0],self.coords[1],self.a_to_num(a)] = r
        # (1-self.alpha)*self.Q[self.coords[0],self.coords[1],self.a_to_num(a)] + self.alpha*r

    def move_a(self,a):
        if a=='north':
            self.coords[0] = self.coords[0]-1
        elif a=='south':
            self.coords[0] = self.coords[0]+1
        elif a=='west':
            self.coords[1] = self.coords[1]-1
        elif a=='east':
            self.coords[1] = self.coords[1]+1

    def move_and_reward(self):
        a = self.pi()
        yx = self.coords.copy()
        if self.coords[0]==0 and a=='north':
            r = -1
        elif self.coords[0]==self.ylim-1 and a=='south':
            r = -1
        elif self.coords[1]==0 and a=='west':
            r = -1
        elif self.coords[1]==self.xlim-1 and a=='east':
            r = -1
        elif self.coords[0]==self.A[0] and self.coords[1]==self.A[1]:
            r = 10
            self.coords[0]=self.A0[0]
            self.coords[1]=self.A0[1]
        elif self.coords[0]==self.B[0] and self.coords[1]==self.B[1]:
            r = 5
            self.coords[0]=self.B0[0]
            self.coords[1]=self.B0[1]
        else:
            r = 0
            self.move_a(a)
        
        rn = np.mean(self.Q[self.coords[0],self.coords[1],:]) # this is the mean of action values from the resulting state, which is like the state value.
        # rn = self.Q[self.coords[0],self.coords[1],self.a_to_num(a)]
        self.update( a , r+self.gamma*rn , yx )

        return r

    def show_Q(self):

        import matplotlib.pyplot as plt
        import seaborn as sns; sns.set_theme()

        f,(ax1,ax2,ax3,ax4) = plt.subplots(1,4,sharey=True)
        g1 = sns.heatmap(self.Q[:,:,0],cbar=False,ax=ax1)
        g1.set_title('north')
        g1.set_ylabel('')
        g1.set_xlabel('')
        g2 = sns.heatmap(self.Q[:,:,1],cbar=False,ax=ax2)
        g2.set_title('south')
        g2.set_ylabel('')
        g2.set_xlabel('')
        g3 = sns.heatmap(self.Q[:,:,2],cbar=False,ax=ax3)
        g3.set_title('east')
        g3.set_ylabel('')
        g3.set_xlabel('')
        g4 = sns.heatmap(self.Q[:,:,3],cbar=False,ax=ax4)
        g4.set_title('west')
        g4.set_ylabel('')
        g4.set_xlabel('')
        plt.show()

if __name__=='__main__':
    grid = Grid(epsilon=0.3)
    for n in range(1000000):
        grid.move_and_reward()
    grid.show_Q()


# not optimistic
# epsilon = 0
# return, rmse(q):
# best:
# [2.64116375e+04 4.02388909e+00]
# average:
# [2.64093392e+04 4.02388910e+00]

# epsilon = 0.01
# return, rmse(q):
# best:
# [1.45446858e+07 1.46552723e+03]
# average:
# [4.46794160e+06 4.31773892e+02]

# epsilon = 0.1
# return, rmse(q):
# best:
# [6.40017776e+05 3.66658409e+00]
# average:
# [-51275.68519079    263.74720341]



# optimistic:
# epsilon = 0
# return, rmse(q):
# best:
# [6.79483756e+06 1.01248472e+03]
# average:
# [1.79379047e+06 4.22704868e+02]

# epsilon = 0.01
# return, rmse(q):
# best:
# [42553.27432712   239.01711839]
# average:
# [13537.03693308    97.79685173]

# epsilon = 0.1
# return, rmse(q):
# best:
# [-224629.25699686     572.94849696]
# average:
# [-465818.96374092    1189.93927427]