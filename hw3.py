'''
For the homework in week 3, you will implement the random policy for gridworld, as described in the text. The policy is not being updated, only the values.

1. For the random policy, you know the probabilities for each action, and you know the reward and next state for each state-action pair, you can use dynamic programming and Bellman's equation to update the value for each state iteratively.

2. [optional] Again for the random policy, compute the values of each state using a Monte Carlo simulation, choosing one of the random actions, performing it, and updating the value of the state. You could even imagine simulating 2 actions away from the state.
'''

import numpy as np

class Grid():
    def __init__(self, ylim=10, xlim=10, A0=(0,0), A=(3,7), B0=(9,9), B=(8,7), coords=[0,0], alpha=0.01, gamma=0.9):
        self.xlim=xlim
        self.ylim=ylim
        self.A0,self.A,self.B0,self.B=A0,A,B0,B
        self.coords=coords
        self.Q = np.zeros( (ylim,xlim) )
        self.alpha = alpha
        self.gamma = gamma

    def move_a(self,a):
        if a=='north':
            self.coords[0] = self.coords[0]-1
        elif a=='south':
            self.coords[0] = self.coords[0]+1
        elif a=='west':
            self.coords[1] = self.coords[1]-1
        elif a=='east':
            self.coords[1] = self.coords[1]+1

    def state_val_iter(self):
        for y in range(self.ylim):
            for x in range(self.xlim):
                self.coords[0]=y
                self.coords[1]=x
                newQ = 0
                for a in ['north','south','east','west']:
                    self.coords[0]=y
                    self.coords[1]=x
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
                    
                    v = self.Q[self.coords[0],self.coords[1]] # value of next state

                    newQ = newQ + (r+self.gamma*v)/4

                self.Q[y,x] = newQ

    def pi(self):
        return np.random.choice(['north','south','east','west'])

    def a_to_num(self,a):
        return {'north':0,'south':1,'east':2,'west':3}[a]

    def update(self,r,yx):
        self.Q[ yx[0], yx[1] ] = (1-self.alpha)*self.Q[ yx[0], yx[1] ]+self.alpha*r

    def monte_carlo(self,N):
        for n in range(N):
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
            
            v = self.Q[self.coords[0],self.coords[1]] # value of next state

            self.update( r+self.gamma*v , yx )

    def show_Q(self):

        import matplotlib.pyplot as plt
        import seaborn as sns; sns.set_theme()

        sns.heatmap(self.Q)
        plt.show()

    def show_iter_Q(self):
        import matplotlib.pyplot as plt
        import seaborn as sns; sns.set_theme()

        for n in range(100):
            self.state_val_iter()
            sns.heatmap(self.Q)
            plt.title("Iterative Bellman updates")
            plt.pause(0.0001)
            plt.clf()
        # plt.show()

    def show_monte_carlo(self):
        for n in range(100):
            grid.monte_carlo(100)
            import matplotlib.pyplot as plt
            import seaborn as sns; sns.set_theme()
            sns.heatmap(self.Q)
            plt.title("Monte Carlo Updates")
            plt.pause(0.0001)
            plt.clf()

if __name__=='__main__':

    # Iterative state value updates:

    grid = Grid(alpha=1,gamma=.99)
    grid.show_iter_Q()

    # Monte Carlo:

    # grid = Grid(alpha=.1,gamma=.99)
    # grid.show_monte_carlo()

