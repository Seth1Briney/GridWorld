'''
Seth Briney

Team:
    Bryce Waggener
    Mitchell White

For the lab in week 1, you will work in a team to implement the gridworld simulator. The goal is to create a testbed environment to test policy evaluation and policy improvement.  This week the goal is to test policy evaluation for the random policy. Here is what you will turn in:

1. Who was on your team?

    Bryce Waggener
    Mitchell White

2. What did you learn?

    I'm having this issue when I go to update q values:
    lab3.py:105: RuntimeWarning: invalid value encountered in double_scalars

    I'm confused about why this is happening.

    I realized what was happening is I was setting a new position to self.position, then modifying it. Since the equality is by reference I was accidentally modifying self.position, then dividing by 0 since there are 0's in counts where agent hadn't been yet. I changed it to new_position=self.position.copy(), and this fixed the error.

3. How far did you get in implementing k-bandits?

    I did this week 1. I also finished gridworld.

4. What questions do you have?

    Did the assignment change mid-way? At first I thought it said to track average returns over runs, like in the k-bandits problem. I'm going to do that anyway, since I already started.

Each team member should keep a copy of the Python file with the shared code. You will complete this for homework.

 I will discuss the implementation of the random policy for 5x5 gridworld described in Chapter 3. Description of the program:

1. The program should simulate policy evaluation until the max change in value is less than theta (see pseudocode in 4.1). You can set theta to 0.01

2. The program should compute the average return at each time step for each run

'''


'''

List of probs for each direction representing probability of direction.

continue until delta<theta, where delta is the maximum difference between state values, and theta is some threshold. Perhaps 0.01.

'''


'''
grid_world1.py
5x5 grid world from Chapter 3 Sutton and Barto
We will solve it using DP, so we will compute the expected reward for 
each action

lab3.py:105: RuntimeWarning: invalid value encountered in double_scalars
  new_val = old_val+(r-old_val)/count

'''

import numpy as np

def argmax_rand_tie(x):
    x = np.array(x)
    x=x.reshape(-1,)
    shuf = np.array([n for n in range(len(x))]).reshape(-1,)
    np.random.shuffle(shuf)

    argmax=np.argmax(x[shuf])

    return shuf[argmax]

class Grid:
    '''
    2D array of Grid_Cells
    loop through all of the cells, update the values based on the expected reward and discounted value.
    v(s) = E(r + gamma * v(s'))
    the action a determines the reward r and the next state, but it can bex different for different states
    = sum over a of (r + gamma * v(s')) * p(a)

    '''

    def __init__(self, size = 5, theta = 0.01, gamma = .9):

        self.size = size
        self.position = [0,0]
        self.special = {(0,1):(+10,[4,1]),(0,3):(+5,[2,3])}
        self.gamma = gamma
        self.theta = theta

    def transition(self,direction):
        new_pos = self.position.copy()
        if direction=='up':
            new_pos[0]=new_pos[0]-1
        elif direction=='down':
            new_pos[0]=new_pos[0]+1
        elif direction=='left':
            new_pos[1]=new_pos[1]-1
        elif direction=='right':
            new_pos[1]=new_pos[1]+1
        return new_pos

    def step(self,direction):

        if tuple(self.position) in self.special:
            r, new_pos = self.special[tuple(self.position)]
        elif direction=='up' and self.position[0]==0:
            r = -1
            new_pos = self.position.copy()
        elif direction=='down' and self.position[0]==self.size-1:
            r = -1
            new_pos = self.position.copy()
        elif direction=='left' and self.position[1]==0:
            r = -1
            new_pos = self.position.copy()
        elif direction=='right' and self.position[1]==self.size-1:
            r = -1
            new_pos = self.position.copy()
        else:
            r = 0
            new_pos = self.transition(direction)

        return r, new_pos

    def pi(self,q,epsilon):
        # epsilon greedy policy

        y = self.position[0]
        x = self.position[1]

        if np.random.rand()<epsilon:
            return np.random.choice(['up','down','left','right'])
        else:
            if y==0:
                up = q[y,x]
            else:
                up = q[y-1,x]
            if y==self.size-1:
                down = q[y,x]
            else:
                down = q[y+1,x]
            if x==0:
                left = q[y,x]
            else:
                left = q[y,x-1]
            if x==self.size-1:
                right = q[y,x]
            else:
                right = q[y,x+1]

            best = argmax_rand_tie([up,down,right,left])

            return ['up','down','right','left'][best]


    def update(self, old_val, r, count):
        new_val = old_val+(r-old_val)/count
        return new_val

    def simulate(self, runs=1000, time=1000, p=.995):

        epsilon = 1 # this will eponentially decrease over runs by p
        q_estimates = np.zeros((self.size,self.size))
        R = np.zeros((runs,))
        counts = np.zeros((self.size,self.size))

        for run in range(runs):

            tmp_q_estimates = q_estimates.copy()

            for step in range(time):

                counts[self.position[0],self.position[1]] = counts[self.position[0],self.position[1]] + 1
                direction = self.pi(q_estimates,epsilon)
                r, new_pos = self.step(direction)
                r = r + self.gamma*q_estimates[new_pos[0],new_pos[1]]
                R[run] = R[run] + r
                old_val = q_estimates[self.position[0],self.position[1]]
                count = counts[self.position[0],self.position[1]]
                q_estimates[self.position[0],self.position[1]] = self.update(old_val,r, count)
                self.position = new_pos

            epsilon = p*epsilon

            if np.linalg.norm(q_estimates - tmp_q_estimates)<self.theta:
                return q_estimates, R, run

        return q_estimates, R, runs


if __name__ == '__main__':

    theta = 0.01
    grid = Grid(theta=theta)
    q, R, runs = grid.simulate(1000,10000)
    x = np.arange(0,R.size,1)
    import matplotlib.pyplot as plt
    import seaborn as sns; sns.set_theme()
    plt.plot(x,R)
    plt.title('reward per run, exit on threshold theta='+str(theta)+' after '+str(runs)+' runs.')
    plt.ylabel('R')
    plt.xlabel('run')
    plt.show()
    sns.heatmap(q,cbar=False)
    plt.title('final state values once max threshold of theta reached')
    plt.show()

    
