import argparse
import numpy as np
import random
import matplotlib.pyplot as plt
from environment import MountainCar, GridWorld

from typing import Union, Tuple, Optional # for type annotations


def set_seed(seed: int):
    
    np.random.seed(seed)


def round_output(places: int):
    def wrapper(fn):
        def wrapped_fn(*args, **kwargs):
            return np.round(fn(*args, **kwargs), places)
        return wrapped_fn
    return wrapper


def parse_args() -> Tuple[str, str, str, str, int, int, float, float, float]:
    """
    Parses all args and returns them. Returns:

        (1) env_type : A string, either "mc" or "gw" indicating the type of 
                    environment you should use
        (2) mode : A string, either "raw" or "tile"
        (3) weight_out : The output path of the file containing your weights
        (4) returns_out : The output path of the file containing your returns
        (5) episodes : An integer indicating the number of episodes to train for
        (6) max_iterations : An integer representing the max number of iterations 
                    your agent should run in each episode
        (7) epsilon : A float representing the epsilon parameter for 
                    epsilon-greedy action selection
        (8) gamma : A float representing the discount factor gamma
        (9) lr : A float representing the learning rate
    
    Usage:
        (env_type, mode, weight_out, returns_out, 
         episodes, max_iterations, epsilon, gamma, lr) = parse_args()
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("env", type=str, choices=["mc", "gw"])
    parser.add_argument("mode", type=str, choices=["raw", "tile"])
    parser.add_argument("weight_out", type=str)
    parser.add_argument("returns_out", type=str)
    parser.add_argument("episodes", type=int)
    parser.add_argument("max_iterations", type=int)
    parser.add_argument("epsilon", type=float)
    parser.add_argument("gamma", type=float)
    parser.add_argument("learning_rate", type=float)

    args = parser.parse_args()

    return (args.env, args.mode, args.weight_out, args.returns_out, 
            args.episodes, args.max_iterations, 
            args.epsilon, args.gamma, args.learning_rate)


@round_output(5)
def Q(W: np.ndarray, state: np.ndarray, 
      action: Optional[int] = None) -> Union[float, np.ndarray]:
    '''
    Helper function to compute Q-values for function-approximation 
    Q-learning.

    Note: Do not delete or move the `@round_output(5)` line on top of 
          this function. This just ensures that your Q-value is rounded to 5
          decimal places, which avoids some *pernicious* cross-platform 
          rounding errors.

    Parameters:
        W     (np.ndarray): Weight matrix with folded-in bias with 
                            shape (action_space, state_space+1).
        state (np.ndarray): State encoded as vector with shape (state_space,).
        action       (int): Action taken. Satisfies 0 <= action < action_space.

    Returns:
        If action argument was provided, returns float Q(state, action).
        Otherwise, returns array of Q-values for all actions from state,
        [Q(state, a_0), Q(state, a_1), Q(state, a_2)...] for all a_i.
    '''
   
    if action is not None:
        #selects weight for given action - is (1, 1)
        return np.dot(W[action], np.append(1,state))
    else:
        #gives q for each action hence- (3,1)
        return np.dot(W, np.append(1,state))
    


if __name__ == "__main__":
    set_seed(10301) 

    # Read in arguments
    (env_type, mode, weight_out, returns_out, 
     episodes, max_iterations, epsilon, gamma, lr) = parse_args()

    # Create environment
    if env_type == "mc":
        env =  MountainCar(mode=mode) 
    elif env_type == "gw":
        env = GridWorld(mode=mode)
    else: 
        raise Exception(f"Invalid environment type {env_type}")

    weightmatrix = np.zeros((env.action_space, env.state_space + 1))
    returns=[]
    for episode in range(episodes):

        curr_state = env.reset()

        curr_return=0.0

        for iteration in range(max_iterations):

            if epsilon==0:
                print("WOAH")
                qval = Q(weightmatrix, curr_state)
                action = np.argmax(qval) 
            elif np.random.rand() > epsilon: 
                qval = Q(weightmatrix, curr_state)
                action = np.argmax(qval) 
            else:
                action = np.random.randint(env.action_space)
    
           
            
            next_state, reward, done_flag = env.step(action)
            curr_return+=reward

                       rhs = reward + gamma * np.max(Q(weightmatrix, next_state))
            lhs = Q(weightmatrix, curr_state, action) - rhs
            weightmatrix[action] =weightmatrix[action]-  lr * lhs * np.append(1,curr_state)  
            curr_state = next_state
             
            

            if done_flag==True:
                break
        returns.append(curr_return)
                
        
        
    np.savetxt(weight_out, weightmatrix, fmt="%.18e", delimiter=" ")
    
    rewardsoutput = open(returns_out, 'w')
    for i in returns:
        rewardsoutput.write(str(i))
        rewardsoutput.write("\n")
    rewardsoutput.close()
    meanreturns=[]
    for i in range(1,len(returns)+1):
        if i<=24:
            print(i)
            meanreturns.append(sum(returns[:i])/len(returns[:i]))
        else:
            meanreturns.append(sum(returns[i-25:i])/25)


    plt.plot([i for i in range(2500)],returns) 
    plt.plot([i for i in range(2500)],meanreturns,label='rolling mean over 25') 
    plt.xlabel("Episodes") 
    plt.ylabel("Returns") 
    plt.title("Return and rolling mean returns over episodes") 
    plt.legend() 
    plt.show() 



    



