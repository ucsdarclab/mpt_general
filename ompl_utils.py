''' Useful ompl files
'''
import numpy as np

try:
    from ompl import base as ob
    from ompl import geometric as og
    from ompl import util as ou
except ImportError:
    raise "Run code from a container with OMPL installed"


def get_ompl_state(space, state):
    ''' Returns an OMPL state
    '''
    ompl_state = ob.State(space)
    for i in range(7):
        ompl_state[i] = state[i]
    return ompl_state

def get_numpy_state(state):
    ''' Return the state as a numpy array.
    :param state: An ob.State from ob.RealVectorStateSpace
    :return np.array:
    '''
    return np.array([state[i] for i in range(7)])