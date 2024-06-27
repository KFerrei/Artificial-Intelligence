"""
Nom du fichier : viterbi_3D.py 
Auteur : Kevin Ferreira
Date de creation : 11 mai 2023

Description : Assignement 3

"""
# IMPORTS
import numpy as np
import sys

# READING AND WRITING FILES
def read_file(file_path):
    """
    Read a file and create all the parameters
    Input: path to the file
    Output: map_size, map_data, nb_observations, sensor_observations, sensor_error
    """
    file = open(file_path, "r")
    
    # Read the size of the map
    map_size = tuple(map(int, file.readline().strip().split()))

    # Read the map data
    map_data = np.zeros((map_size[0], map_size[1], map_size[2]), dtype='object')
    for i in range(map_size[0]):
        row = file.readline().strip().split()
        for j in range(map_size[1]):
            for h in range(map_size[2]):
                if row[j+h*map_size[1]] == 'X':
                    map_data[i,j,h] = 'X'
                else:
                    map_data[i,j,h] = int(row[j+h*map_size[1]])

    # Read the number of sensor observations
    nb_observations = int(file.readline().strip().split()[0])

    # Read the sensor observations
    sensor_observations = []
    for i in range(nb_observations):
        observation = file.readline().strip()
        sensor_observations.append(observation)

    # Read the sensor error rate
    sensor_error = float(file.readline().strip())
    
    return map_size, map_data, nb_observations, sensor_observations, sensor_error

def write_npz(nb_obs, map_size, K, S, trellis):
    """
    Write the output in a npz file
    Input: Number of observation, the size of the map, number of traversable points, trellis array
    Output: 
    """
    # Create the numpy array of the same size as the map
    list_t = []
    for t in range(nb_obs):
        obs_t = np.zeros((map_size[0], map_size[1], map_size[2]))
        for k in range(K):
            obs_t[S[k][0], S[k][1], S[k][2]] = trellis[k,t]
        list_t.append(obs_t)
    # Write the output in a npz file
    np.savez("output.npz", *list_t)

# INITIALIZATION OF VARIABLES
# Init O
def get_O():
    N = 64
    return N, [format(i, '06b') for i in range(N)]

#Init S 
def get_S(map_size, map_data):
    S = []
    for i in range(map_size[0]):
        for j in range(map_size[1]):
            for h in range(map_size[2]):
                if map_data[i,j,h] != 'X':
                    S.append((i,j,h))
    return S, len(S)

# Init Q
def get_Q(K):
    return np.full((K,), 1/K)

#Init Tm
def get_Tm(K, map_size, map_data, S):
    Tm = np.zeros((K, K))
    for k in range(K):
        i, j, h = S[k]
        neighbors = []
        for ni, nj, nh in [(i, j-1, h), (i, j+1, h), (i-1, j, h), (i+1, j, h), (i, j, h-1), (i, j, h+1)]:
            if 0 <= ni < map_size[0] and 0 <= nj < map_size[1] and 0 <= nh < map_size[2] and map_data[ni, nj, nh] != 'X':
                neighbors.append((ni, nj, nh))
        nb_neighbors = len(neighbors)
        for neighbor in neighbors:
            Tm[k, S.index(neighbor)] = 1 / nb_neighbors
    return Tm

#Init Em
def get_Em(K, N, S, O, map_data, map_size, sensor_error):
    Em = np.zeros((K, N))
    for k in range(K):
        pos = S[k]
        for n in range(N):
            d = 0
            obs = O[n]
            true_N = int(pos[0] == 0 or map_data[pos[0]-1, pos[1], pos[2]]=='X')
            true_S = int(pos[0] == (map_size[0]-1) or map_data[pos[0]+1, pos[1], pos[2]]=='X')
            true_W = int(pos[1] == 0 or map_data[pos[0], pos[1]-1, pos[2]]=='X')
            true_E = int(pos[1] == (map_size[1]-1) or map_data[pos[0], pos[1]+1, pos[2]]=='X')
            true_T = int(pos[2] == 0 or map_data[pos[0], pos[1], pos[2]-1]=='X')
            true_B = int(pos[2] == (map_size[2]-1) or map_data[pos[0], pos[1], pos[2]+1]=='X')
            d = int(true_N != int(obs[0])) + int(true_S != int(obs[1])) + int(true_W != int(obs[2])) + int(true_E != int(obs[3])) + int(true_T != int(obs[4])) + int(true_B != int(obs[5]))
            Em[k,n] = ((1-sensor_error)**(6-d))*(sensor_error**d)
    return Em

# VIRTERBI ALGORITHM 
def viterbi(O, S, Q, Y, Tm, Em):
    """
    Input:
        O: Observation space O = {o1, o2,...,oN}
        S: State space S = {s1,s2,...,sK}, where K refers to the traversable positions.
        Q: Array of initial probabilities 
        Y: A sequence of observations Y = (y1,y2,...,yT).
        Tm: Transition matrix of size K x K.
        Em: Emission matrix of size K x N.
    Output: Trellis matrix
    """
    K = len(S)  # Number of states
    T = len(Y)  # Length of observation sequence
    
    # Initialize trellis matrix
    trellis = np.zeros((K, T))

    for i in range(K):
        trellis[i, 0] = Q[i] * Em[i, O.index(Y[0])]
    for j in range(1, T):
        for i in range(K):
            temp = [trellis[k, j - 1] * Tm[k, i] * Em[i, O.index(Y[j])] for k in range(K)]
            trellis[i, j] = max(temp)
    return trellis

# MAIN        
def main():
    name, file_path = sys.argv
    map_size, map_data, nb_obs, sensor_observations, sensor_error = read_file(file_path)
    N, O = get_O()
    S, K = get_S(map_size, map_data)
    Q = get_Q(K)
    Tm = get_Tm(K, map_size, map_data, S)
    Em = get_Em(K, N, S, O, map_data, map_size, sensor_error)
    trellis = viterbi(O, S, Q, sensor_observations, Tm, Em)
    write_npz(nb_obs, map_size, K, S, trellis)



if __name__ == "__main__":
    main()