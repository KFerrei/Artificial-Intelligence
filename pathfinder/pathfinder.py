import numpy as np
import sys

def read_file(file_path):
    text_maze = open(file_path, "r")
    text_maze = text_maze.read()
    text_maze = text_maze.split("\n")

    size = text_maze[0].split(" ")
    size = (int(size[0]), int(size[1]))

    start_pos = text_maze[1].split(" ")
    start_pos = (int(start_pos[0])-1, int(start_pos[1])-1)

    end_pos = text_maze[2].split(" ")
    end_pos = (int(end_pos[0])-1, int(end_pos[1])-1)

    maze = np.array([text_maze[i].split(" ") for i in range(3, 3+size[0])],dtype=object)
    for i in range(size[0]):
        for j in range(size[1]):
            if maze[i,j] != 'X':
                maze[i,j] = int(maze[i,j])
    return size, start_pos, end_pos, maze

def write_path(maze, route, size):
    if route == "null":
        print(route)
    else:
        path = np.copy(maze)
        for r in route:
            path[r[0],r[1]] = '*'
    
        path_text = ''
        for i in range(size[0]):
            for j in range(size[1]):
                path_text +=  str(path[i,j]) +" "
            path_text = path_text[:-1]
            path_text += "\n"
        print(path_text)
        
def neighor(pos, maze,size):
    neighbors = {}
    if pos[0] != 0:
        if maze[pos[0]-1, pos[1]] != 'X':
            neighbors[(pos[0]-1, pos[1])] = maze[pos[0]-1, pos[1]]  
    
    if pos[0] != (size[0]-1):
        if maze[pos[0]+1, pos[1]] != 'X':
            neighbors[(pos[0]+1, pos[1])] = maze[pos[0]+1, pos[1]]
    
    if pos[1] != 0:
        if maze[pos[0], pos[1]-1] != 'X':
            neighbors[(pos[0], pos[1]-1)] = maze[pos[0], pos[1]-1]
    
    if pos[1] != (size[1]-1):
        if maze[pos[0], pos[1]+1] != 'X':
            neighbors[(pos[0], pos[1]+1)] = maze[pos[0], pos[1]+1]
    return neighbors

def traversal_BFS(start_pos, maze, size):
     # Initialize structure with initial vertex, at distance 0, with no predecessor
    queuing_structure = []
    queuing_structure.append((0, start_pos, None))

    # Initialize routing 
    explored_vertices = {}
    routing_table = {}

    while len(queuing_structure) > 0 :
        (distance_to_current_vertex, current_vertex, parent) = queuing_structure.pop(0)
        if current_vertex not in explored_vertices :
            # Store route to it        
            explored_vertices[current_vertex] = distance_to_current_vertex
            routing_table[current_vertex] = parent
            neighors = neighor(current_vertex, maze, size)
            for n in neighors:
                if n not in explored_vertices : 
                    cost = maze[n] - maze[current_vertex]
                    if cost < 0:
                        cost = 0
                    distance_to_neighbor = distance_to_current_vertex + 1 + cost
                    queuing_structure.append((distance_to_neighbor, n, current_vertex))
    return routing_table

def traversal_UCS(start_pos, maze, size):
    # Initialize structure with initial vertex, at distance 0, with no predecessor
    queuing_structure = []
    queuing_structure.append((0, start_pos, None))

    # Initialize routing 
    explored_vertices = {}
    routing_table = {}

    while len(queuing_structure) > 0 :
        (distance_to_current_vertex, current_vertex, parent) = queuing_structure.pop(0)
        if current_vertex not in explored_vertices :
            # Store route to it        
            explored_vertices[current_vertex] = distance_to_current_vertex
            routing_table[current_vertex] = parent
            neighors = neighor(current_vertex, maze, size)
            for n in neighors:
                if n not in explored_vertices : 
                    cost = maze[n] - maze[current_vertex]
                    if cost < 0:
                        cost = 0
                    distance_to_neighbor = distance_to_current_vertex + 1 + cost
                    queuing_structure.append((distance_to_neighbor, n, current_vertex))
        queuing_structure.sort(key=lambda x: x[0])
    return routing_table

def traversal_A_euclidean(start_pos, end_pos, maze, size):
    # Initialize structure with initial vertex, at distance 0, with no predecessor
    queuing_structure = []
    distance_cv_to_end = euclidean_distance(start_pos, end_pos)
    queuing_structure.append((distance_cv_to_end, start_pos, None))

    # Initialize routing 
    explored_vertices = {}
    routing_table = {}

    while len(queuing_structure) > 0 :
        (distance_to_current_vertex, current_vertex, parent) = queuing_structure.pop(0)
        if current_vertex not in explored_vertices :
            # Store route to it        
            explored_vertices[current_vertex] = distance_to_current_vertex
            routing_table[current_vertex] = parent
            neighors = neighor(current_vertex, maze, size)
            for n in neighors:
                if n not in explored_vertices : 
                    cost = maze[n] - maze[current_vertex]
                    if cost < 0:
                        cost = 0
                    distance_to_neighbor = distance_to_current_vertex + 1 + cost
                    distance_to_neighbor_2 = distance_to_neighbor + euclidean_distance(n, end_pos)
                    queuing_structure.append((distance_to_neighbor_2, n, current_vertex))
        queuing_structure.sort(key=lambda x: x[0])
    return routing_table

def traversal_A_manhattan(start_pos, end_pos, maze, size):
    # Initialize structure with initial vertex, at distance 0, with no predecessor
    queuing_structure = []
    man_dist = manhattan_distance(start_pos, end_pos,maze,size)
    distance_cv_to_end = man_dist
    queuing_structure.append((distance_cv_to_end, start_pos, None, man_dist))
    # Initialize routing 
    explored_vertices = {}
    routing_table = {}

    while len(queuing_structure) > 0 :
        (distance_to_current_vertex, current_vertex, parent, man_dist) = queuing_structure.pop(0)
        distance_to_current_vertex = distance_to_current_vertex - man_dist
        if current_vertex not in explored_vertices :
            # Store route to it        
            explored_vertices[current_vertex] = distance_to_current_vertex
            routing_table[current_vertex] = parent
            neighors = neighor(current_vertex, maze, size)
            for n in neighors:
                if n not in explored_vertices : 
                    cost = maze[n] - maze[current_vertex]
                    if cost < 0:
                        cost = 0
                    distance_to_neighbor = distance_to_current_vertex + 1 + cost 
                    man_dist = manhattan_distance(n, end_pos, maze,size)
                    queuing_structure.append((distance_to_neighbor+man_dist, n, current_vertex,man_dist))
        queuing_structure.sort(key=lambda x: x[0])
    return routing_table


def find_path(start_pos, end_pos, routing_table):
    if end_pos not in routing_table :
        return "null"
    route = [end_pos]
    while route[0] != start_pos :
        route.insert(0, routing_table[route[0]])
    return route

def euclidean_distance(pos1, pos2):
    distance = ((pos1[0]-pos2[0])**2 + (pos1[1]-pos2[1])**2)**0.5
    return distance

def manhattan_distance(pos1, pos2, maze,size):
    routing_table = traversal_UCS(pos1, maze, size)
    route = find_path(pos1, pos2, routing_table)
    distance = 0
    for r in range(1, len(route)):
        c = maze[route[r]] - maze[route[r-1]]
        if c<0:
            c = 0
        distance += 1+c
    return distance

def main():
    if len(sys.argv) == 4:
        name, map, algorithm, heuristic = sys.argv
    elif len(sys.argv) == 3:
        name, map, algorithm = sys.argv
    size, start_pos, end_pos, maze = read_file(map)
    if algorithm == "bfs":
        routing_table = traversal_BFS(start_pos, maze, size)     
        
    elif algorithm == "ucs":
        routing_table = traversal_UCS(start_pos, maze, size)
        
    elif algorithm == "astar":
        if heuristic == "euclidean":
            routing_table = traversal_A_euclidean(start_pos, end_pos, maze, size)   
        elif heuristic == "manhattan":
            routing_table = traversal_A_manhattan(start_pos, end_pos, maze, size)     
    route = find_path(start_pos, end_pos, routing_table)
    write_path(maze, route, size)


if __name__ == "__main__":
    main()