import math
from queue import PriorityQueue as pq
import random
import numpy as np
import matplotlib.pyplot as plt

class point :
    def __init__(self, x, y) :
        (self.x, self.y) = (x, y)
    def __repr__(self) :
        return 'point({}, {})'.format(self.x, self.y)
    def __lt__(self, other) :
        if self.x != other.x :
            return self.x < other.x
        return self.y < other.y
    def __eq__(self, other) :
        return self.x == other.x and self.y == other.y
    def __hash__(self) :
        return hash((self.x, self.y))

class rectangle :
    def __init__(self, x1, y1, x2, y2) :
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
    def __repr__(self) :
        return 'rectangle({}, {}, {}, {})'.format(self.x1, self.y1, self.x2, self.y2)
    def __getitem__(self, index) :
        if index == 0 :
            return point(self.x1, self.y1)
        if index == 1 :
            return point(self.x2, self.y1)
        if index == 2 :
            return point(self.x2, self.y2)
        if index == 3 :
            return point(self.x1, self.y2)
    def __contains__(self, point) :
        for i in range(4) :
            if point == self[i] :
                return True
        return False
    def width(self) :
        return self.x2 - self.x1
    def height(self) :
        return self.y2 - self.y1
    def point(self, index) :
        return self[index]

class graph :
    def __init__(self) :
        self.graph = dict()
    def __repr__(self) :
        return 'graph({})'.format(self.graph)
    def __getitem__(self, vertex) :
        return self.graph[vertex]
    def connect(self, vertex, edge) :
        if vertex not in self.graph :
            self.graph[vertex] = set()
        self.graph[vertex].add(edge)

class room :
    def __init__(self, index) :
        # index
        self.index = index
        # partition
        self.proportion = None
        self.prefered_ratio = None
        self.subrooms = list()
        self.rectangle = None
        # connection
        self.connection = set()
        self.geometry = list()
    def __repr__(self) :
        return 'room({})'.format(self.index)
    def __lt__(self, other) :
        #return self.index < other.index
        return False
    def __eq__(self, other) :
        return self.index == other.index
    def __hash__(self) :
        return hash((room, self.index))

def partition(subrooms, boundary_rectangle, partition_depth, partition_graph) :
    for i in range(4) :
        partition_graph.connect(boundary_rectangle[i], (boundary_rectangle[(i - 1) % 4], partition_depth))
        partition_graph.connect(boundary_rectangle[i], (boundary_rectangle[(i + 1) % 4], partition_depth))
    if len(subrooms) == 0 :
        pass
    elif len(subrooms) == 1 :
        subrooms[0].rectangle = boundary_rectangle
        partition(subrooms[0].subrooms, boundary_rectangle, partition_depth + 1, partition_graph)
    elif False and len(subrooms) == 2 :
        # special case for 2 ?
        pass
    else :
        # rectangle parameters
        if boundary_rectangle.width() >= boundary_rectangle.height() :
            (width_start, width_end, width) = (boundary_rectangle.x1, boundary_rectangle.x2, boundary_rectangle.width())
            (height_start, height_end, height) = (boundary_rectangle.y1, boundary_rectangle.y2, boundary_rectangle.height())
        else :
            (width_start, width_end, width) = (boundary_rectangle.y1, boundary_rectangle.y2, boundary_rectangle.height())
            (height_start, height_end, height) = (boundary_rectangle.x1, boundary_rectangle.x2, boundary_rectangle.width())
        # greedy partition test
        total_proportion = 0
        for subroom in subrooms :
            total_proportion += subroom.proportion
        cumulative_proportion = subrooms[0].proportion
        partition_proportion = cumulative_proportion
        best_width = (cumulative_proportion / total_proportion) * width
        best_ratio_error = abs(abs(math.log2(height / best_width)) - subrooms[0].prefered_ratio)
        ###
        #best_ratio_error += abs(math.log2(height / (width - best_width)))
        ###
        partition_count = 1
        for i in range(1, len(subrooms)) :
            cumulative_proportion += subrooms[i].proportion
            current_width = (cumulative_proportion / total_proportion) * width
            current_ratio_error = abs(abs(math.log2(height / ((i + 1) * current_width))) - subrooms[i].prefered_ratio)
            ###
            #if i + 1 < len(subrooms) : current_ratio_error += abs(math.log2(height / (width - current_width)))
            ###
            if current_ratio_error <= best_ratio_error :
                partition_proportion = cumulative_proportion
                best_width = current_width
                best_ratio_error = current_ratio_error
                partition_count = i + 1
            else :
                break
        # partition parameters
        if partition_count < len(subrooms) :
            partition_width = width_start + best_width
        else :
            partition_width = width_end
        partition_heights = [height_start]
        cumulative_proportion = 0
        for i in range(partition_count - 1) :
            cumulative_proportion += subrooms[i].proportion
            partition_heights.append(height_start + (cumulative_proportion / partition_proportion) * height)
        partition_heights.append(height_end)
        # commit partition
        for i in range(partition_count) :
            # subpartition
            if boundary_rectangle.width() >= boundary_rectangle.height() :
                partition_rectangle = rectangle(width_start, partition_heights[i], partition_width, partition_heights[i + 1])
            else :
                partition_rectangle = rectangle(partition_heights[i], width_start, partition_heights[i + 1], partition_width)
            subrooms[i].rectangle = partition_rectangle
            partition(subrooms[i].subrooms, partition_rectangle, partition_depth + 1, partition_graph)
        if partition_count < len(subrooms) :
            # remaining area partition
            if boundary_rectangle.width() >= boundary_rectangle.height() :
                partition_rectangle = rectangle(partition_width, height_start, width_end, height_end)
            else :
                partition_rectangle = rectangle(height_start, partition_width, height_end, width_end)
            partition(subrooms[partition_count:], partition_rectangle, partition_depth, partition_graph)

def geometrize(partition_graph) :
    pass

def pathfinding(proper_rooms, partition_graph) :
    #print('pathfinding')
    connected = set()
    paths = list()
    for start_room in proper_rooms :
        #print('proper room')
        for end_room in start_room.connection :
            #print('end room')
            if (start_room, end_room) in connected or (end_room, start_room) in connected :
                continue
            least_cost = dict()
            queue = pq()
            for i in range(4) :
                least_cost[start_room.rectangle[i]] = (0, None)
                queue.put((0, start_room.rectangle[i]))
            while queue.empty() == False :
                (cumulative_cost, current_point) = queue.get()
                #print('pop', cumulative_cost, current_point)
                if current_point in end_room.rectangle :
                    end_point = current_point
                    #print('pop path found')
                    break
                if least_cost[current_point][0] < cumulative_cost :
                    #print('pop better partial path exist')
                    continue
                for (adjacent_point, weight) in partition_graph[current_point] :
                    if weight == 0 :
                        continue
                    cost = least_cost[current_point][0] + math.pow(2, weight) * math.sqrt(math.pow(adjacent_point.x - current_point.x, 2) + math.pow(adjacent_point.y - current_point.y, 2))
                    #print('edge', cost, adjacent_point)
                    if adjacent_point in least_cost and least_cost[adjacent_point][0] < cost :
                        #print('better partial path exist')
                        continue
                    #print('push')
                    least_cost[adjacent_point] = (cost, current_point)
                    queue.put((cost, adjacent_point))
            else :
                print('error: queue empty')
                input()
            #print('realize path')
            path = list()
            path_point = end_point
            while path_point not in start_room.rectangle :
                #print('path point', path_point)
                path.append(path_point)
                path_point = least_cost[path_point][1]
            path.append(path_point)
            connected.add((start_room, end_room))
            paths.append(((start_room, end_room), path))
            #print('path complete')
    return paths

cumulative_index = 0

def generate_house(breadth, depth, room_dictionary) :
    global cumulative_index
    subrooms = list()
    if breadth < 2 or depth < 0 :
        return subrooms
    b = random.randint(2, breadth)
    for i in range(b) :
        d = random.randint(0, depth)
        room_dictionary[cumulative_index] = room(cumulative_index)
        subroom = room_dictionary[cumulative_index]
        cumulative_index += 1
        subroom.proportion = (1 + random.random()) * 4
        subroom.prefered_ratio = 0
        subroom.subrooms = sorted(generate_house(b - 1, d - 1, room_dictionary), key = (lambda subroom : subroom.proportion))
        subrooms.append(subroom)
    return subrooms

def randomize_room_connection(count, room_dictionary) :
    proper_room = list()
    for (room_index, room) in room_dictionary.items() :
        if len(room.subrooms) == 0 :
            proper_room.append(room)
    for i in range(count) :
        start_room = proper_room[random.randint(0, len(proper_room) - 1)]
        end_room = proper_room[random.randint(0, len(proper_room) - 1)]
        if start_room != end_room :
            start_room.connection.add(end_room)
            end_room.connection.add(start_room)
    return proper_room

def color(depth) :
    return (math.cos(math.tau * depth / 8 + 0 * math.tau / 3) / 2 + 0.5, math.cos(math.tau * depth / 8 + 1 * math.tau / 3) / 2 + 0.5, math.cos(math.tau * depth / 8 + 2 * math.tau / 3) / 2 + 0.5)

def print_partition_hierarchy(subrooms, depth = 0) :
    for subroom in subrooms :
        print('{}{}'.format('    ' * depth, subroom.rectangle))
        print_partition_hierarchy(subroom.subrooms, depth + 1)

def draw_partition(subrooms, array, depth = 0) :
    for subroom in subrooms :
        if len(subroom.subrooms) > 0 :
            draw_partition(subroom.subrooms, array, depth + 1)
        else :
            (x1, y1, x2, y2) = (int(subroom.rectangle.x1), int(subroom.rectangle.y1), int(subroom.rectangle.x2), int(subroom.rectangle.y2))
            array[x1 : x2, y1 : y2, :] = 1
            array[x1 + 1 : x2, y1 + 1 : y2, :] = 0

def draw_paths(paths, array) :
    for i in range(len(paths)) :
        path = paths[i][1]
        for j in range(len(path) - 1) :
            (x1, y1) = (int(path[j].x), int(path[j].y))
            (x2, y2) = (int(path[j + 1].x), int(path[j + 1].y))
            if x1 > x2 :
                (x1, x2) = (x2, x1)
            if y1 > y2 :
                (y1, y2) = (y2, y1)
            array[x1 : x2 + 1, y1 : y2 + 1, :] = 1 - np.array(color(i))

def draw_vertex(array, partition_graph) :
    min_vertex_value = dict()
    for (vertex, edges) in partition_graph.graph.items() :
        for (v, w) in edges :
            if v not in min_vertex_value or min_vertex_value[v] > w :
                min_vertex_value[v] = w
                array[int(v.x), int(v.y), :] = color(w)

sample = np.zeros((8, 1, 3))
for i in range(8) :
    sample[i, 0] = color(i)

while True :

    cumulative_index = 0
    room_dictionary = dict()

    room_dictionary[cumulative_index] = room(cumulative_index)
    house = room_dictionary[cumulative_index]
    cumulative_index += 1
    house.proportion = 1
    house.prefered_ratio = 0
    house.subrooms = generate_house(11, 7, room_dictionary)

    proper_room = randomize_room_connection(15, room_dictionary)

    boundary_rectangle = rectangle(0, 0, (1 + random.random()) * 100, (1 + random.random()) * 100)
    partition_graph = graph()
    partition([house], boundary_rectangle, 0, partition_graph)

    paths = pathfinding(proper_room, partition_graph)

    array = np.ones((int(boundary_rectangle.width()) + 1, int(boundary_rectangle.height()) + 1, 3))
    draw_partition([house], array)
    draw_paths(paths, array)
    draw_vertex(array, partition_graph)

    plt.subplot(122)
    plt.imshow(array)
    plt.subplot(121)
    plt.imshow(sample)
    #print_partition_hierarchy([house])
    #print(partition_graph)
    plt.show()