import math
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

class area :
    def __init__(self, index) :
        # index
        self.index = index
        self.is_room = False
        # partition
        self.proportion = None
        self.prefered_ratio = None
        self.subareas = list()
        self.rectangle = None
        # connection
        self.connection = set()
        self.geometry = list()
        self.door_segment = list()
    def __repr__(self) :
        return 'area({})'.format(self.index)
    def __lt__(self, other) :
        #return self.index < other.index
        return False
    def __eq__(self, other) :
        return self.index == other.index
    def __hash__(self) :
        return hash((area, self.index))

# partition: assign rectangle to area
# ??? todo or not todo: assign rectangle to room only ???
# return partition graph
def partition(area_hierarchy, boundary_rectangle, partition_depth, partition_graph) :
    for i in range(4) :
        partition_graph.connect(boundary_rectangle[i], (boundary_rectangle[(i - 1) % 4], partition_depth))
        partition_graph.connect(boundary_rectangle[i], (boundary_rectangle[(i + 1) % 4], partition_depth))
    if len(area_hierarchy) == 0 :
        pass
    elif len(area_hierarchy) == 1 :
        area_hierarchy[0].rectangle = boundary_rectangle
        partition(area_hierarchy[0].subareas, boundary_rectangle, partition_depth + 1, partition_graph)
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
        for subarea in area_hierarchy :
            total_proportion += subarea.proportion
        cumulative_proportion = area_hierarchy[0].proportion
        partition_proportion = cumulative_proportion
        best_width = (cumulative_proportion / total_proportion) * width
        best_ratio_error = abs(abs(math.log2(height / best_width)) - area_hierarchy[0].prefered_ratio)
        partition_count = 1
        for i in range(1, len(area_hierarchy)) :
            cumulative_proportion += area_hierarchy[i].proportion
            current_width = (cumulative_proportion / total_proportion) * width
            current_ratio_error = abs(abs(math.log2(height / ((i + 1) * current_width))) - area_hierarchy[i].prefered_ratio)
            if current_ratio_error <= best_ratio_error :
                partition_proportion = cumulative_proportion
                best_width = current_width
                best_ratio_error = current_ratio_error
                partition_count = i + 1
            else :
                break
        # partition parameters
        if partition_count < len(area_hierarchy) :
            partition_width = width_start + best_width
        else :
            partition_width = width_end
        partition_heights = [height_start]
        cumulative_proportion = 0
        for i in range(partition_count - 1) :
            cumulative_proportion += area_hierarchy[i].proportion
            partition_heights.append(height_start + (cumulative_proportion / partition_proportion) * height)
        partition_heights.append(height_end)
        # commit partition
        for i in range(partition_count) :
            # subpartition
            if boundary_rectangle.width() >= boundary_rectangle.height() :
                partition_rectangle = rectangle(width_start, partition_heights[i], partition_width, partition_heights[i + 1])
            else :
                partition_rectangle = rectangle(partition_heights[i], width_start, partition_heights[i + 1], partition_width)
            area_hierarchy[i].rectangle = partition_rectangle
            partition(area_hierarchy[i].subareas, partition_rectangle, partition_depth + 1, partition_graph)
        if partition_count < len(area_hierarchy) :
            # remaining area partition
            if boundary_rectangle.width() >= boundary_rectangle.height() :
                partition_rectangle = rectangle(partition_width, height_start, width_end, height_end)
            else :
                partition_rectangle = rectangle(height_start, partition_width, height_end, width_end)
            partition(area_hierarchy[partition_count:], partition_rectangle, partition_depth, partition_graph)

# geometrize: define geometry
# return room point dictionary
def geometrize(room_index_dictionary, partition_graph, room_point_dictionary) :
    # interception
    intercept_x = dict()
    intercept_y = dict()
    for current_point in partition_graph.graph :
        if current_point.x not in intercept_x :
            intercept_x[current_point.x] = set()
        intercept_x[current_point.x].add(current_point.y)
        if current_point.y not in intercept_y :
            intercept_y[current_point.y] = set()
        intercept_y[current_point.y].add(current_point.x)
        for (adjacent_point, _) in partition_graph[current_point] :
            if current_point.x == adjacent_point.x :
                intercept_x[current_point.x].add(adjacent_point.y)
            else :
                intercept_y[current_point.y].add(adjacent_point.x)
    # sorting interception
    for x in intercept_x :
        intercept_x[x] = sorted(intercept_x[x])
    for y in intercept_y :
        intercept_y[y] = sorted(intercept_y[y])
    # geometrize areas
    for (index, room) in room_index_dictionary.items() :
        geometry = list()
        (x1, y1, x2, y2) = (room.rectangle.x1, room.rectangle.y1, room.rectangle.x2, room.rectangle.y2)
        index_x1 = intercept_y[y1].index(x1)
        while intercept_y[y1][index_x1] < x2 :
            geometry.append(point(intercept_y[y1][index_x1], y1))
            index_x1 += 1
        index_y1 = intercept_x[x2].index(y1)
        while intercept_x[x2][index_y1] < y2 :
            geometry.append(point(x2, intercept_x[x2][index_y1]))
            index_y1 += 1
        index_x2 = intercept_y[y2].index(x2)
        while intercept_y[y2][index_x2] > x1 :
            geometry.append(point(intercept_y[y2][index_x2], y2))
            index_x2 -= 1
        index_y2 = intercept_x[x1].index(y2)
        while intercept_x[x1][index_y2] > y1 :
            geometry.append(point(x1, intercept_x[x1][index_y2]))
            index_y2 -= 1
        room.geometry = geometry
        for vertex in geometry :
            if vertex not in room_point_dictionary :
                room_point_dictionary[vertex] = set()
            room_point_dictionary[vertex].add(room)
    #return room_point_dictionary

cumulative_area_index = None

# return area hierarchy, room dictionary
#min_area_breadth, max_area_breadth, min_area_depth, max_area_depth
def generate_area_hierarchy(recursive_depth, area_breadth, area_depth, area_index_dictionary) :
    global cumulative_area_index
    if recursive_depth == 0 :
        current_area_index = cumulative_area_index
        cumulative_area_index += 1
        area_index_dictionary[current_area_index] = area(current_area_index)
        area_index_dictionary[current_area_index].proportion = 1 + random.random()
        area_index_dictionary[current_area_index].prefered_ratio = 1
        area_index_dictionary[current_area_index].subareas = generate_area_hierarchy(recursive_depth + 1, area_breadth, area_depth, area_index_dictionary)
        return [area_index_dictionary[current_area_index]]
    else :
        if area_depth[1] <= 0 :
            return list()
        subareas = list()
        current_area_breadth = random.randint(area_breadth[0], area_breadth[1])
        for i in range(current_area_breadth) :
            current_area_index = cumulative_area_index
            cumulative_area_index += 1
            area_index_dictionary[current_area_index] = area(current_area_index)
            area_index_dictionary[current_area_index].proportion = 1 + random.random()
            area_index_dictionary[current_area_index].prefered_ratio = random.random() - 0.5
            next_max_area_depth = max(0, random.randint(area_depth[0], area_depth[1]) - 1)
            area_index_dictionary[current_area_index].subareas = generate_area_hierarchy(recursive_depth + 1, area_breadth, (area_depth[0] - 1, area_depth[1] - 1), area_index_dictionary)
            subareas.append(area_index_dictionary[current_area_index])
        return sorted(subareas, key = (lambda area : area.proportion))

def identify_room(area_hierarchy, room_index_dictionary) :
    for area in area_hierarchy :
        if len(area.subareas) == 0 :
            room_index_dictionary[area.index] = area
        else :
            identify_room(area.subareas, room_index_dictionary)

def generate_connection(room_index_dictionary, room_point_dictionary, index) :
    door_segment = list()
    connected = {room_index_dictionary[index]}
    available = dict()
    for i in range(len(room_index_dictionary[index].geometry)) :
        first_point = room_index_dictionary[index].geometry[i]
        second_point = room_index_dictionary[index].geometry[(i + 1) % len(room_index_dictionary[index].geometry)]
        if abs(first_point.x - second_point.x) + abs(first_point.y - second_point.y) <= 2 :
            continue
        for incident_room in room_point_dictionary[first_point] :
            if incident_room not in connected and incident_room in room_point_dictionary[second_point] :
                if incident_room not in available :
                    available[incident_room] = set()
                available[incident_room].add((room_index_dictionary[index], first_point, second_point))
    while len(connected) != len(room_index_dictionary) :
        (second_room, connections) = random.choice(list(available.items()))
        (first_room, first_point, second_point) = random.choice(list(connections))
        door_segment.append((first_room, second_room, first_point, second_point))
        connected.add(second_room)
        available.pop(second_room)
        for i in range(len(second_room.geometry)) :
            third_point = second_room.geometry[i]
            fourth_point = second_room.geometry[(i + 1) % len(second_room.geometry)]
            if abs(third_point.x - fourth_point.x) + abs(third_point.y - fourth_point.y) <= 2 :
                continue
            for third_room in room_point_dictionary[third_point] :
                if third_room not in connected and third_room in room_point_dictionary[fourth_point] :
                    if third_room not in available :
                        available[third_room] = set()
                    available[third_room].add((second_room, third_point, fourth_point))
    return door_segment



def color_sampler(depth) :
    return (math.cos(math.tau * depth / 8 + 0 * math.tau / 3) / 2 + 0.5, math.cos(math.tau * depth / 8 + 1 * math.tau / 3) / 2 + 0.5, math.cos(math.tau * depth / 8 + 2 * math.tau / 3) / 2 + 0.5)

def print_partition_hierarchy(subareas, depth = 0) :
    for subarea in subareas :
        print('{}{}'.format('    ' * depth, subarea.rectangle))
        print_partition_hierarchy(subarea.subareas, depth + 1)

def draw_partition(array, subareas, depth = 0) :
    for subarea in subareas :
        if len(subarea.subareas) > 0 :
            draw_partition(array, subarea.subareas, depth + 1)
        else :
            (x1, y1, x2, y2) = (int(subarea.rectangle.x1), int(subarea.rectangle.y1), int(subarea.rectangle.x2), int(subarea.rectangle.y2))
            array[x1 : x2, y1 : y2, :] = 1
            array[x1 + 1 : x2, y1 + 1 : y2, :] = 0

def draw_vertex(array, partition_graph) :
    min_vertex_value = dict()
    for (vertex, edges) in partition_graph.graph.items() :
        for (v, w) in edges :
            if v not in min_vertex_value or min_vertex_value[v] > w :
                min_vertex_value[v] = w
                array[int(v.x), int(v.y), :] = color_sampler(w)

def draw_connection(array, door_segment) :
    for (_, _, first_point, second_point) in door_segment :
        (x1, y1, x2, y2) = (int(first_point.x), int(first_point.y), int(second_point.x), int(second_point.y))
        if x1 > x2 :
            (x1, x2) = (x2, x1)
        if y1 > y2 :
            (y1, y2) = (y2, y1)
        if first_point.x == second_point.x :
            array[x1, y1 + 1 : y2, :] = [1, 0, 0]
        else :
            array[x1 + 1 : x2, y2, :] = [1, 0, 0]


sample = np.zeros((8, 1, 3))
for i in range(8) :
    sample[i, 0] = color_sampler(i)

while True :

    cumulative_area_index = 0
    area_index_dictionary = dict()
    area_hierarchy = generate_area_hierarchy(0, (2, 4), (2, 2), area_index_dictionary)

    room_index_dictionary = dict()
    identify_room(area_hierarchy, room_index_dictionary)
    ###
    #room_index_dictionary[0] = area_index_dictionary[0]
    ###

    boundary_rectangle = rectangle(0, 0, (1 + random.random()) * 25, (1 + random.random()) * 25)
    partition_graph = graph()
    room_point_dictionary = dict()

    partition(area_hierarchy, boundary_rectangle, 0, partition_graph)
    geometrize(room_index_dictionary, partition_graph, room_point_dictionary)

    ###
    #room_index_dictionary[0].geometry.reverse()
    ###
    door_segment = generate_connection(room_index_dictionary, room_point_dictionary, list(room_point_dictionary[point(0, 0)])[0].index)
    #door_segment = generate_connection(room_index_dictionary, room_point_dictionary, random.choice(list(room_index_dictionary.keys()))

    array = np.ones((int(boundary_rectangle.width()) + 1, int(boundary_rectangle.height()) + 1, 3))
    draw_partition(array, area_hierarchy)
    draw_vertex(array, partition_graph)
    draw_connection(array, door_segment)

    #plt.subplot(121)
    #plt.imshow(sample)
    #plt.subplot(122)
    plt.imshow(array)
    #print_partition_hierarchy([house])
    #print(partition_graph)
    plt.show()