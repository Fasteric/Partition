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
    def __len__(self) :
        return len(self.graph)
    def __iter__(self) :
        return (vertex for vertex in self.graph)
    def __contains__(self, vertex) :
        return vertex in self.graph
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
        # area properties
        self.proportion = 1
        self.children = list()
        self.rectangle = None
        # room only properties
        self.type = None
        self.geometry = list()
        self.connection = dict()
        #self.door_segment = list()
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
        partition(area_hierarchy[0].children, boundary_rectangle, partition_depth + 1, partition_graph)
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
        best_ratio_error = abs(math.log2(height / best_width))
        partition_count = 1
        for i in range(1, len(area_hierarchy)) :
            cumulative_proportion += area_hierarchy[i].proportion
            current_width = (cumulative_proportion / total_proportion) * width
            current_ratio_error = abs(math.log2(height / ((i + 1) * current_width)))
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
            partition(area_hierarchy[i].children, partition_rectangle, partition_depth + 1, partition_graph)
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

cumulative_area_index = None

# UNUSED
# return area hierarchy, room dictionary
def generate_area_hierarchy(recursive_depth, area_index_dictionary, area_breadth, area_depth) :
    global cumulative_area_index
    if recursive_depth == 0 :
        current_area_index = cumulative_area_index
        cumulative_area_index += 1
        area_index_dictionary[current_area_index] = area(current_area_index)
        area_index_dictionary[current_area_index].proportion = 1 + random.random()
        #area_index_dictionary[current_area_index].prefered_ratio = 1
        area_index_dictionary[current_area_index].children = generate_area_hierarchy(recursive_depth + 1, area_index_dictionary, area_breadth, area_depth)
        return [area_index_dictionary[current_area_index]]
    else :
        if area_depth[1] <= 0 :
            return list()
        children = list()
        current_area_breadth = random.randint(area_breadth[0], area_breadth[1])
        for i in range(current_area_breadth) :
            current_area_index = cumulative_area_index
            cumulative_area_index += 1
            area_index_dictionary[current_area_index] = area(current_area_index)
            area_index_dictionary[current_area_index].proportion = 1 + random.random()
            #area_index_dictionary[current_area_index].prefered_ratio = random.random() - 0.5
            next_max_area_depth = max(0, random.randint(area_depth[0], area_depth[1]) - 1)
            area_index_dictionary[current_area_index].children = generate_area_hierarchy(recursive_depth + 1, area_index_dictionary, area_breadth, (area_depth[0] - 1, area_depth[1] - 1))
            children.append(area_index_dictionary[current_area_index])
        return sorted(children, key = (lambda area : area.proportion))

def phums_area_hierarchy(phums_list, area_index_dictionary, room_index_dictionary) :
    current_index = 0
    root_area = area(current_index)
    area_index_dictionary[current_index] = root_area
    current_index += 1
    root_area.proportion = 1
    for i in range(len(phums_list)) :
        first_depth_area = area(current_index)
        area_index_dictionary[current_index] = first_depth_area
        current_index += 1
        first_depth_area.proportion = 0.75 + random.random() / 2
        for j in range(len(phums_list[i])) :
            second_depth_area = area(current_index)
            area_index_dictionary[current_index] = second_depth_area
            room_index_dictionary[current_index] = second_depth_area
            current_index += 1
            second_depth_area.proportion = 0.75 + random.random() / 2
            second_depth_area.type = phums_list[i][j]
            first_depth_area.children.append(second_depth_area)
        first_depth_area.children.sort(key = lambda area : area.proportion)
        root_area.children.append(first_depth_area)
    root_area.children.sort(key = lambda area : area.proportion)

# UNUSED
def identify_room(area_hierarchy, room_index_dictionary) :
    for area in area_hierarchy :
        if len(area.children) == 0 :
            room_index_dictionary[area.index] = area
        else :
            identify_room(area.children, room_index_dictionary)

# modify: connection_graph
def generate_tree_connection(connection_root_room, room_index_dictionary, room_point_dictionary) :
    connected = {connection_root_room}
    available_connection = dict()
    for i in range(len(connection_root_room.geometry)) :
        first_point = connection_root_room.geometry[i]
        second_point = connection_root_room.geometry[(i + 1) % len(connection_root_room.geometry)]
        if abs(first_point.x - second_point.x) + abs(first_point.y - second_point.y) <= 2 :
            continue
        for adjacent_room in room_point_dictionary[first_point] :
            if adjacent_room in connected :
                continue
            if adjacent_room in connection_root_room.connection :
                continue
            if adjacent_room not in room_point_dictionary[second_point] :
                continue
            if adjacent_room not in available_connection :
                available_connection[adjacent_room] = list()
            available_connection[adjacent_room].append((connection_root_room, (first_point, second_point)))
    while len(connected) != len(room_index_dictionary) :
        if len(available_connection) == 0 :
            print('unreachable room exist')
            break
        connect_to_room = random.choice(list(available_connection.keys()))
        (connect_from_room, (first_point, second_point)) = random.choice(list(available_connection.pop(connect_to_room)))
        if connect_to_room not in connect_from_room.connection :
            connect_from_room.connection[connect_to_room] = list()
        connect_from_room.connection[connect_to_room].append((first_point, second_point))
        if connect_from_room not in connect_to_room.connection :
            connect_to_room.connection[connect_from_room] = list()
        connect_to_room.connection[connect_from_room].append((second_point, first_point))
        connected.add(connect_from_room)
        connected.add(connect_to_room)
        for i in range(len(connect_to_room.geometry)) :
            first_point = connect_to_room.geometry[i]
            second_point = connect_to_room.geometry[(i + 1) % len(connect_to_room.geometry)]
            if abs(first_point.x - second_point.x) + abs(first_point.y - second_point.y) <= 2 :
                continue
            for adjacent_room in room_point_dictionary[first_point] :
                if adjacent_room in connected :
                    continue
                if adjacent_room in connect_to_room.connection :
                    continue
                if adjacent_room not in room_point_dictionary[second_point] :
                    continue
                if adjacent_room not in available_connection :
                    available_connection[adjacent_room] = list()
                available_connection[adjacent_room].append((connect_to_room, (first_point, second_point)))

def phums_additional_tree_connection(connection_root_room, room_point_dictionary, reference_depth, connecting_depth) :
    depth_room_dictionary = {connection_root_room: 0}
    stack = [connection_root_room]
    while len(stack) > 0 :
        first_room = stack.pop()
        for (second_room, _, _) in connection_graph[first_room] :
            if second_room is not None and second_room not in depth_room_dictionary :
                depth_room_dictionary[second_room] = depth_room_dictionary[first_room] + 1
                stack.append(second_room)
    phums_connected = set()
    for (first_room, depth) in depth_room_dictionary.items() :
        if depth < connecting_depth :
            continue
        first_reference_room = first_room
        while depth_room_dictionary[first_reference_room] > reference_depth :
            for (next_reference_room, _, _) in connection_graph[first_reference_room] :
                if depth_room_dictionary[next_reference_room] == depth_room_dictionary[first_reference_room] - 1 :
                    first_reference_room = next_reference_room
                    break
        for i in range(len(first_room.geometry)) :
            first_point = first_room.geometry[i]
            second_point = first_room.geometry[(i + 1) % len(first_room.geometry)]
            if abs(first_point.x - second_point.x) + abs(first_point.y - second_point.y) <= 2 :
                continue
            for second_room in room_point_dictionary[first_point] :
                if second_room == first_room :
                    continue
                if second_room not in room_point_dictionary[second_point] :
                    continue
                #if depth_room_dictionary[second_room] < connecting_depth :
                    #continue
                if (second_room, first_room) in phums_connected :
                    continue
                second_reference_room = second_room
                while depth_room_dictionary[second_reference_room] > reference_depth :
                    for (next_reference_room, _, _) in connection_graph[second_reference_room] :
                        if depth_room_dictionary[next_reference_room] == depth_room_dictionary[second_reference_room] - 1 :
                            second_reference_room = next_reference_room
                            break
                if first_reference_room != second_reference_room :
                    print('connect', first_room, second_room, 'due to tree structure')
                    phums_connected.add((first_room, second_room))
                    connection_graph.connect(first_room, (second_room, first_point, second_point))

def phums_additional_random_connection(room_point_dictionary, connection_graph, connection_chance) :
    phums_connected = set()
    for first_room in connection_graph :
        for i in range(len(first_room.geometry)) :
            first_point = first_room.geometry[i]
            second_point = first_room.geometry[(i + 1) % len(first_room.geometry)]
            if abs(first_point.x - second_point.x) + abs(first_point.y - second_point.y) <= 2 :
                continue
            for second_room in room_point_dictionary[first_point] :
                if first_room == second_room :
                    continue
                if random.random() >= connection_chance :
                    continue
                if second_room not in room_point_dictionary[second_point] :
                    continue
                if (second_room, first_room) in phums_connected :
                    continue
                abort_flag = False
                for (third_room, _, _) in connection_graph[first_room] :
                    if third_room is None or third_room == second_room :
                        abort_flag = True
                        break
                if abort_flag :
                    continue
                print('connect', first_room, second_room, 'from sheer luck')
                phums_connected.add((first_room, second_room))
                connection_graph.connect(first_room, (second_room, first_point, second_point))
                


def color_sampler(index) :
    return np.array([math.cos(math.tau * -index / 12 + i * math.tau / 3) for i in range(3)]) / 2 + 0.5

sample = np.zeros((12, 1, 3))
for i in range(12) :
    sample[i, 0, :] = color_sampler(i)

def print_partition_hierarchy(children, depth = 0) :
    for subarea in children :
        print('{}{}'.format('    ' * depth, subarea.rectangle))
        print_partition_hierarchy(subarea.children, depth + 1)

# should draw_partition prefer list of room too ?
def draw_partition(array, children, depth = 0) :
    for subarea in children :
        if len(subarea.children) > 0 :
            draw_partition(array, subarea.children, depth + 1)
        else :
            (x1, y1, x2, y2) = (int(subarea.rectangle.x1), int(subarea.rectangle.y1), int(subarea.rectangle.x2), int(subarea.rectangle.y2))
            array[x1 : x2, y1 : y2, :] = 1
            if subarea.type is None :
                array[x1 + 1 : x2, y1 + 1 : y2, :] = 0
            else :
                array[x1 + 1 : x2, y1 + 1 : y2, :] = color_sampler(subarea.type) / 4

def draw_vertex(array, partition_graph) :
    min_vertex_value = dict()
    for (vertex, edges) in partition_graph.graph.items() :
        for (v, w) in edges :
            if v not in min_vertex_value or min_vertex_value[v] > w :
                min_vertex_value[v] = w
                array[int(v.x), int(v.y), :] = color_sampler(w)

# prefer list of room over room_index_dictionary
def draw_connection(array, room_index_dictionary, connection_color) :
    drew = set()
    for connect_from_room in room_index_dictionary.values() :
        for connect_to_room in connect_from_room.connection :
            if connect_to_room in drew :
                continue
            for i in range(len(connect_from_room.connection[connect_to_room])) :
                (first_point, second_point) = connect_from_room.connection[connect_to_room][i]
                if first_point.x > second_point.x or first_point.y > second_point.y :
                    (first_point, second_point) = (second_point, first_point)
                (x1, y1, x2, y2) = (int(first_point.x), int(first_point.y), int(second_point.x), int(second_point.y))
                if x1 == x2 :
                    array[x1, random.randint(y1 + 1, y2 - 1), :] = connection_color
                else :
                    array[random.randint(x1 + 1, x2 - 1), y2, :] = connection_color
        drew.add(connect_from_room)


while True :

    print('generate')

    boundary_rectangle = rectangle(0, 0, (1 + random.random()) * 50, (1 + random.random()) * 50)
    # phum's list generation
    (room_count, first_breadth) = (16, 3)
    phums_list = [0.5 + random.random() for i in range(first_breadth)]
    phums_list = [int((sum(phums_list[:i + 1]) / sum(phums_list)) * (room_count - first_breadth)) for i in range(first_breadth)]
    phums_list[1:] = [phums_list[i] - phums_list[i - 1] for i in range(1, first_breadth)]
    phums_list = [1 + phums_list[i] for i in range(first_breadth)]
    random.shuffle(phums_list)
    phums_list = [[random.randint(2, 10) for j in range(phums_list[i])] for i in range(first_breadth)]
    #phums_list = [[1, 2, 3], [4, 5], [6, 7]]
    print(phums_list)

    # generate area hierarchy
    area_index_dictionary = dict()
    room_index_dictionary = dict()
    phums_area_hierarchy(phums_list, area_index_dictionary, room_index_dictionary)

    # partition and geometrize
    partition_root_area = area_index_dictionary[0]
    room_point_dictionary = dict()
    partition_graph = graph()
    partition([partition_root_area], boundary_rectangle, 0, partition_graph)
    geometrize(room_index_dictionary, partition_graph, room_point_dictionary)

    # generate tree-structured connection
    #connection_root = list(room_point_dictionary[point(0, 0)])[0]
    connection_root_room = random.choice(list(room_index_dictionary.values()))
    connection_root_room.type = 0
    generate_tree_connection(connection_root_room, room_index_dictionary, room_point_dictionary)

    array = np.ones((int(boundary_rectangle.width()) + 1, int(boundary_rectangle.height()) + 1, 3))
    draw_partition(array, [partition_root_area])
    #draw_vertex(array, partition_graph)
    draw_connection(array, room_index_dictionary, color_sampler(0))
    
    ###
    plt.imshow(array)
    plt.show()
    continue
    ###

    phums_tree_connection_array = np.copy(array)
    phums_random_connection_array = np.copy(array)
    phums_tree_connection_graph = graph()
    phums_random_connection_graph = graph()
    for (key, value) in connection_graph.graph.items() :
        phums_tree_connection_graph.graph[key] = set(value)
        phums_random_connection_graph.graph[key] = set(value)

    phums_tree_connection(connection_root, room_point_dictionary, phums_tree_connection_graph, 2, 3)
    phums_random_connection(room_point_dictionary, phums_random_connection_graph, 0.1)

    for (key, value) in connection_graph.graph.items() :
        phums_tree_connection_graph.graph[key] -= value
        phums_random_connection_graph.graph[key] -= value
    draw_connection(phums_tree_connection_array, phums_tree_connection_graph, color_sampler(4))
    draw_connection(phums_random_connection_array, phums_random_connection_graph, color_sampler(8))

    #plt.subplot(141)
    #plt.imshow(sample)
    plt.subplot(131)
    plt.imshow(array)
    plt.subplot(132)
    plt.imshow(phums_tree_connection_array)
    plt.subplot(133)
    plt.imshow(phums_random_connection_array)
    #print_partition_hierarchy([house])
    #print(partition_graph)
    plt.show()

    print()