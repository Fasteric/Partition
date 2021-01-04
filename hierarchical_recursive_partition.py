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
    def collide(self, other) :
        if self.x2 >= other.x1 and self.x1 <= other.x2 and self.y2 >= other.y1 and self.y1 <= other.y2 :
            return True
        if other.x2 >= self.x1 and other.x1 <= self.x2 and other.y2 >= self.y1 and other.y1 <= self.y2 :
            print('not enough')
            return True
        return False

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
    def __init__(self) :
        self.rectangle = None
        self.proportion = 1
        self.children = list()
    def __lt__(self, other) :
        return self.proportion < other.proportion

class room :
    def __init__(self, index) :
        self.index = index
        self.area = None
        self.type = None
        self.geometry = list() # [point]
        self.connection = dict() # room -> [segment]
        self.furniture = list() # [furniture]
    def __repr__(self) :
        return 'room({})'.format(self.index)
    def __eq__(self, other) :
        return self.index == other.index
    def __hash__(self) :
        return hash((room, self.index))

class furniture :
    def __init__(self, type, padding_rectangle, margin_rectangle) :
        self.type = type
        self.padding_rectangle = padding_rectangle
        self.margin_rectangle = margin_rectangle
    def __repr__(self) :
        return 'furniture({}, {})'.format(self.type, self.padding_rectangle)


# generate hierarchy from list
# output: room_list
def generate_phums_area_hierarchy(partition_root_area, phums_list, room_list) :
    room_index = 0
    partition_root_area.proportion = 1
    for i in range(len(phums_list)) :
        first_depth_area = area()
        first_depth_area.proportion = 0.75 + random.random() / 2
        for j in range(len(phums_list[i])) :
            second_depth_area = area()
            second_depth_area.proportion = 0.75 + random.random() / 2
            current_room = room(room_index)
            room_index += 1
            current_room.area = second_depth_area
            current_room.type = phums_list[i][j]
            room_list.append(current_room)
            first_depth_area.children.append(second_depth_area)
        first_depth_area.children.sort(key = lambda area : area.proportion)
        partition_root_area.children.append(first_depth_area)
    partition_root_area.children.sort(key = lambda area : area.proportion)

# assign rectangle to area
# output: partition_graph
def partition(area_hierarchy, boundary_rectangle, partition_depth, partition_graph) :
    for i in range(4) :
        partition_graph.connect(boundary_rectangle[i], (boundary_rectangle[(i + 1) % 4], partition_depth))
        partition_graph.connect(boundary_rectangle[i], (boundary_rectangle[(i + 3) % 4], partition_depth))
    if len(area_hierarchy) == 0 :
        return
    elif len(area_hierarchy) == 1 :
        area_hierarchy[0].rectangle = boundary_rectangle
        partition(area_hierarchy[0].children, boundary_rectangle, partition_depth + 1, partition_graph)
    else :
        # rectangle parameters
        if boundary_rectangle.width() >= boundary_rectangle.height() :
            partition_direction = 0
            (width_start, width_end, width) = (boundary_rectangle.x1, boundary_rectangle.x2, boundary_rectangle.width())
            (height_start, height_end, height) = (boundary_rectangle.y1, boundary_rectangle.y2, boundary_rectangle.height())
        else :
            partition_direction = 1
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
            if partition_direction == 0 :
                partition_rectangle = rectangle(width_start, partition_heights[i], partition_width, partition_heights[i + 1])
            else :
                partition_rectangle = rectangle(partition_heights[i], width_start, partition_heights[i + 1], partition_width)
            area_hierarchy[i].rectangle = partition_rectangle
            partition(area_hierarchy[i].children, partition_rectangle, partition_depth + 1, partition_graph)
        if partition_count < len(area_hierarchy) :
            # remaining area partition
            if partition_direction == 0 :
                partition_rectangle = rectangle(partition_width, height_start, width_end, height_end)
            else :
                partition_rectangle = rectangle(height_start, partition_width, height_end, width_end)
            partition(area_hierarchy[partition_count:], partition_rectangle, partition_depth, partition_graph)

# define geometry
# output: room_point_dictionary
def geometrize(room_list, partition_graph, room_point_dictionary) :
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
    for room in room_list :
        geometry = list()
        (x1, y1, x2, y2) = (room.area.rectangle.x1, room.area.rectangle.y1, room.area.rectangle.x2, room.area.rectangle.y2)
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

# minimum connection
# return success
def generate_tree_connection(connection_root_room, room_list, room_point_dictionary) :
    connected = {connection_root_room}
    available_connection = dict() # room -> [(room, (segment))]
    for i in range(len(connection_root_room.geometry)) :
        first_point = connection_root_room.geometry[i]
        second_point = connection_root_room.geometry[(i + 1) % len(connection_root_room.geometry)]
        if first_point.x == second_point.x :
            if abs(first_point.y - second_point.y) <= 4 :
                continue
        else :
            if abs(first_point.x - second_point.x) <= 6 :
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
    while len(connected) != len(room_list) :
        if len(available_connection) == 0 :
            print('unreachable room exist')
            return False
        second_room = random.choice(list(available_connection.keys()))
        (first_room, (first_point, second_point)) = random.choice(list(available_connection.pop(second_room)))
        if second_room not in first_room.connection :
            first_room.connection[second_room] = list()
        first_room.connection[second_room].append((first_point, second_point, 0))
        if first_room not in second_room.connection :
            second_room.connection[first_room] = list()
        second_room.connection[first_room].append((second_point, first_point, 0))
        connected.add(first_room)
        connected.add(second_room)
        for i in range(len(second_room.geometry)) :
            first_point = second_room.geometry[i]
            second_point = second_room.geometry[(i + 1) % len(second_room.geometry)]
            if first_point.x == second_point.x :
                if abs(first_point.y - second_point.y) <= 4 :
                    continue
            else :
                if abs(first_point.x - second_point.x) <= 6 :
                    continue
            for adjacent_room in room_point_dictionary[first_point] :
                if adjacent_room in connected :
                    continue
                if adjacent_room in second_room.connection :
                    continue
                if adjacent_room not in room_point_dictionary[second_point] :
                    continue
                if adjacent_room not in available_connection :
                    available_connection[adjacent_room] = list()
                available_connection[adjacent_room].append((second_room, (first_point, second_point)))
    return True

# additional connection
def generate_phums_additional_connection(connection_root_room, room_list, room_point_dictionary, reference_depth, connecting_depth) :
    # depth room dictionary
    depth_room_dictionary = {connection_root_room: 0} # room -> depth
    stack = [connection_root_room]
    while len(stack) > 0 :
        current_room = stack.pop()
        for adjacent_room in current_room.connection :
            if adjacent_room in depth_room_dictionary :
                continue
            depth_room_dictionary[adjacent_room] = depth_room_dictionary[current_room] + 1
            stack.append(adjacent_room)
    # additional connection
    for first_room in room_list :
        if depth_room_dictionary[first_room] < connecting_depth :
            continue
        # first reference room
        first_reference_room = first_room
        while depth_room_dictionary[first_reference_room] > reference_depth :
            for next_reference_room in first_reference_room.connection :
                if depth_room_dictionary[next_reference_room] == depth_room_dictionary[first_reference_room] - 1 :
                    first_reference_room = next_reference_room
                    break
        for i in range(len(first_room.geometry)) :
            first_point = first_room.geometry[i]
            second_point = first_room.geometry[(i + 1) % len(first_room.geometry)]
            if first_point.x == second_point.x :
                if abs(first_point.y - second_point.y) <= 4 :
                    continue
            else :
                if abs(first_point.x - second_point.x) <= 6 :
                    continue
            for second_room in room_point_dictionary[first_point] :
                if first_room == second_room :
                    continue
                if depth_room_dictionary[second_room] < connecting_depth :
                    #continue
                    pass
                if second_room in first_room.connection :
                    continue
                if second_room not in room_point_dictionary[second_point] :
                    continue
                # second reference room
                second_reference_room = second_room
                while depth_room_dictionary[second_reference_room] > reference_depth :
                    for next_reference_room in second_reference_room.connection :
                        if depth_room_dictionary[next_reference_room] == depth_room_dictionary[second_reference_room] - 1 :
                            second_reference_room = next_reference_room
                            break
                if first_reference_room != second_reference_room :
                    if second_room not in first_room.connection :
                        first_room.connection[second_room] = list()
                    first_room.connection[second_room].append((first_point, second_point, 4))
                    if first_room not in second_room.connection :
                        second_room.connection[first_room] = list()
                    second_room.connection[first_room].append((second_point, first_point, 4))
                    #print('connect', first_room, second_room, 'due to tree structure')

def place_door(room_list) :
    door_placed = set()
    for first_room in room_list :
        for second_room in first_room.connection :
            if (second_room, first_room) in door_placed :
                continue
            for (first_point, second_point, connection_metadata) in first_room.connection[second_room] :
                if first_point.x < second_point.x and first_point.y == second_point.y :
                    door_x = (first_point.x + 3) + (second_point.x - first_point.x - 6) * random.random()
                    door_y = first_point.y
                    door_padding_rectangle = rectangle(door_x, door_y, door_x + 3, door_y + 1)
                    first_room_door_margin_rectangle = rectangle(door_x, door_y, door_x + 3, door_y + 3)
                    second_room_door_margin_rectangle = rectangle(door_x, door_y - 3, door_x + 3, door_y)
                elif first_point.x == second_point.x and first_point.y < second_point.y :
                    door_x = first_point.x
                    door_y = (first_point.y + 1) + (second_point.y - first_point.y - 4) * random.random()
                    door_padding_rectangle = rectangle(door_x, door_y, door_x + 1, door_y + 3)
                    first_room_door_margin_rectangle = rectangle(door_x - 3, door_y, door_x, door_y + 3)
                    second_room_door_margin_rectangle = rectangle(door_x, door_y, door_x + 5, door_y + 3)
                elif first_point.x > second_point.x and first_point.y == second_point.y :
                    door_x = (second_point.x + 3) + (first_point.x - second_point.x - 6) * random.random()
                    door_y = second_point.y
                    door_padding_rectangle = rectangle(door_x, door_y, door_x + 3, door_y + 1)
                    first_room_door_margin_rectangle = rectangle(door_x, door_y - 3, door_x + 3, door_y)
                    second_room_door_margin_rectangle = rectangle(door_x, door_y, door_x + 3, door_y + 3)
                elif first_point.x == second_point.x and first_point.y > second_point.y :
                    door_x = second_point.x
                    door_y = (second_point.y + 1) + (first_point.y - second_point.y - 4) * random.random()
                    door_padding_rectangle = rectangle(door_x, door_y, door_x + 1, door_y + 3)
                    first_room_door_margin_rectangle = rectangle(door_x, door_y, door_x + 5, door_y + 3)
                    second_room_door_margin_rectangle = rectangle(door_x - 3, door_y, door_x, door_y + 3)
                # for door masking
                first_room_door = furniture(connection_metadata - 12, door_padding_rectangle, first_room_door_margin_rectangle)
                second_room_door = furniture(connection_metadata - 12, door_padding_rectangle, second_room_door_margin_rectangle)
                first_room.furniture.append(first_room_door)
                second_room.furniture.append(second_room_door)
                door_placed.add((first_room, second_room))

def furnish(room_list, viable_furniture) :
    for room in room_list :
        (room_x1, room_y1, room_x2, room_y2) = (room.area.rectangle.x1, room.area.rectangle.y1, room.area.rectangle.x2, room.area.rectangle.y2)
        (room_width, room_height) = (room.area.rectangle.width(), room.area.rectangle.height())
        for i in range(11) : # number of furniture
            selected_furniture = random.choice(viable_furniture[room.type])
            (furniture_wall, furniture_type) = selected_furniture[:2]
            (furniture_padding_width, furniture_padding_height) = selected_furniture[2]
            (furniture_margin_width, furniture_margin_height) = selected_furniture[3]
            (furniture_dx, furniture_dy) = (furniture_margin_width - furniture_padding_width, furniture_margin_height - furniture_padding_height)
            if furniture_margin_width >= room_width or furniture_margin_height >= room_height :
                continue
            for j in range(5) : # number of attempt
                if furniture_wall :
                    wall_side = random.randint(0, 3)
                    if wall_side == 0 :
                        furniture_margin_x = (room_x1 + 3 - furniture_dx / 2) + (room_width - furniture_margin_width - 3 + furniture_dx) * random.random()
                        furniture_margin_y = room_y1 + 1 - furniture_dy / 2
                    elif wall_side == 1 :
                        furniture_margin_x = room_x2 - furniture_margin_width + furniture_dx / 2
                        furniture_margin_y = (room_y1 + 1 - furniture_dy / 2) + (room_height - furniture_margin_height - 1 + furniture_dy) * random.random()
                    elif wall_side == 2 :
                        furniture_margin_x = (room_x1 + 3 - furniture_dx / 2) + (room_width - furniture_margin_width - 3 + furniture_dx) * random.random()
                        furniture_margin_y = room_y2 - furniture_margin_height + furniture_dy / 2
                    elif wall_side == 3 :
                        furniture_margin_x = room_x1 + 3 - furniture_dx / 2
                        furniture_margin_y = (room_y1 + 1 - furniture_dy / 2) + (room_height - furniture_margin_height - 1 + furniture_dy) * random.random()
                else :
                    furniture_margin_x = (room_x1 + 3) + (room_width - furniture_margin_width - 3) * random.random()
                    furniture_margin_y = (room_y1 + 1) + (room_height - furniture_margin_height - 1) * random.random()
                furniture_margin_rectangle = rectangle(furniture_margin_x, furniture_margin_y, furniture_margin_x + furniture_margin_width, furniture_margin_y + furniture_margin_height)
                collided = False
                for existed_furniture in room.furniture :
                    if existed_furniture.margin_rectangle.collide(furniture_margin_rectangle) :
                        collided = True
                        break
                if collided :
                    continue
                furniture_padding_x = furniture_margin_x + (furniture_margin_width - furniture_padding_width) / 2
                furniture_padding_y = furniture_margin_y + (furniture_margin_height - furniture_padding_height) / 2
                furniture_padding_rectangle = rectangle(furniture_padding_x, furniture_padding_y, furniture_padding_x + furniture_padding_width, furniture_padding_y + furniture_padding_height)
                placing_furniture = furniture(room.type, furniture_padding_rectangle, furniture_margin_rectangle)
                room.furniture.append(placing_furniture)
                #print('placing', placing_furniture, 'on', room)
                break


def color_sampler(index) :
    return np.array([math.cos(math.tau * index / 12 - i * math.tau / 3) for i in range(3)]) / 2 + 0.5

sample = np.zeros((12, 1, 3))
for i in range(12) :
    sample[i, 0, :] = color_sampler(i)

def print_partition_hierarchy(children, depth = 0) :
    for subarea in children :
        print('{}{}'.format('    ' * depth, subarea.rectangle))
        print_partition_hierarchy(subarea.children, depth + 1)

def draw_partition(array, room_list) :
    for room in room_list :
        (x1, y1, x2, y2) = (round(room.area.rectangle.x1), round(room.area.rectangle.y1), round(room.area.rectangle.x2), round(room.area.rectangle.y2))
        array[x1 + 1 : x1 + 3, y1 + 1 : y2, :] = color_sampler(room.type) / 8
        array[x1 + 3 : x2    , y1 + 1 : y2, :] = color_sampler(room.type) / 4

def draw_vertex(array, partition_graph) :
    min_vertex_value = dict()
    for (vertex, edges) in partition_graph.graph.items() :
        for (v, w) in edges :
            if v not in min_vertex_value or min_vertex_value[v] > w :
                min_vertex_value[v] = w
                array[round(v.x), round(v.y), :] = color_sampler(w)

def draw_connection(array, room_list) :
    drew = set()
    for first_room in room_list :
        for second_room in first_room.connection :
            if second_room in drew :
                continue
            for i in range(len(first_room.connection[second_room])) :
                (first_point, second_point, color) = first_room.connection[second_room][i]
                if first_point.x > second_point.x or first_point.y > second_point.y :
                    (first_point, second_point) = (second_point, first_point)
                (x1, y1, x2, y2) = (round(first_point.x), round(first_point.y), round(second_point.x), round(second_point.y))
                if x1 - x2 == 0 :
                    r = random.randint(y1 + 2, y2 - 2)
                    array[x1, r - 1 : r + 2, :] = np.power(color_sampler(color), 2)
                else :
                    r = random.randint(x1 + 2, x2 - 2)
                    array[r - 1 : r + 2, y2, :] = np.power(color_sampler(color), 2)
        drew.add(first_room)

def draw_furniture(array, room_list) :
    for room in room_list :
        for furniture in room.furniture :
            (furniture_x, furniture_y) = (round(furniture.padding_rectangle.x1), round(furniture.padding_rectangle.y1))
            (furniture_width, furniture_height) = (round(furniture.padding_rectangle.width()), round(furniture.padding_rectangle.height()))
            array[furniture_x : furniture_x + furniture_width, furniture_y : furniture_y + furniture_height, :] = color_sampler(furniture.type)
            if furniture.type < 0 and furniture_width == 1 and room.geometry[0].x == furniture.padding_rectangle.x1 :
                (masking_x, masking_y) = (furniture_x + 1, furniture_y)
                if masking_x + 1 < array.shape[0] :
                    array[masking_x : masking_x + 2, masking_y : masking_y + 3, :] = color_sampler(room.type) / 4


def print_hierarchy(root, depth = 0) :
    for e in root :
        print('{}{}'.format('    ' * depth, e))
        print_hierarchy(e.children, depth + 1)

while True :

    #print('generate')

    # phum's list generation
    (room_count, first_breadth) = (11, 3)
    phums_list = [0.5 + random.random() for i in range(first_breadth)]
    phums_list = [round((sum(phums_list[:i + 1]) / sum(phums_list)) * (room_count - first_breadth)) for i in range(first_breadth)]
    phums_list[1:] = [phums_list[i] - phums_list[i - 1] for i in range(1, first_breadth)]
    phums_list = [1 + phums_list[i] for i in range(first_breadth)]
    random.shuffle(phums_list)
    phums_list = [[random.randint(2, 10) for j in range(phums_list[i])] for i in range(first_breadth)]
    #phums_list = [[1, 2, 3], [4, 5], [6, 7]]
    print(phums_list)

    # generate area hierarchy
    partition_root_area = area()
    room_list = list()
    generate_phums_area_hierarchy(partition_root_area, phums_list, room_list)

    # partition and geometrize
    boundary_rectangle = rectangle(0, 0, 50, 50)
    partition_graph = graph()
    partition([partition_root_area], boundary_rectangle, 0, partition_graph)

    room_point_dictionary = dict()
    geometrize(room_list, partition_graph, room_point_dictionary)

    # generate connection
    connection_root_room = random.choice(room_list)
    connection_root_room.type = 0
    tree_connection_result = generate_tree_connection(connection_root_room, room_list, room_point_dictionary)
    if tree_connection_result == False :
        continue
    generate_phums_additional_connection(connection_root_room, room_list, room_point_dictionary, 2, 3)

    place_door(room_list)

    # wall, type, padding, margin
    viable_furniture_list = list()
    viable_furniture_list.append((False, 0, (1, 1), (3, 3)))
    viable_furniture_list.append((True,  0, (1, 1), (3, 3)))
    viable_furniture_list.append((False, 1, (2, 1), (4, 3)))
    viable_furniture_list.append((True,  1, (2, 1), (4, 3)))
    viable_furniture_list.append((False, 2, (1, 2), (3, 5)))
    viable_furniture_list.append((True,  2, (1, 2), (3, 5)))
    viable_furniture_list.append((False, 3, (3, 2), (5, 4)))
    viable_furniture_list.append((False, 3, (2, 3), (4, 5)))
    viable_furniture = dict()
    for i in range(12) :
        viable_furniture[i] = viable_furniture_list
    furnish(room_list, viable_furniture)

    array = np.ones((round(boundary_rectangle.width()) + 1, round(boundary_rectangle.height()) + 1, 3))
    draw_partition(array, room_list)
    #draw_vertex(array, partition_graph)
    #draw_connection(array, room_list)
    draw_furniture(array, room_list)
    
    plt.imshow(array)
    plt.show()

    print()