import math
import random
import numpy as np
import matplotlib.pyplot as plt

class rectangle :
    # represent rectangular object
    # x2 >= x1, y2 >= y1
    def __init__(self, x1, y1, x2, y2) :
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
    def __repr__(self) :
        return 'rect({}, {}, {}, {})'.format(self.x1, self.y1, self.x2, self.y2)
    def width(self) :
        return self.x2 - self.x1
    def height(self) :
        return self.y2 - self.y1

class node :
    # node structure for graph or tree
    # directional children
    def __init__(self, value, children) :
        self.value = value
        self.children = children
    def __lt__(self, other) :
        return self.value < other.value
    def __repr__(self) :
        return 'node({}, {})'.format(self.value, self.children)

def partition(proportions, rect) :
    # input: list of nodes of proportion value, rectangular area
    # output: list of nodes of partitioned area tree bounded within rect (same tree structure as proportions)
    # recursive
    if len(proportions) == 0 :
        # trivial case: return empty list
        return list()
    elif len(proportions) == 1 :
        # 1 case: partitioned area is the rect itself
        return [node(rect, partition(proportions[0].children, rect))]
    else :
        # n case:
        # determine total proportion value
        total_proportion = 0
        for proportion in proportions :
            total_proportion += proportion.value
        # determine best partition direction
        dx = rect.x2 - rect.x1
        dy = rect.y2 - rect.y1
        if dx >= dy :
            (width_start, width_end, width) = (rect.x1, rect.x2, dx)
            (height_start, height_end, height) = (rect.y1, rect.y2, dy)
        else :
            (width_start, width_end, width) = (rect.y1, rect.y2, dy)
            (height_start, height_end, height) = (rect.x1, rect.x2, dx)
        # initialize partition parameters before search (proportions[0])
        cumulative_proportion = proportions[0].value
        partition_proportion = cumulative_proportion
        best_width = (cumulative_proportion / total_proportion) * width
        proportion_count = 1
        best_ratio_error = math.log2(height / best_width)
        # search for best partition parameters (test for proportions[1:])
        for i in range(1, len(proportions)) :
            cumulative_proportion += proportions[i].value
            if i + 1 < len(proportions) :
                current_width = (cumulative_proportion / total_proportion) * width
            else :
                # if all proportions has been used, partition width should equal to rectangle partition-side width
                # overkill? there is a check after the loop
                current_width = width
            current_ratio_error = math.log2(height / ((i + 1) * current_width))
            if abs(current_ratio_error) <= abs(best_ratio_error) :
                # new parameters promise better aspect ratio
                partition_proportion = cumulative_proportion
                best_width = current_width
                proportion_count = i + 1
                best_ratio_error = current_ratio_error
                if current_ratio_error < 0 :
                    # next test will not resulted in better aspect ratio
                    break
            else :
                # previous test promise better aspect ratio
                break
        # finalize partition width
        if proportion_count < len(proportions) :
            partition_width = width_start + best_width
        else :
            # this might already covered for that overkill if statement inside the loop
            partition_width = width_end
        # finalize rectangle height for each area
        partition_heights = [height_start]
        cumulative_proportion = 0
        for i in range(proportion_count - 1) :
            cumulative_proportion += proportions[i].value
            partition_heights.append(height_start + (cumulative_proportion / partition_proportion) * height)
        partition_heights.append(height_end)
        # commit partition
        partitions = list()
        # partition for proportions[:proportion_count]
        for i in range(proportion_count) :
            if dx >= dy :
                partition_rect = rectangle(width_start, partition_heights[i], partition_width, partition_heights[i + 1])
            else :
                partition_rect = rectangle(partition_heights[i], width_start, partition_heights[i + 1], partition_width)
            partitions.append(node(partition_rect, partition(proportions[i].children, partition_rect)))
        # partition for proportions[proportion_count:]
        if dx >= dy :
            partition_rect = rectangle(partition_width, height_start, width_end, height_end)
        else :
            partition_rect = rectangle(height_start, partition_width, height_end, width_end)
        partitions.extend(partition(proportions[proportion_count:], partition_rect))
        return partitions

def print_tree(nodes, depth = 0) :
    for node in nodes :
        print('    ' * depth + str(node.value))
        print_tree(node.children, depth + 1)

def gen_tree(breadth, depth, valspan) :
    tree = list()
    if breadth < 2 or depth < 1 :
        return tree
    b = random.randint(1, breadth)
    for i in range(b) :
        d = random.randint(1, depth)
        v = (valspan / 4) + random.random() * valspan
        tree.append(node(v, gen_tree(b - 1, d - 1, valspan)))
    return sorted(tree)

def draw_partitions(partitions, array) :
    for partition in partitions :
        (x1, y1, x2, y2) = (partition.value.x1, partition.value.y1, partition.value.x2, partition.value.y2)
        array[int(x1) : int(x2), int(y1) : int(y2), :] = 1
        array[int(x1) + 1 :, int(y1) + 1 :, :] = 0
        draw_partitions(partition.children, array)

#proportions = [node(1, [node(3, []), node(2, [node(1, []), node(1, [])]), node(1, [])])]
while True :
    proportions = [node(1, gen_tree(8, 6, 20))]
    #print_tree(proportions)
    rect = rectangle(0, 0, 50 + 450 * random.random(), 50 + 450 * random.random())
    partitions = partition(proportions, rect)
    print_tree(partitions)
    array = np.ones((int(rect.x2) - int(rect.x1), int(rect.y2) - int(rect.y1), 3), dtype = np.double)
    array[int(rect.x1) + 1 :, int(rect.y1) + 1 :, :] = 0
    draw_partitions(partitions, array)
    plt.imshow(array)
    plt.show()
    print('')