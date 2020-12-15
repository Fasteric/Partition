import math
import numpy as np
import matplotlib.pyplot as plt

class rectangle :
    def __init__(self, x, y, width, height) :
        self.x = x
        self.y = y
        self.width = width
        self.height = height
    def __repr__(self) :
        return 'rect({}, {}, {}, {})'.format(self.x, self.y, self.width, self.height)
    def area(self) :
        return self.width * self.height

class partition_node :
    def __init__(self, proportion, children) :
        self.proportion = proportion
        self.children = children
    def __lt__(self, other) :
        return self.proportion < other.proportion

def partition(nodes, rect, rooms) :
    print('partition', nodes)
    if len(nodes) == 0 :
        print('skip')
        pass
    elif len(nodes) == 1 :
        rooms.append(rect)
        #if points.get(rect.x) == None :
        #    points[rect.x] = set()
        #points[rect.x].add(rect.y + rect.height)
        #if points.get(rect.y) == None :
        #    points[rect.y] = set()
        #points[rect.y].add(rect.x + rect.width)
        partition(nodes[0].children, rect, rooms)
    else :
        total_proportion = 0
        for node in nodes :
            total_proportion += node.proportion
        if rect.width >= rect.height :
            (width_start, height_start, width_length, height_length) = (rect.x, rect.y, rect.width, rect.height)
        else :
            (width_start, height_start, width_length, height_length) = (rect.y, rect.x, rect.height, rect.width)
        cumulative_proportion = nodes[0].proportion
        best_width = (cumulative_proportion / total_proportion) * width_length
        best_heights = [height_length]
        best_ratio_error = math.log2(best_width / height_length)
        for i in range(1, len(nodes)) :
            cumulative_proportion += nodes[i].proportion
            current_width = (cumulative_proportion / total_proportion) * width_length
            current_heights = list()
            for j in range(i + 1) :
                current_heights.append((nodes[i].proportion / cumulative_proportion) * height_length)
            current_ratio_error = math.log2((i + 1) * current_width / height_length)
            if abs(current_ratio_error) <= abs(best_ratio_error) :
                best_width = current_width
                best_heights = current_heights
                best_ratio_error = current_ratio_error
                if current_ratio_error > 0 :
                    break
            else :
                break
        cumulative_height = 0
        for i in range(len(best_heights)) :
            current_height_start = height_start + cumulative_height
            if rect.width >= rect.height :
                current_room = rectangle(width_start, current_height_start, best_width, best_heights[i])
            else :
                current_room = rectangle(current_height_start, width_start, best_heights[i], best_width)
            rooms.append(current_room)
            partition(nodes[i].children, current_room, rooms)
            cumulative_height += best_heights[i]
        if rect.width >= rect.height :
            partition(nodes[len(best_heights):], rectangle(width_start + best_width, height_start, width_length - best_width, height_length), rooms)
        else :
            partition(nodes[len(best_heights):], rectangle(height_start, width_start + best_width, height_length, width_length - best_width))

tree = [partition_node(1, [partition_node(3, list()), partition_node(2, list()), partition_node(1, list())])]
rooms = list()
partition(tree, rectangle(0, 0, 1, 1), rooms)
print(rooms)