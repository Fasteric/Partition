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
    def __repr__(self) :
        return 'partition_node({}, {})'.format(self.proportion, self.children)

class partition_rect :
    def __init__(self, rect, children) :
        self.rect = rect
        self.children = children
    def __repr__(self) :
        return 'partition_rect({}, {})'.format(self.rect, self.children)

def partition(nodes, rect) :
    if len(nodes) == 0 :
        return list()
    elif len(nodes) == 1 :
        return [partition_rect(rect, partition(nodes[0].children, rect))]
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
                # potential rounding error : sum != height_length
                current_heights.append((nodes[j].proportion / cumulative_proportion) * height_length)
            current_ratio_error = math.log2((i + 1) * current_width / height_length)
            if abs(current_ratio_error) <= abs(best_ratio_error) :
                best_width = current_width
                best_heights = current_heights
                best_ratio_error = current_ratio_error
                if current_ratio_error > 0 :
                    break
            else :
                break
        current_partition = list()
        cumulative_height = 0
        for i in range(len(best_heights)) :
            current_height_start = height_start + cumulative_height
            if rect.width >= rect.height :
                current_rect = rectangle(width_start, current_height_start, best_width, best_heights[i])
            else :
                current_rect = rectangle(current_height_start, width_start, best_heights[i], best_width)
            current_partition.append(partition_rect(current_rect, partition(nodes[i].children, current_rect)))
            # potential rounding error ?
            cumulative_height += best_heights[i]
        if rect.width >= rect.height :
            current_partition.extend(partition(nodes[len(best_heights):], rectangle(width_start + best_width, height_start, width_length - best_width, height_length)))
        else :
            current_partition.extend(partition(nodes[len(best_heights):], rectangle(height_start, width_start + best_width, height_length, width_length - best_width)))
        return current_partition

def draw(prect) :
    (x, y, width, height) = (int(prect.rect.x), int(prect.rect.y), int(prect.rect.width), int(prect.rect.height))
    array = np.ones((width, height, 3), dtype = np.double)
    array[1:, 1:, :] = 0
    for c in prect.children :
        (cx, cy, cwidth, cheight) = (int(c.rect.x), int(c.rect.y), int(c.rect.width), int(c.rect.height))
        array[cx : cx + cwidth, cy : cy + cheight, :] = draw(c)
    return array

tree = [partition_node(1, [partition_node(3, [partition_node(1, list()), partition_node(1, list())]), partition_node(2, list()), partition_node(3, list())])]
partition_result = partition(tree, rectangle(0, 0, 100, 175))
print(partition_result)
plt.imshow(draw(partition_result[0]))
plt.show()