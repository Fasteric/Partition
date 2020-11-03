import numpy as np
import matplotlib.pyplot as plt
import random

class roomnode :
    def __init__(self, index, area) :
        self.index = index
        self.area = area
        self.subroom = list()

def partition(subroom, space) :

    # room to space factor
    space_width, space_height, _ = space.shape
    if space_height == 0 :
        return
    space_area = space_width * space_height
    room_area = 0
    for i in range(len(subroom)) :
        room_area += subroom[i].area
    room_to_space_factor = space_area / room_area
    print('room_to_space_factor', room_to_space_factor)

    maximal_worst_aspect_ratio = None
    optimal_partition_width = None
    optimal_subpartition_heights = list()
    for i in range(len(subroom)) :

        # partition_width
        partition_room_area = 0
        for j in range(i + 1) :
            partition_room_area += subroom[j].area
        partition_space_area = partition_room_area * room_to_space_factor
        partition_width = partition_space_area / space_height

        # subpartition_heights and worst_aspect_ratio
        subpartition_heights = list()
        worst_aspect_ratio = None
        for j in range(i + 1) :
            subpartition_height = subroom[j].area * room_to_space_factor / partition_width
            subpartition_heights.append(subpartition_height)
            aspect_ratio = None
            if subpartition_height > partition_width :
                aspect_ratio = subpartition_height / partition_width
            else :
                aspect_ratio = partition_width / subpartition_height
            print('wha', partition_width, subpartition_height, aspect_ratio)
            if worst_aspect_ratio == None or worst_aspect_ratio < aspect_ratio :
                worst_aspect_ratio = aspect_ratio
        if i + 1 < len(subroom) :
            remaining_aspect_ratio = None
            if space_height > space_width - partition_width :
                remaining_aspect_ratio = space_height / (space_width - partition_width)
            else :
                remaining_aspect_ratio = (space_width - partition_width) / space_height
            if worst_aspect_ratio < remaining_aspect_ratio :
                worst_aspect_ratio = remaining_aspect_ratio
            print('wha', space_width - partition_width, space_height, remaining_aspect_ratio)
        print('worst_aspect_ratio', worst_aspect_ratio)
        print('maximal_worst_aspect_ratio', maximal_worst_aspect_ratio)
        
        # ? update maximal : commit parition
        if maximal_worst_aspect_ratio == None or worst_aspect_ratio < maximal_worst_aspect_ratio :
            print('update')
            maximal_worst_aspect_ratio = worst_aspect_ratio
            optimal_partition_width = partition_width
            optimal_subpartition_heights = subpartition_heights
        else :
            break
    
    print('partition', len(optimal_subpartition_heights))
    subpartition_height_offset = 0
    for j in range(len(optimal_subpartition_heights) - 1) :
        print('height partition', subpartition_height_offset + optimal_subpartition_heights[j])
        space[:round(optimal_partition_width), round(subpartition_height_offset + optimal_subpartition_heights[j]), :].fill(1)
        subpartition_height_offset += optimal_subpartition_heights[j]
    if len(optimal_subpartition_heights) < len(subroom) :
        print('width partition', optimal_partition_width)
        space[round(optimal_partition_width), :, :].fill(1)
        partition(subroom[i:], space[round(optimal_partition_width):, :, :].transpose(1, 0, 2))

while True :
    a = roomnode(0, 1)
    n = random.randint(3, 12)
    alist = list()
    for i in range(n) :
        r = random.randint(3, 64)
        while r in alist :
            r = random.randint(3, 64)
        alist.append(r)
    alist.sort()
    for i in range(n) :
        a.subroom.append(roomnode(i + 1, alist[i]))
    img = np.zeros((random.randint(128, 192), random.randint(128, 192),3))
    print('testing', n, 'rooms')
    partition(a.subroom, img)
    plt.imshow(img)
    plt.show()
    print()
