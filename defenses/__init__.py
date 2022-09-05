from advertorch.defenses import MedianSmoothing2D, BitSqueezing, JPEGFilter
from torch import nn

def get_defense(defense_list,):
    bits_squeezing = BitSqueezing(bit_depth=3)
    median_filter = MedianSmoothing2D(kernel_size=3)
    jpeg_filter = JPEGFilter(75)
    if 'bits-squeezing' in defense_list and 'median-filter' in defense_list and 'jpeg-filter' in defense_list:
        defense = nn.Sequential(bits_squeezing, median_filter, jpeg_filter)
    elif 'bits-squeezing' in defense_list and 'median-filter' in defense_list:
        defense = nn.Sequential(bits_squeezing, median_filter)
    elif 'bits-squeezing' in defense_list and 'jpeg-filter' in defense_list:
        defense = nn.Sequential(bits_squeezing, jpeg_filter)
    elif 'median-filter' in defense_list and 'jpeg-filter' in defense_list:
        defense = nn.Sequential(median_filter, median_filter)
        
    elif 'bits-squeezing' in defense_list:
        defense = nn.Sequential(bits_squeezing)
    elif 'median-filter' in defense_list:
        defense = nn.Sequential(median_filter)
    elif 'jpeg-filter' in defense_list:
        defense = nn.Sequential(jpeg_filter)
    else:
        defense = nn.Sequential()
    print('defense: ', defense, '\n')
    return defense