import numpy as np

color_palette_blue = np.array([[  9,   1,  47], # darkest blue
                           [  4,   4,  73], # blue 5
                           [  0,   7, 100], # blue 4
                           [ 12,  44, 138], # blue 3
                           [ 24,  82, 177], # blue 2
                           [ 57, 125, 209], # blue 1
                           [134, 181, 229], # blue 0
                           [211, 236, 248], # lightest blue
                           [241, 233, 191], # lightest yellow
                           [248, 201,  95], # light yellow
                           [255, 170,   0], # dirty yellow
                           [204, 128,   0], # brown 0
                           [153,  64,   0], # brown 1
                           [106,  31,   3],  # brown 2
                           [ 66,  20,  15], # brown 3
                           [ 25,   7,  26]], dtype=np.float32) # dark violett 
color_palette_blue = color_palette_blue / 255