cites_count = 5
rand_seed = 0
duration = 150
big_size = 660000
small_size = 110000
flow_to_msk = 0.15
flow_to_obl = 0.03/4
flow_to_neighbour = 0.01
flow_to_corner = 0.002

pop_files = ['msk', 'mobl', 'mobl', 'mobl', 'mobl']
imports = [10, 0, 0, 0, 0]

import pandas as pd

nodes = {
    'Id' : list(range(cites_count)),
    'Sizes' : [13149803, 8651260//4, 8651260//4, 8651260//4, 8651260//4],
    'Agent_sizes' : [big_size, small_size, small_size, small_size, small_size],
    'Image' : ['images/skyline.png', 'images/home.png', 'images/home.png', 'images/home.png', 'images/home.png']
}

pd.DataFrame(nodes).to_csv('graph/nodes.csv', sep='\t', index=False)

edges = {'Source':[],'Target':[],'Flow':[]}

adjacency_matrix = [[0, flow_to_obl, flow_to_obl, flow_to_obl, flow_to_obl],
                    [flow_to_msk, 0, flow_to_neighbour, flow_to_neighbour, flow_to_corner],
                    [flow_to_msk, flow_to_neighbour, 0, flow_to_corner, flow_to_neighbour],
                    [flow_to_msk, flow_to_neighbour, flow_to_corner, 0, flow_to_neighbour],
                    [flow_to_msk, flow_to_corner, flow_to_neighbour, flow_to_neighbour, 0]]

for i in range(cites_count):
    for j in range(cites_count):
        if i != j:
            edges['Source'].append(i)
            edges['Target'].append(j)
            edges['Flow'].append(adjacency_matrix[i][j])


pd.DataFrame(edges).to_csv('graph/edges.csv', sep='\t', index=False)