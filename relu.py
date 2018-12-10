def relu(input):  
    relu_out = np.zeros(input.shape)  
    for map_num in range(input.shape[-1]):  
        for r in np.arange(0,input.shape[0]):  
            for c in np.arange(0, input.shape[1]):  
                relu_out[r, c, map_num] = np.max(input[r, c, map_num], 0) 