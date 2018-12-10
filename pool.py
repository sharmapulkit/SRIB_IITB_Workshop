def pooling(input, size=2, stride=2):  
    pool_out = np.zeros((np.uint16((input.shape[0] - size + 1)/stride),  
                            np.uint16((input.shape[1] - size + 1)/stride),  
                            input.shape[-1]))  
    for map_num in range(input.shape[-1]):  
        r2 = 0  
        for r in np.arange(0, input.shape[0] - size - 1, stride):  
            c2 = 0  
            for c in np.arange(0, input.shape[1] - size-1, stride):  
                pool_out[r2, c2, map_num] = np.max(input[r:r+size, c:c+size])  
                c2 = c2 + 1  
            r2 = r2 +1  
	return pool_out