class Activation:
    ## act_type: 0 = ReLU
    ##           1 = Softmax
    ##           2 = Sigmoid
    def __init__(self, ip, act_type):
        self.ip = ip
        self.act_type = act_type
    
    def relu(self):
        input = self.ip  
        relu_out = np.zeros(input.shape)  
        for map_num in range(input.shape[-1]):  
            for r in np.arange(0,input.shape[0]):  
                for c in np.arange(0, input.shape[1]):  
                    relu_out[r, c, map_num] = np.max(input[r, c, map_num], 0) 
        return relu_out 

    def softmax(self):
        out = np.exp(self.ip) 
        return out / np.sum(out)

    def forward_pass(self):
        if (self.act_type == 0):
            return self.relu()
        if (self.act_type == 1):
            return self.softmax()