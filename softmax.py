def softmax(input):
    out = np.exp(input) 
    return out / np.sum(out)