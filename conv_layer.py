class ConvLayer():
    def __init__(self, layer_type, ip, kernels, stride=1, padding=1):
        self.layer_type = layer_type
        self.ip = ip
        self.padding = padding
        self.stride = stride
        self.kernels = kernels
        return
    
    def _conv_maps(self, cmap, kernel):
        ip_size, ip_size = cmap.shape
        kernel_size, kernel_size = kernel.shape
        output_size = int((ip_size - kernel_size + self.padding)/self.stride)
        #### Create a new padded input map
        padded_size = ip_size + self.padding
        cmap_padded = np.zeros([padded_size, padded_size])
        for col in range(padded_size):
            for row in range(padded_size):
                if (row >= ip_size) or (col >= ip_size):
                    break
                cmap_padded[row + self.padding//2, col + self.padding//2] = cmap[row, col]

        #### Compute the output map
        output_map = np.zeros([ip_size, ip_size])
        try:
            for c in range(ip_size):
                for r in range(ip_size):
                    m1 = cmap_padded[c:c+(kernel_size - self.padding//2 + 1), r:r + (kernel_size - self.padding//2 + 1)]
                    output_map[c][r] = multiply_matrices(m1, kernel)
        except Exception as e:
            print(e)
        return output_map
      
    def _conv_3d_maps(self, cmap, kernel):
      num_channels, ip_size, ip_size = cmap.shape
      num_kernel_channels, kernel_size, kernel_size = kernel.shape
      if not (num_channels == num_kernel_channels):
        print("Mismatch in number of channels of kernel and input image")
      output_size = int((ip_size - kernel_size + self.padding)/self.stride)
      output_size = output_size
      #### Create a new padded input map
      padded_size = ip_size + self.padding
      cmap_padded = np.zeros([num_channels, padded_size, padded_size])
      for c in range(num_channels):
        for col in range(padded_size):
            for row in range(padded_size):
                if (row >= ip_size) or (col >= ip_size):
                    break
                cmap_padded[c, row + self.padding//2, col + self.padding//2] = cmap[c, row, col]

      #### Compute the output map
      output_size, output_size = padded_size//self.stride, padded_size//self.stride
      output_map_3d = np.zeros([output_size, output_size])
      stride = self.stride
      try:
          cmap_iter_row = self.padding//2
          cmap_iter_col = self.padding//2
          out_col, out_row = 0, 0
          while (out_row < output_size):
              while (out_col < output_size):
                  if (cmap_iter_col + kernel_size//2 + 1 > padded_size) or (cmap_iter_row + kernel_size//2 + 1 > padded_size):
                    break
                  m1 = cmap_padded[:, cmap_iter_row - kernel_size//2:cmap_iter_row + kernel_size//2 + 1, cmap_iter_col - kernel_size//2:cmap_iter_col + kernel_size//2 + 1]
                  output_map_3d[out_row][out_col] = np.sum(multiply_matrices(m1, kernel))                  
                  cmap_iter_col += stride
                  out_col += 1
              out_row += 1
              cmap_iter_row += stride

        #   for c in range(ip_size):
        #       for r in range(ip_size):
        #           m1 = cmap_padded[:, c:c+(kernel_size - self.padding//2 + 1), r:r + (kernel_size - self.padding//2 + 1)]
        #           output_map_3d[c][r] = np.sum(multiply_matrices(m1, kernel))
      except Exception as e:
          print(e)
          
      return output_map_3d

    def forward_pass(self):
        this_input = self.ip
        ip_depth, ip_size, ip_size = this_input.shape
        num_kernels, kernel_depth, kernel_size, kernel_size = self.kernels.shape
        outputs = []
        for kernel_id in range(num_kernels):
          out_map = np.zeros((ip_size, ip_size))
          out_map = self._conv_3d_maps(this_input, self.kernels[kernel_id])
#           for map_id, ip_map in enumerate(this_input):
#               out_map_ = self._conv_maps(ip_map, self.kernels[kernel_id][map_id])
#               out_map += out_map_
          outputs.append(out_map)
        return np.array(outputs)

    def test(self):
        cmap = np.ones([5, 10, 10])
        print(cmap.shape)
        self.ip = cmap
        out = self.forward_pass()
        print(out[0].shape)

