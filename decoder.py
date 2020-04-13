#!/usr/bin/env python3

import os
#os.system('pip3 install tensorflow_compression-1.2-cp36-cp36m-manylinux1_x86_64.whl')
from glob import glob
from network import decompress

def main():  

  for image_file in glob('images/*.npz'):
  
    print(image_file[7:])
    
    if not os.path.isdir('./recon'):
      os.makedirs('./recon')
    input = image_file
    
    output = 'recon/' + image_file[7:-4] + '.png'
    num_filters = 128
    checkpoint_dir = 'models'
    decompress(input, output, num_filters, checkpoint_dir)
    


if __name__ == "__main__":
  main()





