#!/usr/bin/env python3

import os
os.system('pip3 install tensorflow_compression-1.2-cp36-cp36m-manylinux1_x86_64.whl')


from glob import glob
#from network import compress
from network import compress



def main():

  if not os.path.isdir('./images'):
    os.makedirs('./images')
    
 
  for image_file in glob('valid/*.png'):
    print(image_file[6:])
    
    
    input = image_file
    output = 'images/'+ image_file[6:-4] + '.npz'
    num_filters = 128
    checkpoint_dir = 'models'    
    compress(input, output, num_filters, checkpoint_dir)  

    
if __name__ == "__main__":
  main()
