#!/usr/bin/python

import sys
import os
import matplotlib.pyplot as plt
import numpy as np
import csv

from skimage.segmentation import felzenszwalb, slic, quickshift
from skimage import color
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float
from skimage import io

def doSegment(param, img):
  segments_res = []
  if param[0] == 'slic':
    segments_res = slic(img, n_segments=int(param[1]), compactness=int(param[2]), sigma=int(param[3]), convert2lab=True, enforce_connectivity=True)
  elif param[0] == 'pff':
    segments_res = felzenszwalb(img, scale=int(param[1]), sigma=float(param[2]), min_size=int(param[3]))
  elif param[0] == 'quick':
    segments_res = quickshift(img, kernel_size=int(param[1]), max_dist=int(param[2]), ratio=float(param[3]), convert2lab=True)
  return segments_res

def ensure_path(path):
  # directory = path
  directory = path[:path.rfind('/')]
  if not os.path.exists(directory):
    os.makedirs(directory)

def segmentToCsv(path_output, segments):
  with open(path_output, 'w') as the_file:
    writer = csv.writer(the_file)
    writer.writerows(segments)

def segmentsToFile(path_output, segments):
  map_segment = {}
  for i in range(np.max(segments)+1):
    map_segment.setdefault(i, [])
  
  for row in range(len(segments)):
    for col in range(len(segments[row])):
      map_segment[segments[row][col]].append(row*len(segments[row]) + col)
  str_buf = ''
  for x in xrange(len(map_segment)):    
    str_buf += str(map_segment[x]).strip('[]').replace(" ", "") + '\n'
  ensure_path(path_output)
  with open(path_output, 'w') as the_file:
    the_file.write(str_buf)

def main():
  # param 1 : path-list-gambar, param 2 : path-list-param, param 3 : dir-image, param4 : dir-output
  if len(sys.argv) != 5:
    assert False, "INSUFFICIENT ARGUMENTS!"
    
  filenames = [line.strip() for line in open(sys.argv[1])]

  count = 1;
  for filename in filenames:
    print 'segment image : ' + str(count) + '/' + str(len(filenames))
    count+=1
    file_path = (sys.argv[3]).strip() + '/' + filename
    for line in open(sys.argv[2]):      
      image = img_as_float(io.imread(file_path))
      param = line.strip().split('-')
      segments_res = doSegment(param, img=image)
      # path_output=sys.argv[4] + filename[:len(filename)-4] + '/' + filename[:len(filename)-4]+'-'+'-'.join(param)+'.sup'
      path_output=sys.argv[4] + filename[:len(filename)-4] + '/' + filename[:len(filename)-4]+'-'+'-'.join(param)+'.csv'
      seg_img = mark_boundaries(image, segments_res)
      io.imsave(sys.argv[4] + '/image/' + filename[:len(filename)-4]+'-'+'-'.join(param)+'.bmp', seg_img)
      # segmentsToFile(path_output, segments_res)    
      segmentToCsv(path_output, segments_res)    

if __name__ == "__main__":
  main();
