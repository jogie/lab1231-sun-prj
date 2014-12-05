import subprocess
import csv
import numpy as np 

def detect_region(image_file, output_file, thres=1000):
  command_shell = './../../external/region-detector/detect_points.ln -haraff -i {} -o {} -thres {}'.format(image_file, output_file, thres)
  p = subprocess.Popen(command_shell, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
  retval = p.wait()

def compute_description_region(image_file, haraff_file, output_file, method='sift'):
  command_shell = './../../external/region-descriptor/compute_descriptors.ln -{} -i {} -p1 {} -o1 {}'.format(method, image_file, haraff_file, output_file)
  p = subprocess.Popen(command_shell, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
  for line in p.stdout.readlines():
    print line,
  retval = p.wait()

def csv2ListOfSegment(csv_path):
  with open(csv_path) as f:
    reader = csv.reader(f)
    segments = list(reader)
  for row in range(len(segments)):
    for col in range(len(segments[row])):
      segments[row][col] = int(segments[row][col])
  max_segment = np.max(segments)+1
  result = [[] for x in range(max_segment)]
  for row in range(len(segments)):
    for col in range(len(segments[row])):
      result[segments[row][col]].append( (row, col) )  
  return result