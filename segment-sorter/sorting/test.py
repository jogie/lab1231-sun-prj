from util_region import detect_region
from util_region import compute_description_region
from util_region import csvToListOfSegment

# print csvToListOfSegment('/home/jogie/sun4/exp/overlap-segment/superpixel-3/9_9_s/9_9_s-slic-13-10-1.csv')

# target_listpath_msrc = '/home/jogie/sun3/dataset/msrc/unmix/Images/list.txt'
target_listpath = '/home/jogie/sun4/exp/segment-sorter/meta/voc600.txt'

# img_dir_msrc = '/home/jogie/sun3/dataset/msrc/unmix/Images/ppm/'
img_dir = '/home/jogie/sun3/dataset/voc2010/Images/ppm/'

haraff_dir = '/home/jogie/sorter_exp/descriptor/voc2010/haraff/'
sift_dir = '/home/jogie/sorter_exp/descriptor/voc2010/sift/'

with open(target_listpath) as f:
  target_list = f.readlines()
target_list = [x.strip('\n') for x in target_list]
  
index = 0
for target in target_list:
  image_file = img_dir + target + '.ppm'
  haraff_file = haraff_dir+ target + '.haraff'
  sift_file = sift_dir + target + '.sift'
  detect_region(image_file, haraff_file, thres=1000)
  print index,'/region' 
  compute_description_region(image_file, haraff_file, sift_file, method='sift')
  print index,'/sift'
  index += 1
