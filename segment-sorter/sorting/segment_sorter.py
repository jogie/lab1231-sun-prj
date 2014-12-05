import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.append('/home/jogie/git-jogie/lab1231-sun-prj/segmenter/src-py')
from segment import doSegment

import pickle
from util_region import detect_region
from util_region import compute_description_region
from util_region import csv2ListOfSegment

from sklearn.cluster import KMeans
from skimage.util import img_as_float
from skimage.segmentation import mark_boundaries
from skimage import io

img_dir = '/home/jogie/sun3/dataset/msrc/unmix/Images/ppm/'
haraff_dir = '/home/jogie/sorter_exp/haraff/'
sift_dir = '/home/jogie/sorter_exp/sift/'

def sift_file2list(sift_file):
    with open(sift_file) as f:
        list_sift_raw = f.readlines()  
    list_sift_raw2 = [element.strip('\n').split(' ') for element in list_sift_raw[2:len(list_sift_raw)] ]
    list_sift = [element[5:len(element)] for element in list_sift_raw2]
    return [map(int, element) for element in list_sift] 

def get_center_region(haraff_file):
    with open(haraff_file) as f:
        list_haraff_raw = f.readlines()
    n_region = int(list_haraff_raw[1])
    result = []
    for i in range(n_region):
        elips_param = list_haraff_raw[i+2].split(' ')
        x,y = (float(elips_param[0]), float(elips_param[1]) )
        result.append( (int(y), int(x)) )
    return result

def calculate_prob_word_given_segment(list_word, center_region, segments):
    prob_word_segment = {}
    mix_word_region = zip(list_word, center_region)
    print 'jumlah region ',len(mix_word_region)
    index_segment = 0
    for segment in segments:
        word_hist = {}
        word_in_segment = []
        for word, center_reg in mix_word_region:
            if center_reg in segment:
                word_in_segment.append(word)
        print 'word in segment', word_in_segment
        for word in set(word_in_segment):
            word_hist[word] = float(len([w for w in word_in_segment if w == word])) / float(len(word_in_segment))
        prob_word_segment[index_segment] = word_hist
        index_segment += 1
    return prob_word_segment

def main():
    target = '3_12_s'
    kmeans = pickle.load( open( "/home/jogie/sorter_exp/kmeans-model/test.kmeans", "rb" ) )
    image_path = img_dir + target + '.ppm'
    haraff_file = haraff_dir+ target + '.haraff'
    sift_file = sift_dir + target + '.sift'
    superpixel_file = '/home/jogie/sun4/exp/overlap-segment/superpixel-3/3_12_s/3_12_s-slic-5-10-1.csv'

    list_sift = sift_file2list(sift_file)
    list_word = [kmeans.predict(el_sift) for el_sift in list_sift]
    
    list_word = [a[0] for a in list_word]
    center_region = get_center_region(haraff_file)
    segments = csv2ListOfSegment(superpixel_file)

    prob_word_segment = calculate_prob_word_given_segment(list_word, center_region, segments) #dict key : index segment
    print prob_word_segment

if __name__ == "__main__":
    main();