import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.append('/home/jogie/git-jogie/lab1231-sun-prj/segmenter/src-py')
from segment import doSegment

import csv
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
param_segment_dir = '/home/jogie/sun4/exp/overlap-segment/superpixel-3'
kmeans_file = "/home/jogie/sorter_exp/kmeans-model/kmeans.20141207.1326.model"
model_LDA_file = '/home/jogie/sorter_exp/lda-model/training.20141207.093541/model-final.phi'

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

def normalize_hist(word_hist, n_element, epsilon=0.00001):
    normal_hist = {}
    n_non_zero = len(word_hist)
    n_zero = n_element - n_non_zero
    if n_zero == n_element:
        for i in range(n_element):
            normal_hist[i] = 1.0 / float(n_element)
        return normal_hist
    for i in range(n_element):
        if i in word_hist:
            normal_hist[i] = word_hist[i] - (float(epsilon*n_zero)/float(n_non_zero))
        else:
            normal_hist[i] = epsilon
    return normal_hist

def calculate_prob_word_given_segment(list_word, center_region, segments):
    prob_word_segment = {}
    mix_word_region = zip(list_word, center_region)
    n_word = len(mix_word_region)
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
        normalize_word_hist = normalize_hist(word_hist, 2000)
        if len(word_in_segment) != 0: 
            prob_word_segment[index_segment] = normalize_word_hist
        index_segment += 1
    return prob_word_segment

def KLD(prob_p, prob_q):
    KLD_result = 0
    for i in range(len(prob_q)):
        KLD_result += prob_p[i] * np.log2(prob_p[i] / prob_q[i])
    return KLD_result

def rank_segment(prob_word_segments, model_LDA_file, n_topic):
    result = {}
    with open(model_LDA_file) as f:
        reader = csv.reader(f, delimiter=' ')
        model_LDA = list(reader)

    model_LDA = [row[0:len(row)-1] for row in model_LDA] #remove element null
    for row in range(len(model_LDA)):
        for col in range(len(model_LDA[row])):
            model_LDA[row][col] = float(model_LDA[row][col])

    for index_segment in prob_word_segments:
        result[index_segment] = []
        for topic in range(n_topic):
            result[index_segment].append(KLD(prob_word_segments[index_segment], model_LDA[topic]))
    return result

def prob_topic_segments(param):
    kmeans = pickle.load( open(param['kmeans_file'], "rb" ) )
    list_sift = sift_file2list(param['sift_file'])
    
    list_word = [kmeans.predict(el_sift) for el_sift in list_sift]
    list_word = [a[0] for a in list_word] #second element is dtype, ignore

    center_region = get_center_region(param['haraff_file'])
    segments = csv2ListOfSegment(param['superpixel_file'])

    prob_word_segments = calculate_prob_word_given_segment(list_word, center_region, segments) #dict key : index segment
    
    result = rank_segment(prob_word_segments, param['model_LDA_file'], 21)
    return result

def main():
    list_target = ['3_12_s']
    list_param_segment = ['slic-7-10-1', 'slic-3-10-1', 'slic-5-10-1', 'slic-7-10-1', 'slic-9-10-1', 'slic-11-10-1', 'slic-13-10-1', 'slic-3-20-1', 'slic-5-20-1', 'slic-7-20-1', 'slic-9-20-1', 'slic-11-20-1', 'slic-13-20-1']
    all_result = {}
    for target in list_target:
        all_result[target] = {}
        for param_segment in list_param_segment:
            print '------ ',param_segment, '-----------'
            param = {}
            param['haraff_file'] = haraff_dir+ target + '.haraff'
            param['sift_file'] = sift_dir + target + '.sift'
            param['superpixel_file'] = param_segment_dir+'/'+target+'/'+target+'-'+param_segment+'.csv'
            param['kmeans_file'] = kmeans_file
            param['model_LDA_file'] = model_LDA_file
            all_result[target][param_segment] = prob_topic_segments(param)
    print all_result

if __name__ == "__main__":
    main();
