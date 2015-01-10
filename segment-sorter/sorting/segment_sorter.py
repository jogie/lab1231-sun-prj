import matplotlib.pyplot as plt
import numpy as np
import sys
import os

sys.path.append('/home/jogie/git-jogie/lab1231-sun-prj/segmenter/src-py')
from segment import doSegment
from make_figure import csv2Array

import csv
import pickle
from operator import itemgetter
import math 

from util_region import csvToListOfSegment

from sklearn.cluster import KMeans
from skimage.util import img_as_float
from skimage.segmentation import mark_boundaries
from skimage import io

def ensure_path(path):
  directory = path[:path.rfind('/')]
  if not os.path.exists(directory):
    os.makedirs(directory)

def writeCsv(buff, output_path):
    with open(output_path, 'w') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=',')
        for item_buff in buff:
            spamwriter.writerow(item_buff)

#best n segment in cross all topic
def create_best_n_segment_image(name_file, segment_dir, corpus_meta, prob_topic_doc, n_topic, n_segment, output_dir, img_dir):
    buff_segment = []
    
    ind_segment = [int(item_corpus[0]) for item_corpus in corpus_meta if name_file == item_corpus[1].split('-')[0]]
    val_segment = [(ind, prob_topic_doc[ind][topic]) for ind in ind_segment for topic in range(n_topic)]
    val_segment = sorted(val_segment,key=itemgetter(1), reverse=True)

    bag_of_best_segment = val_segment[0:n_segment]
    for index, prob in bag_of_best_segment:
        img = img_as_float(io.imread(img_dir+name_file+'.ppm'))
        n_col = len(img[0])
        list_of_segment = csvToListOfSegment(segment_dir+'/'+name_file+'/'+ corpus_meta[index][1])
        no_segment = int(corpus_meta[index][2])
        buff_segment.append([row*n_col+col for row, col in list_of_segment[no_segment]])
    
    writeCsv(buff_segment, output_dir+'/'+name_file+'.sup')

#sort segment per topic
def sort_segment_based_topic(name_file, segment_dir, corpus_meta, prob_topic_doc, n_topic, n_segment, output_dir, img_dir):
    buff_segment = []    
    ind_segment = [int(item_corpus[0]) for item_corpus in corpus_meta if name_file == item_corpus[1].split('-')[0]]
    val_per_topic = []
    for topic in range(n_topic):
        val = [(ind, prob_topic_doc[ind][topic]) for ind in ind_segment]
        sort_val = sorted(val, key=itemgetter(1), reverse=True)
        val_per_topic.append(sort_val)    
    sort_val_per_topic =  sorted(val_per_topic, key=lambda x: x[0][1], reverse=True)

    first_val = [fi[0][1] for fi in sort_val_per_topic[0:4]]
    quota_per_topic = [int(math.floor(fi/sum(first_val) * n_segment)) for fi in first_val]
    print quota_per_topic

    bag_of_best_segment = [sort_val_per_topic[ind][0:quota] for quota, ind in zip(quota_per_topic, range(len(quota_per_topic)))]

    for row in bag_of_best_segment:
        for index, prob in row:
            img = img_as_float(io.imread(img_dir+name_file+'.ppm'))
            n_col = len(img[0])
            list_of_segment = csvToListOfSegment(segment_dir+'/'+name_file+'/'+ corpus_meta[index][1])
            no_segment = int(corpus_meta[index][2])
            buff_segment.append([row*n_col+col for row, col in list_of_segment[no_segment]])
    writeCsv(buff_segment, output_dir+'/'+name_file+'.sup')

#sort segment cross_topic
def sort_segment_cross_topic(name_file, segment_dir, corpus_meta, prob_topic_doc, n_topic, n_segment, output_dir, img_dir):
    buff_segment = []    
    ind_segment = [int(item_corpus[0]) for item_corpus in corpus_meta if name_file == item_corpus[1].split('-')[0]]
    val_per_topic = []
    for topic in range(n_topic):
        val = [(ind, prob_topic_doc[ind][topic]) for ind in ind_segment]
        sort_val = sorted(val, key=itemgetter(1), reverse=True)
        val_per_topic.append(sort_val)    
    sort_val_per_topic =  sorted(val_per_topic, key=lambda x: x[0][1], reverse=True)

    candidate_segment = []
    for i in range(4):
        candidate_segment += sort_val_per_topic[i]

    candidate_segment = sorted(candidate_segment,key=itemgetter(1), reverse=True)
    bag_of_best_segment = candidate_segment[0:n_segment]

    for index, prob in bag_of_best_segment:
        img = img_as_float(io.imread(img_dir+name_file+'.ppm'))
        n_col = len(img[0])
        list_of_segment = csvToListOfSegment(segment_dir+'/'+name_file+'/'+ corpus_meta[index][1])
        no_segment = int(corpus_meta[index][2])
        buff_segment.append([row*n_col+col for row, col in list_of_segment[no_segment]])    
    writeCsv(buff_segment, output_dir+'/'+name_file+'.sup')

def create_segmentfile_from_binary_string(name_file, segment_dir, corpus_meta, output_dir, img_dir, ind_segment):
    buff_segment = []
    for index in ind_segment:
        img = img_as_float(io.imread(img_dir+name_file+'.ppm'))
        n_col = len(img[0])
        segments_res = csv2Array(segment_dir+'/'+name_file+'/'+ corpus_meta[index][1])
        list_of_segment = csvToListOfSegment(segment_dir+'/'+name_file+'/'+ corpus_meta[index][1])
        no_segment = int(corpus_meta[index][2])
        img[segments_res != no_segment] = (0.0,0.0,0.9)
        buff_segment.append([row*n_col+col for row, col in list_of_segment[no_segment]])
        ensure_path(output_dir+'/image/'+name_file+'/')
        io.imsave(output_dir+'/image/'+name_file+'/'+str(index)+'.png', img)
    writeCsv(buff_segment, output_dir+'/'+name_file+'.sup')


def main():
    img_dir = '/home/jogie/sun3/dataset/msrc/unmix/Images/ppm/'
    segment_dir = '/home/jogie/sun4/exp/segment-sorter/superpixel/msrc/'
    segmen_param_path = '/home/jogie/sun4/exp/segment-sorter/meta/segment-param.list'
    list_path = '/home/jogie/sun4/exp/segment-sorter/meta/93.txt'
    theta_path = '/home/jogie/sorter_exp/lda-model/training.20141210.010751/corpus.testing.data.theta'
    corpus_meta_path = '/home/jogie/sorter_exp/lda-model/training.20141210.010751/corpus.testing.meta'
    output_dir = '/home/jogie/sun4/exp/segment-sorter/best-superpixel/'
    n_topic = 21
    n_segment = 200

    print 'create prob_topic_doc...'
    prob_topic_doc = [ line.strip().split(' ') for line in open(theta_path)]
    
    for row in range(len(prob_topic_doc)):
        for col in range(len(prob_topic_doc[row])):
            prob_topic_doc[row][col] = float(prob_topic_doc[row][col])

    print 'create corpus_meta...'
    corpus_meta = [ line.strip().split(' ') for line in open(corpus_meta_path)]

    print 'create list file and param...'
    list_file = [ line.strip('\n') for line in open(list_path)]
    list_file = [ item for item in list_file if item not in ['1_4_s','1_6_s', '1_7_s']]

    list_param = [ line.strip('\n') for line in open(segmen_param_path)]
    
    print 'create best segment...'
    # n_segment_list = [50, 100, 150, 250, 300, 350]
    n_segment_list = [200]
    for name_file, ctr in zip(list_file, range(len(list_file))):
        print ctr+1,'/',len(list_file)
        for n_segment in n_segment_list:
            print n_segment,
            # ensure_path(output_dir+'cross-topic/'+str(n_segment)+'/')
            # sort_segment_cross_topic(name_file, segment_dir, corpus_meta, prob_topic_doc, n_topic, n_segment, output_dir+'cross-topic/'+str(n_segment), img_dir)
            # ensure_path(output_dir+'based-topic/'+str(n_segment)+'/')            
            # sort_segment_based_topic(name_file, segment_dir, corpus_meta, prob_topic_doc, n_topic, n_segment, output_dir+'based-topic/'+str(n_segment), img_dir)
            ensure_path(output_dir+'all-topic/'+str(n_segment)+'/')
            create_best_n_segment_image(name_file, segment_dir, corpus_meta, prob_topic_doc, n_topic, n_segment, output_dir+'all-topic/'+str(n_segment), img_dir)
        print '\n'

if __name__ == "__main__":
    main();