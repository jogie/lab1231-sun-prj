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
from operator import itemgetter

img_dir = '/home/jogie/sun3/dataset/msrc/unmix/Images/ppm/'

def ensure_path(path):
  directory = path[:path.rfind('/')]
  if not os.path.exists(directory):
    os.makedirs(directory)

def get_index_bestdoc(val, N=25):
    list_index = []
    ind = range(0, len(val))
    zip_ind_val = zip(ind, val)
    zip_ind_val = sorted(zip_ind_val,key=itemgetter(1), reverse=True)
    prob = [val[1] for val in zip_ind_val[0:N]]
    for i in range(N):
        list_index.append(zip_ind_val[i][0])
    return list_index

def csv2Array(csv_path):
    with open(csv_path) as f:
        reader = csv.reader(f)
        arr = list(reader)
    for row in range(len(arr)):
        for col in range(len(arr[row])):
            arr[row][col] = int(arr[row][col])
    return np.asarray(arr)

def createFigure4(list_file, list_param, corpus_meta, prob_topic_doc, segment_dir, output_dir):
    for topic in range(21):
        print topic
        for file_item in list_file:
            for param in list_param:
                segment_in_file = [item_corpus for item_corpus in corpus_meta if str(file_item+'-'+param) in item_corpus[1]]
                segments_res = csv2Array(segment_dir+'/'+file_item+'/'+file_item+'-'+param+'.csv')
                img = img_as_float(io.imread(img_dir+file_item+'.ppm'))
                output = np.zeros( (len(img), len(img[0])) )
                for segment in segment_in_file:
                    output[segments_res == int(segment[2])] = prob_topic_doc[int(segment[0])][topic] 
                output = mark_boundaries(output, segments_res)
                ensure_path(output_dir+'/'+file_item+'/'+str(topic)+'/')
                io.imsave(output_dir+'/'+file_item+'/'+str(topic)+'/'+file_item+'-'+str(topic) + '-'+param+'.png', output)  

def createFigure5(prob_topic_doc, corpus_meta, segment_dir, output_dir):
    all_result = {}
    for i in range(21):
        all_result[i] = []
        arr = [doc[i] for doc in prob_topic_doc]
        list_index_bestdoc = get_index_bestdoc(arr, N=25)
        for ind in list_index_bestdoc:
            all_result[i].append( (corpus_meta[ind][1], int(corpus_meta[ind][2])) )
    for i in range(21):
        print i
        for param, ind in all_result[i]:
            img_name = param.split('-')[0]
            segments_res = csv2Array(segment_dir+'/'+img_name+'/'+param)
            img = img_as_float(io.imread(img_dir+img_name+'.ppm'))  
            img[segments_res != ind] = 0
            io.imsave(output_dir + '/' + str(i)+'/topic-'+str(i)+'-'+param+'-'+ str(ind) + '.png', img)

def main():
    output_dir = '/home/jogie/sorter_exp/exp_result/'
    segment_dir = '/home/jogie/sun4/exp/overlap-segment/superpixel-3-slic-only'

    theta_final_path = '/home/jogie/sorter_exp/lda-model/training.20141207.093541/model-final.theta'
    prob_topic_doc = [ line.strip().split(' ') for line in open(theta_final_path)]
    
    for row in range(len(prob_topic_doc)):
        for col in range(len(prob_topic_doc[row])):
            prob_topic_doc[row][col] = float(prob_topic_doc[row][col])

    corpus_meta_path = '/home/jogie/sorter_exp/lda-model/training.20141207.093541/corpus.20141207.085033.meta'
    corpus_meta = [ line.strip().split(' ') for line in open(corpus_meta_path)]
    
    # createFigure5(prob_topic_doc, corpus_meta, segment_dir, output_dir+'figure5/')
    
    list_file = ['1_9_s']
    list_param = ['slic-3-10-1', 'slic-5-10-1', 'slic-7-10-1', 'slic-9-10-1', 'slic-11-10-1', 'slic-13-10-1', 'slic-3-20-1', 'slic-5-20-1', 'slic-7-20-1', 'slic-9-20-1', 'slic-11-20-1', 'slic-13-20-1']
    createFigure4(list_file, list_param, corpus_meta, prob_topic_doc, segment_dir, output_dir+'figure4/')

if __name__ == "__main__":
    main();

   # list_target = ['2_16_s']
    # param = ['slic-3-10-1']
    # for i in range(21): #per topic
    #     img = img_as_float(io.imread('/home/jogie/sun3/dataset/msrc/unmix/Images/ppm/'+list_target[0]+'.ppm'))
    #     segments_res = slic(img, n_segments=3, compactness=10, sigma=1, convert2lab=True)
    #     img[segments_res == 1] = 0
    #     io.imsave('local_logo.png', img)

        # for i in range(21):
    #     print i
    #     for param, ind in all_result[i]:
    #         img_name = param.split('-')[0]
    #         segments_res = csv2Array(segment_dir+'/'+img_name+'/'+param)
    #         img = img_as_float(io.imread('/home/jogie/sun3/dataset/msrc/unmix/Images/ppm/'+img_name+'.ppm'))  
    #         img[segments_res != ind] = 0
    #         io.imsave(str(i)+'/topic-'+str(i)+'-'+param+'-'+ str(ind) + '.png', img)
    # min_val = min([min(row) for row in prob_topic_doc])
    # print 'min', min_val
    # max_val = max([max(row) for row in prob_topic_doc])
    # print 'max', max_val

        # for topic in range(21):
    #     for file_item in list_file:
    #         for param in list_param:
    #             segment_in_file = [item_corpus for item_corpus in corpus_meta if str(file_item+'-'+param) in item_corpus[1]]
    #             segments_res = csv2Array(segment_dir+'/'+file_item+'/'+file_item+'-'+param+'.csv')
    #             img = img_as_float(io.imread('/home/jogie/sun3/dataset/msrc/unmix/Images/ppm/'+file_item+'.ppm'))
    #             output = np.zeros( (len(img), len(img[0])) )
    #             for segment in segment_in_file:
    #                 output[segments_res != int(segment[2])] = (prob_topic_doc[int(segment[0])][topic] - min_val) / max_val
    #             output = mark_boundaries(output, segments_res)
    #             ensure_path(output_dir+'/'+file_item+'/'+str(topic)+'/')
    #             io.imsave(output_dir+'/'+file_item+'/'+str(topic)+'/'+file_item+'-'+str(topic) + '-'+param+'.png', output)  
