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
from util_region import csvToListOfSegment

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

def createFigure4(list_file, list_param, corpus_meta, prob_topic_doc, segment_dir, n_topic, output_dir):
    for file_item in list_file:
        print file_item
        for topic in range(n_topic):
            print topic
            fig = plt.figure()
            img = img_as_float(io.imread(img_dir+file_item+'.ppm'))
            ax1 = fig.add_subplot(4,4, 1, axisbg='grey')
            ax1.set_xticks(()), ax1.set_yticks(())
            ax1.imshow(img)
            index=2
            for param in list_param:
                if 'slic' in param:
                    # print 'test', index
                    # print corpus_meta[0][1].split('-')[0]
                    segment_in_file = [item_corpus for item_corpus in corpus_meta if file_item == item_corpus[1].split('-')[0]]
                    print len(segment_in_file)
                    segments_res = csv2Array(segment_dir+'/'+file_item+'/'+file_item+'-'+param+'.sup')
                    img = img_as_float(io.imread(img_dir+file_item+'.ppm'))
                    output = np.zeros( (len(img), len(img[0])) )
                    for segment in segment_in_file:
                        # print prob_topic_doc[int(segment[0])][topic]
                        output[segments_res == int(segment[2])] = prob_topic_doc[int(segment[0])][topic]
                    output = mark_boundaries(output, segments_res)
                    ax1 = fig.add_subplot(4,4, index, axisbg='grey')
                    # ax1 = fig.add_subplot(5,10, index, axisbg='grey')
                    ax1.set_xticks(()), ax1.set_yticks(())
                    ax1.imshow(output)
                    index += 1 
            ensure_path(output_dir+'/'+file_item+'/')
            plt.savefig(output_dir+'/'+file_item+'/topic-'+str(topic)+'-'+file_item)
            plt.clf()
            plt.close()

def createFigure5(prob_topic_doc, corpus_meta, segment_dir, n_topic, output_dir):
    all_result = {}
    for i in range(n_topic):
        all_result[i] = []
        arr = [doc[i] for doc in prob_topic_doc]
        list_index_bestdoc = get_index_bestdoc(arr, N=25)
        for ind in list_index_bestdoc:
            all_result[i].append( (corpus_meta[ind][1], int(corpus_meta[ind][2])) )
    for i in range(n_topic):
        fig = plt.figure()
        index = 1
        print i
        for param, ind in all_result[i]:
            img_name = param.split('-')[0]
            segments_res = csv2Array(segment_dir+'/'+img_name+'/'+param)
            img = img_as_float(io.imread(img_dir+img_name+'.ppm'))  
            img[segments_res != ind] = (0.0,0.0,0.9)
            ax1 = fig.add_subplot(5,5, index, axisbg='grey')
            ax1.set_xticks(()), ax1.set_yticks(())
            ax1.imshow(img)
            index += 1    
        plt.savefig(output_dir+'/topic-'+str(i)+'.pdf')
        plt.clf()
        plt.close()

def plot_top_N_segment_in_topic(list_file, segment_dir, corpus_meta, prob_topic_doc, n_topic, N, output_dir):
    for name_file in list_file:
        print name_file
        for topic in range(n_topic):
            print topic
            fig = plt.figure()
            index_plt = 1
            ind_segment = [int(item_corpus[0]) for item_corpus in corpus_meta if name_file == item_corpus[1].split('-')[0]]
            val_segment = [prob_topic_doc[ind][topic] for ind in ind_segment]
            ind_of_ind_segment = get_index_bestdoc(val_segment, 10)
            for index in ind_of_ind_segment:
                segments_res = csv2Array(segment_dir+'/'+name_file+'/'+ corpus_meta[ind_segment[index]][1])
                no_segment = int(corpus_meta[ind_segment[index]][2])
                output = img_as_float(io.imread(img_dir+name_file+'.ppm'))
                output[segments_res != no_segment] = (0.0,0.0,0.9)
                ax1 = fig.add_subplot(4,3, index_plt, axisbg='grey')
                ax1.set_xticks(()), ax1.set_yticks(())
                ax1.set_title(prob_topic_doc[ind_segment[index]][topic])
                ax1.imshow(output)
                index_plt += 1
            ensure_path(output_dir+'/'+name_file+'/')
            plt.savefig(output_dir+'/'+name_file+'/'+str(topic))
            plt.clf()
            plt.close()            

def writeCsv(buff, output_path):
    with open(output_path, 'w') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=',')
        for item_buff in buff:
            spamwriter.writerow(item_buff)

def create_best_segment_image(list_file, segment_dir, corpus_meta, prob_topic_doc, n_topic, n_segment, output_dir):
    for name_file in list_file:
        print name_file
        buff_segment = []
        ind_segment = [int(item_corpus[0]) for item_corpus in corpus_meta if name_file == item_corpus[1].split('-')[0]]
        val_segment = [(ind, prob_topic_doc[ind][topic]) for ind in ind_segment for topic in range(n_topic)]
        val_segment = sorted(val_segment,key=itemgetter(1), reverse=True)
        bag_of_best_segment = val_segment[0:n_segment]
        fig = plt.figure()
        index_plt = 1
        for index, prob in bag_of_best_segment:
            print 'jogie'
            img = img_as_float(io.imread(img_dir+name_file+'.ppm'))
            n_col = len(img[0])
            list_of_segment = csvToListOfSegment(segment_dir+'/'+name_file+'/'+ corpus_meta[index][1])
            no_segment = int(corpus_meta[index][2])
            buff_segment.append([row*n_col+col for row, col in list_of_segment[no_segment]])
            segments_res = csv2Array(segment_dir+'/'+name_file+'/'+ corpus_meta[index][1])
            img[segments_res != no_segment] = (0.0,0.0,0.9)
            ax1 = fig.add_subplot(6,5, index_plt, axisbg='grey')
            ax1.set_xticks(()), ax1.set_yticks(())
            ax1.imshow(img)
            index_plt += 1
        plt.savefig('coba/'+name_file)
        plt.clf()
        plt.close()
        # writeCsv(buff_segment, output_dir+'/'+name_file+'.sup')

def main():
    #to do -> call with args
    output_dir = '/home/jogie/sorter_exp/exp_result/msrc/141221-500K/'
    segment_dir = '/home/jogie/sun4/exp/segment-sorter/superpixel/msrc/'
    sup_param_dir = '/home/jogie/sun4/exp/segment-sorter/meta/segment-param.list'
    list_path = '/home/jogie/sun4/exp/segment-sorter/meta/test591.list'
    theta_final_path = '/home/jogie/sorter_exp/lda-model/training.20141221.182404/model-final.theta'
    corpus_meta_path = '/home/jogie/sorter_exp/lda-model/training.20141221.182404/corpus.20141221.145856.meta'
    
    print 'craete prob_topic_doc...'
    prob_topic_doc = [ line.strip().split(' ') for line in open(theta_final_path)]
    
    for row in range(len(prob_topic_doc)):
        for col in range(len(prob_topic_doc[row])):
            prob_topic_doc[row][col] = float(prob_topic_doc[row][col])

    print 'craete corpus_meta...'
    corpus_meta = [ line.strip().split(' ') for line in open(corpus_meta_path)]
    
    print 'craete list file and param...'
    list_file = [ line.strip('\n') for line in open(list_path)]
    list_file = [ item for item in list_file if item not in ['1_4_s','1_6_s', '1_7_s']]

    list_param = [ line.strip('\n') for line in open(sup_param_dir)]
    
    print 'craete best segment...'
    list_file = ['10_13_s']
    # create_best_segment_image(list_file, segment_dir, corpus_meta, prob_topic_doc, 21, 30, output_dir+'/best-superpixel/')

    # createFigure5(prob_topic_doc, corpus_meta, segment_dir, 21 ,output_dir+'figure5/')

    createFigure4(list_file, list_param, corpus_meta, prob_topic_doc, segment_dir, 21, output_dir+'figure4/')
    # plot_top_N_segment_in_topic(list_file, segment_dir, corpus_meta, prob_topic_doc, 21, 10, output_dir+'figureNew')

if __name__ == "__main__":
    main();