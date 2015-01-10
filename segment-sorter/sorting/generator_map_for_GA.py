from util_region import csvToListOfSegment
from skimage import io
from skimage.util import img_as_float
import cPickle as pickle

segment_dir = '/home/jogie/sun4/exp/segment-sorter/superpixel/msrc/'
img_dir = '/home/jogie/sun3/dataset/msrc/unmix/Images/ppm/'

def generate_map_segment(name_file, param):
    global cache_list_segment
    map_segment_init = {}
    map_segment_for_GA = {}
    segment_in_image = [int(item_corpus[0]) for item_corpus in param['corpus_meta'] if name_file == item_corpus[1].split('-')[0]]
    cache = ('key','list_of_segment') #remember 1 before
    for i in range(len(segment_in_image)):
        index = segment_in_image[i]        
        n_col = len(img_as_float(io.imread(param['img_dir']+name_file+'.ppm'))[0])
        if param['corpus_meta'][index][1] == cache[0]:
            list_of_segment = cache[1]
        else:            
            list_of_segment = csvToListOfSegment(param['segment_dir']+'/'+name_file+'/'+ param['corpus_meta'][index][1])
            cache = (param['corpus_meta'][index][1], list_of_segment)
        no_segment = int(param['corpus_meta'][index][2])
        pixel_in_segment = [row*n_col+col for row, col in list_of_segment[no_segment]]
        max_prob_topic = max([prob_topic for prob_topic in param['prob_topic_doc'][index]])
        if len(pixel_in_segment) >= 1000:
            map_segment_init[index] = (max_prob_topic, pixel_in_segment)
    map_segment_for_GA[name_file] = map_segment_init
    return map_segment_for_GA

def main():    
    theta_path = '/home/jogie/sorter_exp/lda-model/training.20141210.010751/corpus.testing.data.theta'
    corpus_meta_path = '/home/jogie/sorter_exp/lda-model/training.20141210.010751/corpus.testing.meta'
    list_path = '/home/jogie/sun4/exp/segment-sorter/meta/93.txt'

    print 'create list file and param...'
    # list_file = [ line.strip('\n') for line in open(list_path)]
    # list_file = [ item for item in list_file if item not in ['1_4_s','1_6_s', '1_7_s']]
    list_file = ['12_7_s']

    print 'create prob_topic_doc...'
    prob_topic_doc = [ line.strip().split(' ') for line in open(theta_path)]

    for row in range(len(prob_topic_doc)):
        for col in range(len(prob_topic_doc[row])):
            prob_topic_doc[row][col] = float(prob_topic_doc[row][col])

    print 'create corpus_meta...'
    corpus_meta = [ line.strip().split(' ') for line in open(corpus_meta_path)]

    for ind, name_file in zip(range(len(list_file)), list_file):
        print ind,'/',len(list_file)
        generate_map_segment(name_file, corpus_meta, prob_topic_doc, segment_dir, img_dir)

    pickle.dump(map_segment_for_GA, open("map_segment_for_GA.pickle", "wb"))

if __name__ == "__main__":
    main();