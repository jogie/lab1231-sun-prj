from util_region import csvToListOfSegment
from skimage import io
from skimage.util import img_as_float

from segment_sorter import create_segmentfile_from_binary_string
from generator_map_for_GA import generate_map_segment

from pyevolve import G1DBinaryString
from pyevolve import GSimpleGA
from pyevolve import Selectors
from pyevolve import Mutators
import math
import cPickle as pickle

map_segment = {}
name_file_current = ''
index_converter = []

segment_dir = '/home/jogie/sun4/exp/segment-sorter/superpixel/msrc/'
img_dir = '/home/jogie/sun3/dataset/msrc/unmix/Images/ppm/'
img = []

def calc_probability_score(chromosome, ind_bit_1):
    global index_converter
    total_prob_in_segment = sum([index_converter[ind][1][0] for ind in ind_bit_1])    
    return float(total_prob_in_segment) / len(ind_bit_1)

def calc_coverage_score(pixel_in_segment, total_pixel_in_image):    
    total_setpixel_in_segment = len(set(pixel_in_segment))
    return float(total_setpixel_in_segment) / total_pixel_in_image

def calc_nonoverlap_score(pixel_in_segment, ind_bit_1):
    total_pixel_in_segment = len(pixel_in_segment)
    total_set_pixel_in_segment = len(set(pixel_in_segment))
    return float(total_set_pixel_in_segment) / total_pixel_in_segment

def calc_mean_pixel_in_segment(total_pixel_in_segment, total_pixel_in_image, ind_bit_1):
    mean_s = float(total_pixel_in_segment) / len(ind_bit_1)
    return mean_s / total_pixel_in_image

# This function is the evaluation function, we want to maximize coverage area and probability, minimize overlap area
def eval_func(chromosome):
    ind_bit_1 = [ind for ind, bit in zip(range(len(chromosome)),chromosome) if bit == 1]    
    probability_score = calc_probability_score(chromosome, ind_bit_1)
    
    pixel_in_segment = []
    for ind in ind_bit_1:
        pixel_in_segment += index_converter[ind][1][1]    
    total_pixel_in_segment = len(pixel_in_segment)
    total_pixel_in_image = len(img) * len(img[0])

    coverage_score = calc_coverage_score(pixel_in_segment, total_pixel_in_image)
    nonoverlap_score = calc_nonoverlap_score(pixel_in_segment, ind_bit_1)
    # mean_size_segment_score = calc_mean_pixel_in_segment(total_pixel_in_segment, total_pixel_in_image, ind_bit_1)

    return coverage_score + nonoverlap_score + probability_score 

def do_selection(name_file):
    global name_file_current
    name_file_current = name_file
    global index_converter
    index_converter = [(ind, segment_info) for ind, segment_info in sorted(map_segment[name_file].iteritems())]
    
    print len(index_converter)
    genome = G1DBinaryString.G1DBinaryString(len(index_converter))

    # The evaluator function (objective function)
    genome.evaluator.set(eval_func)

    #G1DBinaryStringMutatorFlip, G1DBinaryStringMutatorSwap
    genome.mutator.set(Mutators.G1DBinaryStringMutatorFlip)

    # Genetic Algorithm Instance
    ga = GSimpleGA.GSimpleGA(genome)

    #GRankSelector, GRouletteWheel, GRouletteWheel_PrepareWheel, GTournamentSelector, GUniformSelector
    ga.selector.set(Selectors.GRankSelector)

    #G1DBinaryStringXSinglePoint, G1DBinaryStringXTwoPoint, G1DBinaryStringXUniform
    # ga.crossover.set(Crossovers.G1DBinaryStringXTwoPoint)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 

    ga.setGenerations(100)
    # ga.setMultiProcessing(True)
    # Do the evolution, with stats dump
    # frequency of 10 generations
    ga.evolve(freq_stats=10)

    # Best individual
    best = ga.bestIndividual()
    output = [index_converter[ind][0] for ind, i in zip(range(len(best)),best) if i == 1]            
    print best
    return output    

def segment_selection_using_GA(name_file, param):
    global map_segment
    map_segment = generate_map_segment(name_file, param)
    global img
    img = img_as_float(io.imread(param['img_dir']+name_file+'.ppm'))
    ind_segment = do_selection(name_file)
    create_segmentfile_from_binary_string(name_file, param['segment_dir'], param['corpus_meta'], param['output_dir'], param['img_dir'], ind_segment)

def main():
    print 'start'
    list_path = '/home/jogie/sun4/exp/segment-sorter/meta/93.txt'
    segment_dir = '/home/jogie/sun4/exp/segment-sorter/superpixel/msrc/'
    output_dir = '/home/jogie/sun4/exp/segment-sorter/best-superpixel/genetic-algorithm/100.exp4/'
    img_dir = '/home/jogie/sun3/dataset/msrc/unmix/Images/ppm/'
    theta_path = '/home/jogie/sorter_exp/lda-model/training.20141221.182404/model-final.theta'
    corpus_meta_path = '/home/jogie/sorter_exp/lda-model/training.20141221.182404/corpus.20141221.145856.meta'        

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
    # list_file = ['1_12_s', '1_13_s']#, '1_14_s', '11_14_s']
    param = {}
    for ind, name_file in zip(range(len(list_file)),list_file):
        print ind+1, '/', len(list_file)
        param['corpus_meta'] = corpus_meta
        param['prob_topic_doc'] = prob_topic_doc
        param['segment_dir'] = segment_dir
        param['img_dir'] = img_dir
        param['output_dir'] = output_dir
        segment_selection_using_GA(name_file, param)
    
if __name__ == "__main__":
    main();