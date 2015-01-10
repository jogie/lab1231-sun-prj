#!/usr/bin/python
'''
Goal:
Given a set of images, make a gibbslda-compatible data format, see http://gibbslda.sourceforge.net/

Format:
[M]
[document(segment)1]
[document(segment)2]
...
[document(segment)M]

'''
from os import listdir
from os.path import isfile, join
import sys
sys.path.append('/home/jogie/git-jogie/lab1231-sun-prj/segment-sorter/common')
import segment_sorter_util as util


def get_contained_region(regions, segment):
    '''
    For now: a simple check if the int of region center is in the segment

    '''
    contained_regions = [region for region in regions if (int(region['loc'][1]),int(region['loc'][0])) in segment]

    return contained_regions

def write_gibbslda_data(corpus, data_filepath, metadata_filepath):
    '''
    The corpus data format follows http://gibbslda.sourceforge.net/
    '''
    data_fo = open(data_filepath, "wb")
    meta_fo = open(metadata_filepath, "wb")
    
    data_fo.write( str(len(corpus))+'\n' )
    meta_fo.write( str(len(corpus))+'\n' )
    for i,doc in enumerate(corpus):
        #
        words = ''.join(i+' ' for i in doc['words']) + '\n'
        data_fo.write(words)

        #
        doc_id = str(i)
        segmentation_filename = doc['meta']['segmentation_filename']
        segment_id = doc['meta']['segment_id']
        meta = doc_id + ' ' + segmentation_filename + ' ' + str(segment_id) + '\n'
        meta_fo.write(meta)

    data_fo.close()
    meta_fo.close()

def main(argv):
    list_filepath = argv[1]
    with open(list_filepath) as f:
        content = f.readlines()
    content = [i.strip('\n') for i in content]

    segment_dir = argv[2]
    clustered_region_dir = argv[3]
    corpus = []

    for i, img_id in enumerate(content):
        print('Processing: %s (%i/%i)' % (img_id,i+1,len(content)))
        region_filepath = clustered_region_dir + '/' + img_id + '.sift.cls'
        regions = util.get_region_from_regfile(region_filepath, clustered=True)

        local_segment_dir = segment_dir + '/' + img_id
        segmentation_filenames = [f for f in listdir(local_segment_dir) if isfile(join(local_segment_dir,f))]        

        for segmentation_filename in segmentation_filenames:# as for one image, we may do multiple segmentations
            segmentation_filepath = local_segment_dir + '/' + segmentation_filename
            segments = util.get_segment_from_supfile(segmentation_filepath)

            for segment in segments:# NOTE: _only_ segments that contain region become documents
                if len(segment['pixels']) >= 1000:
                    doc = {}
                    contained_regions = get_contained_region(regions, segment['pixels'])            
                    if len(contained_regions) != 0:
                        words = ['word'+str(region['label']) for region in contained_regions]

                        meta = {}
                        meta['segmentation_filename'] = segmentation_filename
                        meta['segment_id'] = segment['id']

                        doc['words'] = words
                        doc['meta'] = meta
                        corpus.append(doc)

    print('n_document(=n_segment)= %i' % (len(corpus)))
    corpus_filepath = argv[4]
    metacorpus_filepath = argv[5]
    write_gibbslda_data(corpus, corpus_filepath, metacorpus_filepath)

if __name__ == '__main__':
    main(sys.argv)