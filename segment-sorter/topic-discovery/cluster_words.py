#!/usr/bin/python
'''
Goal:
Cluster (quantize) regions (raw-words)

Input:
List of region-data file, e.g. .sift

Output:
KMeans model
'''
import sys
import numpy as np
import cPickle
from sklearn.cluster import KMeans
import sys
sys.path.append('/home/jogie/git-jogie/lab1231-sun-prj/segment-sorter/common')
import segment_sorter_util as util

def read_region_from_list(list_filepath, region_dir):
    with open(list_filepath) as f:
        content = f.readlines()
    content = [i.strip('\n') for i in content]

    region_list = []
    for i, filename in enumerate(content):
        print('reading region data from: %s (%i of %i)' % (filename, i+1, len(content)))
        filepath = region_dir + '/' + filename + '.sift'
        local_region_list = util.get_region_from_regfile(filepath, clustered=False)

        local_region_list_with_id = []
        for region in local_region_list:
            new_region = region
            new_region['id'] = filename
            local_region_list_with_id.append(new_region)

        region_list = region_list + local_region_list_with_id

    return region_list

def write_clustered_region(region_list, clustered_region_dir):
    filenames = set( [region['id'] for region in region_list] )

    for filename in filenames:
        filepath = clustered_region_dir + '/' + filename + '.sift.cls'
        fo = open(filepath, "wb")

        local_region_list = [region for region in region_list if region['id']==filename]
        n_sift_param = len(local_region_list[0]['sift'])
        fo.write(str(n_sift_param)+'\n')
        fo.write(str(len(local_region_list))+'\n')
        for region in local_region_list:
            loc_param_str = ''.join(str(i)+' ' for i in region['loc'])
            sift_param_str = ''.join(str(i)+' ' for i in region['sift'])
            label_param_str = str(region['label']) + ' '
            id_param_str = region['id']

            line = loc_param_str + sift_param_str + label_param_str + id_param_str + '\n'
            fo.write(line)
        fo.close()

def main(argv):
    # Load
    list_filepath = argv[1]
    region_dir = argv[2]
    clustered_region_dir = argv[3]
    model_filepath = argv[4]
    X_filepath = argv[5]

    region_list = read_region_from_list(list_filepath, region_dir)

    # Cluster
    X = np.asarray( [ i['sift'] for i in region_list ] )
    np.savetxt(X_filepath, X, delimiter=",")
    print('n_sample= %i' % (X.shape[0]))
    print('n_feature= %i' % (X.shape[1]))

    kmeans = KMeans(n_clusters=int(argv[6]),max_iter=int(argv[7]),n_init=int(argv[8]),init=argv[9],n_jobs=-1)

    print('kmeans param: %s' % (kmeans.get_params))
    print('kmeans.fit()...')
    kmeans.fit(X)

    # Attach the cluster label to the region data
    for i, region in enumerate(region_list):
        region['label'] = kmeans.labels_[i]

    # 
    write_clustered_region(region_list, clustered_region_dir)
    # TODO: Visualize

    # Save the k-means model
    with open(model_filepath, 'wb') as fid:
        cPickle.dump(kmeans, fid) 

if __name__ == '__main__':
    main(sys.argv)