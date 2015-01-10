#!/bin/sh
#

timestamp="$(date +'%Y%m%d.%H%M%S')"
dataset=voc2010
region_desc=sift

root_dir=/home/jogie/sorter_exp/lda-model-$dataset

#
list_filepath=$root_dir/meta/$dataset.$region_desc.list
region_dir=$root_dir/region/$dataset/$region_desc

clustered_region_dir=$root_dir/region-clustered/$dataset/$region_desc
rm -rf $clustered_region_dir
mkdir -p $clustered_region_dir

kmeans_model_dir=$root_dir/kmeans-model/$dataset
mkdir -p $kmeans_model_dir
kmeans_model_filepath=$kmeans_model_dir/kmeans.$timestamp.model
kmeans_input_filepath=$kmeans_model_dir/kmeans.$timestamp.input

#
n_word_cluster=2000
max_iter=300
n_init=10
init=k-means++

export PYTHONPATH=$PYTHONPATH:/home/tor/sun/ws/lab1231-sun-prj/segment-sorter/common
exe=../../../segment-sorter/topic-discovery/cluster_words.py

python  $exe \
		$list_filepath $region_dir \
		$clustered_region_dir $kmeans_model_filepath $kmeans_input_filepath\
		$n_word_cluster $max_iter $n_init $init
