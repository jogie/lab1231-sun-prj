#!/bin/sh
#
root_dir=/home/jogie/sorter_exp/lda-model
timestamp="$(date +'%Y%m%d.%H%M%S')"
dataset=msrc21

#
corpus_name=corpus.20141221.145856
corpus_dir=$root_dir/corpus/$dataset
tr_data_filepath=$corpus_dir/$corpus_name.data
tr_metadata_filepath=$corpus_dir/$corpus_name.meta

model_dir=$root_dir/lda-model/$dataset/training.$timestamp
mkdir -p $model_dir
cp $tr_data_filepath $model_dir #as the LDAexe puts the resulted model in the same dir as the inputted training_data
cp $tr_metadata_filepath $model_dir

#
dfile=$model_dir/$corpus_name.data
ntopics=21

alpha=0.5 #from [Russel, 2006]
beta=0.5 #from [Russel, 2006]

niters=5000 #from [Russel, 2006]
savestep=1000
twords=10

lda=../../../external/GibbsLDA++-0.2/src/lda 
$lda -est -dfile $dfile -ntopics $ntopics -alpha $alpha -beta $beta -niters $niters -savestep $savestep -twords $twords 
