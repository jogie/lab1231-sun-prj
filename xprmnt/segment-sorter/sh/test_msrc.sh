#!/bin/sh
curr_dir="$(dirname "$0")"

#
. "$curr_dir/set_dataset_param_msrc.sh"

#
num=$1
segdir=$2"/"$3;
echo $segdir
result_dir=/home/jogie/sun4/exp/segment-sorter/ann/msrc-test-$num-$segdir"/"
test_img_list_filepath=/home/jogie/sun4/exp/segment-sorter/meta/msrc$num.txt
superpixel_dir=/home/jogie/sun4/exp/segment-sorter/best-superpixel/$segdir"/"
log_fi=log"/"$segdir"/"log.txt
# Constant params across num
exe=../../../segment-sorter/build/main 

# Annotate
# rm -rf $result_dir
mkdir -p $result_dir
mkdir -p log"/"$segdir"/"
$exe $dataset_name $n_label $ori_img_dir $test_img_list_filepath $result_dir $unary_philipp_dir $superpixel_dir > $log_fi
