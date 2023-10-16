#!/bin/bash
BASE_OPTION="--backbone resnet18"
if [ $1 = "PFBP-SCUT5500" ]
then
  DATASET_OPTION="--train-way 5 --dataset FBP5500"
elif [ $1 = "PFBP-SCUT500" ]
then
  DATASET_OPTION="--train-way 3 --dataset FBPSCUT"

elif [ $1 = "PFBP-US10K" ]
then
  DATASET_OPTION="--train-way 5 --dataset US10K"
else
  echo 'the {dataset} arg should be one of "PFBP-SCUT5500", "PFBP-SCUT500", or "PFBP-US10K"'
  echo "Usage: bash train_fea.sh {arg1=dataset}"
  exit 1
fi
MERGE_OPTION="$BASE_OPTION $DATASET_OPTION"
echo $MERGE_OPTION
python3 train_fea.py $MERGE_OPTION