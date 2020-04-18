#!/usr/bin/env bash

# input: jobname
jobname=$1
# set working directory to this script's location
cd $(dirname $0)

# retrieve parameters from master list
params=$(cat joblist | grep $jobname)
arch=$(cut -d' ' -f2 <<< $params)
cores=$(cut -d' ' -f3 <<< $params)
h_rt=$(cut -d' ' -f4 <<< $params)
h_data=$(cut -d' ' -f5 <<< $params)

# copy from the template
cat template.sh > $jobname.sh

# replace parameters
# note: cluster uses GNU sed version 4.2.1
sed -i -e 's/$jobname/'$jobname'/g' $jobname.sh
sed -i -e 's/$arch/'$arch'/g'       $jobname.sh
sed -i -e 's/$cores/'$cores'/g'     $jobname.sh
sed -i -e 's/$h_rt/'$h_rt'/g'       $jobname.sh
sed -i -e 's/$h_data/'$h_data'/g'   $jobname.sh

# submit the job and delete the temporary script
qsub -N $jobname $jobname.sh
rm $jobname.sh
