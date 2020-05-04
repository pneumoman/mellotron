xcmd=$(ps -ef | grep train | head -1 | awk -F: '{ print $1"~"$4 }')
cmdtext=$(echo $xcmd | awk -F"~.." '{ print $2 }')
xpid=$(echo $xcmd | awk '{ print $2 }')

kill ${xpid}

echo "process id ${xpid}, $cmdtext killed"

model_dir=$(ls -t my_models | head -1)
model_file=$(ls -t my_models/${model_dir} | head -1)

echo "./my_models/${model_dir}/${model_file}"

