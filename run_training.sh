run_dir=$(date +run-%b%d-%H%M)
out_dir=/workspace/mellotron/my_models/${run_dir}
log_dir=/workspace/logs/${run_dir}
mkdir $out_dir
mkdir $log_dir

if [ -f "training.out" ]
then
  echo "removing previous log file"
  rm training.out
fi

nohup python train.py --output_directory=${out_dir} --log_directory=${log_dir} $* >> training.out &

tail -100f training.out

