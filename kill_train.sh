xcmd=$(ps -ef | grep train | head -1 | awk -F: '{ print $1"~"$4 }')
cmdtext=$(echo $xcmd | awk -F"~.." '{ print $2 }')
xpid=$(echo $xcmd | awk '{ print $2 }')

kill ${xpid}

echo "process id ${xpid}, $cmdtext killed"

