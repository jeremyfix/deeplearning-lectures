#!/bin/bash

if [ $# -ne 2 ] && [ $# -ne 3 ]
then
    echo "Usage : "
    echo `basename $0` " login <mode> <jobnum>"
    echo " mode = 0 : within CentraleSupelec"
    echo " mode = 1 : externally"
    echo "<jobnum> is optional; if you do not provide it, I set it to 1"
    exit
fi

login=$1
mode=$2
if [ $# -ne 3 ]; then
    job_num=1
else
    job_num=$3
fi

GREEN="\\e[1;32m" 
NORMAL="\\e[0;39m" 
RED="\\e[1;31m"
BLUE="\\e[1;34m" 
MAGENTA="\\e[1;35m"

display_info() {
    echo -e "$BLUE $1 $NORMAL"
}
display_wait() {
    echo -e "$MAGENTA $1 $NORMAL"
}
display_success() {
    echo -e "$GREEN $1 $NORMAL"
}
display_error() {
    echo -e "$RED $1 $NORMAL"
}


if [ -f job_id$job_num ]
then
    display_info "The file job_id$job_num exists. I will kill the previous reservation in case it is running"
    if [ $mode == 0 ]
    then
	ssh $login@term2.grid "oardel `cat job_id$job_num`"
    else
	ssh_options="-o ProxyCommand=ssh -W %h:%p $login@ghome.metz.supelec.fr"
	ssh "$ssh_options" $login@term2.grid "oardel `cat job_id$job_num`"
    fi
    
    if [ $? -ne 0 ]; then
	display_error "[Error] Stopping there"
	exit 1
    fi
    display_info "Waiting for the previous job to be killed"
    # We wait until the job is really killed
    # we might check the status of the job with oarstat -j `cat job_id` -s
    # returns JOB_ID: {Running, Finishing, Error}
    sleep 3
    display_success "Done"
    rm job_id$job_num
fi
