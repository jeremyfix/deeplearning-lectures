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


#### Some function declaration
if [ $mode == 0 ]
then
    ssh_options=
else
    # Bounce over the proxy
    ssh_options="-o ProxyCommand=ssh -W %h:%p $login@ghome.metz.supelec.fr"
fi

test_job_state ()
{
    if [ $mode == 0 ]
    then
	ssh $login@term2.grid "oarstat -s -j `cat job_id$job_num` " | awk -F ": " '{print $NF}' -
    else
	# Bounce over the proxy
	ssh "$ssh_options" $login@term2.grid "oarstat -s -j `cat job_id$job_num` " | awk -F ": " '{print $NF}' -
    fi    
}

if [ -f job_id$job_num ]; then
    display_info "The file job_id$job_num exists. I am checking the reservation is still valid"
    job_state=`test_job_state`
    if [ "$job_state" != "Running" ]; then
	display_error "   The reservation is not running yet or anymore. Please book a machine"
	exit 0
    fi
    display_success "   The reservation is still running"
    display_info "Logging to the booked node"
    if [ $mode == 0 ]
    then
	ssh -t $login@term2.grid oarsub -C `cat job_id$job_num`
    else
	ssh "$ssh_options" -t $login@term2.grid oarsub -C `cat job_id$job_num`
    fi
    display_info "Unlogged"
else
    display_error "   The file job_id$job_num does not exist. Please book a machine"
    exit 0
fi
