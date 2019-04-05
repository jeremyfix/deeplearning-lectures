#!/bin/bash

usage_m="Usage :
Logs to an already booked node on the GPU cluster of CentraleSupelec Metz

   -u, --user <login>          login to connect to CS Metz
   -j, --jobid <JOB_ID>        The JOB_ID to which to connect. If not provided
                               a list of your booked JOB_ID will be displayed
   -h, --help                  prints this help message
"

# Parse the command line arguments
USER=
JOBID=

while [[ $# -gt 0 ]]
do
    key="$1"
    case $key in
        -u|--user)
            USER="$2"
            shift # pass argument
            shift # pass value
            ;;
        -j|--jobid)
            JOBID="$2"
            shift
            shift
            ;;
        -h|--help)
            exec echo "$usage_m";;
        *)
            exec echo "Unrecognized option $key"
    esac
done




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

FRONTAL_NODE=term2.grid

# test_job_state job_id
# returns
test_job_state ()
{
    ssh "$ssh_options" $USER@$FRONTAL_NODE "oarstat -s -j $1" | awk -F ": " '{print $NF}' -
}

list_job_id ()
{
    ssh "$ssh_options" $USER@$FRONTAL_NODE "oarstat -u $USER"
}

if [ -z $USER ]
then
    display_error "A login is required. Specify it with -u|--user, run with -h for help"
    exec echo "$usage_m"
    exit
fi

# Bounce over the proxy
ssh_options="-o ProxyCommand=ssh -W %h:%p $USER@ghome.metz.supelec.fr"

if [ -z $JOBID ]
then
    display_error "No job_id is specified, you must provide one. Call with -h for more help  "
    display_info "Listing your current reservations"
    list_job_id
    exit
fi

display_info "I am checking if the reservation $JOBID is still valid"
job_state=`test_job_state $JOBID`
if [ "$job_state" != "Running" ]; then
    display_error "   The reservation is not running yet or anymore."
    display_error "   please select a valid job id"
    list_job_id
    exit 0
fi

display_success "   The reservation $JOBID is still running"
display_info "Logging to the booked node"

ssh "$ssh_options" -t $USER@$FRONTAL_NODE oarsub -C $JOBID

display_info "Unlogged"

