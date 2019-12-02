#!/bin/bash

usage_m="Usage :
Deletes a reservation on the GPU cluster of CentraleSupelec Metz

   -u, --user <login>          Login to connect to CS Metz
   -j, --jobid <JOB_ID>        The JOB_ID to delete. If not provided
                               a list of your booked JOB_ID will be displayed
   -j, --jobid all             Will kill all the jobs booked by <login>
   -h, --help                  Prints this help message
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

# Bounce over the proxy
ssh_options="-o ProxyCommand=ssh -W %h:%p $USER@ghome.metz.supelec.fr"

test_job_state ()
{
    ssh "$ssh_options" $USER@term2.grid "oarstat -s -j $1" | awk -F ": " '{print $NF}' -
}

list_job_id ()
{
    ssh "$ssh_options" $USER@term2.grid "oarstat -u $USER"
}

if [ -z $USER ]
then
    display_error "A login is required. Specify it with -u|--user, run with -h for help"
    exec echo "$usage_m"
    exit
fi

if [[ -z $JOBID  &&  "$JOBID" != "all" ]]
then
    display_error "No job_id is specified, you must provide one. Call with -h for more help  "
    display_info "Listing your current reservations"
    list_job_id
    exit
fi


if [[ "$JOBID" = "all" ]]
then
    display_info "Collecting all the jobid booked for $USER"
    JOBID=`list_job_id | tail +3 | awk '{print $1}' | tr '\n' ' '`
    echo "JOB ID : $JOBID"
fi

if [[ -z $JOBID ]]
then
    display_info "No job to kill for $USER. I'm leaving"
    display_success "Done"
    exit
fi


#display_info "I am checking if the reservation $JOBID is still valid"
#job_state=`test_job_state $JOBID`
#if [ "$job_state" != "Running" ]; then
#    display_error "   The reservation is not running yet or anymore."
#    display_error "   please select a valid job id"
#    list_job_id
#    exit 0
#fi

display_info "Killing the reservation(s) $JOBID"
ssh "$ssh_options" $USER@term2.grid "oardel $JOBID"

display_success "Done"
