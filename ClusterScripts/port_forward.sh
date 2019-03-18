#!/bin/bash

usage_m="Usage :
Forward a port from a machine you booked to your local computer

   -u, --user <login>          login to connect to CS Metz
   -j, --jobid <JOB_ID>        The JOB_ID to which to connect. If not provided
                               a list of your booked JOB_ID will be displayed
   -m, --machine <MACHINE>     The booked hostname.
   -k, --key <PATH_TO_KEY>     Use the provided ssh key for connection
   -p, --port <PORT>           The distant port <PORT> will be binded to localhost:PORT

   -h, --help                  prints this help message
"

# Parse the command line arguments
USER=
JOBID=
MACHINE=
SSHKEY=
PORT=


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
        -m|--machine)
            MACHINE="$2"
            shift
            shift
            ;;
        -k|--key)
            SSHKEY="$2"
            shift
            shift
            ;;
        -p|--port)
            PORT="$2"
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

SSHKEY_COMMAND=
if [ ! -z $SSHKEY ]
then
    display_info "I will use the file $SSHKEY as ssh key for the connection"
    if [ ! -f $SSHKEY ]
    then
        display_error "The provided file $SSHKEY does not exist."
        exit -1
    fi
    # Ok, the file exists, we need to check the permissions are 600
    permissions=`stat -c %a $SSHKEY`
    if [ ! $permissions == 600 ]
    then
        display_error "The provided ssh key has permissions $permissions"
        display_error "but must have permissions 600. Executing the following should make it :"
        display_error "chmod 600 $SSHKEY"
        exit -1
    fi
    SSHKEY_COMMAND="-i $SSHKEY"
fi

# Bounce over the proxy
ssh_options_term2="-o ProxyCommand=ssh $SSHKEY_COMMAND -W %h:%p $USER@ghome.metz.supelec.fr"
ssh_options_node="-o ProxyCommand=ssh $SSHKEY_COMMAND -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -W %h:%p \"-o ProxyCommand=ssh $SSHKEY_COMMAND -W %h:%p $USER@ghome.metz.supelec.fr\" $USER@term2.grid"


# get_booked_host job_id
get_booked_host ()
{
	ssh "$ssh_options_term2" $USER@term2.grid "oarstat -f -j $1 " | grep assigned_hostnames | awk -F " = " '{print $NF}'
}

# test_job_state job_id
test_job_state ()
{
	ssh "$ssh_options_term2" $USER@term2.grid "oarstat -s -j $1 " | awk -F ": " '{print $NF}' -
}

list_job_id ()
{
    ssh "$ssh_options_term2" $USER@term2.grid "oarstat -u $USER"
}

if [ -z $USER ]
then
    display_error "A login is required. Specify it with -u|--user, run with -h for help"
    exec echo "$usage_m"
    exit -1
fi

if [ -z $PORT ]
then
    display_error "A port is required. Specify it with -p|--port, run with -h for help"
    exec echo "$usage_m"
    exit -1
fi

if [ -z $JOBID ] && [ -z $MACHINE ]
then
    display_error "You must specify a machine with -m <MACHINE>  or a job id with -j <JOB_ID>"
    display_error "more info with $0 -h"
    display_info "Your current reservations are listed below :"
    list_job_id
    exit -1
fi
if [ ! -z $JOBID ] && [ ! -z $MACHINE ]
then
    display_error "You cannot specify both a machine and a jobid"
    exit -1
fi

if [ ! -z $MACHINE ]
then
    host=$MACHINE
else
    # Check the status of the job
    display_info "Checking the status of the reservation JOBID=$JOBID"
    job_state=`test_job_state $JOBID`
    if [ "$job_state" != "Running" ]; then
        display_error "   The reservation is not running yet or anymore. Please book a machine"
        exit -1
    fi
    display_success "   The reservation $JOBID is still running"
    # Request the hostname
    host=`get_booked_host $JOBID`
fi

if [ -z $host ]
then
    display_error "Error while trying to get the booked hostname"
    exit -1
fi

display_info "Activating port forwarding from host $host:$PORT to localhost:$PORT"

ssh $SSHKEY_COMMAND "$ssh_options_node" -N -L $PORT:localhost:$PORT $USER@$host

