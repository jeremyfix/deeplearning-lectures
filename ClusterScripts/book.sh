#!/bin/bash

usage_m="Usage :
Books a node on the GPU cluster of CentraleSupelec Metz

   -u, --user <login>          login to connect to CS Metz
   -m, --machine <machine>     OPTIONAL, a specific machine
   -c, --cluster <cluster>     uSkynet, cameron, john, tx (default: uSkynet)
   -w, --walltime <walltime>   in hours (default: 48)
   -h, --help                  prints this help message
"

# Parse the command line arguments
USER=
CLUSTER=uSkynet
WALLTIME=48
MACHINE=

while [[ $# -gt 0 ]]
do
    key="$1"
    case $key in
        -u|--user)
            USER="$2"
            shift # pass argument
            shift # pass value
            ;;
        -c|--cluster)
            CLUSTER="$2"
            shift
            shift
            ;;
        -w|--walltime)
            WALLTIME="$2"
            shift
            shift
            ;;
        -m|--machine)
            MACHINE="$2"
            shift
            shift
            ;;
        -h|--help)
            exec echo "$usage_m";;
        *)
            exec echo "Unrecognized option $key"
    esac
done

declare -A oar_properties
oar_properties[uSkynet]="(cluster='uSkynet' and host in ('sh01', 'sh02', 'sh03', 'sh04','sh05','sh06', 'sh07','sh08','sh09','sh10','sh11', 'sh12','sh13','sh14','sh15','sh16'))"
oar_properties[cameron]="(cluster='cameron' and host in ('cam00', 'cam01', 'cam02', 'cam03', 'cam04','cam05','cam06', 'cam07','cam08','cam09','cam10','cam11', 'cam12','cam13','cam14','cam15','cam16'))"
oar_properties[tx]="(cluster='tx' and host in ('tx00', 'tx01', 'tx02', 'tx03', 'tx04','tx05','tx06', 'tx07','tx08','tx09','tx10','tx11', 'tx12','tx13','tx14','tx15','tx16'))"



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



# test_job_state job_id
# returns
test_job_state ()
{
    ssh "$ssh_options" $USER@term2.grid "oarstat -s -j $1" | awk -F ": " '{print $NF}' -
}

# book_node properties
book_node ()
{
	ssh "$ssh_options" $USER@term2.grid "oarsub -r \"$(date +'%F %T')\" -p \"$1\" -l nodes=1,walltime=$WALLTIME:00:00" > reservation.log
}


if [ -z $USER ]
then
    display_error "A login is required. Specify it with -u|--user, run with -h for help"
    exec echo "$usage_m"
    exit
fi

case $CLUSTER in
    "uSkynet"|"cameron"|"tx"|"john") ;;
    *)
        display_error "The cluster must be one of uSkynet, cameron, tx"
        exit;;
esac

# Bounce over the proxy
ssh_options="-o ProxyCommand=ssh -W %h:%p $USER@ghome.metz.supelec.fr"

display_info "Booking a node for $USER, on cluster $CLUSTER, with walltime $WALLTIME, machine is $MACHINE"

# Book a node
if [ -z $MACHINE ]
then
    book_node "${oar_properties[$CLUSTER]}"
else
    book_node "(host='$MACHINE' and cluster='$CLUSTER')"
fi

# Check the status of the reservation
resa_status=`cat reservation.log | grep "Reservation valid" | awk -F "--> " '{print $NF}' -`
if [ "$resa_status" == "OK" ]
then
    display_success "Reservation successfull"
else
    display_error "Reservation failed"
    exit
fi

job_id=`cat reservation.log | grep OAR_JOB_ID | awk -F "=" '{ print $2}' -`
display_info "Booking requested : OAR_JOB_ID = $job_id"


display_info "Waiting for the reservation to be running, might last few seconds"
job_state=`test_job_state $job_id`
while [ "$job_state" != "Running" ]
do
    display_wait "   The reservation is not yet running "
    sleep 1
    job_state=`test_job_state $job_id`
done

display_success "The reservation is running"
