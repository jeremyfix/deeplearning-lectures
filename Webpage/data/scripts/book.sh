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

#### Some function declaration
if [ $mode == 0 ]
then
    ssh_options=
else
    # Bounce over the proxy
    #ssh_options='-o "ProxyCommand=ssh -W %h:%p'" $login"'@ghome.metz.supelec.fr"'
    ssh_options="-o ProxyCommand=ssh -W %h:%p $login@ghome.metz.supelec.fr"
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

test_job_state ()
{
    if [ $mode == 0 ]; then
	ssh $login@term2.grid "oarstat -s -j `cat job_id$job_num` " | awk -F ": " '{print $NF}' -
    else
	ssh "$ssh_options" $login@term2.grid "oarstat -s -j `cat job_id$job_num` " | awk -F ": " '{print $NF}' -
    fi
}

oar_properties="(cluster='uSkynet' and host in ('sh01', 'sh02', 'sh03', 'sh04','sh05','sh06', 'sh07','sh08','sh09','sh10','sh11', 'sh12','sh13','sh14','sh15','sh16')) or (cluster='cameron' and host in ('cam00', 'cam01', 'cam02', 'cam03', 'cam04','cam05','cam06', 'cam07','cam08','cam09','cam10','cam11', 'cam12','cam13','cam14','cam15','cam16'))"

book_node ()
{
    if [ $mode == 0 ]; then
	ssh $login@term2.grid "oarsub -r \"$(date +'%F %T')\" -p \"$oar_properties\" -l nodes=1,walltime=96:00:00" > reservation$job_num.log
    else
	ssh "$ssh_options" $login@term2.grid "oarsub -r \"$(date +'%F %T')\" -p \"$oar_properties\" -l nodes=1,walltime=96:00:00" > reservation$job_num.log
    fi
}


# Kill a previous reservation if any
./kill_reservation.sh $login $mode $job_num

display_info "Booking a node"
book_node

resa_status=`cat reservation$job_num.log | grep "Reservation valid" | awk -F "--> " '{print $NF}' -`
if [ "$resa_status" == "OK" ]
then
    display_success "Reservation successfull"
else
    display_error "Reservation failed"
    exit
fi

cat reservation$job_num.log | grep OAR_JOB_ID | awk -F "=" '{ print $2}' - > job_id$job_num
display_info "Booking requested : OAR_JOB_ID = `cat job_id$job_num`"

display_info "Waiting for the reservation to be running, might last few seconds"
job_state=`test_job_state`
while [ "$job_state" != "Running" ]
do
    display_wait "   The reservation is not yet running "
    sleep 1
    job_state=`test_job_state`
done

display_success "The reservation is running"
