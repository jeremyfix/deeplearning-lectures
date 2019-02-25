#!/bin/bash

if [ $# -ne 3 ]
then
    echo "Usage : "
    echo `basename $0` " login <host> <port> "
    echo " port : the remote port to forward. It will mapped locally to localhost:port"
    echo "host : booked hostname"
    echo " Example ports :   8888 for jupyter lab"
    echo "                   6006 for tensorboard"
    exit
fi

login=$1
host=$2
port=$3

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
ssh_options_term2="-o ProxyCommand=ssh -W %h:%p $login@ghome.metz.supelec.fr"
ssh_options_node="-o ProxyCommand=ssh  -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -W %h:%p \"-o ProxyCommand=ssh -W %h:%p $login@ghome.metz.supelec.fr\" $login@term2.grid"


display_info "Activation port forwarding"

ssh "$ssh_options_node" -v -N -L $port:localhost:$port $login@$host

