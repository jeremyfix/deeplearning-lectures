#!/bin/bash

set -e

usage_m="Usage :
Cleans the home of a group of users

   -p, --prefix <login prefix> login prefix of the group of users
   -r, --range <min> <max>     range to consider
   -h, --help                  prints this help message
"

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

while [[ $# -gt 0 ]]
do
    key="$1"
    case $key in
        -p|--prefix)
            PREFIXLOGIN="$2"
            shift # pass argument
            shift # pass value
            ;;
		-r|--range)
			RANGEMIN=$2
			RANGEMAX=$3
			shift  # pass argument
			shift  # pass value min
			shift  # pass value max
			;;
        -h|--help)
			display_info "$usage_m"
			exit -1
			;;
		*)
			display_error "Unrecognized option $key"
			exit -1
			;;
	esac
done

# Check if the required fields have been specified
if [ -z $PREFIXLOGIN ]
then
	display_error "A login prefix is required, see --help"
	exit -1
fi
if [ -z $RANGEMIN ] || [ -z $RANGEMAX ]
then
	display_error "A range is required, see --help"
	exit -1
fi

for i in $(seq $RANGEMIN $RANGEMAX)
do
	display_info "==== User $PREFIXLOGIN$i ===="
	ssh $PREFIXLOGIN$i@term2 'rm -rf *'
done

