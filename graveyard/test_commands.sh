#!/bin/bash

# Read in arguments:
config_dict_key=None
config_file=None
output_folder=None
n_networks=1

while [ ! $# -eq 0 ]
    do
        case "$1" in
            --config_file | -c)
                config_file=$2
                ;;
            --config_dict_key | -k)
                config_dict_key=$2
                ;;
            --output_folder | -o)
                output_folder=$2
                ;;
            --n_networks | -n)
                echo "passing number of networks $2"
                n_networks=$2
                ;;
        esac
        shift 2
    done

echo "The config file supplied is: $config_file"
echo "The config dictionary key supplied is: $config_dict_key"

if [[ $config_file == None ]]
then 
    echo "config_file is None"
else
    echo "config_file is $config_file"
fi


x='teststr'
#mynewvar='hello'
if [ -z ${mynewvar+x} ];
then
    echo "mynewvar wasn't set"
else
    echo "mynewvar was set to $mynewvar"
fi