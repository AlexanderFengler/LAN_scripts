#!/bin/bash

while getopts c:d: flag
    do 
        echo "passed through"
        case "${flag}" in
            c) config_file=${OPTARG};;
            d) config_dict_key=${OPTARG};;
        esac
    done 

echo "$config_file"
echo "$config_dict_key"