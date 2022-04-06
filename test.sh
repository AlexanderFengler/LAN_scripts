#!/bin/bash

base='/users'\
        "/users/" #+ '/users'
base2='/users'\
$base
base3=100
echo $base$base2$base3'_additions'

my_variable=1

if [[ $my_variable -eq 1 ]]
    then
        echo 'passed through'
    fi
    
for ((i = 0; i <= 5; i++))
    do
        echo $i
    done