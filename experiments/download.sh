#!/usr/bin/env bash

src="alandero@hoffman2.idre.ucla.edu"
maindir="~/ProximalDistanceAlgorithms/experiments"
dir="$1"

scp -r "$src:$maindir/$dir" ./
