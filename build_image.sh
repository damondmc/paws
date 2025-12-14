#!/bin/bash

echo "Building image from"

apptainer build paws.sif paws.def

echo "Done"
