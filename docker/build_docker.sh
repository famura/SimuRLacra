#!/bin/bash
docker build . -t simurlacra:sacher --file ./Dockerfile.sacher
docker build . -t simurlacra:blackforest --file ./Dockerfile.blackforest
