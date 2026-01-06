#!/bin/bash

sqsh_location=$(readlink -f $(dirname $0))/sqsh_files
sandbox_name=sglang_v0.5.4.post2
docker_image=lmsysorg/sglang:v0.5.4.post2

while [[ $# -gt 0 ]]; do
    case $1 in
        --docker_image)
            docker_image=$2
            shift 2
            ;;
        --sandbox_name)
            sandbox_name=$2
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            echo "Usage: $0 --docker_image <docker_image> --sandbox_name <sandbox_name>"
            exit 1
            ;;
    esac
done

mkdir -p $sqsh_location
enroot import -o $sqsh_location/$sandbox_name.sqsh docker://$docker_image
enroot create --name $sandbox_name $sqsh_location/$sandbox_name.sqsh
# enroot start --mount $(pwd):$(pwd) --root --rw $sandbox_name
