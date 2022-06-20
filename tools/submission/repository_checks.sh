#!/bin/bash

# Make several checks to confirm that a submission can be uploaded to a
# GitHub repository.

FILES_GREATER_THAN_50=$(find $1 -type f -size +50M)
SYMBOLIC_LINKS=$(find $1 -type l)
BAD_FILE_NAMES=$(find $1 -type f -name ".*")
SPACE_FILE_NAMES=$(find $1 -type f -name "* *")
BAD_FOLDER_NAMES=$(find $1 -type d -name ".*")
SPACE_FOLDER_NAMES=$(find $1 -type d -name "* *")

if [ ${#FILES_GREATER_THAN_50} -gt 0 ] || 
    [ ${#SYMBOLIC_LINKS} -gt 0 ] || 
    [ ${#BAD_FILE_NAMES} -gt 0 ] || 
    [ ${#SPACE_FILE_NAMES} -gt 0 ] || 
    [ ${#BAD_FOLDER_NAMES} -gt 0 ] ||
    [ ${#SPACE_FILE_NAMES} -gt 0 ]
then
    errors="ERRORS:\n;
FILES GREATER THAN 50MB:
${FILES_GREATER_THAN_50};
SYMBOLIC LINKS:
${SYMBOLIC_LINKS};
BAD FILE NAMES:
${BAD_FILE_NAMES};
FILES CONTAINING SPACES:
${SPACE_FILE_NAMES};
BAD FOLDER NAMES:
${BAD_FOLDER_NAMES};
FOLDER CONTAINING SPACES:
${SPACE_FOLDER_NAMES}"
    echo "$errors"
fi
