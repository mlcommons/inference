#================================================================
# HEADER
#================================================================
# DESCRIPTION
#    This is a script containing some commands to automatically 
#    format the c/c++ code contained in one folder. 
#    This will help to maintain good quality code in the github 
#    repository. 
# SET UP
#    You need to allow the correct permissions for this file.
#    This can be done by running:
#    chmod u+x <path to file>
# REQUIREMENTS
#    clang-format
#================================================================
# END_OF_HEADER
#================================================================


# Checks all the modified c/c++ files, format them and adds them
# to the commit.
for FILE in $(git diff upstream/$1 --name-only | grep -E '.*\.py$')
do
    autopep8 --in-place -a $FILE
    git add $FILE
done