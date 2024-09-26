# Git Hooks
## Description
This folder contains git hooks to help with the development and code maintenance of this repository. Each of the hooks is triggered by a specific git action.
## Setup
Set this folder as a custom hooks directory, so git will automatically apply the hooks in this folder. This can be done with git >= 2.9 with the command ```git config core.hooksPath tools/githooks```. Then set the hook file as an executable by running ```chmod u+x <path to file>```