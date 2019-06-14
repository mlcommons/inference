#!/usr/bin/env bash
# Script to download Librispeech Dataset
# For best results allow downloader to use all dataset files.
# For the inference baseline, use the clean set.
#
# ARG		CHOICE				HELP
# ${1} = {all, clean, ...}		See the if statements for details.. selectively downloads data form librispeech

if [ "${1}" = "all" ] 
then
	echo "Downloading all..."
	python librispeech.py
	exit 0
fi
if [ "${1}" = "training" ]
then
	echo "Downloading all (but test-other and dev-other)..."
	python librispeech.py --files_to_use train-clean-100.tar.gz,train-clean-360.tar.gz,train-other-500.tar.gz,dev-clean.tar.gz,test-clean.tar.gz
	exit 0
fi
if [ "${1}" = "clean" ]
then
	echo "Downloading clean only..."
	python librispeech.py --files_to_use train-clean-100.tar.gz,train-clean-360.tar.gz,dev-clean.tar.gz,test-clean.tar.gz
	exit 0
fi
if [ "${1}" = "clean_dev" ]
then
	echo "Downloading clean dev only..."
	python librispeech.py --files_to_use train-clean-100.tar.gz,dev-clean.tar.gz,test-clean.tar.gz
	exit 0
fi
if [ "${1}" = "other" ]
then
	echo "Downloading noisy, aka other only..."
	python librispeech.py --files_to_use train-other-500.tar.gz,dev-other.tar.gz,test-other.tar.gz
	exit 0
fi
if [ "${1}" = "clean_test" ]
then
	echo "Downloading clean test only..."
	python librispeech.py --files_to_use test-clean.tar.gz
	exit 0
fi

