pip install -r requirements.txt
git_dir=$(git rev-parse --show-toplevel)
pip install $git_dir/loadgen