
CWD=$(pwd)

mkdir -p ${CWD}/patch/saxml/server/pax/lm/params/
mkdir -p ${CWD}/patch/saxml/server/tf

FILE_PATH=saxml/server/tf/BUILD
FILE_PATH=saxml/server/pax/lm/params/BUILD
FILE_PATH=saxml/server/pax/lm/lm_tokenizer.py
FILE_PATH=saxml/server/pax/lm/servable_lm_model.py
FILE_PATH=saxml/server/pax/lm/params/lm_cloud.py
FILE_PATH=saxml/server/pax/lm/params/template.py

wget https://raw.githubusercontent.com/google/saxml/f134a5863c1f89c4354e7b6c6c2132594478f3d5/${FILE_PATH} -O ${CWD}/patch/${FILE_PATH}
