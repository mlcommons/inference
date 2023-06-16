
set -ex;
FIDDLE_HEAD_COMMIT="${FIDDLE_HEAD_COMMIT:=HEAD}"
JAX_HEAD_COMMIT="${JAX_HEAD_COMMIT:=HEAD}"
PAXML_HEAD_COMMIT="${PAXML_HEAD_COMMIT:=HEAD}"
PRAXIS_HEAD_COMMIT="${PRAXIS_HEAD_COMMIT:=HEAD}"
SEQIO_HEAD_COMMIT="${SEQIO_HEAD_COMMIT:=HEAD}"

CWD=$(pwd)
REQUIREMENTS_TXT="${CWD}/patch/requirements.txt"
rm -rf ${REQUIREMENTS_TXT}
wget https://raw.githubusercontent.com/${SAXML_GIT_USER}/${SAXML_GIT_REPO}/${SAXML_GIT_COMMIT}/requirements.txt -O ${REQUIREMENTS_TXT}

if [ $FIDDLE_HEAD_COMMIT == "HEAD" ];
then
  git clone https://github.com/google/fiddle
  cd ${CWD}/fiddle
  FIDDLE_HEAD_COMMIT=$(git rev-parse HEAD)
  cd ${CWD}
  rm -rf fiddle
fi

if [ $JAX_HEAD_COMMIT == "HEAD" ];
then
  git clone https://github.com/google/jax
  cd ${CWD}/jax
  JAX_HEAD_COMMIT=$(git rev-parse HEAD)
  cd ${CWD}
  rm -rf jax
fi

if [ $PAXML_HEAD_COMMIT == "HEAD" ];
then
  git clone https://github.com/google/paxml
  cd ${CWD}/paxml
  PAXML_HEAD_COMMIT=$(git rev-parse HEAD)
  cd ${CWD}
  rm -rf paxml
fi

if [ $PRAXIS_HEAD_COMMIT == "HEAD" ];
then
  git clone https://github.com/google/praxis
  cd ${CWD}/praxis
  PRAXIS_HEAD_COMMIT=$(git rev-parse HEAD)
  cd ${CWD}
  rm -rf praxis
fi

if [ $SEQIO_HEAD_COMMIT == "HEAD" ];
then
  git clone https://github.com/google/seqio
  cd ${CWD}/seqio
  SEQIO_HEAD_COMMIT=$(git rev-parse HEAD)
  cd ${CWD}
  rm -rf seqio
fi

echo ${FIDDLE_HEAD_COMMIT}
echo ${JAX_HEAD_COMMIT}
echo ${PAXML_HEAD_COMMIT}
echo ${PRAXIS_HEAD_COMMIT}
echo ${SEQIO_HEAD_COMMIT}


FIDDLE_GITHUB_HEAD="git+https://github.com/google/fiddle"
JAX_GITHUB_HEAD="git+https://github.com/google/jax"
PAXML_GITHUB_HEAD="git+https://github.com/google/paxml"
PRAXIS_GITHUB_HEAD="git+https://github.com/google/praxis"
SEQIO_GITHUB_HEAD="git+https://github.com/google/seqio"

sed -i 's#'${FIDDLE_GITHUB_HEAD}'#'${FIDDLE_GITHUB_HEAD}'@'${FIDDLE_HEAD_COMMIT}'#g' ${REQUIREMENTS_TXT};
sed -i 's#'${JAX_GITHUB_HEAD}'#'${JAX_GITHUB_HEAD}'@'${JAX_HEAD_COMMIT}'#g' ${REQUIREMENTS_TXT};
sed -i 's#'${PAXML_GITHUB_HEAD}'#'${PAXML_GITHUB_HEAD}'@'${PAXML_HEAD_COMMIT}'#g' ${REQUIREMENTS_TXT};
sed -i 's#'${PRAXIS_GITHUB_HEAD}'#'${PRAXIS_GITHUB_HEAD}'@'${PRAXIS_HEAD_COMMIT}'#g' ${REQUIREMENTS_TXT};
sed -i 's#'${SEQIO_GITHUB_HEAD}'#'${SEQIO_GITHUB_HEAD}'@'${SEQIO_HEAD_COMMIT}'#g' ${REQUIREMENTS_TXT};
