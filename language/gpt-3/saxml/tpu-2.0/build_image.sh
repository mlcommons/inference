
set -e;
SAXML_VERSION_GIT_BRANCH="${SAXML_VERSION_GIT_BRANCH:=main}"
SAXML_BUILD_SOURCE="github"
SAXML_GIT_USER="google"
SAXML_GIT_REPO="saxml"
SAXML_GIT_COMMIT="f134a5863c1f89c4354e7b6c6c2132594478f3d5"


docker build \
  --pull \
  --target runtime-admin-server-image \
  -t ${SAX_ADMIN_SERVER_IMAGE_NAME} \
  -f Dockerfile . \
  --build-arg=SAXML_BUILD_SOURCE="${SAXML_BUILD_SOURCE}" \
  --build-arg=SAXML_VERSION_GIT_BRANCH="${SAXML_VERSION_GIT_BRANCH}" \
  --build-arg=SAXML_GIT_COMMIT="${SAXML_GIT_COMMIT}" ;


docker build \
  --pull \
  --target runtime-model-server-image \
  -t ${SAX_MODEL_SERVER_IMAGE_NAME} \
  -f Dockerfile . \
  --build-arg=SAXML_BUILD_SOURCE="${SAXML_BUILD_SOURCE}" \
  --build-arg=SAXML_VERSION_GIT_BRANCH="${SAXML_VERSION_GIT_BRANCH}" \
  --build-arg=SAXML_GIT_COMMIT="${SAXML_GIT_COMMIT}" ;

