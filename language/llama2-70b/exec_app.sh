MLCOMMONS_REPO_PATH="$(dirname "$(dirname "$PWD")")"

# Add any volume mounts here with the following syntax
# /path/to/src:/path/to/dir/in/container
MOUNTS=(
    $MLCOMMONS_REPO_PATH:$MLCOMMONS_REPO_PATH,
    /share:/share,
    /usr/bin/srun:/usr/bin/srun,
    /usr/bin/sinfo:/usr/bin/sinfo,
    /share/software/spack/opt/spack/linux-rocky8-zen/gcc-8.5.0/slurm-23-11-1-1-yh4vs4sr7xks2nbzffs2hdwe7pqfovsg:/opt/slurm,
    /var/spool/slurm/d/conf-cache:/var/spool/slurm/d/conf-cache
)

CI_BUILD_USER=$(id -u -n)
CI_BUILD_UID=$(id -u)
CI_BUILD_GROUP=$(id -g -n)
CI_BUILD_GID=$(id -g)

# Build mount flags
declare -a MOUNT_FLAGS
for _mount in ${MOUNTS[@]}; do
    _split=($(echo $_mount | tr ':' '\n'))
    MOUNT_FLAGS+=("--bind" "${_split[0]}:${_split[1]}")
done

set -x
apptainer exec --nv --ipc --writable-tmpfs \
  --pwd $PWD \
  "${MOUNT_FLAGS[@]}" \
  --env CI_BUILD_USER=$CI_BUILD_USER \
  --env CI_BUILD_UID=$CI_BUILD_UID \
  --env CI_BUILD_GROUP=$CI_BUILD_GROUP \
  --env CI_BUILD_GID=$CI_BUILD_GID \
  llm_gpubringup.sif \
  bash ./with_the_same_user
