
set -ex;

lsblk;

# Format the attached persistent disk now
mkfs.ext4 -m 0 -E lazy_itable_init=0,lazy_journal_init=0,discard /dev/${PERSISTENT_SSD_DISK};
# Create a directory to mount the persistent disk
mkdir -p /mnt/disks/persist;
# Mount the file system to the directory
mount -o discard,defaults /dev/${PERSISTENT_SSD_DISK} /mnt/disks/persist;

lsblk;

# Change Docker root directory /var/lib/docker to ssd
systemctl stop docker.service;
systemctl stop docker.socket;
sed -i 's+ExecStart=/usr/bin/dockerd -H fd://+ExecStart=/usr/bin/dockerd --data-root /mnt/disks/persist -H fd://+g' /lib/systemd/system/docker.service;
rsync -aqxP /var/lib/docker/ /mnt/disks/persist;
systemctl daemon-reload;
systemctl start docker;

ps aux | grep -i docker | grep -v grep;