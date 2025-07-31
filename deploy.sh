#!/bin/bash
source .env
source ./scripts/utils.sh

sshpass -p "$REMOTE_PW" rsync -avz --progress -e \
    "ssh -p $REMOTE_PORT" "./bin" "$REMOTE_USER@$REMOTE_HOST:$REMOTE_DIR/"
sshpass -p "$REMOTE_PW" rsync -avz --progress -e \
    "ssh -p $REMOTE_PORT" "./lib" "$REMOTE_USER@$REMOTE_HOST:$REMOTE_DIR/"