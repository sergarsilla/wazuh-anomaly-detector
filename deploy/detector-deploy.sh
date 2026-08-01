#!/bin/sh
# Privileged deployment hook: takes no arguments and makes no decisions, so the
# CI account can be granted this one fixed command via sudo instead of docker
# group membership (root-equivalent on this host).
#
# Install manually as root; the copy in the source tree is inert until then:
#   install -o root -g root -m 755 deploy/detector-deploy.sh \
#           /usr/local/sbin/deploy-wazuh-anomaly-detector
#
# Grant (/etc/sudoers.d/jenkins-deploy, mode 0440):
#   deploy ALL=(root) NOPASSWD: /usr/local/sbin/deploy-wazuh-anomaly-detector
#
# Training is not part of this hook: weights, datasets and the calibrated
# threshold are excluded from the CI sync and only change on a manual run.
set -eu

APP_DIR=/opt/wazuh-anomaly-detector

cd "$APP_DIR"

docker compose up -d --build detector
