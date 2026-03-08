#!/usr/bin/env bash
set -euo pipefail

NS="pcapns"
VETH_HOST="veth0"

echo "Rimuovo solo le risorse create dallo script..."

if ip link show "$VETH_HOST" >/dev/null 2>&1; then
  sudo ip link delete "$VETH_HOST"
  echo "Interfaccia $VETH_HOST rimossa"
else
  echo "Interfaccia $VETH_HOST non presente"
fi

if ip netns list | grep -qw "$NS"; then
  sudo ip netns delete "$NS"
  echo "Namespace $NS rimosso"
else
  echo "Namespace $NS non presente"
fi

echo "Pulizia completata."