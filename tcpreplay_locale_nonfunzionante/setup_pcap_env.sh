#!/usr/bin/env bash
set -euo pipefail

NS="pcapns"
VETH_HOST="veth0"
VETH_NS="veth1"
HOST_IP="10.10.10.1/24"
NS_IP="10.10.10.2/24"

if ip netns list | grep -qw "$NS"; then
  echo "Namespace $NS già esistente"
  exit 1
fi

if ip link show "$VETH_HOST" >/dev/null 2>&1; then
  echo "Interfaccia $VETH_HOST già esistente"
  exit 1
fi

echo "Creo namespace $NS e coppia $VETH_HOST <-> $VETH_NS"

sudo ip netns add "$NS"
sudo ip link add "$VETH_HOST" type veth peer name "$VETH_NS"
sudo ip link set "$VETH_NS" netns "$NS"

sudo ip addr add "$HOST_IP" dev "$VETH_HOST"
sudo ip link set "$VETH_HOST" up

sudo ip netns exec "$NS" ip addr add "$NS_IP" dev "$VETH_NS"
sudo ip netns exec "$NS" ip link set "$VETH_NS" up
sudo ip netns exec "$NS" ip link set lo up

echo
echo "Ambiente creato."
echo "Host:      $VETH_HOST -> 10.10.10.1"
echo "Namespace: $NS / $VETH_NS -> 10.10.10.2"
echo
echo "Per avviare la tua app Python:"
echo "sudo ip netns exec $NS python3 tua_app.py"
echo
echo "Per fare replay:"
echo "sudo tcpreplay -i $VETH_HOST file.pcap"