"""Test połączenia — strona MASTER (serwer, nasłuchuje; komputer z GPU / serwerem BANC).

    python scripts/net_master.py            # tcp://*:5555
    python scripts/net_master.py --port 6000

Wymaga reguły zapory na TCP 5555 (PowerShell jako administrator):
    New-NetFirewallRule -DisplayName "NeuroFly ZMQ" -Direction Inbound -Protocol TCP -LocalPort 5555 -Action Allow
Potem na drugim komputerze: python scripts/net_slave.py --host <IP mastera>
"""

import sys

from net_test import main

if __name__ == "__main__":
    sys.argv.insert(1, "server")
    main()
