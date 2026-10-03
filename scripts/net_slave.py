"""Test połączenia — strona SLAVE (klient, łączy się z masterem; komputer z symulatorem).

    python scripts/net_slave.py --host <IP mastera>
    python scripts/net_slave.py --host 100.77.85.7 -n 500 --fps 30

Wysyła pary klatek w rozmiarze oczu drona (2×512×450×3) i podaje czas obiegu oraz ocenę dla 30 FPS.
"""

import sys

from net_test import main

if __name__ == "__main__":
    sys.argv.insert(1, "client")
    main()
