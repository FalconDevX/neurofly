# NeuroFly

Hackathon: zamknięta pętla wzrok → connectome Drosophila → sterowanie → symulowany dron → nowy obraz.

| Osoba | Zakres |
|---|---|
| 1 | Kamera RGB → FlyGym Retina → RetinaMapper → FlyVis → neurony BANC |
| 2 | BANC/VNC → motoneurony skrzydeł → thrust / roll / pitch / yaw (`banc_control/`) |
| 3 | Gazebo Sim, quadcopter, kamera, IMU, epizody treningowe |

- Plan Osoby 2: [docs/osoba2-plan.md](docs/osoba2-plan.md)
- Testy: `pip install -e .[dev] && pytest`
