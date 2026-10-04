# NeuroFly: 3D simulation and training, voice-over script (EN)

Video: `data/videos/neurofly_world.mp4`, 99 s, 1920×1080. Read at a calm pace (about 2.5 words per second). Each block starts with its scene.

## 00:00 to 00:06  Flying On A Fly Brain

> This is our 3D world, and this drone is piloted by a real fruit fly brain.

On screen: *Our 3D world, piloted by a real fruit fly connectome*

## 00:06 to 00:16  A new world every flight

> Every flight happens in a new random world: hills, buildings, wind, and a dark red mast as the target. Nothing is scripted.

On screen: *Hills, buildings, wind, and a dark red mast as the target*

## 00:16 to 00:28  Step by step

> We trained the brain's readout step by step. First it learned to turn toward a target while hovering, then to do it precisely, trained on two GPUs at once.

On screen: *Each model takes over more of the flight*

## 00:28 to 00:42  Homing

> Next, it learned to fly across the world. A teacher pilot shows the way, then steps back. In the end, one hundred thirty-eight of one hundred fifty flights reached the target with no teacher at all.

On screen: *Flies across the world to the target*

## 00:42 to 00:52  Take the help away

> Then we took away the drone's own sensors. Height and speed had to come from the fly brain alone. At first, it got lost.

On screen: *No drone sensors: height and speed only from the fly brain*

## 00:52 to 01:04  Train again

> So we trained again, on two laptops in parallel, learning from every single flight. After one hundred sixty flights, it reached the target in five of six test worlds, with buildings in the way.

On screen: *Two laptops, two GPUs, learning from every flight*

## 01:04 to 01:20  Freefly

> Now it holds its height, steers past the buildings and finds the mast. GPS points the way from afar, the last metres come from what its eyes see and the fly's own wiring.

On screen: *Height, speed and the final approach from the fly brain*

## 01:20 to 01:33  Under the hood

> Under the hood, the fly brain itself is never trained. We only learn how to read it: a simple decoder watches single flight neurons and learns from a teacher, with every flight added to one shared dataset.

On screen: *The brain is never trained. We only learn how to read it.*

## 01:33 to 01:39  From A Fly'S Brain

> From a fly's brain to a flying drone. NeuroFly.

On screen: *to a flying drone.*

## Evolution of the algorithm (short)

1. **Average the wing muscle neurons**: could not tell left from right.
2. **Read 375 single flight neurons**: the target side shows up clearly.
3. **Imitate a teacher pilot**: DAgger: the teacher steps back over time.
4. **Learn from every flight**: one shared dataset, ridge regression, two GPUs.
5. **Remove the drone's sensors**: height and speed from the fly brain alone.
