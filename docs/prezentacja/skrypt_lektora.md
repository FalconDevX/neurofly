# NeuroFly — demo voice-over script

Time: about 4 min (about 620 words of voice-over). Square brackets: what is on screen (not read aloud).
Numbers come from BANC v888 and our own measurements (`CLAUDE.md`, `docs/prezentacja/assets/data.json`).

---

## 1. Hook (0:00–0:20)

[Fly with the connectome inside, next to the X2 drone]

> A fruit fly has a brain smaller than a poppy seed. Yet it flies, dodges obstacles and finds its target.
> So we asked a simple question: what if a drone were steered by the real nervous system of a fly — not by a network that only pretends to be one?

## 2. The three main parts (0:20–0:45)

[Three panels side by side: fly eye / connectome / drone]

> NeuroFly has three main parts, just like a living fly: eyes, a brain and a body.
> The eyes are the drone's two cameras, seen through a model of the fly's eye.
> The brain is the fly's connectome.
> And the body is a quadcopter drone in a physics simulator.
> Each of us on the team built one of these parts, and the hand-overs between them.

## 3. What a connectome is (0:45–1:20)

[Explorer: the whole connectome rotating; then highlight optic lobes, central brain, neck, nerve cord]

> So what is a connectome? It is a complete wiring diagram of a nervous system: every neuron, and every connection between them.
> Ours is BANC, the Brain And Nerve Cord connectome of the fruit fly. Researchers cut the fly's nervous system into ultra-thin slices,
> imaged them with an electron microscope, and traced every neuron and every synapse — with AI and many hours of human proofreading.
> The result: one hundred seventy-five thousand neurons and over eighteen million synapses.
> It has a few main parts. The optic lobes, behind the eyes, process what the fly sees. The central brain combines it and decides.
> Descending neurons run through the neck and carry the brain's commands down. And the nerve cord — the fly's spinal cord —
> holds the motor neurons that drive the wing muscles, and receives signals from the halteres, the fly's built-in gyroscopes.

## 4. How we built it (1:20–1:55)

[Explorer in "Flight circuit" mode; then code / panels: FlyVis, BANC on GPU, MuJoCo drone]

> We took the official BANC release, version 888, exactly as published. We do not train its connections — not a single weight.
> Each connection becomes a weight: the number of synapses, with a plus or minus sign from the predicted neurotransmitter.
> On top of this wiring we run a simple model of neuron activity, on a GPU, fast enough to keep up with the cameras.
> Every group we use — visual neurons, descending flight neurons, haltere sensors, wing motor neurons —
> comes from BANC's own annotations. Nothing is guessed.
> On the eye side, a published model of fly vision, FlyVis, tells us how the fly's visual neurons react to the camera image.
> On the body side, the drone flies in the MuJoCo physics simulator.

## 5. The loop, step by step (1:55–2:30)

[Slide "How it works: one loop, 8 steps"; highlight each step as it is named]

> Now we close the loop. One camera frame goes around it like this.
> One: the two cameras each see a wide view, as wide as a fly's eye.
> Two: the image is cut into seven hundred twenty-one small hexagons — the facets of a fly's eye.
> Three: FlyVis computes how the fly's visual neurons respond.
> Four: that activity is fed into the matching neurons of the connectome — over twenty-two thousand of them.
> Five: the signal spreads through the whole fly brain and nerve cord.
> Six: we read the flight neurons — the brain's flight commands.
> Seven: a small decoder turns them into four numbers: power, tilt, pitch and turn.
> Eight: the drone moves, the cameras see a new image, and we are back at step one. One full lap takes about twenty milliseconds.

## 6. Where the "decision" is (2:30–2:55)

[Live brain panel: signal from vision to DNs to MNs, left / right neuron votes]

> Our first finding surprised us. If you average the wing motor neurons, they cannot tell you where the target is.
> The direction is carried by individual descending neurons — the ones running from the brain, through the neck, down to the body.
> From them we can read the side of the target ninety-nine percent of the time.
> The only thing we learn is a small linear readout: from those neurons to the drone's turn.

## 7. Demo: flight (2:55–3:35)

[Video `demo_planB_dn.mp4`: hover, target to the side, gust; BANC panel alongside]

> Let's see it fly. The drone hovers in place. A target appears to one side — only one eye can see it.
> The signal travels through the connectome, the descending neurons "vote" for a turn, and the drone turns toward the target.
> After training, the average heading error dropped from over twenty degrees to just a few.
> Now a gust of wind. The drone is knocked off course — and is back on heading in under four seconds.

## 8. Flight in the world (3:35–3:55)

[`sim.run_env --banc` window: flight to a mast with noisy sensors]

> Finally, flying to a target in an open world — with noisy sensors, wind and motor lag.
> Before training, the drone reached none of the targets. After three hundred episodes on two GPUs, it reaches five out of six.
> The direction is chosen by the connectome alone. The drone's sensors just keep it in the air — much like a fly's fast wing reflexes.

## 9. Honest limits and closing (3:55–4:15)

[Slide "From BANC / Our assumptions", then the logo]

> We say openly what is ours: the neuron dynamics model, the mapping from wings to motors, and the linear readout. And for now, this is a simulation.
> But the wiring is real. It all runs on a laptop, and it's open.
> NeuroFly: a real fly brain at the controls of a drone.

---

### Notes for the speaker

- Calm pace, pause after every number.
- Pronunciation: "BANC" /bæŋk/ ("bank"), "FlyVis" "fly-vis", "connectome" /kəˈnɛktoʊm/, "haltere" /ˈhɔːltɪər/.
- If the demo must be cut to 3 min: skip section 8, and in section 4 keep only the first two and the last two sentences.
