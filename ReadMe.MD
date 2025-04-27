TVSO (Temporal Variance Shadow Optimizer for URP)

Dynamic, intelligent shadow cascade optimization for Unity URP.

Features

GPU-powered shadow variance detection

Dynamic adjustment of URP cascade splits

Smarter cascade management based on scene activity

Lightweight compute shader + AsyncGPUReadback

Plug-and-Play: Just add it to your main camera

Debugging overlays and variance logging

Why Use TVSO?

In URP, shadow cascades are static: one-size-fits-all, even when your scene varies wildly.

TVSO fixes that.

TVSO detects dynamic areas and tightens cascades for sharper shadows

Detects static areas and relaxes cascades for better performance

No manual tweaking. No heavy scripting overhead.

Result:

Smoother framerate

Lower GPU load

Cleaner visuals when it matters most

Initial Real-World Test Results

Test

FPS in heavy city scenes
----------------------------------------------------------------------------

Before TVSO
29–37 FPS

Frame times
24–29ms

Stability
Wide swings

Garbage Collection impact
Noticeable spikes

----------------------------------------------------------------------------

After TVSO
30–45 FPS

Frame times
21–27ms

Stability
Smoother recovery

Garbage Collection impact
Softer recovery

TVSO uses an intelligent threshold (default: 10%) to trigger re-optimization only when needed.

----------------------------------------------------------------------------

Setup

Import the TemporalVarianceShadowOptimizerURP.cs script.

Assign the VarianceComputeURP_Atlas.compute shader.

Add the TemporalVarianceShadowOptimizerURP component to your Main Camera.

Configure sample count (default: 16) and frame dispatch settings (default: every 3 frames).

(Optional) Enable Debug Mode to see live variance logs.

Roadmap

Threshold auto-tuning and learning mode

Shadow bias and normal bias optimization

Realtime debug heatmap overlay

HDRP and Built-In Render Pipeline support (future)

'Aggressive Mode' for hard-tuning cascade saves


Credits

Created by:

David Alex Cappleman

Original System Design and Implementation: David Alex Cappleman

License

This project is licensed under the MIT License. Feel free to use, modify, and contribute!

MIT License
Copyright (c) 2025

TVSO isn't just about "better shadows." It's about smarter shadows.

Smoother. Faster. Sharper. Adaptive. Tasty. Shadows

Powered by variance, protected by Sunny.

