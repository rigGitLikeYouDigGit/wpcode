
# idem
Linking a 3d scene live across multiple DCCs at the same time

We take inspiration from Arras by DreamWorks https://github.com/dreamworksanimation/arras4_core to gather and propagate deltas ("impulses") to a scene, from any client to all other clients.

## the dream
- Building a character or viewing a scene, adjusting pose on a single frame in Maya while another window of Mantra renders it lit
- Sculpting face shapes in ZBrush or Blender while seeing how they blend in a live ROM in Maya
- Tweaking a cloth drape in Marvellous while passing it live through some texture generators in Houdini
- Adjusting proxy geo in Maya, passing it through some Houdini generators and bringing it live back into Maya (HEngine is expensive)
- Multiple people animating a single scene at the same time? Lunacy?

## the reality
- Currently works on a toy scale, syncing updates between camera matrices in Houdini and Maya.
- I set out to get the server/client architecture working, checking connections, flipping into active vs waiting states, and it does. Now it's all integration, working up the different kinds of scene data to send.
  - This led to more questions on how you save scene data, break it up etc.

## Why not just Arras?
- Skill issue. Couldn't compile it.
- Also Arras is pure C++ which obviously makes it more difficult to connect with python events

## Why not just Omniverse?
- Omniverse displays a single common USD file across all clients - this is great, if all you need to do is work on a single USD file
- Not everything can be easily represented in USD
- Maybe you don't want to hard-couple yourself to Nvidia?


# todo:
- get protobuffers / capnproto / some kind of shared memory working, so we don't have to go through sockets
- later balance deltas and absolute updates in case any of them don't make it over network