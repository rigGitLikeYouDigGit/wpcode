# Chimaera

Chimaera is/was the python equivalent of a Strandbeest - a system of information and evaluation able to mutate and reflect any and every part of itself once set in motion.

I was frustrated that references and live links to other parts of programs are often an afterthought, and so there is always a tight limit on how intelligent its behaviour can be, and how that behaviour can be expressed.

In Chimaera each evaluation node is a tree of data, where every branch, and indeed the branches to be resolved, can be augmented and changed through expression logic. 

Data flowing in the graph also takes the form of a tree.

Nodes themselves can be collapsed to their trees, merged with the data stream, operated on as data, then expanded, in turn to operate on data.

I'm not interested in how useful this will or will not be. I'm interested in a system that can represent anything, and can represent the logic that governs anything.

## APEX

APEX in Houdini is a graph that builds and compiles an evaluation graph.
Chimaera is similar, but with any level of depth - the data given to a node might be raw 3D transform data, or it might be compacted node settings, which are edited as one, expanded to full nodes, and then in turn expanded - this happens inline with the evaluation of the overall graph.


## Use cases
- To be clear, the platonic ideal of Chimaera as set out above, is of little use outside of very narrow situations. If the final "compiled" form is a sequence of rigging operations in Maya, or pipeline steps to build a scene, 99% of the time we won't need it.
  - We would benefit in characters that have hundreds of repeated elements though (and by luck I've had to do a couple of those - octopus and Monstrous Nightmare).
- Ephemeral rigging is where it gets interesting - that is, dynamically generating a user interface to a character, depending on an instant situation. 
  - Hopefully you can see how in this case, where the current state may be very complex, and our output needs to be equally complex, a more flexible way of reasoning might be useful.

## Future
The goal isn't wrong, but I think this venture probably is. With the visitor/WpDex/Modelled system, I'm interested in embedding expressions in any arbitrary python structure, and then registering changes/overrides on that as deltas, over the base state. 

That way we can have a reference in a node's parametres, or an entire set of nodes can be referenced, saving only explicit changes to their parametres.

I dream of a single, holistic system, knowing every piece of data in an entire studio, able to draw from or edit it as needed.


## Can't you just build a normal rigging system
Sure I can, I did it in a weekend for work, but I didn't get here by being normal

