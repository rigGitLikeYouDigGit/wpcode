# wptree
Trees and tree-adjacent accessories

- Very useful structure in VFX - joints, ASTs, blend trees, node attributes, nested graphs etc
- Each item has **name**, **value** and **branches**.

## `interface.TreeInterface`
 - Mixin base class to let objects behave *like* trees - interpreting `name`, `value` and `branches` in however way they will. Makes no assumptions on how data is stored, only requires overriding methods to retrieve it

## `main.Tree`
 - Actual Tree object - this one can store its own data, and stand alone as a python data structure.

## ui
 - old tests for a tree editing Qt widget, this is superseded by WpDex

## todo:
- This was one of the earliest "good" ideas I worked on, but there are several older parts showing their naivety now - 
  - **Overrides, and looking up values on parent branches** - this stumped me for a long time (tree pun) - on surface it's a good idea, but then the question is how far up should you look? Should a branch point to a `root` that isn't the absolute root of the hierarchy, should there be some kind of breakpoint system to stop inherited lookups at a certain branch, etc.
    - If that ever comes back, I have the `wplib.object.OverrideProvider` mixin but still unsure of it 