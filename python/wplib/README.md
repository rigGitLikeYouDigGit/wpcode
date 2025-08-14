# wplib
Main abstract python library

## wplib.coderef
- Super simple way of saving reference to a code type as a string, and then retrieving it. Useful for serialising objects and restoring them with the same type.
  - Todo: add some kind of hook for user to resolve any missing refs manually (eg if file definitions have moved around after file saved) 


## wplib.inheritance
- Lib functions for checking bases of types, registering and overriding logic against a common parent class (not necessarily a base class)
- `_MetaResolver` to have multiple-metaclass types "just work" (no refunds)

## wplib.serial
- Robust way to serialise any python structure to a human-readable string, and load it back identically into code. Handles custom types.
- No support yet for circular references, in practice hasn't been an issue.
  - Adaptors let each type handle its own regeneration, so you could easily flatten and restore references by custom logic if needed.

## wplib.codegen
- Rudimentary but effective ways to generate python code blocks and syntax in text - used in wpm for custom classes and definitions for maya nodes

## `wplib.object.Visitor`
- Fundamental object to a few higher functions like expressions, delta tracking, serialisation etc 
- `VisitAdaptor` classes used to register logic against arbitrary types, without having to extend them
- Invoking it is a bit verbose / over-engineered (see object.visitor.main for example):
  - `visitor = DeepVisitor()`
  - `params = VisitPassParams( **settings )`
  - `result = visitor.dispatchPass( targetDeepObject, params )`
  - which is fine for the heavier cases but sometimes you just want to check if a structure contains any expressions, and it's a bit too clunky
- Object itself should support:
  - direction: top-down or bottom-up
  - priority: depth-first or breadth-first
  - operation: iterate, apply in place, or copy/transform
    - I'm gonna be honest I tried to leetcode everything into one non-recursive function and it just wasn't on the cards, so instead I went full caveman and copied out each permutation as I needed it.
      - Hence function names like `_transformRecursiveTopDownDepthFirst()`,eat your heart out Byron

## `wplib.pathable.Pathable`
- Indexing into objects using string values and patterns.
- I found a way to make it obscenely complicated.
- This does work, I use it for trees, maya nodes, plugs, dex delta paths etc
  - but to be honest it's really gotten away from me in complexity


## `wplib.object.TypeNamespace`
 - Enums but better / subclasses that can act as enums.
 - I've found it rare that you need to pass an enum as an option value, without also needing specific logic relating to that enum option.
 - So this is a slight extension to help using type objects as enum-esque values, while also hanging methods on them.
 - Also enforces a namespace lookup on subclasses, eg `Sentinel.FailToFind` - forces you to always give some context on the meaning.
 - Maybe this is super bad practice? But it makes more sense to me.

## `wplib.object.Adaptor`
 - Type-specific logic for general operations, without modifying those types.
 - Essentially same effect as function overloading in C, using a type-specialised process to get similar results/logic 
 - It's become fundamental to how I split apart problems - probably some overuse, but it's always helpful to me to silo off type-specific processing from the overall logic of a system



## `wplib.object.Broadcaster`
- Attempt to generalise logic of numpy broadcasting
  - Expanding source values to destinations, slice destinations to fit sources
- Kind of works with maya plugs, but still in progress

## `wplib.object.EventDisptcher`
- Mixin to pass events to connected objects
  - I think formally this should be called messaging? Unclear on the difference

## `wplib.object.Proxy`
- A questionable idea
- Completely wrap a python object in a perfect invisible view, including proper type spoofing, with optional intercept hook to transform the object between the source and view's result.
- Used further in WpDex

## `wplib.object.Signal`
- Super simple way to attach external functions to a single trigger


## `wplib.object.validation`
- Richer way of checking multiple rules against an input, giving proper reports on problems, suggestions for fixing etc.



## `wplib.object.WpCallback`
- "WpCallback" to contrast explicitly with subclasses "MayaCallback" etc
- Helping to manage persistent callback-triggered funtions - pausing, removing, reusing the same hook on multiple functions etc.
