# wpdex

### Make any python structure reactive, in-place, invisibly to user code.


## `wpdex.WpDex`
- Main metadata building block, handles pathing and access via a framework "above" the actual data object, so that we don't accidentally add keys to dicts, we can visit the framework via separate logic to how we would visit the actual object (if needed)

## `wpdex.WpDexProxy`
- we get real wild with it
- Extend the normal wplib.object.Proxy to automatically create and maintain a WpDex framework around the wrapped object, in place, AND provide hooks around original methods to track deltas and send signals via the `rx` reactive framework.
- `WpDex` handles auto-generation of UIs; `WpDexProxy` handles automatic updates to those UIs when state changes.

## `wpdex.Modelled`
- Starting a tool or program from scratch, you can use a Modelled object to represent its overall state with a single Python data structure.
- Guarantees WpDex has visibility from the root, and guarantees any changes to the state will be reflected in any affected UI views.

## `wpdex.ui.AtomicWindow`
- Fire-and-forget solution to display and edit any arbitrarily nested python data structure in Qt (provided that structure is wrapped in a WpDex).
- The ui side of it is quite crazy and needs refining, but fundamentally it's sound - we define a base `AtomicWidgetInterface` to define common logic for all UI widgets:
  - Receiving new value
  - Transforming it for display in UI
  - Value edited in UI
  - Transforming that back into proper data
- We then specialise it:
  - `AtomicCheckBox` is just a boolean view.
  - `AtomicCanvasScene` and `AtomicCanvasElement` are fully reactive versions of QGraphicsScenes, integrated with the `WpCanvas` classes from wpui. This lets us do reactive nodegraph views.
- Also includes `MetaResolver` used in anger to get Qt to play nice with the Adaptor classes

## about `rx`
- Having done quite a deep dive to integrate it here, I'm not too fond of it. Chaining functions from reactive values is extremely cool, but I haven't found myself using it that much in UI - at least, not more than a one-off lambda could accomplish.
- The indispensible use case would be if you needed to treat a static value and a reactive value agnostically through the same modifier function. I haven't run into that yet; you're always aware if what you have is reactive or not.


in the brief windows when this works, it's the coolest thing in the world