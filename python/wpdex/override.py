from __future__ import annotations
import typing as T

"""override system

consider a syntax processing pass to make condition statements
easier to read?

("a").AND("b").OR("c").AND("d")
 vs
("a").AND(
	("b").OR("c").AND("d")
	)  ?

is there a way to do full python syntax parsing?

example - register custom widget to use for
a certain type of object

2 parts - matching a lookup to a set of overrides,
and actually applying each override

registerOverride(
	toMatch={
		"purpose" : "WidgetType").AND(
		"obj" : isinstance(obj, dict)
		,
	toApply={
		"widget" : DictWidget,
		
		}
	)

match="f["purpose"]==WidgetType and isinstance(f["obj"], dict)"

match=lambda f, obj: f["purpose"]==WidgetType and isinstance(f["obj"], dict)
match={"purpose" : "WidgetType", "objType" : dict}, matchCombine="any"
match=["path", "to", "obj"]		

"""



