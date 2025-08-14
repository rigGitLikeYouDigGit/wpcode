# wpexp
older tests on evaluating, parsing syntax and expressions

Goal is to have a single "toolbox" object to represent a set of rules and functions available in expressions.

## example:
In a maya tool, you want to list all controls on the character except those on the arms (maybe some funky 1st person viewmodel stuff).

Wouldn't it be nice to have a way to understand string matching to nodes, in this context? To write:

`" character_:*_CTL - character_:*Arm_CTL "`

This package helps you accomplish this - by overriding different stages of syntax and evaluation, we can pack in checks for those characters, and effects when we find them in a parsed AST Name object.

## wpexp.ExpEvaluator
Main object, holds lists of passes to run over raw string, over parsed AST etc.

## wpexp.syntax.SyntaxPasses
TypeNamespace for custom syntax transformations


## wpexp.syntax.SyntaxPasses.NameToStrPass
Special mention for by far the most useful one:

converts a literal dict definition of 

`" { myVar : myValue } "`
into 

`" { 'myVar' : 'myValue' } "`

so that Python can eval it without having a fit (tunable with a whitelist of valid var names)

## wpexp.new
- First tests for a more advanced system of expressions and references, including "compiling" functions, chaining values etc.
- As usual, too complicated, needs reconsidering in the context of other wp code

# Unknown:
 - How best to pass an expression as a contained object into an evaluation context. Currently we have an Evaluator object, and that's what you subclass for different evaluation logic based on the syntax