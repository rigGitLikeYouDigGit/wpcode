

from wplib.serial import SerialRegister

from wplib.coderef import CodeRef

from wplib.object.traversable import Traversable

from enum import Enum
# print(getCodeRef(Test))
# print(getCodeRef(Test.methodTest))
# print(getCodeRef(Test.InternalTest))
# print(getCodeRef(Test.InternalTest.internalMethod))
#
# print(Test.InternalTest.__qualname__)

# internalRef = getCodeRef(Test.InternalTest.internalMethod)
#
# resolved = resolveCodeRefSequence(internalRef)
# print(resolved)

internalRef = CodeRef.get(Test.InternalTest.internalMethod)
print(internalRef)
resolved = CodeRef.get(internalRef)
print(resolved)


