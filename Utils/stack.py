# -*- coding: utf-8 -*-
# @Time     : 2019/05/01 16:39
# @Author   : Devin Hua

class Stack:

    def __init__(self):
        self.stack = []

    # Use list append method to add element
    def add(self, dataval):
#         if dataval not in self.stack:
#             self.stack.append(dataval)
#             return True
#         else:
#             return False
        self.stack.append(dataval)

    # Use peek to look at the top of the stack
    def peek(self):
        return self.stack[-1]

    def pop(self):
        if len(self.stack) <= 0:
            print ("No element in the Stack")
            return -1
        else:
            return self.stack.pop()

if __name__ == "__main__":
    AStack = Stack()
    AStack.add("Mon")
    AStack.add("Tue")
    print(AStack.peek())
    AStack.add("Wed")
    AStack.add("Thu")
    print(AStack.peek())
    print(AStack.pop())
    print(AStack.pop())
    print(AStack.pop())
    print(AStack.pop())
    print(AStack.pop())