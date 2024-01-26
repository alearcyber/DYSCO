"""
This file only serves the purpose of making it easier to allow the user to select from a series of different test from
the same Python script.
This primary routine will take a dialogue tree as input and walk the user through it. The root node is a list,
the intermediate nodes are also lists, and the leaf nodes are strings
"""
from typing import Callable



#################################################################################################
# These functions and the list below are an example input data structure.
#################################################################################################
def example1():
    print("example 1 called...")

def example2():
    print("example 2 called...")

def example3():
    print("This is the first option nested within the third option of the root")

def example4():
    print("This is the second option nested within the third option of the root")

example_dtree = [
    ("example1", example1),
    ("example2", example2),
    ("more", [
        ("example3", example3),
        ("example4", example4)
    ])

]


def ask(dtree):
    if isinstance(dtree[1], Callable):
        print("Executing " + dtree[0])
        dtree[1]()
    else:
        prompt = "Make A Selection..."
        for i in range(len(dtree)):
            prompt += f'\n\t{i + 1}) {dtree[i][0]}.'
        choice = int(input(prompt + '\nInput Selection:'))
        ask(dtree[choice - 1][1])







if __name__ == "__main__":
    ask(example_dtree)
