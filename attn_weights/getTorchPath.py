# This script enables to locate the torch installation directory
import os
import torch
import inspect
def f():
    p=os.path.dirname(inspect.getfile(torch))
    print(p)

f()
