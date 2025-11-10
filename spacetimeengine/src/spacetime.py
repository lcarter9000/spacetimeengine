#!/usr/bin/env python
from sympy import *
from spacetimeengine.src.solutions import Solution  # Adjust the import path as needed
import matplotlib.pyplot as plt
import numpy as np
import os
from mpl_toolkits.mplot3d import Axes3D

import sys
import pdb

"""
# ┌────────┬────────────────────────────────────────────────────────────┐
# │ Command│ Description                                                │
# ├────────┼────────────────────────────────────────────────────────────┤
# │ c      │ Continue execution until next breakpoint or trace call     │
# │ n      │ Step to the next line in the current function              │
# │ s      │ Step into the next function call                           │
# │ r      │ Continue until the current function returns                │
# │ q      │ Quit the debugger and stop execution                       │
# │ l      │ List source code around the current line                   │
# │ p var  │ Print the value of variable `var`                          │
# │ b line │ Set breakpoint at line number `line`                       │
# │ b func │ Set breakpoint at function `func`                          │
# │ cl     │ Clear all breakpoints                                      │
# │ h      │ Show help on commands                                      │
# └────────┴────────────────────────────────────────────────────────────┘
"""
def trace_calls(frame, event, arg):
    if event == 'call':
        code = frame.f_code
        func_name = code.co_name
        filename = code.co_filename
        lineno = frame.f_lineno
        print(f"\n🔍 Pausing at function: {func_name} ({filename}:{lineno})")
        pdb.set_trace()  # Pause execution here
    return trace_calls  # Continue tracing deeper calls


class SpaceTime:

    # Run at object creation.
    def __init__(self, solution, suppress_printing = False):
            
        # Initializes coordinate set class object.
        self.coordinate_set = solution[1]
        # Integer amount of dimensions associated with metric solution.
        self.dimension_count = len(self.coordinate_set)
        # Simple array for counting through tensor indices.
        self.dimensions = range(len(self.coordinate_set))
        # Upon a SpaceTime object creation, the user may choose to print the terms as they are computed.
        self.suppress_printing = suppress_printing
        
        # Sets the metric tensor and its inverse.
        self.metric_index_config = solution[2]
        if (self.metric_index_config == "uu"):
            self.metric_tensor_uu = solution[0]
            self.metric_tensor_dd = simplify(solution[0].inv())
        elif(self.metric_index_config == "dd"):
            self.metric_tensor_dd = solution[0]
            self.metric_tensor_uu = simplify(solution[0].inv())
        else:
            print("Invalid index_config string.")
        
        # Declares ( gravitational field ) connection class object.
        self.christoffel_symbols_udd = Matrix([
                                                 [
                                                     [ 0, 0, 0, 0 ], 
                                                     [ 0, 0, 0, 0 ], 
                                                     [ 0, 0, 0, 0 ], 
                                                     [ 0, 0, 0, 0 ]
                                                 ],
                                                 [
                                                     [ 0, 0, 0, 0 ], 
                                                     [ 0, 0, 0, 0 ], 
                                                     [ 0, 0, 0, 0 ], 
                                                     [ 0, 0, 0, 0 ]
                                                 ],
                                                 [
                                                     [ 0, 0, 0, 0 ], 
                                                     [ 0, 0, 0, 0 ], 
                                                     [ 0, 0, 0, 0 ], 
                                                     [ 0, 0, 0, 0 ]
                                                 ],
                                                 [
                                                     [ 0, 0, 0, 0 ], 
                                                     [ 0, 0, 0, 0 ], 
                                                     [ 0, 0, 0, 0 ], 
                                                     [ 0, 0, 0, 0 ]
                                                 ]
                                             ])
        # Declares the Christoffel symbols of the first kind class object.
        self.christoffel_symbols_ddd = Matrix([
                                                 [
                                                     [ 0, 0, 0, 0 ], 
                                                     [ 0, 0, 0, 0 ], 
                                                     [ 0, 0, 0, 0 ], 
                                                     [ 0, 0, 0, 0 ]
                                                 ],
                                                 [
                                                     [ 0, 0, 0, 0 ], 
                                                     [ 0, 0, 0, 0 ], 
                                                     [ 0, 0, 0, 0 ], 
                                                     [ 0, 0, 0, 0 ]
                                                 ],
                                                 [
                                                     [ 0, 0, 0, 0 ], 
                                                     [ 0, 0, 0, 0 ], 
                                                     [ 0, 0, 0, 0 ], 
                                                     [ 0, 0, 0, 0 ]
                                                 ],
                                                 [
                                                     [ 0, 0, 0, 0 ], 
                                                     [ 0, 0, 0, 0 ], 
                                                     [ 0, 0, 0, 0 ], 
                                                     [ 0, 0, 0, 0 ]
                                                 ]
                                             ])
        
        # Declares Riemann curvature tensor class object.
        self.riemann_tensor_uddd = Matrix([    
                                               [    
                                                    [
                                                        [ 0, 0, 0, 0 ], 
                                                        [ 0, 0, 0, 0 ], 
                                                        [ 0, 0, 0, 0 ], 
                                                        [ 0, 0, 0, 0 ]
                                                    ],
                                                    [
                                                        [ 0, 0, 0, 0 ], 
                                                        [ 0, 0, 0, 0 ], 
                                                        [ 0, 0, 0, 0 ], 
                                                        [ 0, 0, 0, 0 ]
                                                    ],
                                                    [
                                                        [ 0, 0, 0, 0 ], 
                                                        [ 0, 0, 0, 0 ], 
                                                        [ 0, 0, 0, 0 ], 
                                                        [ 0, 0, 0, 0 ]
                                                    ],
                                                    [
                                                        [ 0, 0, 0, 0 ], 
                                                        [ 0, 0, 0, 0 ], 
                                                        [ 0, 0, 0, 0 ], 
                                                        [ 0, 0, 0, 0 ]
                                                    ]
                                               ],
                                               [    
                                                    [
                                                        [ 0, 0, 0, 0 ], 
                                                        [ 0, 0, 0, 0 ], 
                                                        [ 0, 0, 0, 0 ], 
                                                        [ 0, 0, 0, 0 ]
                                                    ],
                                                    [
                                                        [ 0, 0, 0, 0 ], 
                                                        [ 0, 0, 0, 0 ], 
                                                        [ 0, 0, 0, 0 ], 
                                                        [ 0, 0, 0, 0 ]
                                                    ],
                                                    [
                                                        [ 0, 0, 0, 0 ], 
                                                        [ 0, 0, 0, 0 ], 
                                                        [ 0, 0, 0, 0 ], 
                                                        [ 0, 0, 0, 0 ]
                                                    ],
                                                    [
                                                        [ 0, 0, 0, 0 ], 
                                                        [ 0, 0, 0, 0 ], 
                                                        [ 0, 0, 0, 0 ], 
                                                        [ 0, 0, 0, 0 ]
                                                    ]
                                               ],
                                               [    
                                                    [
                                                        [ 0, 0, 0, 0 ], 
                                                        [ 0, 0, 0, 0 ], 
                                                        [ 0, 0, 0, 0 ], 
                                                        [ 0, 0, 0, 0 ]
                                                    ],
                                                    [
                                                        [ 0, 0, 0, 0 ], 
                                                        [ 0, 0, 0, 0 ], 
                                                        [ 0, 0, 0, 0 ], 
                                                        [ 0, 0, 0, 0 ]
                                                    ],
                                                    [
                                                        [ 0, 0, 0, 0 ], 
                                                        [ 0, 0, 0, 0 ], 
                                                        [ 0, 0, 0, 0 ], 
                                                        [ 0, 0, 0, 0 ]
                                                    ],
                                                    [
                                                        [ 0, 0, 0, 0 ], 
                                                        [ 0, 0, 0, 0 ], 
                                                        [ 0, 0, 0, 0 ], 
                                                        [ 0, 0, 0, 0 ]
                                                    ]
                                               ],
                                               [    
                                                    [
                                                        [ 0, 0, 0, 0 ], 
                                                        [ 0, 0, 0, 0 ], 
                                                        [ 0, 0, 0, 0 ], 
                                                        [ 0, 0, 0, 0 ]
                                                    ],
                                                    [
                                                        [ 0, 0, 0, 0 ], 
                                                        [ 0, 0, 0, 0 ], 
                                                        [ 0, 0, 0, 0 ], 
                                                        [ 0, 0, 0, 0 ]
                                                    ],
                                                    [
                                                        [ 0, 0, 0, 0 ], 
                                                        [ 0, 0, 0, 0 ], 
                                                        [ 0, 0, 0, 0 ], 
                                                        [ 0, 0, 0, 0 ]
                                                    ],
                                                    [
                                                        [ 0, 0, 0, 0 ], 
                                                        [ 0, 0, 0, 0 ], 
                                                        [ 0, 0, 0, 0 ], 
                                                        [ 0, 0, 0, 0 ]
                                                    ]
                                               ]    
                                           ])  

        # Declares Riemann curvature tensor "dddd" type class object.
        self.riemann_tensor_dddd = Matrix([    
                                                   [    
                                                        [
                                                            [ 0, 0, 0, 0 ], 
                                                            [ 0, 0, 0, 0 ], 
                                                            [ 0, 0, 0, 0 ], 
                                                            [ 0, 0, 0, 0 ]
                                                        ],
                                                        [
                                                            [ 0, 0, 0, 0 ], 
                                                            [ 0, 0, 0, 0 ], 
                                                            [ 0, 0, 0, 0 ], 
                                                            [ 0, 0, 0, 0 ]
                                                        ],
                                                        [
                                                            [ 0, 0, 0, 0 ], 
                                                            [ 0, 0, 0, 0 ], 
                                                            [ 0, 0, 0, 0 ], 
                                                            [ 0, 0, 0, 0 ]
                                                        ],
                                                        [
                                                            [ 0, 0, 0, 0 ], 
                                                            [ 0, 0, 0, 0 ], 
                                                            [ 0, 0, 0, 0 ], 
                                                            [ 0, 0, 0, 0 ]
                                                        ]
                                                   ],
                                                   [    
                                                        [
                                                            [ 0, 0, 0, 0 ], 
                                                            [ 0, 0, 0, 0 ], 
                                                            [ 0, 0, 0, 0 ], 
                                                            [ 0, 0, 0, 0 ]
                                                        ],
                                                        [
                                                            [ 0, 0, 0, 0 ], 
                                                            [ 0, 0, 0, 0 ], 
                                                            [ 0, 0, 0, 0 ], 
                                                            [ 0, 0, 0, 0 ]
                                                        ],
                                                        [
                                                            [ 0, 0, 0, 0 ], 
                                                            [ 0, 0, 0, 0 ], 
                                                            [ 0, 0, 0, 0 ], 
                                                            [ 0, 0, 0, 0 ]
                                                        ],
                                                        [
                                                            [ 0, 0, 0, 0 ], 
                                                            [ 0, 0, 0, 0 ], 
                                                            [ 0, 0, 0, 0 ], 
                                                            [ 0, 0, 0, 0 ]
                                                        ]
                                                   ],
                                                   [    
                                                        [
                                                            [ 0, 0, 0, 0 ], 
                                                            [ 0, 0, 0, 0 ], 
                                                            [ 0, 0, 0, 0 ], 
                                                            [ 0, 0, 0, 0 ]
                                                        ],
                                                        [
                                                            [ 0, 0, 0, 0 ], 
                                                            [ 0, 0, 0, 0 ], 
                                                            [ 0, 0, 0, 0 ], 
                                                            [ 0, 0, 0, 0 ]
                                                        ],
                                                        [
                                                            [ 0, 0, 0, 0 ], 
                                                            [ 0, 0, 0, 0 ], 
                                                            [ 0, 0, 0, 0 ], 
                                                            [ 0, 0, 0, 0 ]
                                                        ],
                                                        [
                                                            [ 0, 0, 0, 0 ], 
                                                            [ 0, 0, 0, 0 ], 
                                                            [ 0, 0, 0, 0 ], 
                                                            [ 0, 0, 0, 0 ]
                                                        ]
                                                   ],
                                                   [    
                                                        [
                                                            [ 0, 0, 0, 0 ], 
                                                            [ 0, 0, 0, 0 ], 
                                                            [ 0, 0, 0, 0 ], 
                                                            [ 0, 0, 0, 0 ]
                                                        ],
                                                        [
                                                            [ 0, 0, 0, 0 ], 
                                                            [ 0, 0, 0, 0 ], 
                                                            [ 0, 0, 0, 0 ], 
                                                            [ 0, 0, 0, 0 ]
                                                        ],
                                                        [
                                                            [ 0, 0, 0, 0 ], 
                                                            [ 0, 0, 0, 0 ], 
                                                            [ 0, 0, 0, 0 ], 
                                                            [ 0, 0, 0, 0 ]
                                                        ],
                                                        [
                                                            [ 0, 0, 0, 0 ], 
                                                            [ 0, 0, 0, 0 ], 
                                                            [ 0, 0, 0, 0 ], 
                                                            [ 0, 0, 0, 0 ]
                                                        ]
                                                   ]    
                                               ])  

        # Declares Weyl curvature tensor "dddd" type class object.
        self.weyl_tensor_dddd = Matrix([    
                                                   [    
                                                        [
                                                            [ 0, 0, 0, 0 ], 
                                                            [ 0, 0, 0, 0 ], 
                                                            [ 0, 0, 0, 0 ], 
                                                            [ 0, 0, 0, 0 ]
                                                        ],
                                                        [
                                                            [ 0, 0, 0, 0 ], 
                                                            [ 0, 0, 0, 0 ], 
                                                            [ 0, 0, 0, 0 ], 
                                                            [ 0, 0, 0, 0 ]
                                                        ],
                                                        [
                                                            [ 0, 0, 0, 0 ], 
                                                            [ 0, 0, 0, 0 ], 
                                                            [ 0, 0, 0, 0 ], 
                                                            [ 0, 0, 0, 0 ]
                                                        ],
                                                        [
                                                            [ 0, 0, 0, 0 ], 
                                                            [ 0, 0, 0, 0 ], 
                                                            [ 0, 0, 0, 0 ], 
                                                            [ 0, 0, 0, 0 ]
                                                        ]
                                                   ],
                                                   [    
                                                        [
                                                            [ 0, 0, 0, 0 ], 
                                                            [ 0, 0, 0, 0 ], 
                                                            [ 0, 0, 0, 0 ], 
                                                            [ 0, 0, 0, 0 ]
                                                        ],
                                                        [
                                                            [ 0, 0, 0, 0 ], 
                                                            [ 0, 0, 0, 0 ], 
                                                            [ 0, 0, 0, 0 ], 
                                                            [ 0, 0, 0, 0 ]
                                                        ],
                                                        [
                                                            [ 0, 0, 0, 0 ], 
                                                            [ 0, 0, 0, 0 ], 
                                                            [ 0, 0, 0, 0 ], 
                                                            [ 0, 0, 0, 0 ]
                                                        ],
                                                        [
                                                            [ 0, 0, 0, 0 ], 
                                                            [ 0, 0, 0, 0 ], 
                                                            [ 0, 0, 0, 0 ], 
                                                            [ 0, 0, 0, 0 ]
                                                        ]
                                                   ],
                                                   [    
                                                        [
                                                            [ 0, 0, 0, 0 ], 
                                                            [ 0, 0, 0, 0 ], 
                                                            [ 0, 0, 0, 0 ], 
                                                            [ 0, 0, 0, 0 ]
                                                        ],
                                                        [
                                                            [ 0, 0, 0, 0 ], 
                                                            [ 0, 0, 0, 0 ], 
                                                            [ 0, 0, 0, 0 ], 
                                                            [ 0, 0, 0, 0 ]
                                                        ],
                                                        [
                                                            [ 0, 0, 0, 0 ], 
                                                            [ 0, 0, 0, 0 ], 
                                                            [ 0, 0, 0, 0 ], 
                                                            [ 0, 0, 0, 0 ]
                                                        ],
                                                        [
                                                            [ 0, 0, 0, 0 ], 
                                                            [ 0, 0, 0, 0 ], 
                                                            [ 0, 0, 0, 0 ], 
                                                            [ 0, 0, 0, 0 ]
                                                        ]
                                                   ],
                                                   [    
                                                        [
                                                            [ 0, 0, 0, 0 ], 
                                                            [ 0, 0, 0, 0 ], 
                                                            [ 0, 0, 0, 0 ], 
                                                            [ 0, 0, 0, 0 ]
                                                        ],
                                                        [
                                                            [ 0, 0, 0, 0 ], 
                                                            [ 0, 0, 0, 0 ], 
                                                            [ 0, 0, 0, 0 ], 
                                                            [ 0, 0, 0, 0 ]
                                                        ],
                                                        [
                                                            [ 0, 0, 0, 0 ], 
                                                            [ 0, 0, 0, 0 ], 
                                                            [ 0, 0, 0, 0 ], 
                                                            [ 0, 0, 0, 0 ]
                                                        ],
                                                        [
                                                            [ 0, 0, 0, 0 ], 
                                                            [ 0, 0, 0, 0 ], 
                                                            [ 0, 0, 0, 0 ], 
                                                            [ 0, 0, 0, 0 ]
                                                        ]
                                                   ]    
                                               ])
        
        # Declares Riemann curvature tensor "uddd" type class object.
        self.weyl_tensor_uddd = Matrix([    
                                                   [    
                                                        [
                                                            [ 0, 0, 0, 0 ], 
                                                            [ 0, 0, 0, 0 ], 
                                                            [ 0, 0, 0, 0 ], 
                                                            [ 0, 0, 0, 0 ]
                                                        ],
                                                        [
                                                            [ 0, 0, 0, 0 ], 
                                                            [ 0, 0, 0, 0 ], 
                                                            [ 0, 0, 0, 0 ], 
                                                            [ 0, 0, 0, 0 ]
                                                        ],
                                                        [
                                                            [ 0, 0, 0, 0 ], 
                                                            [ 0, 0, 0, 0 ], 
                                                            [ 0, 0, 0, 0 ], 
                                                            [ 0, 0, 0, 0 ]
                                                        ],
                                                        [
                                                            [ 0, 0, 0, 0 ], 
                                                            [ 0, 0, 0, 0 ], 
                                                            [ 0, 0, 0, 0 ], 
                                                            [ 0, 0, 0, 0 ]
                                                        ]
                                                   ],
                                                   [    
                                                        [
                                                            [ 0, 0, 0, 0 ], 
                                                            [ 0, 0, 0, 0 ], 
                                                            [ 0, 0, 0, 0 ], 
                                                            [ 0, 0, 0, 0 ]
                                                        ],
                                                        [
                                                            [ 0, 0, 0, 0 ], 
                                                            [ 0, 0, 0, 0 ], 
                                                            [ 0, 0, 0, 0 ], 
                                                            [ 0, 0, 0, 0 ]
                                                        ],
                                                        [
                                                            [ 0, 0, 0, 0 ], 
                                                            [ 0, 0, 0, 0 ], 
                                                            [ 0, 0, 0, 0 ], 
                                                            [ 0, 0, 0, 0 ]
                                                        ],
                                                        [
                                                            [ 0, 0, 0, 0 ], 
                                                            [ 0, 0, 0, 0 ], 
                                                            [ 0, 0, 0, 0 ], 
                                                            [ 0, 0, 0, 0 ]
                                                        ]
                                                   ],
                                                   [    
                                                        [
                                                            [ 0, 0, 0, 0 ], 
                                                            [ 0, 0, 0, 0 ], 
                                                            [ 0, 0, 0, 0 ], 
                                                            [ 0, 0, 0, 0 ]
                                                        ],
                                                        [
                                                            [ 0, 0, 0, 0 ], 
                                                            [ 0, 0, 0, 0 ], 
                                                            [ 0, 0, 0, 0 ], 
                                                            [ 0, 0, 0, 0 ]
                                                        ],
                                                        [
                                                            [ 0, 0, 0, 0 ], 
                                                            [ 0, 0, 0, 0 ], 
                                                            [ 0, 0, 0, 0 ], 
                                                            [ 0, 0, 0, 0 ]
                                                        ],
                                                        [
                                                            [ 0, 0, 0, 0 ], 
                                                            [ 0, 0, 0, 0 ], 
                                                            [ 0, 0, 0, 0 ], 
                                                            [ 0, 0, 0, 0 ]
                                                        ]
                                                   ],
                                                   [    
                                                        [
                                                            [ 0, 0, 0, 0 ], 
                                                            [ 0, 0, 0, 0 ], 
                                                            [ 0, 0, 0, 0 ], 
                                                            [ 0, 0, 0, 0 ]
                                                        ],
                                                        [
                                                            [ 0, 0, 0, 0 ], 
                                                            [ 0, 0, 0, 0 ], 
                                                            [ 0, 0, 0, 0 ], 
                                                            [ 0, 0, 0, 0 ]
                                                        ],
                                                        [
                                                            [ 0, 0, 0, 0 ], 
                                                            [ 0, 0, 0, 0 ], 
                                                            [ 0, 0, 0, 0 ], 
                                                            [ 0, 0, 0, 0 ]
                                                        ],
                                                        [
                                                            [ 0, 0, 0, 0 ], 
                                                            [ 0, 0, 0, 0 ], 
                                                            [ 0, 0, 0, 0 ], 
                                                            [ 0, 0, 0, 0 ]
                                                        ]
                                                   ]    
                                               ])  
        
        # Declares Riemann curvature tensor "dduu" type class object.        
        self.weyl_tensor_dduu = Matrix([    
                                                   [    
                                                        [
                                                            [ 0, 0, 0, 0 ], 
                                                            [ 0, 0, 0, 0 ], 
                                                            [ 0, 0, 0, 0 ], 
                                                            [ 0, 0, 0, 0 ]
                                                        ],
                                                        [
                                                            [ 0, 0, 0, 0 ], 
                                                            [ 0, 0, 0, 0 ], 
                                                            [ 0, 0, 0, 0 ], 
                                                            [ 0, 0, 0, 0 ]
                                                        ],
                                                        [
                                                            [ 0, 0, 0, 0 ], 
                                                            [ 0, 0, 0, 0 ], 
                                                            [ 0, 0, 0, 0 ], 
                                                            [ 0, 0, 0, 0 ]
                                                        ],
                                                        [
                                                            [ 0, 0, 0, 0 ], 
                                                            [ 0, 0, 0, 0 ], 
                                                            [ 0, 0, 0, 0 ], 
                                                            [ 0, 0, 0, 0 ]
                                                        ]
                                                   ],
                                                   [    
                                                        [
                                                            [ 0, 0, 0, 0 ], 
                                                            [ 0, 0, 0, 0 ], 
                                                            [ 0, 0, 0, 0 ], 
                                                            [ 0, 0, 0, 0 ]
                                                        ],
                                                        [
                                                            [ 0, 0, 0, 0 ], 
                                                            [ 0, 0, 0, 0 ], 
                                                            [ 0, 0, 0, 0 ], 
                                                            [ 0, 0, 0, 0 ]
                                                        ],
                                                        [
                                                            [ 0, 0, 0, 0 ], 
                                                            [ 0, 0, 0, 0 ], 
                                                            [ 0, 0, 0, 0 ], 
                                                            [ 0, 0, 0, 0 ]
                                                        ],
                                                        [
                                                            [ 0, 0, 0, 0 ], 
                                                            [ 0, 0, 0, 0 ], 
                                                            [ 0, 0, 0, 0 ], 
                                                            [ 0, 0, 0, 0 ]
                                                        ]
                                                   ],
                                                   [    
                                                        [
                                                            [ 0, 0, 0, 0 ], 
                                                            [ 0, 0, 0, 0 ], 
                                                            [ 0, 0, 0, 0 ], 
                                                            [ 0, 0, 0, 0 ]
                                                        ],
                                                        [
                                                            [ 0, 0, 0, 0 ], 
                                                            [ 0, 0, 0, 0 ], 
                                                            [ 0, 0, 0, 0 ], 
                                                            [ 0, 0, 0, 0 ]
                                                        ],
                                                        [
                                                            [ 0, 0, 0, 0 ], 
                                                            [ 0, 0, 0, 0 ], 
                                                            [ 0, 0, 0, 0 ], 
                                                            [ 0, 0, 0, 0 ]
                                                        ],
                                                        [
                                                            [ 0, 0, 0, 0 ], 
                                                            [ 0, 0, 0, 0 ], 
                                                            [ 0, 0, 0, 0 ], 
                                                            [ 0, 0, 0, 0 ]
                                                        ]
                                                   ],
                                                   [    
                                                        [
                                                            [ 0, 0, 0, 0 ], 
                                                            [ 0, 0, 0, 0 ], 
                                                            [ 0, 0, 0, 0 ], 
                                                            [ 0, 0, 0, 0 ]
                                                        ],
                                                        [
                                                            [ 0, 0, 0, 0 ], 
                                                            [ 0, 0, 0, 0 ], 
                                                            [ 0, 0, 0, 0 ], 
                                                            [ 0, 0, 0, 0 ]
                                                        ],
                                                        [
                                                            [ 0, 0, 0, 0 ], 
                                                            [ 0, 0, 0, 0 ], 
                                                            [ 0, 0, 0, 0 ], 
                                                            [ 0, 0, 0, 0 ]
                                                        ],
                                                        [
                                                            [ 0, 0, 0, 0 ], 
                                                            [ 0, 0, 0, 0 ], 
                                                            [ 0, 0, 0, 0 ], 
                                                            [ 0, 0, 0, 0 ]
                                                        ]
                                                   ]    
                                               ])          
        
        # Declares the covariant Ricci curvature tensor class object.
        self.ricci_tensor_dd = Matrix([
                                            [ 0, 0, 0, 0 ], 
                                            [ 0, 0, 0, 0 ], 
                                            [ 0, 0, 0, 0 ], 
                                            [ 0, 0, 0, 0 ]
                                        ])
        
        # Declares the contravariant Ricci curvature tensor class object.
        self.ricci_tensor_uu = Matrix([
                                            [ 0, 0, 0, 0 ], 
                                            [ 0, 0, 0, 0 ], 
                                            [ 0, 0, 0, 0 ], 
                                            [ 0, 0, 0, 0 ]
                                        ])
        
        # Declares the mixed Ricci curvature tensor class object.
        self.ricci_tensor_ud = Matrix([
                                            [ 0, 0, 0, 0 ], 
                                            [ 0, 0, 0, 0 ], 
                                            [ 0, 0, 0, 0 ], 
                                            [ 0, 0, 0, 0 ]
                                        ])
        
        # Declares Ricci curvature tensor class object.
        self.ricci_scalar = sin(self.coordinate_set[1]) * cos(self.coordinate_set[2])
        
        # Declares the covariant Einstein curvature tensor class object.
        self.einstein_tensor_dd = Matrix([    
                                                [ 0, 0, 0, 0 ], 
                                                [ 0, 0, 0, 0 ], 
                                                [ 0, 0, 0, 0 ], 
                                                [ 0, 0, 0, 0 ]
                                           ])
        
        # Declares the contravariant Einstein curvature tensor class object.
        self.einstein_tensor_uu = Matrix([    
                                                [ 0, 0, 0, 0 ], 
                                                [ 0, 0, 0, 0 ], 
                                                [ 0, 0, 0, 0 ], 
                                                [ 0, 0, 0, 0 ]
                                           ])
        
        # Declares the mixed Einstein curvature tensor class object.
        self.einstein_tensor_ud = Matrix([    
                                                [ 0, 0, 0, 0 ], 
                                                [ 0, 0, 0, 0 ], 
                                                [ 0, 0, 0, 0 ], 
                                                [ 0, 0, 0, 0 ]
                                           ])
        
        # Declares the covariant stress-energy tensor class object.
        self.stress_energy_tensor_dd = Matrix([
                                                     [ 0, 0, 0, 0 ], 
                                                     [ 0, 0, 0, 0 ], 
                                                     [ 0, 0, 0, 0 ], 
                                                     [ 0, 0, 0, 0 ]
                                                 ])
        
        # Declares the contravariant stress-energy tensor class object.
        self.stress_energy_tensor_uu = Matrix([
                                                     [ 0, 0, 0, 0 ], 
                                                     [ 0, 0, 0, 0 ], 
                                                     [ 0, 0, 0, 0 ], 
                                                     [ 0, 0, 0, 0 ]
                                                 ])
        
        # Declares the mixed stress-energy tensor class object.
        self.stress_energy_tensor_ud = Matrix([
                                                     [ 0, 0, 0, 0 ], 
                                                     [ 0, 0, 0, 0 ], 
                                                     [ 0, 0, 0, 0 ], 
                                                     [ 0, 0, 0, 0 ]
                                                 ])

        # Declares the contravariant Schouten tensor class object.
        self.schouten_tensor_uu = Matrix([
                                                [ 0, 0, 0, 0 ], 
                                                [ 0, 0, 0, 0 ], 
                                                [ 0, 0, 0, 0 ], 
                                                [ 0, 0, 0, 0 ]
                                            ])

        # Declares the covariant Schouten tensor class object.
        self.schouten_tensor_dd = Matrix([
                                                [ 0, 0, 0, 0 ], 
                                                [ 0, 0, 0, 0 ], 
                                                [ 0, 0, 0, 0 ], 
                                                [ 0, 0, 0, 0 ]
                                            ])

        # Declares cosmological constant class object.
        self.cosmological_constant = 0
        
        # Acceleration vectors.
        self.proper_acceleration = [ 0, 0, 0, 0 ]
        self.coordinate_acceleration = [ 0, 0, 0, 0 ]
        self.geodesic_deviation_acceleration = [ 0, 0, 0, 0 ]

        # Velocity vectors.
        self.proper_velocity = [ 0, 0, 0, 0 ]
        self.coordinate_velocity = [ 0, 0, 0, 0 ]
        self.geodesic_velocity= [ 0, 0, 0, 0 ]

        # Position vectors.
        self.proper_position = [ 0, 0, 0, 0 ]
        self.coordinate_position = [ 0, 0, 0, 0 ]
        self.geodesic_deviation_position = [ 0, 0, 0, 0 ]

        """
        Initializing object functions
        =============================
        """

        # TODO
        # finish all of these functions.

        self.set_all_metric_coefficients("dd")
        #self.set_all_metric_coefficients("uu")
        self.set_all_connection_coefficients("udd")
        #self.set_all_connection_coefficients("ddd")
        self.set_all_riemann_coefficients("uddd")
        #self.set_all_riemann_coefficients("dddd")
        self.set_all_ricci_coefficients("dd")
        #self.set_all_weyl_coefficients("dddd")
        #self.set_all_weyl_coefficients("uddd")
        self.set_all_schouten_coefficients("dd")
        #self.set_all_cotton_coefficients("ddd")
        #self.set_all_ricci_coefficients("uu")
        #self.set_all_ricci_coefficients("ud")
        self.set_ricci_scalar()
        self.set_all_einstein_coefficients("dd")
        #self.set_all_einstein_coefficients("uu")
        #self.set_all_einstein_coefficients("ud")
        self.set_all_stress_energy_coefficients("dd")
        #self.set_all_stress_energy_coefficients("uu")
        #self.set_all_stress_energy_coefficients("ud")
        #self.set_cosmological_constant(solution[3])
        self.set_all_proper_time_geodesic_accelerations()
        self.set_all_coordinate_time_geodesic_accelerations()
        self.set_all_geodesic_deviation_accelerations()
        
    """
    Metric coefficient functions
    ============================
    """
    
    def get_metric_coefficient(self, index_config, mu, nu):
        """
        Description
        ===========
        Gets a single metric coefficient from class object for a given index configuration and index value pair.

        Example
        =======
        >> newtonian = SpaceTime(Solution().weak_field_approximation(), True)
        >> print(newtonian.get_metric_coefficient("uu",1,1))
        >>

        LaTeX representation
        ====================
        g_{ij}
        g^{ij}

        URL Reference
        =============
        https://en.wikipedia.org/wiki/Metric_tensor

        TODOs
        =====
        - Link example with test.
        - Need higher quality tests.
        """

        if (index_config == "uu"):
            return self.metric_tensor_uu[mu, nu]
        elif(index_config == "dd"):
            return self.metric_tensor_dd[mu, nu]
        else:
            print("Invalid index_config string.")
    
    def set_metric_coefficient(self, index_config, mu, nu, expression):
        """
        Description
        ===========
        Sets a single metric coefficient equal to a given expression.
        WARNING: This function is used for memory managment purposes and is not reccomended. for regular use since it can easily create contradictions within a solution easily. This may have more uses in the future.
        
        Example
        =======
        >> newtonian = SpaceTime(Solution().weak_field_approximation(), True)
        >> print(newtonian.set_metric_coefficient("uu",1,1),0)

        LaTeX representation
        ====================
        g_{23} = # set_metric_coefficient("dd",2,3,0)
        g^{03} = # set_metric_coefficient("uu",0,3,0)

        URL Reference
        =============
        https://en.wikipedia.org/wiki/Metric_tensor

        TODOs
        =====
        - Link example with test.
        - Need higher quality tests.
        """

        if (index_config == "uu"):
            self.metric_tensor_uu[mu,nu] = expression
        elif(index_config == "dd"):
            self.metric_tensor_dd[mu,nu] = expression
        else:
            print("Invalid index_config string.")
            
            
    def set_all_metric_coefficients(self, index_config):
        """
        Description
        ===========
        Sets all metric coefficients for a given index configuration. It retrieves these values from the solution input.
        * Effectively this function only is needed when the user specifies a print on object creation.

        Example
        =======
        >> newtonian = SpaceTime(Solution().weak_field_approximation(), True)
        >> newtonian.set_all_metric_coefficients("uu") # Redundant becasue function is called at creation of SpaceTime object.

        LaTeX representation
        ====================
        g_{ij}
        g^{ij}

        URL Reference
        =============
        https://en.wikipedia.org/wiki/Metric_tensor

        TODOs
        =====
        - Link example with test.
        - Need higher quality tests.
        """

        if (index_config == "uu"):
            if(self.suppress_printing == False):
                print("")
                print("")
                print("Metric tensor coefficients (uu)")
                print("===============================")
                for mu in self.dimensions:
                    for nu in self.dimensions:
                        self.print_metric_coefficient(index_config, mu, nu)
        elif(index_config == "dd"):
            if(self.suppress_printing == False):
                print("")
                print("")
                print("Metric tensor coefficients (dd)")
                print("===============================")
                for mu in self.dimensions:
                    for nu in self.dimensions:
                        self.print_metric_coefficient(index_config, mu, nu)
        else:
            print("Invalid index_config string.")

    def print_metric_coefficient(self, index_config, mu, nu):
        """
        Description
        ===========
        Prints a single metric tensor coefficient.

        Example
        =======
        >> newtonian = SpaceTime(Solution().weak_field_approximation(), True)
        >> newtonian.print_metric_coefficient("uu",3,1) 
        0
        LaTeX representation
        ====================
        g_{ij}
        g^{ij}

        URL Reference
        =============
        https://en.wikipedia.org/wiki/Metric_tensor

        TODOs
        =====
        - Link example with test.
        - Need higher quality tests.
        """

        if (index_config == "uu"):
            pprint(Eq(Symbol('g^%s%s' % (mu, nu)), self.get_metric_coefficient(index_config, mu, nu)))
        elif(index_config == "dd"):
            pprint(Eq(Symbol('g_%s%s' % (mu, nu)), self.get_metric_coefficient(index_config, mu, nu)))
        else:
            print("Invalid index_config string.")
            
    def print_all_metric_coefficients(self, index_config):
        """
        Description
        ===========
        Prints all metric tensor coefficients.

        Example
        =======
        >> newtonian = SpaceTime(Solution().weak_field_approximation(), True)
        >> newtonian.print_all_metric_coefficients("uu")
        ...
        LaTeX representation
        ====================
        g_{ij}
        g^{ij}

        URL Reference
        =============
        https://en.wikipedia.org/wiki/Metric_tensor

        TODOs
        =====
        - Link example with test.
        - Need higher quality tests.
        """

        if (index_config == "uu"):
            for mu in self.dimensions:
                for nu in self.dimensions:
                    self.print_metric_coefficient(index_config, mu, nu)
        elif(index_config == "dd"):
            for mu in self.dimensions:
                for nu in self.dimensions:
                    self.print_metric_coefficient(index_config, mu, nu)
        else:
            print("Invalid index_config string.")

    """
    Connection coefficient functions
    ================================
    """       
    
    def get_connection_coefficient(self, index_config, i, k, l):
        r"""
        Description
        ===========
        Gets a single connection coefficients from class object.

        Example
        =======
        >> newtonian = SpaceTime(Solution().weak_field_approximation(), True)
        >> print(newtonian.get_connection_coefficient("udd",1,1,1))
        2*G*M*(G*M/(c**2*r) - 1/2)/(c**2*r**2*(-2*G*M/(c**2*r) + 1)**2)

        LaTeX representation
        ====================
        \Gamma^{i}_{kl}
        \Gamma_{ikl}

        URL Reference
        =============
        https://en.wikipedia.org/wiki/Christoffel_symbols

        TODOs
        =====
        - Link example with test.
        - Need higher quality tests.
        """

        if(index_config == "udd"):
            return self.christoffel_symbols_udd[i,k][l]
        elif(index_config == "ddd"):
            return self.christoffel_symbols_ddd[i,k][l]
        else:
            print("Invalid index_config string.")     
        
    def set_connection_coefficient(self, index_config, i, k, l, expression):
        r"""
        Description
        ===========
        Sets a single connection coefficient equal to a given expression.
        WARNING: This function is used for memory managment purposes and is not reccomended for regular use since it can easily create contradictions within a solution easily. This may have more uses in the future.

        Example
        =======
        >> newtonian = SpaceTime(Solution().weak_field_approximation(), True)
        >> newtonian.set_connection_coefficient("ud",1,1,1,0)
        
        LaTeX representation
        ====================
        \Gamma^{i}_{kl},
        \Gamma_{ikl}

        URL Reference
        =============
        https://en.wikipedia.org/wiki/Christoffel_symbols

        TODOs
        =====
        - Link example with test.
        - Need higher quality tests.
        """

        if(index_config == "udd"):
            self.christoffel_symbols_udd[i,k][l] = expression
        elif(index_config == "ddd"):
            self.christoffel_symbols_ddd[i,k][l] = expression
        else:
            print("Invalid index_config string.")        
    
    def set_all_connection_coefficients(self, index_config):
        r"""
        Description
        ===========
        Sets all connection coefficient values for reuse. Allows for the removal of redundant calculations.
        WARNING: Redundant since this is called at creation of SpaceTime object.

        Example
        =======
        >> newtonian = SpaceTime(Solution().weak_field_approximation(), True)
        >> newtonian.set_all_connection_coefficients("udd")
        
        LaTeX representation
        ====================
        \Gamma^{i}_{kl}
        \Gamma_{ikl}

        URL Reference
        =============
        https://en.wikipedia.org/wiki/Christoffel_symbols

        TODOs
        =====
        - Link example with test.
        - Need higher quality tests.
        """

        if(index_config == "udd"):
            if(self.suppress_printing == False):
                print("")
                print("")
                print("Connection coefficients (udd)")
                print("=============================")
            for i in self.dimensions:
                for k in self.dimensions:
                    for l in self.dimensions:
                        self.set_connection_coefficient(index_config, i, k, l, self.compute_connection_coefficient(index_config, i, k, l))
                        if(self.suppress_printing == False):
                            self.print_connection_coefficient(index_config, i, k, l )
        elif(index_config == "ddd"):
            if(self.suppress_printing == False):
                print("")
                print("")
                print("Connection coefficients (ddd)")
                print("=============================")
            for i in self.dimensions:
                for k in self.dimensions:
                    for l in self.dimensions:
                        self.set_connection_coefficient(index_config, i, k, l, self.compute_connection_coefficient(index_config, i, k, l))
                        if(self.suppress_printing == False):
                            self.print_connection_coefficient(index_config, i, k, l )
        else:
            print("Invalid index_config string.")

    def compute_connection_coefficient(self, index_config, i, k, l):
        r"""
        Description
        ===========
        Computes a single connection coefficient.

        Example
        =======
        >> newtonian = SpaceTime(Solution().weak_field_approximation(), True)
        >> newtonian.compute_connection_coefficients("udd",0,0,0)
        
        LaTeX representation
        ====================
        \Gamma^{i}_{kl}
        \Gamma_{ikl}

        URL Reference
        =============
        https://en.wikipedia.org/wiki/Christoffel_symbols

        TODOs
        =====
        - Link example with test.
        - Need higher quality tests.
        """

        connection = 0
        if index_config == "udd":
            for m in self.dimensions:
                connection = connection+Rational('1/2')*self.metric_tensor_uu[m,i]*(diff(self.metric_tensor_dd[k,m], self.coordinate_set[l])+diff(self.metric_tensor_dd[l,m], self.coordinate_set[k])-diff(self.metric_tensor_dd[k,l], self.coordinate_set[m]))
            return connection
        elif index_config == "ddd":
            connection = Rational('1/2')*(diff(self.metric_tensor_dd[i,k], self.coordinate_set[l])+diff(self.metric_tensor_dd[i,l], self.coordinate_set[k])-diff(self.metric_tensor_dd[k,l], self.coordinate_set[i]))
            return simplify(connection)
        else:
            print("Invalid index_config string.")
    
    def print_connection_coefficient(self, index_config, i, j, k ):
        r"""
        Description
        ===========
        Prints a single connection coefficient.

        Example
        =======
        >> newtonian = SpaceTime(Solution().weak_field_approximation(), True)
        >> newtonian.print_connection_coefficient("udd",0,0,0)
        
        Γ⁰₀₀ = 0

        LaTeX representation
        ====================
        \Gamma^{i}_{kl}
        \Gamma_{ikl}

        URL Reference
        =============
        https://en.wikipedia.org/wiki/Christoffel_symbols

        TODOs
        =====
        - Link example with test.
        - Need higher quality tests.
        """

        if(index_config == "udd"):
            pprint(Eq(Symbol('Gamma^%s_%s%s' % (i, j, k)),self.get_connection_coefficient(index_config, i, j, k )))
        elif(index_config == "ddd"):
            # TODO
            # MUST TEST
            pprint(Eq(Symbol('Gamma_%s%s%s' % (i, j, k)),self.get_connection_coefficient(index_config, i, j, k )))
        else:
            print("Invalid index_config string.")
    
    # Prints all connection coefficients.
    def print_all_connection_coefficients(self, index_config):
        r"""
        Description
        ===========
        Prints all connection coefficients for a given index configuration.

        Example
        =======
        >> newtonian = SpaceTime(Solution().weak_field_approximation(), True)
        >> newtonian.print_all_connection_coefficients("udd")
        
        Γ⁰₀₀ = 0

        ...
        ...
        ...

        Γ³₃₃ = 0

        LaTeX representation
        ====================
        \Gamma^{i}_{kl}
        \Gamma_{ikl}

        URL Reference
        =============
        https://en.wikipedia.org/wiki/Christoffel_symbols

        TODOs
        =====
        - Link example with test.
        - Need higher quality tests.
        """

        if(index_config == "udd"):
            for lam in self.dimensions:
                for mu in self.dimensions:
                    for nu in self.dimensions:
                        print("")
                        self.print_connection_coefficient(index_config, lam, mu, nu )
        elif(index_config == "ddd"):
            for lam in self.dimensions:
                for mu in self.dimensions:
                    for nu in self.dimensions:
                        print("")
                        self.print_connection_coefficient(index_config, lam, mu, nu )
        else:
            print("Invalid index_config string.")
    
    """
    Riemann coefficient functions
    =============================
    """      
    
    def get_riemann_coefficient(self, index_config, rho, sig, mu, nu):
        """
        Description
        ===========
        Gets a single Riemann coefficients from class object.

        Example
        =======
        >> newtonian = SpaceTime(Solution().weak_field_approximation(), True)
        >> pprint(newtonian.get_riemann_coefficient("uddd",0,1,2,3))
        0

        LaTeX representation
        ====================
        R^{i}_{jkl} -> (i,j,k,l) = (rho,sig,mu,nu)
        R_{ijkl} -> (i,j,k,l) = (rho,sig,mu,nu)

        URL Reference
        =============
        https://en.wikipedia.org/wiki/List_of_formulas_in_Riemannian_geometry#Riemann_curvature_tensor
        https://en.wikipedia.org/wiki/Riemann_curvature_tensor

        TODOs
        =====
        - Link example with test.
        - Need higher quality tests.
        - Needs functionality for other index configurations.
        """

        if(index_config == "uddd"):
            return self.riemann_tensor_uddd[int(rho*16/self.dimension_count+sig)][mu][nu]
        elif(index_config == "dddd"):
            return self.riemann_tensor_dddd[int(rho*16/self.dimension_count+sig)][mu][nu]
        else:
            print("Invalid index_config string.")  
    
    def set_riemann_coefficient(self, index_config, rho, sig, mu, nu, expression):
        """
        Description
        ===========
        Sets a single Riemann coefficient equal to a given expression.
        WARNING: This function is used for memory managment purposes and is not reccomended for interactive use.

        Example
        =======
        >> newtonian = SpaceTime(Solution().weak_field_approximation(), True)
        >> pprint(newtonian.set_riemann_coefficient("uddd",0,1,2,3),0)
        
        LaTeX representation
        ====================
        R^{i}_{jkl} -> (i,j,k,l) = (rho,sig,mu,nu)
        R_{ijkl} -> (i,j,k,l) = (rho,sig,mu,nu)

        URL Reference
        =============
        https://en.wikipedia.org/wiki/List_of_formulas_in_Riemannian_geometry#Riemann_curvature_tensor
        https://en.wikipedia.org/wiki/Riemann_curvature_tensor

        TODOs
        =====
        - Link example with test.
        - Need higher quality tests.
        - Needs functionality for other index configurations.
        """

        if(index_config == "uddd"):
            self.riemann_tensor_uddd[int(rho*16/self.dimension_count+sig)][mu][nu] = expression
        elif(index_config == "dddd"):
            # TODO
            # MUST TEST
            self.riemann_tensor_dddd[int(rho*16/self.dimension_count+sig)][mu][nu] = expression
        else:
            print("Invalid index_config string.")        
    
    def set_all_riemann_coefficients(self, index_config):
        """
        Description
        ===========
        Sets all Riemann coefficients values for reuse. Allows for the removal of redundant calculations.
        WARNING: Redundant becasue function is already called at creation of SpaceTime object.

        Example
        =======
        >> newtonian = SpaceTime(Solution().weak_field_approximation(), True)
        >> pprint(newtonian.set_all_riemann_coefficients("uddd")
        
        LaTeX representation
        ====================
        R^{i}_{jkl} -> (i,j,k,l) = (rho,sig,mu,nu)
        R_{ijkl} -> (i,j,k,l) = (rho,sig,mu,nu)

        URL Reference
        =============
        https://en.wikipedia.org/wiki/List_of_formulas_in_Riemannian_geometry#Riemann_curvature_tensor
        https://en.wikipedia.org/wiki/Riemann_curvature_tensor

        TODOs
        =====
        - Link example with test.
        - Need higher quality tests.
        - Needs functionality for other index configurations.
        """

        if index_config == "uddd":
            if(self.suppress_printing == False):
                print("")
                print("")
                print("Riemann curvature tensor coefficients (uddd)")
                print("============================================")
            for rho in self.dimensions:
                for sig in self.dimensions:
                    for mu in self.dimensions:
                        for nu in self.dimensions:
                            self.set_riemann_coefficient(index_config, rho, sig, mu, nu, self.compute_riemann_coefficient(index_config, rho, sig, mu, nu))
                            if(self.suppress_printing == False):
                                self.print_riemann_coefficient(index_config, rho, sig, mu, nu)
        elif index_config == "dddd":
            if(self.suppress_printing == False):
                print("")
                print("")
                print("Riemann curvature tensor coefficients (dddd)")
                print("============================================")
            for rho in self.dimensions:
                for sig in self.dimensions:
                    for mu in self.dimensions:
                        for nu in self.dimensions:
                            # TODO
                            # MUST TEST
                            self.set_riemann_coefficient(index_config, rho, sig, mu, nu, self.compute_riemann_coefficient(index_config, rho, sig, mu, nu))
                            if(self.suppress_printing == False):
                                self.print_riemann_coefficient(index_config, rho, sig, mu, nu)
        else:
            print("Invalid index_config string.")
    
    def compute_riemann_coefficient(self, index_config, rho, sig, mu, nu):
        """
        Description
        ===========
        Computes a single Riemann tensor coefficient.

        Example
        =======
        >> newtonian = SpaceTime(Solution().weak_field_approximation(), True)
        >> print(newtonian.compute_riemann_coefficient("uddd",0,2,2,0))
        G*M/(c**2*r)

        LaTeX representation
        ====================
        R^{i}_{jkl} -> (i,j,k,l) = (rho,sig,mu,nu)
        R_{ijkl} -> (i,j,k,l) = (rho,sig,mu,nu)

        URL Reference
        =============
        https://en.wikipedia.org/wiki/List_of_formulas_in_Riemannian_geometry#Riemann_curvature_tensor
        https://en.wikipedia.org/wiki/Riemann_curvature_tensor

        TODOs
        =====
        - Link example with test.
        - Need higher quality tests.
        - Needs functionality for other index configurations.
        """

        riemann_coefficient = 0
        if index_config == "uddd":
            riemann_coefficient = diff(self.get_connection_coefficient("udd", rho, nu, sig), self.coordinate_set[mu]) - diff(self.get_connection_coefficient("udd", rho, mu, sig), self.coordinate_set[nu])    
            for lam in self.dimensions:
                riemann_coefficient = riemann_coefficient + self.get_connection_coefficient("udd", rho, mu, lam)*self.get_connection_coefficient("udd", lam, nu, sig) - self.get_connection_coefficient("udd", rho, nu, lam)*self.get_connection_coefficient("udd", lam, mu, sig)
            riemann_coefficient = simplify(riemann_coefficient)
            return riemann_coefficient
        elif index_config == "dddd":
            riemann_coefficient = Rational('1/2')*(self.get_metric_coefficient("dd", rho, nu).diff(self.coordinate_set[sig]).diff(self.coordinate_set[mu]) + self.get_metric_coefficient("dd", sig, mu).diff(self.coordinate_set[rho]).diff(self.coordinate_set[nu])-self.get_metric_coefficient("dd", rho, mu).diff(self.coordinate_set[sig]).diff(self.coordinate_set[nu])-self.get_metric_coefficient("dd", sig, nu).diff(self.coordinate_set[rho]).diff(self.coordinate_set[mu]))
            for n in self.dimensions:
                for p in self.dimensions:
                    riemann_coefficient = riemann_coefficient + self.get_metric_coefficient("dd", n, p)*(self.get_connection_coefficient("udd", n, sig, mu)*self.get_connection_coefficient("udd", p, rho, nu)-self.get_connection_coefficient("udd", n, sig, nu)*self.get_connection_coefficient("udd", p, rho, mu))
            riemann_coefficient = simplify(riemann_coefficient)
            return riemann_coefficient
        else:
            print("Invalid index_config string.")
        
    def print_riemann_coefficient(self, index_config, rho, sig, mu, nu):
        """
        Description
        ===========
        Prints a single Riemann coefficient.

        Example
        =======
        >> newtonian = SpaceTime(Solution().weak_field_approximation(), True)
        >> newtonian.print_riemann_coefficient("uddd",0,2,2,0)

                G⋅M
        R⁰₂₂₀ = ────
                2
                c ⋅r

        LaTeX representation
        ====================
        R^{i}_{jkl} -> (i,j,k,l) = (rho,sig,mu,nu)
        R_{ijkl} -> (i,j,k,l) = (rho,sig,mu,nu)

        URL Reference
        =============
        https://en.wikipedia.org/wiki/List_of_formulas_in_Riemannian_geometry#Riemann_curvature_tensor
        https://en.wikipedia.org/wiki/Riemann_curvature_tensor

        TODOs
        =====
        - Link example with test.
        - Need higher quality tests.
        - Needs functionality for other index configurations.
        """

        if index_config == "uddd":
            pprint(Eq(Symbol('R^%s_%s%s%s' % (rho, sig, mu, nu)), self.get_riemann_coefficient(index_config, rho, sig, mu, nu)))
        elif index_config == "dddd":
            pprint(Eq(Symbol('R_%s%s%s%s' % (rho, sig, mu, nu)), self.get_riemann_coefficient(index_config, rho, sig, mu, nu)))
        else:
            print("Invalid index_config string.")
            
    def print_all_riemann_coefficients(self, index_config):
        """
        Description
        ===========
        Prints all connection coefficients.

        Example
        =======
        >> newtonian = SpaceTime(Solution().weak_field_approximation(), True)
        >> newtonian.print_all_riemann_coefficients("uddd")

        R⁰₀₀₀ = 0

        R⁰₀₀₁ = 0

        R⁰₀₀₂ = 0

        ...

        LaTeX representation
        ====================
        R^{i}_{jkl} -> (i,j,k,l) = (rho,sig,mu,nu)
        R_{ijkl} -> (i,j,k,l) = (rho,sig,mu,nu)

        URL Reference
        =============
        https://en.wikipedia.org/wiki/List_of_formulas_in_Riemannian_geometry#Riemann_curvature_tensor
        https://en.wikipedia.org/wiki/Riemann_curvature_tensor

        TODOs
        =====
        - Link example with test.
        - Need higher quality tests.
        - Needs functionality for other index configurations.
        """
        
        if index_config == "uddd":
            for rho in self.dimensions:
                for sig in self.dimensions:
                    for mu in self.dimensions:
                        for nu in self.dimensions:
                            self.print_riemann_coefficient(index_config, rho, sig, mu, nu)
        elif index_config == "dddd":
            for rho in self.dimensions:
                for sig in self.dimensions:
                    for mu in self.dimensions:
                        for nu in self.dimensions:
                            self.print_riemann_coefficient(index_config, rho, sig, mu, nu)
        else:
            print("Invalid index_config string.")
    
    
    """
    Weyl coefficient functions
    ==========================
    """ 
    
    def get_weyl_coefficient(self, index_config, i, k, l, m):
        """
        Description
        ===========
        Gets a single Weyl coefficients from class object.

        Example
        =======
        >> newtonian = SpaceTime(Solution().weak_field_approximation(), True)
        >> pprint(newtonian.get_weyl_coefficient("dddd",0,1,2,3))
        0

        LaTeX Representation
        ====================
        C_{iklm}

        URL Reference
        =============
        https://en.wikipedia.org/wiki/Weyl_tensor

        TODOs
        =====
        - Link example with test.
        - Need higher quality tests.
        - Needs functionality for other index configurations.
        """

        if(index_config == "uddd"):
            # TODO
            # MUST TEST
            return self.weyl_tensor_uddd[int(i*16/self.dimension_count+k)][l][m]
        elif(index_config == "dduu"):
            # TODO
            # MUST TEST
            return self.weyl_tensor_dduu[int(i*16/self.dimension_count+k)][l][m]
        elif(index_config == "dddd"):
            # TODO
            # MUST TEST
            return self.weyl_tensor_dddd[int(i*16/self.dimension_count+k)][l][m]
        else:
            print("Invalid index_config string.") 
    
    def set_weyl_coefficient(self, index_config, i, k, l, m, expression):
        """
        Description
        ===========
        Sets a single Weyl coefficient from class object equal to the value of a given expression.

        Example
        =======
        >> newtonian = SpaceTime(Solution().weak_field_approximation(), True)
        >> pprint(newtonian.set_weyl_coefficient("dddd",0,1,2,3,0))

        LaTeX Representation
        ====================
        C_{iklm}

        URL Reference
        =============
        https://en.wikipedia.org/wiki/Weyl_tensor

        TODOs
        =====
        - Link example with test.
        - Need higher quality tests.
        - Needs functionality for other index configurations.
        """

        if(index_config == "uddd"):
            # TODO
            # MUST TEST
            self.weyl_tensor_uddd[int(i*16/self.dimension_count+k)][l][m] = expression
        elif(index_config == "dduu"):
            # TODO
            # MUST TEST
            self.weyl_tensor_dduu[int(i*16/self.dimension_count+k)][l][m] = expression
        elif(index_config == "dddd"):
            # TODO
            # MUST TEST
            self.weyl_tensor_dddd[int(i*16/self.dimension_count+k)][l][m] = expression
        else:
            print("Invalid index_config string.") 
    
    def set_all_weyl_coefficients(self, index_config):
        """
        Description
        ===========
        Sets and computes all Weyl coefficients.

        Example
        =======
        >> newtonian = SpaceTime(Solution().weak_field_approximation(), True)
        >> pprint(newtonian.set_all_weyl_coefficients("dddd"))

        LaTeX Representation
        ====================
        C_{iklm}

        URL Reference
        =============
        https://en.wikipedia.org/wiki/Weyl_tensor

        TODOs
        =====
        - Link example with test.
        - Need higher quality tests.
        - Needs functionality for other index configurations.
        """

        if index_config == "uddd":
            if(self.suppress_printing == False):
                print("")
                print("")
                print("Weyl curvature tensor coefficients (uddd)")
                print("=========================================")
            for i in self.dimensions:
                for k in self.dimensions:
                    for l in self.dimensions:
                        for m in self.dimensions:
                            self.set_weyl_coefficient(index_config, i, k, l, m, self.compute_weyl_coefficient(index_config, i, k, l, m))
                            if(self.suppress_printing == False):
                                self.print_weyl_coefficient(index_config, i, k, l, m)
        elif index_config == "dduu":
            if(self.suppress_printing == False):
                print("")
                print("")
                print("Weyl curvature tensor coefficients (dduu)")
                print("=========================================")
            for i in self.dimensions:
                for k in self.dimensions:
                    for l in self.dimensions:
                        for m in self.dimensions:
                            # TODO
                            # MUST TEST
                            self.set_weyl_coefficient(index_config, i, k, l, m, self.compute_weyl_coefficient(index_config, i, k, l, m))
                            if(self.suppress_printing == False):
                                self.print_weyl_coefficient(index_config, i, k, l, m)
        elif index_config == "dddd":
            if(self.suppress_printing == False):
                print("")
                print("")
                print("Weyl curvature tensor coefficients (dddd)")
                print("=========================================")
            for i in self.dimensions:
                for k in self.dimensions:
                    for l in self.dimensions:
                        for m in self.dimensions:
                            # TODO
                            # MUST TEST
                            self.set_weyl_coefficient(index_config, i, k, l, m, self.compute_weyl_coefficient(index_config, i, k, l, m))
                            if(self.suppress_printing == False):
                                self.print_weyl_coefficient(index_config, i, k, l, m)
        else:
            print("Invalid index_config string.")
    
    def compute_weyl_coefficient(self, index_config, i, k, l, m):
        """
        Description
        ===========
        Sets and computes all Weyl coefficients.

        Example
        =======
        >> newtonian = SpaceTime(Solution().weak_field_approximation(), True)
        >> pprint(newtonian.compute_weyl_coefficient("dddd",0,1,2,3))
        0
        
        LaTeX Representation
        ====================
        C_{iklm}

        URL Reference
        =============
        https://en.wikipedia.org/wiki/Weyl_tensor

        TODOs
        =====
        - Link example with test.
        - Need higher quality tests.
        - Needs functionality for other index configurations.
        """

        n = len(self.coordinate_set)
        weyl_coefficient = 0
        if(index_config == "uddd"):
            # TODO
            # MUST TEST
            pass
        elif(index_config == "dduu"):
            # TODO
            # MUST TEST
            pass
        elif(index_config == "dddd"):   
            weyl_coefficient = self.get_riemann_coefficient("dddd", i, k, l, m) + Rational('1/'+str(n-2))*(self.get_ricci_coefficient("dd",i,m)*self.get_metric_coefficient("dd",k,l)-self.get_ricci_coefficient("dd",i,l)*self.get_metric_coefficient("dd",k,m)+self.get_ricci_coefficient("dd",k,l)*self.get_metric_coefficient("dd",i,m)-self.get_ricci_coefficient("dd",k,m)*self.get_metric_coefficient("dd",i,l))+Rational('1/'+str(int((n-1)*(n-2))))*self.get_ricci_scalar()*(self.get_metric_coefficient("dd", i, l)*self.get_metric_coefficient("dd", k, m)-self.get_metric_coefficient("dd", i, m)*self.get_metric_coefficient("dd", k, l))
            return simplify(weyl_coefficient)
        else:
            print("Invalid index_config string.") 
    
    def print_weyl_coefficient(self, index_config, i, k, l, m):
        """
        Description
        ===========
        Prints a single Weyl coefficient.

        Example
        =======
        >> newtonian = SpaceTime(Solution().weak_field_approximation(), True)
        >> pprint(newtonian.compute_weyl_coefficient("dddd",0,0,0,0))
        0
        
        LaTeX Representation
        ====================
        C_{iklm}

        URL Reference
        =============
        https://en.wikipedia.org/wiki/Weyl_tensor

        TODOs
        =====
        - Link example with test.
        - Need higher quality tests.
        - Needs functionality for other index configurations.
        """

        if(index_config == "uddd"):
            pprint(Eq(Symbol('C^%s_%s%s%s' % (i, k, l, m)), self.get_weyl_coefficient(index_config, i, k, l, m)))
        elif(index_config == "dduu"):
            pprint(Eq(Symbol('C_%s%s^%s%s' % (i, k, l, m)), self.get_weyl_coefficient(index_config, i, k, l, m)))
        elif(index_config == "dddd"):
            pprint(Eq(Symbol('C_%s%s%s%s' % (i, k, l, m)), self.get_weyl_coefficient(index_config, i, k, l, m)))
        else:
            print("Invalid index_config string.") 
    
    def print_all_weyl_coefficients(self, index_config):
        """
        Description
        ===========
        Prints all Weyl coefficients.

        Example
        =======
        >> newtonian = SpaceTime(Solution().weak_field_approximation(), True)
        >> pprint(newtonian.print_all_weyl_coefficients("dddd"))

        C₀₀₀₀ = 0

        C₀₀₀₁ = 0

        C₀₀₀₂ = 0

        ...
        
        C₃₃₃₂ = 0

        C₃₃₃₃ = 0

        LaTeX Representation
        ====================
        C_{iklm}

        URL Reference
        =============
        https://en.wikipedia.org/wiki/Weyl_tensor

        TODOs
        =====
        - Link example with test.
        - Need higher quality tests.
        - Needs functionality for other index configurations.
        """

        if index_config == "uddd":
            for i in self.dimensions:
                for k in self.dimensions:
                    for l in self.dimensions:
                        for m in self.dimensions:
                            self.print_weyl_coefficient(index_config, i, k, l, m)
        elif index_config == "dduu":
            for i in self.dimensions:
                for k in self.dimensions:
                    for l in self.dimensions:
                        for m in self.dimensions:
                            # TODO
                            # MUST TEST
                            self.print_weyl_coefficient(index_config, i, k, l, m)
        elif index_config == "dddd":
            for i in self.dimensions:
                for k in self.dimensions:
                    for l in self.dimensions:
                        for m in self.dimensions:
                            # TODO
                            # MUST TEST
                            self.print_weyl_coefficient(index_config, i, k, l, m)
    
    """
    Ricci coefficient functions
    =============================
    """      
    
    def get_ricci_coefficient(self, index_config, mu, nu):
        """
        Description
        ===========
        Gets a single Ricci coefficient from class object.

        Example
        =======
        >> newtonian = SpaceTime(Solution().weak_field_approximation(), True)
        >> pprint(newtonian.get_ricci_coefficient("dd",0,0))
        0
        
        LaTeX Representation
        ====================
        R_{m,n}
        R^{m,n}

        URL Reference
        =============
        https://en.wikipedia.org/wiki/Ricci_curvature

        TODOs
        =====
        - Link example with test.
        - Need higher quality tests.
        - Needs functionality for other index configurations.
        """

        if (index_config == "uu"):
            return self.ricci_tensor_uu[mu,nu]
        elif(index_config == "dd"):
            return self.ricci_tensor_dd[mu,nu]
        else:
            print("Invalid index_config string.")
    
    def set_ricci_coefficient(self, index_config, mu, nu, expression):
        """
        Description
        ===========
        Sets a single Ricci coefficient from class object.

        Example
        =======
        >> newtonian = SpaceTime(Solution().weak_field_approximation(), True)
        >> newtonian.set_ricci_coefficient("dd",0,0,0)
        
        LaTeX Representation
        ====================
        R_{m,n}
        R^{m,n}

        URL Reference
        =============
        https://en.wikipedia.org/wiki/Ricci_curvature

        TODOs
        =====
        - Link example with test.
        - Need higher quality tests.
        - Needs functionality for other index configurations.
        """

        if (index_config == "uu"):
            self.ricci_tensor_uu[mu,nu] = expression
        elif(index_config == "dd"):
            self.ricci_tensor_dd[mu,nu] = expression
        else:
            print("Invalid index_config string.")
    
    # Sets all Ricci coefficient values for reuse. Allows for the removal of redundant calculations.
    def set_all_ricci_coefficients(self, index_config):
        """
        Description
        ===========
        Computes and sets all Ricci tensor class object coefficients. Runs at object creation.

        Example
        =======
        >> newtonian = SpaceTime(Solution().weak_field_approximation(), True)
        >> newtonian.set_all_ricci_coefficients("dd")
        
        LaTeX Representation
        ====================
        R_{m,n}
        R^{m,n}

        URL Reference
        =============
        https://en.wikipedia.org/wiki/Ricci_curvature

        TODOs
        =====
        - Link example with test.
        - Need higher quality tests.
        - Needs functionality for other index configurations.
        """

        if(index_config == "uu"):
            if(self.suppress_printing == False):
                print("")
                print("")
                print("Ricci curvature tensor coefficients (uu)")
                print("========================================")
            for mu in self.dimensions:
                for nu in self.dimensions:
                    self.set_ricci_coefficient(index_config, mu, nu, self.compute_ricci_coefficient(index_config, mu, nu))
                    if(self.suppress_printing == False):
                        self.print_ricci_coefficient(index_config, mu, nu)
        elif(index_config == "dd"):
            if(self.suppress_printing == False):
                print("")
                print("")
                print("Ricci curvature tensor coefficients (dd)")
                print("========================================")
            for mu in self.dimensions:
                for nu in self.dimensions:
                    self.set_ricci_coefficient(index_config, mu, nu, self.compute_ricci_coefficient(index_config, mu, nu))
                    if(self.suppress_printing == False):
                        self.print_ricci_coefficient(index_config, mu, nu)
    
    def compute_ricci_coefficient(self, index_config, mu, nu):
        """
        Description
        ===========
        Computes a single Ricci tensor coefficient.

        Example
        =======
        >> newtonian = SpaceTime(Solution().weak_field_approximation(), True)
        >> print(newtonian.compute_ricci_coefficient("dd",0,0))
        0
        
        LaTeX Representation
        ====================
        R_{m,n}
        R^{m,n}

        URL Reference
        =============
        https://en.wikipedia.org/wiki/Ricci_curvature

        TODOs
        =====
        - Link example with test.
        - Need higher quality tests.
        - Needs functionality for other index configurations.
        """

        ricci_coefficient = 0
        if index_config == "dd":
            for lam in self.dimensions:
                ricci_coefficient = ricci_coefficient + self.get_riemann_coefficient("uddd", lam, mu, lam, nu)
            ricci_coefficient = simplify(ricci_coefficient)
        elif index_config == "uu":
            print("")
        elif index_config == "ud" or index_config == "du":
            print("")
        else:
            print("Invalid index_config string.")

        return ricci_coefficient
    
    def print_ricci_coefficient(self, index_config, mu, nu):
        """
        Description
        ===========
        Prints a single Ricci coefficient.

        Example
        =======
        >> newtonian = SpaceTime(Solution().weak_field_approximation(), True)
        >> newtonian.print_ricci_coefficient("dd",0,0)

        R₀₀ = 0
        
        LaTeX Representation
        ====================
        R_{m,n}
        R^{m,n}

        URL Reference
        =============
        https://en.wikipedia.org/wiki/Ricci_curvature

        TODOs
        =====
        - Link example with test.
        - Need higher quality tests.
        - Needs functionality for other index configurations.
        """

        if (index_config == "uu"):
            pprint(Eq(Symbol('R^%s%s' % (mu, nu)), self.get_ricci_coefficient(index_config, mu, nu)))
        elif(index_config == "dd"):
            pprint(Eq(Symbol('R_%s%s' % (mu, nu)), self.get_ricci_coefficient(index_config, mu, nu)))
        else:
            print("Invalid index_config string.")
    
    # Prints all Ricci coefficients.
    def print_all_ricci_coefficients(self, index_config):
        """
        Description
        ===========
        Prints a single Weyl coefficient.

        Example
        =======
        >> newtonian = SpaceTime(Solution().weak_field_approximation(), True)
        >> pprint(newtonian.print_all_ricci_coefficients("dd"))

        R₀₀ = 0

        R₀₁ = 0

        R₀₂ = 0
        
        LaTeX Representation
        ====================
        R_{m,n}
        R^{m,n}

        URL Reference
        =============
        https://en.wikipedia.org/wiki/Ricci_curvature

        TODOs
        =====
        - Link example with test.
        - Need higher quality tests.
        - Needs functionality for other index configurations.
        """
        for mu in self.dimensions:
            for nu in self.dimensions:
                self.print_ricci_coefficient(index_config, mu, nu)
    
    """
    Ricci scalar functions
    ======================
    """
    
    def get_ricci_scalar(self):
        """
        Description
        ===========
        Gets the Ricci scalar class object.

        Example
        =======
        >> newtonian = SpaceTime(Solution().weak_field_approximation(), True)
        >> pprint(newtonian.get_ricci_scalar())
        0

        LaTeX Representation
        ====================
        R = g^{mn} R_{mn} 

        URL Reference
        =============
        https://en.wikipedia.org/wiki/Ricci_curvature

        TODOs
        =====
        - Link example with test.
        - Need higher quality tests.
        - Needs functionality for other index configurations.
        """

        return self.ricci_scalar  
    
    def set_ricci_scalar(self):
        """
        Description
        ===========
        # Sets Ricci scalar from class object.

        Example
        =======
        >> newtonian = SpaceTime(Solution().weak_field_approximation(), True)
        >> pprint(newtonian.set_ricci_scalar())

        LaTeX Representation
        ====================
        R = g^{mn} R_{mn} 

        URL Reference
        =============
        https://en.wikipedia.org/wiki/Ricci_curvature

        TODOs
        =====
        - Link example with test.
        - Need higher quality tests.
        - Needs functionality for other index configurations.
        """

        self.ricci_scalar = self.compute_ricci_scalar()
        if(self.suppress_printing == False):
            print("")
            print("")
            print("Ricci curvature scalar")
            print("======================")
            self.print_ricci_scalar()
    
    def compute_ricci_scalar(self):
        """
        Description
        ===========
        # Computes the Ricci scalar.

        Example
        =======
        >> newtonian = SpaceTime(Solution().weak_field_approximation(), True)
        >> pprint(newtonian.compute_ricci_scalar())

        LaTeX Representation
        ====================
        R = g^{mn} R_{mn} 

        URL Reference
        =============
        https://en.wikipedia.org/wiki/Ricci_curvature

        TODOs
        =====
        - Link example with test.
        - Need higher quality tests.
        - Needs functionality for other index configurations.
        """

        ricci_scalar = 0
        for mu in self.dimensions:
            for nu in self.dimensions:
                ricci_scalar = ricci_scalar + self.metric_tensor_uu[mu, nu] * self.get_ricci_coefficient("dd", mu, nu)
        ricci_scalar = simplify(ricci_scalar)
        return ricci_scalar
    
    def print_ricci_scalar(self):
        """
        Description
        ===========
        # Prints Ricci scalar.

        Example
        =======
        >> newtonian = SpaceTime(Solution().weak_field_approximation(), True)
        >> pprint(newtonian.print_ricci_scalar())

        R = 0

        LaTeX Representation
        ====================
        R = g^{mn} R_{mn} 

        URL Reference
        =============
        https://en.wikipedia.org/wiki/Ricci_curvature

        TODOs
        =====
        - Link example with test.
        - Need higher quality tests.
        - Needs functionality for other index configurations.
        """

        pprint(Eq(Symbol('R'), self.get_ricci_scalar()))
    
    """
    Einstein tensor functions
    =========================
    """
    
    def get_einstein_coefficient(self, index_config, mu, nu):
        """
        Description
        ===========
        Gets a single Einstein coefficient from class object.

        Example
        =======
        >> newtonian = SpaceTime(Solution().weak_field_approximation(), True)
        >> pprint(newtonian.get_einstein_coefficient("dd",0,0))
        0
        
        LaTeX Representation
        ====================
        G = R_{mn} - R/2 g_{mn} 

        URL Reference
        =============
        https://en.wikipedia.org/wiki/Einstein_tensor

        TODOs
        =====
        - Link example with test.
        - Need higher quality tests.
        - Needs functionality for other index configurations.
        """

        if (index_config == "uu"):
            # TODO
            # MUST TEST
            return self.einstein_tensor_uu[mu, nu]
        elif(index_config == "dd"):
            return self.einstein_tensor_dd[mu, nu]
        else:
            print("Invalid index_config string.")
    
    def set_einstein_coefficient(self, index_config, mu, nu, expression):
        """
        Description
        ===========
        Sets a single Ricci coefficient from class object.

        Example
        =======
        >> newtonian = SpaceTime(Solution().weak_field_approximation(), True)
        >> newtonian.set_einstein_coefficient("dd",0,0,0)

        LaTeX Representation
        ====================
        G = R_{mn} - R/2 g_{mn} 

        URL Reference
        =============
        https://en.wikipedia.org/wiki/Einstein_tensor

        TODOs
        =====
        - Link example with test.
        - Need higher quality tests.
        - Needs functionality for other index configurations.
        """

        if (index_config == "uu"):
            # TODO
            # MUST TEST
            self.einstein_tensor_uu[mu, nu] = expression 
        elif(index_config == "dd"):
            self.einstein_tensor_dd[mu, nu] = expression 
        else:
            print("Invalid index_config string.")  
    
    def set_all_einstein_coefficients(self, index_config):
        """
        Description
        ===========
        Sets all Einstein coefficient values for reuse. Allows for the removal of redundant calculations.

        Example
        =======
        >> newtonian = SpaceTime(Solution().weak_field_approximation(), True)
        >> newtonian.set_all_einstein_coefficients("dd")

        LaTeX Representation
        ====================
        G = R_{mn} - R/2 g_{mn} 

        URL Reference
        =============
        https://en.wikipedia.org/wiki/Einstein_tensor

        TODOs
        =====
        - Link example with test.
        - Need higher quality tests.
        - Needs functionality for other index configurations.
        """

        if (index_config=="uu"):
            if(self.suppress_printing == False):
                print("")
                print("")
                print("Einstein curvature tensor coefficients (uu)")
                print("===========================================")
            for mu in self.dimensions:
                for nu in self.dimensions:
                    self.set_einstein_coefficient(index_config, mu, nu, self.compute_einstein_coefficient(index_config, mu, nu))
                    if(self.suppress_printing == False):
                        self.print_einstein_coefficient(index_config, mu, nu)
        elif (index_config == "dd"):
            if(self.suppress_printing == False):
                print("")
                print("")
                print("Einstein curvature tensor coefficients (dd)")
                print("===========================================")
            for mu in self.dimensions:
                for nu in self.dimensions:
                    self.set_einstein_coefficient(index_config, mu, nu, self.compute_einstein_coefficient(index_config, mu, nu))
                    if(self.suppress_printing == False):
                        self.print_einstein_coefficient(index_config, mu, nu)
        else:
            print("Invalid index_config string.") 
                    
    def compute_einstein_coefficient(self, index_config, mu, nu):
        """
        Description
        ===========
        Computes a single Einstein tensor coefficient.

        Example
        =======
        >> newtonian = SpaceTime(Solution().weak_field_approximation(), True)
        >> print(Newtonian.compute_einstein_coefficient("dd",0,0))
        0

        LaTeX Representation
        ====================
        G = R_{mn} - R/2 g_{mn} 

        URL Reference
        =============
        https://en.wikipedia.org/wiki/Einstein_tensor

        TODOs
        =====
        - Link example with test.
        - Need higher quality tests.
        - Needs functionality for other index configurations.
        """

        einstein_coefficient = 0
        if index_config == "dd":
            einstein_coefficient = self.get_ricci_coefficient("dd", mu, nu) - Rational('1/2') * self.get_ricci_scalar() * self.metric_tensor_dd[mu,nu]
            einstein_coefficient = simplify(einstein_coefficient)
        elif index_config == "uu":
            # TODO
            # MUST TEST
            print("")
        elif index_config == "ud" or index_config == "du":
            # TODO
            # MUST TEST
            print("")
        else:
            print("Invalid index_config string.")
        return einstein_coefficient
    
    def print_einstein_coefficient(self, index_config, mu, nu):
        """
        Description
        ===========
        Prints a single Einstein coefficient.

        Example
        =======
        >> newtonian = SpaceTime(Solution().weak_field_approximation(), True)
        >> print(newtonian.print_einstein_coefficient("dd",0,0))

        G₀₀ = 0
        
        LaTeX Representation
        ====================
        G = R_{mn} - R/2 g_{mn} 

        URL Reference
        =============
        https://en.wikipedia.org/wiki/Einstein_tensor

        TODOs
        =====
        - Link example with test.
        - Need higher quality tests.
        - Needs functionality for other index configurations.
        """

        if (index_config == "uu"):
            # TODO
            # MUST TEST
            pprint(Eq(Symbol('G^%s%s' % (mu, nu)), self.get_einstein_coefficient(index_config, mu, nu)))
        elif(index_config == "dd"):
            pprint(Eq(Symbol('G_%s%s' % (mu, nu)), self.get_einstein_coefficient(index_config, mu, nu)))
        else:
            print("Invalid index_config string.")  
    
    def print_all_einstein_coefficients(self, index_config):
        """
        Description
        ===========
        Prints all Einstein coefficients.

        Example
        =======
        >> newtonian = SpaceTime(Solution().weak_field_approximation(), True)
        >> pprint(newtonian.print_all_einstein_coefficients("dd"))


        G₀₀ = 0

        G₀₁ = 0

        G₀₂ = 0

        G₀₃ = 0

        ...

        G₃₂ = 0

        G₃₃ = 0
        
        LaTeX Representation
        ====================
        G = R_{mn} - R/2 g_{mn} 

        URL Reference
        =============
        https://en.wikipedia.org/wiki/Einstein_tensor

        TODOs
        =====
        - Link example with test.
        - Need higher quality tests.
        - Needs functionality for other index configurations.
        """

        for mu in self.dimensions:
            for nu in self.dimensions:
                self.print_einstein_coefficient(index_config, mu, nu)
    
    
    """
    Stress-energy-momentum tensor functions
    =======================================
    """

    def get_stress_energy_coefficient(self, index_config, mu, nu):
        r"""
        Description
        ===========
        Returns a stress-energy coefficient for a given associated index pair and index configuration.

        Example
        =======
        >> newtonian = SpaceTime(Solution().weak_field_approximation(), True)
        >> pprint(newtonian.get_stress_energy_coefficient("dd",0,0))
        0

        LaTeX Representation
        ====================
        T_{mn} = frac{c^{4}}{8 \pi G} G_{mn}

        URL Reference
        =============
        https://en.wikipedia.org/wiki/Stress%E2%80%93energy_tensor

        TODOs
        =====
        - Link example with test.
        - Need higher quality tests.
        - Needs functionality for other index configurations.
        """

        if (index_config == "uu"):
            # TODO
            # MUST TEST
            return self.stress_energy_tensor_uu[mu, nu]
        elif(index_config == "dd"):
            return self.stress_energy_tensor_dd[mu, nu]
        else:
            print("Invalid index_config string.")
    
    def set_stress_energy_coefficient(self, index_config, mu, nu, expression):
        r"""
        Description
        ===========
        Sets a single stress-energy coefficient from class object.

        Example
        =======
        >> newtonian = SpaceTime(Solution().weak_field_approximation(), True)
        >> newtonian.set_stress_energy_coefficient("dd",0,0,0)

        LaTeX Representation
        ====================
        T_{mn} = frac{c^{4}}{8 \pi G} G_{mn}

        URL Reference
        =============
        https://en.wikipedia.org/wiki/Stress%E2%80%93energy_tensor

        TODOs
        =====
        - Link example with test.
        - Need higher quality tests.
        - Needs functionality for other index configurations.
        """

        if (index_config == "uu"):
            # TODO
            # MUST TEST
            self.stress_energy_tensor_uu[mu, nu] = expression
        elif(index_config == "dd"):
            self.stress_energy_tensor_dd[mu, nu] = expression
        else:
            print("Invalid index_config string.")
    
    def set_all_stress_energy_coefficients(self, index_config):
        r"""
        Description
        ===========
        Sets all stress-energy coefficient values for reuse. Allows for the removal of redundant calculations.

        Example
        =======
        >> newtonian = SpaceTime(Solution().weak_field_approximation(), True)
        >> newtonian.set_all_stress_energy_coefficients("dd")

        LaTeX Representation
        ====================
        T_{mn} = frac{c^{4}}{8 \pi G} G_{mn}

        URL Reference
        =============
        https://en.wikipedia.org/wiki/Stress%E2%80%93energy_tensor

        TODOs
        =====
        - Link example with test.
        - Need higher quality tests.
        - Needs functionality for other index configurations.
        """

        if (index_config=="uu"):
            if(self.suppress_printing == False):
                print("")
                print("")
                print("Stress-energy-momentum tensor coefficients (uu)")
                print("===============================================")
            for mu in self.dimensions:
                for nu in self.dimensions:
                    self.set_stress_energy_coefficient(index_config, mu, nu, self.compute_stress_energy_coefficient(index_config, mu, nu))
                    if(self.suppress_printing == False):
                        self.print_stress_energy_coefficient(index_config, mu, nu)
        elif (index_config == "dd"):
            if(self.suppress_printing == False):
                print("")
                print("")
                print("Stress-energy-momentum tensor coefficients (dd)")
                print("===============================================")
            for mu in self.dimensions:
                for nu in self.dimensions:
                    self.set_stress_energy_coefficient(index_config, mu, nu, self.compute_stress_energy_coefficient(index_config, mu, nu))   
                    if(self.suppress_printing == False):
                        self.print_stress_energy_coefficient(index_config, mu, nu)
        else:
            print("Invalid index_config string.")
    
    def compute_stress_energy_coefficient(self, index_config, mu, nu):
        r"""
        Description
        ===========
        Sets all stress-energy coefficient values for reuse. Allows for the removal of redundant calculations.

        Example
        =======
        >> newtonian = SpaceTime(Solution().weak_field_approximation(), True)
        >> print(newtonian.compute_stress_energy_coefficient("dd",0,0))
        0

        LaTeX Representation
        ====================
        T_{mn} = frac{c^{4}}{8 \pi G} G_{mn}

        URL Reference
        =============
        https://en.wikipedia.org/wiki/Stress%E2%80%93energy_tensor

        TODOs
        =====
        - Link example with test.
        - Need higher quality tests.
        - Needs functionality for other index configurations.
        """

        stress_energy_coefficient = 0
        c, G = symbols('c G')
        if index_config == "dd":
            stress_energy_coefficient = c**4/(8*pi*G)*self.get_einstein_coefficient(index_config, mu, nu) + c**4/(8*pi*G) * self.cosmological_constant * self.metric_tensor_dd[mu,nu]
        elif index_config == "uu":
            stress_energy_coefficient = c**4/(8*pi*G)*self.get_einstein_coefficient(index_config, mu, nu)
        elif index_config == "ud" or index_config == "du":
            pass
        else:
            print("Invalid index_config string.")
        return simplify(stress_energy_coefficient)

    def print_stress_energy_coefficient(self, index_config, mu, nu):
        r"""
        Description
        ===========
        Prints a single stress-energy coefficient.

        Example
        =======
        >> newtonian = SpaceTime(Solution().weak_field_approximation(), True)
        >> newtonian.print_stress_energy_coefficient("dd",0,0)
        
        T₀₀ = 0

        LaTeX Representation
        ====================
        T_{mn} = frac{c^{4}}{8 \pi G} G_{mn}

        URL Reference
        =============
        https://en.wikipedia.org/wiki/Stress%E2%80%93energy_tensor

        TODOs
        =====
        - Link example with test.
        - Need higher quality tests.
        - Needs functionality for other index configurations.
        """

        if (index_config == "uu"):
            pprint(Eq(Symbol('T^%s%s' % (mu, nu)), self.get_stress_energy_coefficient(index_config, mu, nu)))
        elif(index_config == "dd"):
            pprint(Eq(Symbol('T_%s%s' % (mu, nu)), self.get_stress_energy_coefficient(index_config, mu, nu)))
        else:
            print("Invalid index_config string.")
    
    def print_all_stress_energy_coefficients(self, index_config):
        r"""
        Description
        ===========
        Prints all stress-energy coefficients.

        Example
        =======
        >> newtonian = SpaceTime(Solution().weak_field_approximation(), True)
        >> newtonian.print_all_stress_energy_coefficients("dd")

        T₀₀ = 0

        T₀₁ = 0

        T₀₂ = 0

        ...
        
        T₃₂ = 0

        T₃₃ = 0

        LaTeX Representation
        ====================
        T_{mn} = frac{c^{4}}{8 \pi G} G_{mn}

        URL Reference
        =============
        https://en.wikipedia.org/wiki/Stress%E2%80%93energy_tensor

        TODOs
        =====
        - Link example with test.
        - Need higher quality tests.
        - Needs functionality for other index configurations.
        """
        for mu in self.dimensions:
            for nu in self.dimensions:
                self.print_stress_energy_coefficient(index_config, mu, nu)
                
    """
    Cosmological constant functions
    ===============================
    """
    def get_cosmological_constant(self):
        r"""
        Description
        ===========
        Prints all stress-energy coefficients.

        Example
        =======
        >> newtonian = SpaceTime(Solution().weak_field_approximation(), True)
        >> print(newtonian.get_cosmological_constant())
        0

        LaTeX Representation
        ====================
        \Lambda 

        URL Reference
        =============
        https://en.wikipedia.org/wiki/Cosmological_constant
        https://en.wikipedia.org/wiki/Friedmann_equations
        https://en.wikipedia.org/wiki/Friedmann%E2%80%93Lema%C3%AEtre%E2%80%93Robertson%E2%80%93Walker_metric
        https://en.wikipedia.org/wiki/Dark_energy
        https://en.wikipedia.org/wiki/Accelerating_expansion_of_the_universe

        TODOs
        =====
        - Link example with test.
        - Need higher quality tests.
        - Needs functionality for other index configurations.
        """
                
        return self.cosmological_constant  
    
    def set_cosmological_constant(self, expression):
        r"""
        Description
        ===========
        Sets cosmological constant from class object.

        Example
        =======
        >> newtonian = SpaceTime(Solution().weak_field_approximation(), True)
        >> print(newtonian.set_cosmological_constant(0))
    
        LaTeX Representation
        ====================
        \Lambda 

        URL Reference
        =============
        https://en.wikipedia.org/wiki/Cosmological_constant
        https://en.wikipedia.org/wiki/Friedmann_equations
        https://en.wikipedia.org/wiki/Friedmann%E2%80%93Lema%C3%AEtre%E2%80%93Robertson%E2%80%93Walker_metric
        https://en.wikipedia.org/wiki/Dark_energy
        https://en.wikipedia.org/wiki/Accelerating_expansion_of_the_universe

        TODOs
        =====
        - Link example with test.
        - Need higher quality tests.
        - Needs functionality for other index configurations.
        """
        self.cosmological_constant = expression
        if(self.suppress_printing == False):
            print("")
            print("")
            print("Cosmological constant")
            print("=====================")
            self.print_cosmological_constant()
            
    def print_cosmological_constant(self):
        r"""
        Description
        ===========
        Sets cosmological constant from class object.

        Example
        =======
        >> newtonian = SpaceTime(Solution().weak_field_approximation(), True)
        >> newtonian.print_cosmological_constant()
    
        LaTeX Representation
        ====================
        \Lambda 

        URL Reference
        =============
        https://en.wikipedia.org/wiki/Cosmological_constant
        https://en.wikipedia.org/wiki/Friedmann_equations
        https://en.wikipedia.org/wiki/Friedmann%E2%80%93Lema%C3%AEtre%E2%80%93Robertson%E2%80%93Walker_metric
        https://en.wikipedia.org/wiki/Dark_energy
        https://en.wikipedia.org/wiki/Accelerating_expansion_of_the_universe

        TODOs
        =====
        - Link example with test.
        - Need higher quality tests.
        - Needs functionality for other index configurations.
        """
        pprint(Eq(Symbol('Lambda'), self.get_cosmological_constant()))


    """
    Proper geodesic functions
    =========================
    """

    def get_proper_time_geodesic_acceleration(self, lam):
        return self.proper_acceleration[lam]

    def set_proper_time_geodesic_acceleration(self, lam, expression):
        self.proper_acceleration[lam] = expression

    def set_all_proper_time_geodesic_accelerations(self):
        if(self.suppress_printing == False):
            print("")
            print("")
            print("Proper acceleration vectors")
            print("============================")
        for lam in self.dimensions:
            self.set_proper_time_geodesic_acceleration(lam, self.compute_proper_time_geodesic_acceleration(lam))
            if(self.suppress_printing == False):
                self.print_proper_time_geodesic_acceleration(lam)

    def compute_proper_time_geodesic_acceleration(self, lam):
        acceleration = 0
        for mu in self.dimensions:
            for nu in self.dimensions:
                acceleration = acceleration + -1*self.get_connection_coefficient("udd",lam,mu,nu)*Derivative(self.coordinate_set[mu],Symbol('tau'))*Derivative(self.coordinate_set[nu],Symbol('tau'))
        return simplify(acceleration)

    def print_proper_time_geodesic_acceleration(self, lam):
        pprint(Eq(Derivative(Derivative(self.coordinate_set[lam],Symbol('tau')),Symbol('tau')), self.get_proper_time_geodesic_acceleration(lam)))

    def print_all_proper_time_geodesic_accelerations(self):
        for lam in self.dimensions:
            self.print_proper_time_geodesic_acceleration(lam)

    """
    Coordinate geodesic functions
    =============================
    """

    def get_coordinate_time_geodesic_acceleration(self, lam):
        return self.coordinate_acceleration[lam]

    def set_coordinate_time_geodesic_acceleration(self, lam, expression):
        self.coordinate_acceleration[lam] = expression

    def set_all_coordinate_time_geodesic_accelerations(self):
        if(self.suppress_printing == False):
            print("")
            print("")
            print("Coordinate acceleration vectors")
            print("===============================")
        for lam in self.dimensions:
            self.set_coordinate_time_geodesic_acceleration(lam, self.compute_coordinate_time_geodesic_acceleration(lam))
            if(self.suppress_printing == False):
                self.print_coordinate_time_geodesic_acceleration(lam)

    def compute_coordinate_time_geodesic_acceleration(self, lam):
        acceleration = 0
        for mu in self.dimensions:
            for nu in self.dimensions:
                acceleration = acceleration + -1*self.get_connection_coefficient("udd",lam,mu,nu)*diff(self.coordinate_set[mu],self.coordinate_set[0])*diff(self.coordinate_set[nu],self.coordinate_set[0])+self.get_connection_coefficient("udd",0,mu,nu)*Derivative(self.coordinate_set[mu],self.coordinate_set[0])*Derivative(self.coordinate_set[nu],self.coordinate_set[0])*Derivative(self.coordinate_set[lam],self.coordinate_set[0])
        return simplify(acceleration)

        # Velocity
        #pprint(Eq(Derivative(self.coordinate_set[lam],self.coordinate_set[0]), integrate(acc,Symbol('t'))))

    def print_coordinate_time_geodesic_acceleration(self, lam):
        pprint(Eq(Derivative(Derivative(self.coordinate_set[lam],self.coordinate_set[0]),self.coordinate_set[0]), self.get_coordinate_time_geodesic_acceleration(lam)))

    def print_all_coordinate_time_geodesic_accelerations(self):
        for lam in self.dimensions:
            self.print_coordinate_time_geodesic_acceleration(lam)

    """
    Geodesic deviation functions
    ============================
    """

    def get_geodesic_deviation_acceleration(self, lam):
        return self.geodesic_deviation_acceleration[lam]

    def set_geodesic_deviation_acceleration(self, lam, expression):
        self.geodesic_deviation_acceleration[lam] = expression

    def set_all_geodesic_deviation_accelerations(self):
        if(self.suppress_printing == False):
            print("")
            print("")
            print("Geodesic deviation vectors")
            print("==========================")
        for lam in self.dimensions:
            self.set_geodesic_deviation_acceleration(lam, self.compute_geodesic_deviation_acceleration(lam))
            if(self.suppress_printing == False):
                self.print_separation_geodesic_acceleration(lam)

    def compute_geodesic_deviation_acceleration(self, lam):
        acceleration = 0
        for mu in self.dimensions:
            acceleration = 0
            for nu in self.dimensions:
                for rho in self.dimensions:
                    for sig in self.dimensions:
                        acceleration = acceleration + self.get_riemann_coefficient("uddd", mu, nu, rho, sig)*Derivative(self.coordinate_set[nu],Symbol('tau'))*Derivative(self.coordinate_set[rho],Symbol('tau'))*Symbol('xi_'+str(sig))  
        return simplify(acceleration)

    def print_separation_geodesic_acceleration(self, lam):
        pprint(Eq(Derivative(Derivative(Symbol('xi_'+str(lam)),Symbol('tau')),Symbol('tau')), self.get_geodesic_deviation_acceleration(lam)))

    def print_all_separation_geodesic_accelerations(self):
        for lam in self.dimensions:
            self.print_separation_geodesic_acceleration(lam)

    """
    schouten tensor functions
    =======================
    """

    def get_schouten_coefficient(self, index_config, mu, nu):
        r"""
        Description
        ===========
        Returns a schouten coefficient for a given associated index pair and index configuration.

        Example
        =======
        >> newtonian = SpaceTime(Solution().weak_field_approximation(), True)
        >> print(newtonian.get_schouten_coefficient("dd",0,0))
        G*M*(2*G*M - c**2*r)*Derivative(t, t)**2/(c**2*r**3)

        LaTeX Representation
        ====================
        P_{ij} = frac{1}{n-2}\left ( R_{ij} - frac{R}{2d-2}\: g_{ij} )

        URL Reference
        =============
        https://en.wikipedia.org/wiki/Stress%E2%80%93energy_tensor

        TODOs
        =====
        - Link example with test.
        - Need higher quality tests.
        - Needs functionality for other index configurations.
        """

        if (index_config == "uu"):
            # TODO
            # MUST TEST
            return self.schouten_tensor_uu[mu, nu]
        elif(index_config == "dd"):
            return self.schouten_tensor_dd[mu, nu]
        else:
            print("Invalid index_config string.")

    def set_schouten_coefficient(self, index_config, mu, nu, expression):
        r"""
        Description
        ===========
        Sets (computes) a schouten coefficient for a given associated index pair and index configuration.

        Example
        =======
        >> newtonian = SpaceTime(Solution().weak_field_approximation(), True)
        >> newtonian.set_schouten_coefficient("dd",0,0,G*M*(2*G*M - c**2*r)*Derivative(t, t)**2/(c**2*r**3)))

        LaTeX Representation
        ====================
        P_{ij} = frac{1}{n-2}\left ( R_{ij} - frac{R}{2d-2}\: g_{ij} )

        URL Reference
        =============
        https://en.wikipedia.org/wiki/Stress%E2%80%93energy_tensor

        TODOs
        =====
        - Link example with test.
        - Need higher quality tests.
        - Needs functionality for other index configurations.
        """
        if (index_config == "uu"):
            self.schouten_tensor_uu[mu, nu] = expression
        elif(index_config == "dd"):
            self.schouten_tensor_dd[mu, nu] = expression
        else:
            print("Invalid index_config string.")

    def set_all_schouten_coefficients(self, index_config):
        if (index_config=="uu"):
            if(self.suppress_printing == False):
                print("")
                print("")
                print("Schouten tensor coefficients (uu)")
                print("=================================")
            for mu in self.dimensions:
                for nu in self.dimensions:
                    self.set_schouten_coefficient(index_config, mu, nu, self.compute_schouten_coefficient(index_config, mu, nu))
                    if(self.suppress_printing == False):
                        self.print_schouten_coefficient(index_config, mu, nu)
        elif (index_config == "dd"):
            if(self.suppress_printing == False):
                print("")
                print("")
                print("Schouten tensor coefficients (dd)")
                print("=================================")
            for mu in self.dimensions:
                for nu in self.dimensions:
                    self.set_schouten_coefficient(index_config, mu, nu, self.compute_schouten_coefficient(index_config, mu, nu))   
                    if(self.suppress_printing == False):
                        self.print_schouten_coefficient(index_config, mu, nu)
        else:
            print("Invalid index_config string.")

    def compute_schouten_coefficient(self, index_config, mu, nu):
        acceleration = 0
        for lam in self.dimensions:
            acceleration = acceleration + -1*self.get_connection_coefficient("udd",lam,mu,nu)*Derivative(self.coordinate_set[mu],self.coordinate_set[0])*Derivative(self.coordinate_set[nu],self.coordinate_set[0])+self.get_connection_coefficient("udd",0,mu,nu)*Derivative(self.coordinate_set[mu],self.coordinate_set[0])*Derivative(self.coordinate_set[nu],self.coordinate_set[0])*Derivative(self.coordinate_set[lam],self.coordinate_set[0])
        return simplify(acceleration)

    def print_schouten_coefficient(self, index_config, mu, nu):
        if (index_config == "uu"):
            pprint(Eq(Symbol('P^%s%s' % (mu, nu)), self.get_schouten_coefficient(index_config, mu, nu)))
        elif(index_config == "dd"):
            pprint(Eq(Symbol('P_%s%s' % (mu, nu)), self.get_schouten_coefficient(index_config, mu, nu)))
        else:
            print("Invalid index_config string.")

    def print_all_schouten_coefficients(self, index_config):
        for mu in self.dimensions:
            for nu in self.dimensions:
                self.print_schouten_coefficient(index_config, mu, nu)

    def plot_ricci_scalar_grid(self, x_range, y_range, x_index=0, y_index=1,
                               num_points=20, save_path=None, dpi=150):
        """
        Generate a PNG showing Ricci scalar sampled on a 2D grid with each cell
        annotated. x_index,y_index select coordinate symbols from self.coordinate_set.
        """
        if save_path is None:
            save_path = os.path.join(os.getcwd(), "ricci_scalar_grid.png")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        x_sym = self.coordinate_set[x_index]
        y_sym = self.coordinate_set[y_index]

        # Numeric function (falls back to constant if expression independent)
        ricci_expr = self.get_ricci_scalar()
        f_ricci = lambdify((x_sym, y_sym), ricci_expr, "numpy")

        x_vals = np.linspace(x_range[0], x_range[1], num_points)
        y_vals = np.linspace(y_range[0], y_range[1], num_points)
        ricci_grid = np.zeros((num_points, num_points), dtype=float)

        for i, xv in enumerate(x_vals):
            for j, yv in enumerate(y_vals):
                try:
                    ricci_grid[j, i] = float(f_ricci(xv, yv))
                except Exception:
                    ricci_grid[j, i] = np.nan

        vmax = np.nanmax(np.abs(ricci_grid))
        if not np.isfinite(vmax) or vmax == 0:
            vmax = 1.0
        vmin = -vmax

        fig, ax = plt.subplots(figsize=(10, 7), dpi=dpi)
        mesh = ax.pcolormesh(x_vals, y_vals, ricci_grid,
                             cmap="seismic", shading="auto",
                             vmin=vmin, vmax=vmax)
        ax.set_xlabel(str(x_sym))
        ax.set_ylabel(str(y_sym))
        ax.set_title("Spacetime Ricci Scalar Curvature (Grid Squares)\nDiverging Colors Show Curvature")

        # Annotate each cell
        for i, xv in enumerate(x_vals[:-1]):
            for j, yv in enumerate(y_vals[:-1]):
                val = ricci_grid[j, i]
                if np.isfinite(val):
                    ax.text(xv + (x_vals[1]-x_vals[0])/2.0,
                            yv + (y_vals[1]-y_vals[0])/2.0,
                            f"{val:0.2e}",
                            ha="center", va="center", fontsize=7, color="black")

        # Draw grid lines
        for xv in x_vals:
            ax.axvline(xv, color="black", linewidth=0.5)
        for yv in y_vals:
            ax.axhline(yv, color="black", linewidth=0.5)

        cbar = fig.colorbar(mesh, ax=ax)
        cbar.set_label("Ricci Scalar")

        fig.tight_layout()
        fig.savefig(save_path)
        plt.close(fig)
        return save_path

    def plot_metric_tensor_grid(self, x_range, y_range, mu=0, nu=0,
                                x_index=0, y_index=1, num_points=20,
                                save_path=None, dpi=150, index_config="dd"):
        """
        Plot selected metric component in grayscale and save as metric_tensor_plot.png
        (unless a custom save_path is provided).
        """
        if save_path is None:
            save_path = os.path.join(os.getcwd(), "metric_tensor_plot.png")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        x_sym = self.coordinate_set[x_index]
        y_sym = self.coordinate_set[y_index]

        if index_config == "dd":
            comp_expr = self.metric_tensor_dd[mu, nu]
        elif index_config == "uu":
            comp_expr = self.metric_tensor_uu[mu, nu]
        else:
            raise ValueError("index_config must be 'dd' or 'uu'.")

        f_comp = lambdify((x_sym, y_sym), comp_expr, "numpy")

        x_vals = np.linspace(x_range[0], x_range[1], num_points)
        y_vals = np.linspace(y_range[0], y_range[1], num_points)
        comp_grid = np.zeros((num_points, num_points), dtype=float)

        for i, xv in enumerate(x_vals):
            for j, yv in enumerate(y_vals):
                try:
                    comp_grid[j, i] = float(f_comp(xv, yv))
                except Exception:
                    comp_grid[j, i] = np.nan

        vmax = np.nanmax(comp_grid)
        vmin = np.nanmin(comp_grid)
        if not np.isfinite(vmax) or not np.isfinite(vmin) or vmax == vmin:
            vmax = 1.0
            vmin = -1.0

        fig, ax = plt.subplots(figsize=(10, 7), dpi=dpi)

        # Use a very light grayscale colormap to improve readability
        import matplotlib as mpl
        base = mpl.cm.Greys
        light_gray = mpl.colors.ListedColormap(base(np.linspace(0.75, 1.0, 256)))

        mesh = ax.pcolormesh(
            x_vals, y_vals, comp_grid,
            cmap=light_gray, shading="auto",
            vmin=vmin, vmax=vmax
        )
        lab = f"g_{mu}{nu}" if index_config == "dd" else f"g^{mu}{nu}"
        ax.set_xlabel(str(x_sym))
        ax.set_ylabel(str(y_sym))
        ax.set_title(f"Metric Component {lab} (Light Grayscale)")

        for i, xv in enumerate(x_vals[:-1]):
            for j, yv in enumerate(y_vals[:-1]):
                val = comp_grid[j, i]
                if np.isfinite(val):
                    ax.text(
                        xv + (x_vals[1] - x_vals[0]) / 2.0,
                        yv + (y_vals[1] - y_vals[0]) / 2.0,
                        f"{val:0.2e}",
                        ha="center", va="center", fontsize=7, color="black",
                        bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.65)
                    )

        for xv in x_vals:
            ax.axvline(xv, color="black", linewidth=0.4)
        for yv in y_vals:
            ax.axhline(yv, color="black", linewidth=0.4)

        cbar = fig.colorbar(mesh, ax=ax)
        cbar.set_label(lab)

        fig.tight_layout()
        fig.savefig(save_path)
        plt.close(fig)
        return save_path

    def plot_schwarzschild_embedding(self, r_range=(2.1, 10), num_r=300, num_phi=180,
                                     save_path="mnt/data/schwarzschild_embedding.png",
                                     G_val=1.0, M_val=1.0, c_val=1.0):
        """
        Generates a 3D embedding diagram (equatorial slice) for the Schwarzschild spatial geometry.
        Uses metric: ds^2 = (1 - 2GM/(c^2 r))^-1 dr^2 + r^2 dphi^2 at theta = pi/2, t = const.
        Embedding condition (Euclidean): dz^2 + dr^2 = (1 - 2GM/(c^2 r))^-1 dr^2 -> dz/dr = sqrt(A - 1),
        where A = (1 - 2GM/(c^2 r))^-1.
        """
        r_min, r_max = r_range
        G = G_val
        M = M_val
        c = c_val

        # Avoid horizon singularity
        horizon = 2 * G * M / (c ** 2)
        if r_min <= horizon:
            r_min = horizon * 1.05

        r = np.linspace(r_min, r_max, num_r)
        phi = np.linspace(0, 2 * np.pi, num_phi)
        Phi, R = np.meshgrid(phi, r)

        f = 1.0 - 2.0 * G * M / (c ** 2 * r)
        A = 1.0 / f
        dz_dr = np.sqrt(np.clip(A - 1.0, 0.0, None))

        # Numerical integration for z(r)
        z = np.zeros_like(r)
        dr = np.diff(r)
        for i in range(1, len(r)):
            z[i] = z[i - 1] + 0.5 * (dz_dr[i] + dz_dr[i - 1]) * dr[i - 1]

        Z = np.tile(z, (num_phi, 1)).T
        X = R * np.cos(Phi)
        Y = R * np.sin(Phi)

        # Color by radial curvature factor A
        curvature = np.tile(A, (num_phi, 1)).T

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        # Use a Normalize so the surface facecolors and colorbar match
        from matplotlib import cm, colors
        norm = colors.Normalize(vmin=np.nanmin(curvature), vmax=np.nanmax(curvature))
        facecolors = cm.plasma(norm(curvature))

        surf = ax.plot_surface(
            X, Y, Z,
            facecolors=facecolors,
            rstride=2, cstride=2, linewidth=0, antialiased=True
        )

        # Bind the ScalarMappable to this Axes so colorbar can steal space
        mappable = cm.ScalarMappable(norm=norm, cmap=cm.plasma)
        mappable.set_array(curvature)
        cbar = fig.colorbar(mappable, ax=ax, shrink=0.65, pad=0.08)
        cbar.set_label('Radial curvature factor (1 - 2GM/(c^2 r))^-1')

        ax.set_title('Schwarzschild Embedding Diagram (Equatorial Slice)')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z (Embedded)')

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()

# Example: Add this to your main() function or before exit
from sympy import pprint

def main():
    blackhole_solution = Solution().schwarzschild()
    blackhole_spacetime = SpaceTime(blackhole_solution)

    print("Metric tensor (dd):")
    pprint(blackhole_spacetime.metric_tensor_dd)
    print("\nMetric tensor (uu):")
    pprint(blackhole_spacetime.metric_tensor_uu)
    print("\nRicci tensor (dd):")
    pprint(blackhole_spacetime.ricci_tensor_dd)
    print("\nRicci scalar:")
    pprint(blackhole_spacetime.ricci_scalar)

    # Before plotting, for demonstration:
    blackhole_spacetime.metric_tensor_dd[1, 1] = sin(blackhole_spacetime.coordinate_set[1]) * cos(blackhole_spacetime.coordinate_set[2])

    # Your plot call
    blackhole_spacetime.plot_metric_tensor_grid(
    x_range=(2, 200), y_range=(0, 180), mu=1, nu=1, x_index=1, y_index=2, num_points=10,
    save_path="mnt/data/metric_tensor_plot.png"
)

    # New: Embedding diagram (curved space visualization)
    blackhole_spacetime.plot_schwarzschild_embedding(
        r_range=(2.5, 15), num_r=400, num_phi=240,
        save_path="mnt/data/schwarzschild_embedding.png",
        G_val=1.0, M_val=1.0, c_val=1.0
    )

    # Example usage (adjust Solution / indices as needed)
    st = SpaceTime(Solution().weak_field_approximation(), suppress_printing=True)
    st.plot_ricci_scalar_grid(x_range=(0, 200), y_range=(0, 200), x_index=1, y_index=2, num_points=10)
    st.plot_metric_tensor_grid(x_range=(0, 200), y_range=(0, 200), mu=0, nu=0, x_index=1, y_index=2, num_points=10)

sys.settrace(trace_calls) # Start tracing function calls

if __name__ == "__main__":
    main()

sys.settrace(None)  # Stop tracing after main

