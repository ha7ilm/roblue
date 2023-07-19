import sympy
import sympybotics
from math import *
import random
random.seed(10)
rbtdef = sympybotics.RobotDef('Weigand', # robot name
    [ # (alpha,   a,     d,      theta)
        ('pi/2',  0.35,  -0.675, 'q'), 
        (0,       1.15,  -0.189, 'q'), 
        ('-pi/2', 0.041, 0.189,  'q+(pi/2)'), 
        ('pi/2',  0,     -1,     'q'),
        ('-pi/2', 0,     0,      'q'),
        ('pi',    0,     -0.24,  'q'),
    ], dh_convention='standard' # either 'standard' or 'modified' #TODO the handling of this offset can be opposite wrt. Peter Corke
    )
#
rbtdef.frictionmodel = None #{'Coulomb', 'viscous','offset'} # options are None or a combination of 'Coulomb', 'viscous' and 'offset'
rbtdef.gravityacc = sympy.Matrix([0.0, 0.0, 9.81]) #we flipped the gravity upside down because the DH parameters correspond to the robot fixed to the ceiling, default is [0.0, 0.0, -9.81]
rbtdef.driveinertiamodel = 'simplified'
#T_base = sympy.Matrix([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
#rbtdef_geo = sympybotics.geometry.Geometry(rbtdef)
#rbtdef_geo.T[-1]= T_base*rbtdef_geo.T[-1]

def to_file(x, filename):
    with open(filename, 'w') as f:
        f.write(x)

rbt = sympybotics.RobotDynCode(rbtdef, verbose=True)
#rbt.calc_base_parms()

to_file(str(rbtdef.dynparms()),'sympybotexp_dynparms')
#to_file(str(rbt.dyn.baseparms),'sympybotexp_baseparms')
to_file(sympybotics.robotcodegen.robot_code_to_func('py', rbt.C_code, 'C_out', 'C', rbtdef),'sympybotexp_C_code.py')
to_file(sympybotics.robotcodegen.robot_code_to_func('py', rbt.M_code, 'M_out', 'M', rbtdef),'sympybotexp_M_code.py')
to_file(sympybotics.robotcodegen.robot_code_to_func('py', rbt.g_code, 'g_out', 'g', rbtdef),'sympybotexp_g_code.py')
to_file(sympybotics.robotcodegen.robot_code_to_func('py', rbt.H_code, 'H_out', 'H', rbtdef),'sympybotexp_H_code.py')
#to_file(sympybotics.robotcodegen.robot_code_to_func('py', rbt.Hb_code, 'Hb_out', 'Hb', rbtdef),'sympybotexp_Hb_code.py')
#to_file(sympybotics.robotcodegen.robot_code_to_func('py', [(),rbt.dyn.baseparms], 'bp_out', 'bp', rbtdef),'sympybotexp_baseparams_code.py')
to_file(f"M_shape={rbt.dyn.M.shape}\nC_shape={rbt.dyn.C.shape}\ng_shape={rbt.dyn.g.shape}\nH_shape={rbt.dyn.H.shape}\ndynparms_len={rbtdef.dynparms().__len__()}",'sympybotexp_shapes_code.py')

print('every ending is a new beginning')
