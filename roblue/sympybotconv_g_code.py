from torch import *
def g(parms, q, tgt_device='cuda'):
    # type: (Tensor, Tensor, str) -> Tensor 
    g_out = zeros((q.shape[0],6), device=tgt_device)

    x0 = sin(q[:,3])
    x1 = cos(q[:,3])
    x2 = cos(q[:,1])
    x3 = -9.81*x2
    x4 = cos(q[:,2])
    x5 = sin(q[:,2])
    x6 = sin(q[:,1])
    x7 = -9.81*x6
    x8 = -x7
    x9 = x3*x4 + x5*x8
    x10 = x1*x9
    x11 = sin(q[:,4])
    x12 = -x11
    x13 = -x5
    x14 = x13*x3 + x4*x8
    x15 = cos(q[:,4])
    x16 = x10*x12 + x14*x15
    x17 = -x16
    x18 = parms[53]*x16 - parms[64]*x17
    x19 = sin(q[:,5])
    x20 = cos(q[:,5])
    x21 = x0*x9
    x22 = -x21
    x23 = -x22
    x24 = x10*x15 + x11*x14
    x25 = x19*x24 + x20*x23
    x26 = parms[64]*x25
    x27 = x19*x22 + x20*x24
    x28 = parms[64]*x27
    x29 = x20*x28
    x30 = parms[53]*x24 + x19*x26 + x29
    x31 = parms[42]*x10 + x12*x18 + x15*x30
    x32 = -x20
    x33 = x19*x28
    x34 = parms[42]*x21 - parms[53]*x22 - x26*x32 - x33
    x35 = -x34
    x36 = x0*x31 + x1*x35
    x37 = -x36
    x38 = parms[31]*x9 + x0*x34 + x1*x31
    x39 = parms[31]*x14 + parms[42]*x14 + x11*x30 + x15*x18
    x40 = -x4
    x41 = parms[20]*x3 + x13*x39 + x38*x4
    x42 = 0.041*x4**2 + 0.041*x5**2
    x43 = x36*x42
    x44 = -0.24*x26
    x45 = parms[62]*x17 - parms[63]*x25
    x46 = -parms[61]*x17 + parms[63]*x27
    x47 = parms[51]*x16 + parms[52]*x23 + x19*x46 + x20*x44 + x20*x45 + 0.24*x33
    x48 = -parms[61]*x25 + parms[62]*x27
    x49 = parms[50]*x22 - parms[51]*x24 + x48
    x50 = -x14
    x51 = parms[40]*x21 + parms[41]*x50 + x12*x49 + x15*x47
    x52 = -parms[50]*x17 - parms[52]*x24 - x19*x44 - x19*x45 + 0.24*x29 - x32*x46
    x53 = parms[39]*x14 - parms[40]*x10 + x52
    x54 = parms[29]*x14 + x0*x53 + x1*x51 + x36
    x55 = -0.189*x4
    x56 = parms[39]*x22 + parms[41]*x10 + x11*x47 + x15*x49
    x57 = -parms[29]*x9 + x56
    x58 = -x1
    x59 = -parms[28]*x50 - parms[30]*x9 - x0*x35 - x0*x51 - x31*x58 - x53*x58
#
    g_out[:,0] = x2*x37*(-1.15*x2**2 - 1.15*x6**2) - 0.189*x2*(parms[20]*x7 + x13*x38 + x39*x40) + x2*(parms[19]*x7 + x13*x43 + x13*x57 - 0.189*x38*x5 + x39*x55 + x4*x54) - 0.35*x37 + 0.189*x41*x6 + x6*(-parms[19]*x3 + x13*x54 + x38*x55 + 0.189*x39*x5 + x40*x43 + x40*x57)
    g_out[:,1] = parms[17]*x3 + parms[18]*x8 + x39*x42 + 1.15*x41 + x59
    g_out[:,2] = 0.041*x39 + x59
    g_out[:,3] = x56
    g_out[:,4] = x52
    g_out[:,5] = x48
#
    return g_out