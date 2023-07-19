import os

for i in [
    
    "r01_0.0_1000",                        "r51_0.0_1000",    "ra1_0.0_1000",    "rb1_0.0_1000",    "rc1_0.0_100",   "rd1_0.00_100",     
    "r01_0.2_1000",    "r11_0.0_1000",     "r51_0.2_1000",    "ra1_0.2_1000",    "rb1_0.5_1000",    "rc1_0.01_100",  "rd1_0.05_100",     
    "r01_0.5_1000",    "r11_1.0_1000",     "r51_0.5_1000",    "ra1_0.5_1000",    "rb1_1.0_1000",    "rc1_0.1_100",   "rd1_0.01_100",     
    "r01_1.0_1000",    "r11_5.0_1000",     "r51_1.0_1000",    "ra1_1.0_1000",    "rb1_2.0_1000",    "rc1_0.5_100",   "rd1_0.02_100",     
    "r01_5.0_1000",    "r11_10.0_1000",                       "ra1_2.0_1000",    "rb1_5.0_1000",    "rc1_1.0_100",   "rd1_0.10_100",     
                                                                                                    "rc1_10.0_100",  "rd1_0.50_100",     
    ]:
    basedir = '../exp_' + i
    if os.path.exists(basedir): 
        print("the dir already exists, skipping: "+basedir)
        continue
    else: 
        print("creating: "+basedir)
    os.mkdir(basedir)
    symlinked_files = ['roblue.py', 'sympybotconv_C_code.py', 'sympybotconv_g_code.py', 'sympybotconv_M_code.py', 'gravload_150.py', 'inertia_150.py', 'coriolis_150.py','elabee_robot_ftsd.mat','startroblue.sh']
    #create symbolic links to these files from ../deepnn-gerben-vscode-5 to the current directory
    for file in symlinked_files:
        os.symlink('../roblue/' + file, basedir + '/' + file)
