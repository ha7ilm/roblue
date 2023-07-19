import os

for the_dir in sorted(os.listdir('..')):
    if the_dir.startswith('exp_'):
        i = the_dir.split('exp_')[1]
        if len(i)<3: continue
        basedir = '../exp_' + i
        print(basedir)
        #delete any files in basedir that are not a symbolic link:
        for file in os.listdir(basedir):
            if not os.path.islink(basedir + '/' + file):
                #delete recursively the file or directory:
                if os.path.isdir(basedir + '/' + file):
                    os.system('rm -rf ' + basedir + '/' + file)
                else:
                    os.remove(basedir + '/' + file)
