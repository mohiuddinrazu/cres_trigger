#!/gpfs/loomis/apps/avx/software/miniconda/4.7.10/bin/python

import os
import numpy as np
import argparse
import configparser

config = configparser.ConfigParser()
config.read("../clusterCavity.cfg")


def GenerateDirectory(DIRNAME):
    if(os.path.isdir(DIRNAME)):
        print('The Directory by name %s already exists'%DIRNAME)
        return 
    else:
        try:
            os.mkdir(DIRNAME)
        except OSError:
            print("Creation of the directory %s failed" % DIRNAME)
        else:
            print ("Successfully created the directory %s " % DIRNAME)
        return

###----------------------------------- EDIT LINES --------------------------------------##
VERSION='VCCA_trap_LD_8_with_extent_V01'
NSUBARRAYS=1
LOFREQUENCY=25970200000.0
angles=np.linspace(86.0,90.0,2)
radialPositions=np.linspace(0.0,0.007,16)

RUNNAME=f'Trap_{VERSION}_PRtest'
WORKINGDIR=os.path.join(config.get("data", "outputdir"), RUNNAME, 'results')
MANAGEDIR=os.path.join(config.get("data", "kassLocustSimsdir"),f'Cavity_{VERSION}')
HEXBUGDIR=config.get("data", "hexbugdir")
HEXBUGDIR_CONTAINER=config.get("data","container_hexbug") 
GEOMETRYFILE=os.path.join(HEXBUGDIR_CONTAINER, "Phase3", "Trap", f'CavityGeometry_{VERSION}.xml')
LOCUSTEXECUTABLE='LocustSim'
KATYDIDEXECUTABLE='Katydid'
CONTAINER=config.get("software", "singularity_container_locust")
P8CONTAINER=config.get("software", "singularity_container_p8compute")
#SRCFILE1=os.path.join(config.get("software", "locustdir"), "bin", "kasperenv.sh")
SRCFILE1=os.path.join(config.get("software", "kassiopeiadir"), "bin", "kasperenv.sh")
SRCFILE2=os.path.join(config.get("software", "p8computedir"), "setup.sh")
SRCFILE3=os.path.join(config.get("software", "locustdir"), "setup.sh")
LOCALOUTPUTDIR=config.get("data", "container_output")
###----------------------------------- EDIT LINES --------------------------------------##

#seeds=[int(np.round(seed*10)-500) for seed in angles]
seeds=np.linspace(600,600,1,dtype='int')



print(f'Working from the directory {WORKINGDIR} and manage directory {MANAGEDIR}')
DSQFILE=os.path.join(WORKINGDIR, f'joblist_{RUNNAME}.txt')
SBATCHLISTFILE=os.path.join(WORKINGDIR, f'sbatch_{RUNNAME}.sh')
JSONTEMPFILENAME=os.path.join(MANAGEDIR, 'LocustCavityConfig.json')
XMLTEMPFILENAME=os.path.join(MANAGEDIR, 'Kass_config_P8_Cavity_Template.xml')
YAMLTEMPFILENAME=os.path.join(MANAGEDIR, 'KatydidCavityTemplate.json')

#Create working directory if it doesn't exist
GenerateDirectory(os.path.join(config.get("data", "outputdir"), RUNNAME))
GenerateDirectory(WORKINGDIR)

print('Writing file '+SBATCHLISTFILE)
dsqbf = open(DSQFILE,'w')
sbf = open(SBATCHLISTFILE,'w')
sbf.write('#!/bin/bash\n')
for angle, seed in zip(angles, seeds):
        VRANGE=4.00
        for RADIALPOSITION in radialPositions:
            jobdir = os.path.join(WORKINGDIR, f"Seed{seed}_Angle{angle:0.2f}_Pos{RADIALPOSITION:0.3f}")
            GenerateDirectory(jobdir)
            JSONFILENAME=os.path.join(jobdir, 'LocustElectronSimulation.json')
            YAMLFILENAME=os.path.join(jobdir, 'Katydid.json')
            XMLFILENAME=os.path.join(jobdir, 'Project8Cavity_KassParameters.xml')
            tempDict = {"$versionnumber":         VERSION, 
                        "$radialPosition":        RADIALPOSITION,
                        "$pitchangle":            angle,
                        "$seed":                  seed,
                        "$lofrequency":           LOFREQUENCY,
                        "$vrange":                VRANGE, 
                        "$voffset":               -VRANGE/2,
                        "$locustexec":            LOCUSTEXECUTABLE,
                        "$katydidexec":           KATYDIDEXECUTABLE,
                        "$geometryfilename":GEOMETRYFILE,
                        "$jsonfilename":os.path.join(LOCALOUTPUTDIR, 'LocustElectronSimulation.json'),
                        "$yamlfilename":os.path.join(LOCALOUTPUTDIR, 'Katydid.json'),
                        "$xmlfilename":os.path.join(LOCALOUTPUTDIR, 'Project8Cavity_KassParameters.xml'), 
                        "$pitchanglefilename":os.path.join(LOCALOUTPUTDIR, f'pitchangles_Seed{seed}_Angle{angle:0.2f}_Pos{RADIALPOSITION:0.3f}.txt'),
                        "$eggfilename":os.path.join(LOCALOUTPUTDIR, f'locust_mc_Seed{seed}_Angle{angle:0.2f}_Pos{RADIALPOSITION:0.3f}.egg'),
                        "$katydidoutputfilename":os.path.join(LOCALOUTPUTDIR, f'KatydidOutput_Angle{angle:0.2f}_Pos{RADIALPOSITION:0.3f}.root'),
                        "$katydidbasicfilename":os.path.join(LOCALOUTPUTDIR, f'basic_Angle{angle:0.2f}_Pos{RADIALPOSITION:0.3f}.root'),
                        "$hexbugdir":os.path.join(HEXBUGDIR_CONTAINER, "hexbug"),
                       }

            # Locust Config
            print('Writing file '+JSONFILENAME)
            content = open(JSONTEMPFILENAME, 'r').read()
            for k, v in tempDict.items():
               content = content.replace(k, str(v))
            with open(JSONFILENAME, 'w') as open_file:
                open_file.write(content)

            # Kassiopeia Config
            print('Writing file '+XMLFILENAME)
            jf = open(XMLFILENAME,'w')
            for line in open(XMLTEMPFILENAME,'r'):
                for key in tempDict:
                    line = line.replace(key,str(tempDict[key]))
                jf.write(line)
            jf.close()

            # Katydid config
            print('Writing file '+YAMLFILENAME)
            yf = open(YAMLFILENAME,'w')
            for line in open(YAMLTEMPFILENAME,'r'):
                for key in tempDict:
                    line = line.replace(key,str(tempDict[key]))
                yf.write(line)
            yf.close()


            JOBFILENAME=os.path.join(jobdir, 'JOB.sh')
            print('Writing file '+JOBFILENAME)
            LOCALCMDFILE1='localcmd1.sh'
            LOCALCMDFILE2='localcmd2.sh'
            with open(JOBFILENAME,'w') as bf:
                bf.write('#!/bin/bash\n')
                bf.write(f'#SBATCH -J Seed{seed}_Angle{angle:0.2f}_Pos{RADIALPOSITION:0.3f}\n')
                bf.write(f'#SBATCH -o {jobdir}/run_singularity.out\n')
                bf.write(f'#SBATCH -e {jobdir}/run_singularity.err\n')
                bf.write('#SBATCH -p day\n')
                bf.write('#SBATCH -t 20:00:00\n')
                bf.write('#SBATCH --cpus-per-task=2\n')
                bf.write('#SBATCH --ntasks=1\n')
                bf.write('#SBATCH --mem-per-cpu=15000\n')
                bf.write('#SBATCH --requeue\n')
                bf.write('\n')
                bf.write(f'cd {jobdir}\n')
                dsqbf.write(f'cd {jobdir}; ')
                bf.write(f'echo \'#!/bin/bash\' > {LOCALCMDFILE1}\n')
                dsqbf.write(f'echo \'#!/bin/bash\' > {LOCALCMDFILE1}; ')
                bf.write(f'echo source {SRCFILE1} >> {LOCALCMDFILE1}\n')
                dsqbf.write(f'echo source {SRCFILE1} >> {LOCALCMDFILE1}; ')
                bf.write(f'echo source {SRCFILE3} >> {LOCALCMDFILE1}\n')
                dsqbf.write(f'echo source {SRCFILE3} >> {LOCALCMDFILE1}; ')
                bf.write(f'echo \'#!/bin/bash\' > {LOCALCMDFILE2}\n')
                dsqbf.write(f'echo \'#!/bin/bash\' > {LOCALCMDFILE2}; ')
                bf.write(f'echo source {SRCFILE2} >> {LOCALCMDFILE2}\n')
                dsqbf.write(f'echo source {SRCFILE2} >> {LOCALCMDFILE2}; ')
                bf.write('\n')
                bf.write(f'echo exec {LOCUSTEXECUTABLE} config={os.path.basename(JSONFILENAME)} >> {LOCALCMDFILE1}\n')
                dsqbf.write(f'echo exec {LOCUSTEXECUTABLE} config={os.path.basename(JSONFILENAME)} >> {LOCALCMDFILE1}; ')
                bf.write(f'echo exec {KATYDIDEXECUTABLE} -c {os.path.basename(YAMLFILENAME)} >> {LOCALCMDFILE2}\n')
                dsqbf.write(f'echo exec {KATYDIDEXECUTABLE} -c {os.path.basename(YAMLFILENAME)} >> {LOCALCMDFILE2}; ')
                bf.write('\n')
                bf.write(f'chmod +x {LOCALCMDFILE1}\n')
                dsqbf.write(f'chmod +x {LOCALCMDFILE1}; ')
                bf.write(f'chmod +x {LOCALCMDFILE2}\n')
                dsqbf.write(f'chmod +x {LOCALCMDFILE2}; ')
                bf.write('\n')
                bf.write('date>locuststarttime.txt\n')
                dsqbf.write('date>locuststarttime.txt; ')
                bf.write(f'singularity exec --no-home --bind {HEXBUGDIR}:{HEXBUGDIR_CONTAINER},{jobdir}:{LOCALOUTPUTDIR} {CONTAINER} ./{LOCALCMDFILE1}')
                dsqbf.write(f'singularity exec --no-home --bind {HEXBUGDIR}:{HEXBUGDIR_CONTAINER},{jobdir}:{LOCALOUTPUTDIR} {CONTAINER} ./{LOCALCMDFILE1}; ')
                bf.write('\n')                
                bf.write('date>katydidstarttime.txt\n')
                dsqbf.write('date>katydidstarttime.txt; ')
                bf.write(f'singularity exec --no-home --bind {HEXBUGDIR}:{HEXBUGDIR_CONTAINER},{jobdir}:{LOCALOUTPUTDIR} {P8CONTAINER} ./{LOCALCMDFILE2}')
                dsqbf.write(f'singularity exec --no-home --bind {HEXBUGDIR}:{HEXBUGDIR_CONTAINER},{jobdir}:{LOCALOUTPUTDIR} {P8CONTAINER} ./{LOCALCMDFILE2};\n ')

            # Add to array submission
            sbf.write('sbatch %s\n'%JOBFILENAME)
sbf.close()
dsqbf.close()
print("DONE!")
print("\n")
print(f"run as:\n module load dSQ\n dsq --job-file {DSQFILE} -p scavenge --max-jobs 200 --requeue --cpus-per-task=2 --mem-per-cpu 15g  -t 10:00:00 --mail-type ALL")
