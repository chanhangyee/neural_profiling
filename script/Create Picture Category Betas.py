import sys
sys.path.append('/usr/lib/python2.7/dist-packages')
sys.path.append('/home/ubuntu/anaconda2/lib/python2.7/site-packages')
sys.path.append('/usr/local/lib/python2.7/dist-packages/')

import os, sys, time, glob, shutil

import numpy as np
import nipype
import nipype.interfaces.io as nio           # Data i/o
import nipype.interfaces.spm as spm          # spm
import nipype.interfaces.fsl as fsl    # fsl
import nipype.interfaces.utility as util     # utility
import nipype.pipeline.engine as pe          # pypeline engine
import nipype.algorithms.rapidart as ra      # artifact detection
import nipype.algorithms.modelgen as model   # model specification
import nipype.interfaces.nipy.preprocess as preprocess

sys.path.append('/data/hangyee')
import nipype_init

nipype_init.init()

TR = 2.3
method = 'native'

def ensure_dir(ed):
    try: 
        os.makedirs(ed)
    except OSError:
        if not os.path.isdir(ed):
            raise

import commands
ip_addr = commands.getoutput("/sbin/ifconfig").split("\n")[1].split()[1][5:]
np.random.seed(int(ip_addr.split('.')[-1]))
            
os.chdir('/data/hangyee/brand_image/script/spm')
data_dir = '/data/hangyee/brand_image/'
output_dir = '/data/hangyee/brand_image/spm/'+method+'/picture/'

def get_picture_category_info(subj):
    from nipype.interfaces.base import Bunch
    import scipy.io as spio 
    
    subjectinfo = []

    log_file = data_dir+'log/S'+str(subj)+'_picture.csv'
    nuisance_file = data_dir+'nuisance/S'+str(subj)+'_picture_nuisance.txt'
    brightness_file = data_dir+'nuisance/S'+str(subj)+'_picture_brightness.txt'
    contrast_file = data_dir+'nuisance/S'+str(subj)+'_picture_contrast.txt'

    with open(nuisance_file) as f:
        nuisance = f.readlines()
    reg1 = []
    reg2 = []
    for nuisance_line in nuisance:
        a = nuisance_line.split(' ')
        if len(a) == 2:
            reg1.append(float(a[0]))
            reg2.append(float(a[1]))
            
    with open(brightness_file) as f:
        brightness = f.readlines()
    brightness_reg = [np.log(float(b)+1) for b in brightness]

    #with open(contrast_file) as f:
    #    contrast = f.readlines()
    #contrast_reg = [float(c) for c in contrast]

    log = np.recfromcsv(log_file,delimiter=',')
    
    condition_names = np.unique(log.category)
    durations = [[] for i in range(len(condition_names))]
    onsets = [[] for i in range(len(condition_names))]
    
    for log_line in log:
        idx = np.ravel(np.where(condition_names == log_line.category))[0]
        onsets[idx].append(log_line.onset)
        durations[idx].append(float(7))
            
    subjectinfo.append(Bunch(conditions=condition_names.tolist(),
                                 onsets=onsets,
                                 durations=durations,
                                 regressor_names=['R1','R2','R3'],
                                 regressors=(list(reg1),list(reg2),brightness_reg)))
                             
                                 #regressor_names=['R1','R2','R3','R4'],
                                 #regressors=(list(reg1),list(reg2),brightness_reg,contrast_reg)))
        
    return subjectinfo

## Decide which task to do
def next_subj():
    
    time.sleep(np.random.random()*5)
    log_list = os.listdir(output_dir + 'log')
    
    subj_list = np.arange(4,42)
    
    for l in log_list:
        if not l.startswith('.'):
            subj_list = subj_list[subj_list != int(l)]
    
    if len(subj_list) > 0:
        np.random.shuffle(subj_list)

        subj_choice = subj_list[0]
        sys.stdout = open(output_dir+'log/'+('%02d'%subj_choice), 'w')
    
        return subj_choice
    else:
        return -1

subj = next_subj()
print ip_addr

while subj != -1:
    
    ensure_dir(output_dir+'S'+str(subj))
    ensure_dir(output_dir+'S'+str(subj)+'/by_category')
    
    starttime = time.time()

    print "S%d"%subj,
    sys.stdout.flush()

    for f in glob.glob(output_dir+'S'+str(subj)+'/by_category/*.mat'):
        os.remove(f)

    for f in glob.glob(output_dir+'S'+str(subj)+'/by_category/*.nii'):
        os.remove(f)
        
    os.chdir(output_dir+'S'+str(subj)+'/by_category')
        
    print "Specify model",
    sys.stdout.flush()
    
    modelspec = model.SpecifySPMModel()

    modelspec.inputs.input_units='secs'
    modelspec.inputs.output_units='secs'
    modelspec.inputs.time_repetition=TR
    modelspec.inputs.high_pass_filter_cutoff=128
    modelspec.inputs.functional_runs = [data_dir+'nifti/'+method+'/picture/S'+str(subj)+'_picture_'+method+'.nii']
    modelspec.inputs.subject_info = get_picture_category_info(subj)

    out = modelspec.run()

    print "- Design",
    sys.stdout.flush()
    
    level1design = spm.Level1Design()
    level1design.inputs.timing_units = 'secs'
    level1design.inputs.interscan_interval = TR
    level1design.inputs.bases = {'hrf':{'derivs': [0,0]}}
    level1design.inputs.model_serial_correlations = 'AR(1)'
    level1design.inputs.session_info = out.outputs.session_info

    out = level1design.run()

    #shutil.move(out.outputs.spm_mat_file,output_dir+'S'+str(subj)+'/by_category')

    print "- Estimate",
    sys.stdout.flush()
    
    level1estimate = spm.EstimateModel()
    level1estimate.inputs.estimation_method = {'Classical': 1}
    level1estimate.inputs.spm_mat_file = output_dir+'S'+str(subj)+'/by_category/SPM.mat'

    out = level1estimate.run()
    
    print "- Contrast",
    level1contrast = spm.EstimateContrast()
    level1contrast.inputs.spm_mat_file = out.outputs.spm_mat_file
    level1contrast.inputs.beta_images = out.outputs.beta_images
    level1contrast.inputs.residual_image = out.outputs.residual_image
    cont1 = ('family>party','T', ['family','party'],[1,-1])
    cont2 = ('family>sex','T', ['family','sex'],[1,-1])
    cont3 = ('family>work','T', ['family','work'],[1,-1])
    cont4 = ('party>sex','T', ['party','sex'],[1,-1])
    cont5 = ('party>work','T', ['party','work'],[1,-1])
    cont6 = ('sex>work','T', ['sex','work'],[1,-1])
    contrasts = [cont1,cont2,cont3,cont4,cont5,cont6]
    level1contrast.inputs.contrasts = contrasts

    out = level1contrast.run()

    print "- %f"%(time.time()-starttime)
    
    subj = next_subj()
        
os.system('poweroff')