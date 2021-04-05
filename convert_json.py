import os
import json
import ipdb


def convert(filepath):
    data = json.load(open(filepath,'r'))
    measurements = []
    for n, fold_errors in data['test_errors'].items():
        n = int(n)
        errors = list(fold_errors.values())
        measurements.append({
            'num_train_samples': n,
            'test_errors': errors 
        })
    
    return measurements


def main():
    basepath = '/Users/tanmayg/Downloads/learning_curve_results/effect_of_pretr'
    filenames = ['no_pretr_ft.json','pretr_ft.json','no_pretr_linear.json','pretr_linear.json']
    for filename in filenames:
        measurements = convert(os.path.join(basepath,filename))
        tgt_filename = os.path.join('./data',filename)
        with open(tgt_filename,'w') as f:
            json.dump(measurements,f,indent=4,sort_keys=True)


if __name__=='__main__':
    main()
