import json


class ErrorMeasurements():
    def __init__(self,num_train_samples,test_errors):
        self.num_train_samples = num_train_samples
        self.test_errors = test_errors
        self.num_ms = len(self.test_errors)

    def __str__(self):
        return_str = \
            'num_train_samples: ' + str(self.num_train_samples) + '\n' + \
            'test_errors: ' + str(self.test_errors) + '\n' + \
            'num_ms: ' + str(self.num_ms) + '\n'
        return return_str


class CurveMeasurements():
    def __init__(self):
        self.curvems = []
        
    def load_from_json(self,json_path):
        for m in json.load(open(json_path,'r')):
            errms = ErrorMeasurements(
                m['num_train_samples'],
                m['test_errors'])
            self.curvems.append(errms)
        
        self.curvems = sorted(
            self.curvems,
            key=lambda errms: errms.num_train_samples)

    def __len__(self):
        return len(self.curvems)

    def __getitem__(self,i):
        return self.curvems[i]

    def __str__(self):
        return_str = '-'*2 + '\n'
        for errms in self:
            return_str = return_str + str(errms) + '-'*2 + '\n'
        
        return return_str

    def get_train_dataset_sizes(self):
        return [errms.num_train_samples for errms in self]

    def get_errms(self,num_train_samples):
        for errms in self:
            if errms.num_train_samples==num_train_samples:
                return errms
        
        err_msg = 'Error measurements not found for ' + \
            f'num_train_samples={num_train_samples}'
        assert(False), err_msg
        
    def get_num_ms(self,num_train_samples=None):
        num_ms = 0
        if num_train_samples is None:
            for errms in self:
                num_ms += errms.num_ms
            
            return num_ms

        return self.get_errms(num_train_samples)
        
