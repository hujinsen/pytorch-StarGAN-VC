import numpy as np
import pyworld as pw
import os,shutil
import glob
import librosa

class Singleton(type):

    def __init__(self, *args, **kwargs):
        self.__instance = None
        super().__init__(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        if self.__instance is None:
            self.__instance = super().__call__(*args, **kwargs)
            return self.__instance
        else:
            return self.__instance

class CommonInfo(metaclass=Singleton):
    """docstring for CommonInfo."""
    def __init__(self, datadir: str):
        super(CommonInfo, self).__init__()
        self.datadir = datadir
    
    @property
    def speakers(self):
        ''' return current selected speakers for training
        eg. ['SF2', 'TM1', 'SF1', 'TM2']
        '''
        p = os.path.join(self.datadir, "*")
        all_sub_folder = glob.glob(p)
            
        all_speaker = [s.rsplit('/', maxsplit=1)[1] for s in all_sub_folder]
        all_speaker.sort()
        return all_speaker

speakers = CommonInfo('data/speakers').speakers


class Normalizer(object):
    '''Normalizer: convience method for fetch normalize instance'''
    def __init__(self, statfolderpath: str='./etc'):
        
        self.folderpath = statfolderpath

        self.norm_dict = self.normalizer_dict()

    def forward_process(self, x, speakername):
        mean = self.norm_dict[speakername]['coded_sps_mean']
        std = self.norm_dict[speakername]['coded_sps_std']
        mean = np.reshape(mean, [-1,1])
        std = np.reshape(std, [-1,1])
        x = (x - mean) / std

        return x

    def backward_process(self, x, speakername):
        mean = self.norm_dict[speakername]['coded_sps_mean']
        std = self.norm_dict[speakername]['coded_sps_std']
        mean = np.reshape(mean, [-1,1])
        std = np.reshape(std, [-1,1])
        x = x * std + mean

        return x

    def normalizer_dict(self):
        '''return all speakers normailzer parameter'''

        d = {}
        for one_speaker in speakers:

            p = os.path.join(self.folderpath, '*.npz')
            try:
                stat_filepath = [fn for fn in glob.glob(p) if one_speaker in fn][0]
            except:
                raise Exception('====no match files!====')
            # print(f'[load]: {stat_filepath}')
            t = np.load(stat_filepath)
            d[one_speaker] = t

        return d
    
    def pitch_conversion(self, f0, source_speaker, target_speaker):
        '''Logarithm Gaussian normalization for Pitch Conversions'''
        
        mean_log_src = self.norm_dict[source_speaker]['log_f0s_mean']
        std_log_src = self.norm_dict[source_speaker]['log_f0s_std']

        mean_log_target = self.norm_dict[target_speaker]['log_f0s_mean']
        std_log_target = self.norm_dict[target_speaker]['log_f0s_std']

        f0_converted = np.exp((np.ma.log(f0) - mean_log_src) / std_log_src * std_log_target + mean_log_target)

        return f0_converted
    
class GenerateStatistics(object):
    def __init__(self, folder: str ='./data/processed'):
        self.folder = folder
        self.include_dict_npz = {}
        for s in speakers:
            if not self.include_dict_npz.__contains__(s):
                self.include_dict_npz[s] = []

            for one_file in os.listdir(folder):
                if one_file.startswith(s) and one_file.endswith('npz'):
                    self.include_dict_npz[s].append(one_file)
        

    @staticmethod
    def coded_sp_statistics(coded_sps):
        # sp shape (D, T)
        coded_sps_concatenated = np.concatenate(coded_sps, axis = 1)
        coded_sps_mean = np.mean(coded_sps_concatenated, axis = 1, keepdims = False)
        coded_sps_std = np.std(coded_sps_concatenated, axis = 1, keepdims = False)
        return coded_sps_mean, coded_sps_std

    @staticmethod
    def logf0_statistics(f0s):
        log_f0s_concatenated = np.ma.log(np.concatenate(f0s))
        log_f0s_mean = log_f0s_concatenated.mean()
        log_f0s_std = log_f0s_concatenated.std()

        return log_f0s_mean, log_f0s_std

    def generate_stats(self, statfolder: str = 'etc'):
        '''generate all user's statistics used for calutate normalized
           input like sp, f0
           step 1: generate coded_sp mean std
           step 2: generate f0 mean std
         '''
        etc_path = os.path.join(os.path.realpath('.'), statfolder)
        os.makedirs(etc_path, exist_ok=True)

        for one_speaker in self.include_dict_npz.keys():
            f0s = []
            coded_sps = []           
            arr01 = self.include_dict_npz[one_speaker]
            if len(arr01) == 0:
                continue
            for one_file in arr01:
                t = np.load(os.path.join(self.folder, one_file))
                f0_ = np.reshape(t['f0'], [-1,1])

                f0s.append(f0_)
                coded_sps.append(t['coded_sp'])
            
            log_f0s_mean, log_f0s_std = self.logf0_statistics(f0s)
            coded_sps_mean, coded_sps_std = self.coded_sp_statistics(coded_sps)

            print(f'log_f0s_mean:{log_f0s_mean} log_f0s_std:{log_f0s_std}')
            print(f'coded_sps_mean:{coded_sps_mean.shape}  coded_sps_std:{coded_sps_std.shape}')

            filename = os.path.join(etc_path, f'{one_speaker}-stats.npz')
            np.savez(filename, 
            log_f0s_mean=log_f0s_mean, log_f0s_std=log_f0s_std,
            coded_sps_mean=coded_sps_mean, coded_sps_std=coded_sps_std)

            print(f'[save]: {filename}')

  
    def normalize_dataset(self):
        '''normalize dataset run once!'''
        norm  = Normalizer()
        files = librosa.util.find_files(self.folder, ext='npy')

        for p in files:
            filename = os.path.basename(p)
            speaker = filename.split(sep='_', maxsplit=1)[0]
            mcep = np.load(p)
            mcep_normed = norm.forward_process(mcep, speaker)
            os.remove(p)
            np.save(p, mcep_normed)
            print(f'[normalize]:{p}')

if __name__ == "__main__":
    pass