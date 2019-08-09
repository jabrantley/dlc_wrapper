import os
import yaml
from pathlib import Path
import pdb
import deeplabcut

#cfg = {
#    "wd": Path.cwd(),
#        "task": "MRILegmovements",
#        "subject": "AB01",
#        "date": "2019-04-19",
#        "video": ['Calibration_and_Training.mp4'],
#        "config": "config.yaml",
#        "bodyparts":["RK_above", "RK_center", "RK_below","RA_above","RA_center","RA_below", "LK_above","LK_below", "LA_above","LA_below"],
#        "numframes": 40,
#        }

class dlc_wrapper:

    def __init__(self, cfg):

        if type(cfg) is dict:
            # Get attributes from dictionary
            self.task = cfg["task"]
            self.subject = cfg["subject"]
            self.date = cfg["date"]
            self.video = cfg["video"]
            # self.test_video = cfg["test_video"]
            self.config = cfg["config"] if "config" in cfg else "config.yaml"
            self.numframes2pick = cfg["numframes"] if "numframes" in cfg else 40
            self.bodyparts = cfg["bodyparts"]

            # Define project directory
            self.project_dir = Path("-".join([cfg["task"], cfg["subject"], cfg["date"]]))
        else:
            # Load existing project
            self.load_project(cfg)
            # self.thing = self.my_function(cfg)

    def load_project(self, cfg):
        print("Loading project from config file.")
        from deeplabcut.utils import auxiliaryfunctions
        #  load config.yaml
        with open(cfg,'r') as ymlfile:  # not using cfg = auxiliaryfunctions.read_config(cfg) since cfg is commented map
            main_cfg = yaml.load(ymlfile)
        # Get attributes from dictionary
        self.task = main_cfg["Task"]
        self.subject = main_cfg["scorer"]
        self.date = main_cfg["date"]
        self.video = []
        # Add videos
        k = 0
        for vid in main_cfg["video_sets"].keys():
            self.video.append(str(Path(vid).name))
            k += 1

        # self.test_video = main_cfg["test_video"]
        self.config = os.path.basename(cfg) if type(cfg) is str else cfg.name
        self.numframes2pick = main_cfg["numframes2pick"]
        self.bodyparts = main_cfg["bodyparts"]

        # Define project directory
        self.project_dir = Path(os.path.basename(os.path.dirname(str(cfg)))) if type(cfg) is str else Path(
            cfg.parent.name)

        # Update project paths
        self.update_project_paths()
        print("Project loaded.")

    def create_project(self):
        # Get project directory
        project_dir = self.project_dir
        # Check if project exists
        if project_dir.exists():
            print("The project directory already exists.")
        else:
            # Creates new project then renames according to task and subject
            print("The project directory does not exist.")
            #  Create file name as is created in deeplabcut.create_new_project
            from datetime import datetime as dt
            # Deeplabcut uses the creation name in the date. We want to change this to the study date
            date = dt.today().strftime('%Y-%m-%d')
            old_project_dir = Path("-".join([self.task, self.subject, date]))
            # Create new project
            deeplabcut.create_new_project(self.task, self.subject, self.video,
                                          working_directory=self.project_dir.resolve().parent,
                                          copy_videos=True)
            # Rename project to have project date not current date
            old_project_dir.rename(project_dir)

            from deeplabcut.utils import auxiliaryfunctions

            # Get config file path
            main_config_file = Path(self.project_dir).resolve() / self.config

            #  load config.yaml
            main_config = auxiliaryfunctions.read_config(main_config_file)

            #  Update values in config file:
            main_config['bodyparts'] = self.bodyparts
            main_config['date'] = self.date
            main_config['numframes2pick'] = self.numframes2pick

            # Write dictionary to yaml  config file
            auxiliaryfunctions.write_config(main_config_file, main_config)

            # Update project paths
            self.update_project_paths()
        return

    def full_config_path(self):
        # Get config file path
        return Path(self.project_dir).resolve() / self.config

    def update_project_paths(self):

        print('Updating paths..')
        from deeplabcut.utils import auxiliaryfunctions
        
        #  load config.yaml
        main_config = auxiliaryfunctions.read_config(self.full_config_path())

        # Update path in main config
        new_project_dir = self.project_dir.resolve()
        main_config['project_path'] = str(new_project_dir)

        # Update video path. NOTE: the video path name
        for old_vid in list(main_config["video_sets"]):
            new_vid = str(new_project_dir / "videos" / Path(old_vid).name)
            main_config["video_sets"][new_vid] = main_config["video_sets"].pop(old_vid)

        # Write dictionary to yaml  config file
        auxiliaryfunctions.write_config(self.full_config_path(), main_config)

        # Update train and test config.yaml paths
        trainingsetindex = 0
        shuffle = 1
        modelfoldername = auxiliaryfunctions.GetModelFolder(main_config["TrainingFraction"][trainingsetindex], shuffle, main_config)
        path_train_config = os.path.join(main_config['project_path'], Path(modelfoldername), 'train','pose_cfg.yaml')
        path_test_config = os.path.join(main_config['project_path'], Path(modelfoldername), 'test', 'pose_cfg.yaml')
  
        
        # Update training pose_cfg.yaml
        if os.path.exists(path_train_config):
            #train(str(poseconfigfile),displayiters,saveiters,maxiters,max_to_keep=max_snapshots_to_keep) #pass on path and file name for pose_cfg.yaml!
            with open(path_train_config, "r") as ymlfile:
                cfg_train = yaml.load(ymlfile,Loader=yaml.FullLoader)
                
            cfg_train['project_path'] = str(new_project_dir)
            old_dataset_train = os.path.join(*cfg_train['dataset'].split('\\')) #str(Path(cfg_train['dataset']))
            cfg_train['dataset'] = old_dataset_train
            old_metadataset = os.path.join(*cfg_train['metadataset'].split('\\')) #str(Path(cfg_train['metadataset']))
            cfg_train['metadataset'] = old_metadataset
            # if Path(Path.cwd().parent /
            # init_loc = input("Please specificy directory to resnet_v1_50.ckpt")
            cfg_train['init_weights'] = str(Path.cwd().parent / "resnet_v1_50.ckpt")
            with open(path_train_config, 'w') as ymlfile:
                yaml.dump(cfg_train, ymlfile)
    
            # Update MATLAB file contining training files
            if os.path.exists(self.project_dir / cfg_train['dataset']):
                import scipy.io as sio
                # Load Matlab file dataset annotation
                mlab = sio.loadmat(self.project_dir / cfg_train['dataset'])
                num_images = mlab['dataset'].shape[1]
                for i in range(num_images):
                    oldFilePath = mlab['dataset'][0, i][0][0]
                    newFilePath = os.path.join(*oldFilePath.split('\\')) #str(Path(oldFilePath))
                    mlab['dataset'][0, i][0][0] = newFilePath
                # Saving mat file
                sio.savemat(os.path.join(self.project_dir / cfg_train['dataset']), mlab)

        # Update testing pose_cfg.yaml
        if os.path.exists(path_test_config):
            #train(str(poseconfigfile),displayiters,saveiters,maxiters,max_to_keep=max_snapshots_to_keep) #pass on path and file name for pose_cfg.yaml!
            with open(path_test_config, "r") as ymlfile:
                cfg_test = yaml.load(ymlfile,Loader=yaml.FullLoader)
            cfg_test['init_weights'] = str(Path.cwd().parent / "resnet_v1_50.ckpt")
            old_dataset_test = os.path.join(*cfg_test['dataset'].split('\\')) #str(Path(cfg_test['dataset']))
            cfg_test['dataset'] = old_dataset_test
            with open(path_test_config, 'w') as ymlfile:
                yaml.dump(cfg_test, ymlfile)

        print('done.')
                
    def update_checkpoint(self,new_weights="none"):
        
        if new_weights == 'none':
            print('No checkpoint provided.')
        else:
            print('Updating init_weights with %s..' % new_weights)
            from deeplabcut.utils import auxiliaryfunctions
            
            #  load config.yaml
            main_config = auxiliaryfunctions.read_config(self.full_config_path())

            # Update train and test config.yaml paths
            trainingsetindex = 0
            shuffle = 1
            modelfoldername = auxiliaryfunctions.GetModelFolder(main_config["TrainingFraction"][trainingsetindex], shuffle, main_config)
            path_train_config = os.path.join(main_config['project_path'], Path(modelfoldername), 'train', 'pose_cfg.yaml')
            path_test_config = os.path.join(main_config['project_path'], Path(modelfoldername), 'test', 'pose_cfg.yaml')
            
            # Update training pose_cfg.yaml
            if os.path.exists(path_train_config):
                with open(path_train_config, "r") as ymlfile:
                    cfg_train = yaml.load(ymlfile)
                # Update init_weights path
                cfg_train['init_weights'] = os.path.join(main_config['project_path'], Path(modelfoldername), 'train',new_weights)
                with open(path_train_config, 'w') as ymlfile:
                    yaml.dump(cfg_train, ymlfile)
        
            # Update testing pose_cfg.yaml
            # if os.path.exists(path_test_config):
            #    with open(path_test_config, "r") as ymlfile:
            #        cfg_test = yaml.load(ymlfile)
            #    # Update init_weights path
            #    cfg_test['init_weights'] = os.path.join(main_config['project_path'], #Path(modelfoldername), 'test',new_weights)
            #with open(path_test_config, 'w') as ymlfile:
            #        yaml.dump(cfg_test, ymlfile)
                                                                
            print('done.')
                                                                
                
    def extract_frames(self):
        # There probably is a more elegant way to do this but it preseves the ability to run in an interactive session
        # try: # if is iPython
        #    #%matplotlib inline
        # finally:
        deeplabcut.extract_frames(self.full_config_path())

    def label_frames(self):
        # There probably is a more elegant way to do this but it preseves the ability to run in an interactive session
        # try:  # if is iPython
        # %gui wx
        # finally:
        deeplabcut.label_frames(self.full_config_path())

    def check_labels(self):
        deeplabcut.check_labels(self.full_config_path())

    def create_training_dataset(self):
        deeplabcut.create_training_dataset(self.full_config_path())

    def train_network(self):
        deeplabcut.train_network(self.full_config_path(), shuffle=1, gputouse=0)

    def evaluate_network(self):
        deeplabcut.evaluate_network(self.full_config_path(), plotting=False)

    def analyze_videos(self):
        test_videos = self.test_video
        deeplabcut.analyze_videos(self.full_config_path(), [test_videos])
        deeplabcut.create_labeled_video(self.full_config_path(), [test_videos])
