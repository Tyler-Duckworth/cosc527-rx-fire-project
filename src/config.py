import os
import json
import time
from datetime import datetime

class Config:
    def __init__(self, args):
        self.show_anim = not args.hide_animation
        self.output_dir = args.output_dir
        self.output_file = args.output_file
        
        self.generations = args.generations
        self.seed = args.seed
        self.switch_frame = args.switch_frame
        self.buildings = []
        self.fire_origins = []
        self.cell_width = 30
        self.name = ""
        # config file overwrites CLI args (TODO: Reverse?)
        if args.config_file is not None:
            self.load_config(args.config_file)

        output_file_name = f'results_{datetime.now().strftime("%Y_%m_%d__%H%M%S")}.mp4'
        self.output_file = f"{self.output_dir}/{output_file_name}"
        
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir, exist_ok=True)
    
    def load_config(self, config_file_path):
        with open(config_file_path, "r") as fp:
            buf = json.load(fp)
        self.buildings = buf.get("buildings", [])
        self.fire_origins = buf.get("fire_origins", [])
        self.output_dir = buf.get("output_dir", "results/default")
        self.data_file = buf.get("data_file", "data/stacked_ca.tif")
        self.generations = buf.get("generations", 200)
        self.switch_frame = buf.get("switch_frame", 25)
        self.seed = buf.get("seed", int(time.time()))
        self.show_animation = buf.get("show_animation", False)
        self.name = buf.get("name", "")


