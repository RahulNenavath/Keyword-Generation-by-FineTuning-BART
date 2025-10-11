from pathlib import Path

class Config:
    def __init__(self):
        self.project_dir = Path(__file__).parent
        self.data_dir = self.project_dir.parent / 'Data'
    
if __name__ == "__main__":
    config = Config()
    print(f'Project directtory: {config.project_dir}')