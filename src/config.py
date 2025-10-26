import os
import tomllib
from pathlib import Path
from dotenv import load_dotenv

class Config:
    def __init__(self):
        self.project_dir = Path(__file__).parent
        self.data_dir = self.project_dir.parent / 'Data'
        self.env_path = self.project_dir.parent / ".env"
        load_dotenv(dotenv_path=self.env_path)
        self.s3_uri = os.getenv("s3_model_uri")
        self.aws_profile = os.getenv("aws_profile_name")

        self.toml_path = self.project_dir / 'prompt-template.toml'
        with open(self.toml_path, "rb") as f:
            self.prompt_template = tomllib.load(f)
        self.keyword_prompts = self.prompt_template['keyword_refinement']

    def get_toml(self) -> dict:
        """Return the parsed TOML content as a dict (empty if file missing)."""
        return self.toml

    def get_toml_value(self, *keys, default=None):
        """
        Nested lookup into the TOML dict.
        Example: get_toml_value('prompts', 'default') -> toml['prompts']['default']
        """
        cur = self.toml
        for k in keys:
            if not isinstance(cur, dict) or k not in cur:
                return default
            cur = cur[k]
        return cur

if __name__ == "__main__":
    config = Config()
    print(f'Project directory: {config.project_dir}')
    if config.toml:
        print("TOML loaded keys:", list(config.toml.keys()))
    else:
        print("No TOML file found or empty TOML.")
