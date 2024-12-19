"""configuration and setup utils"""

from dataclasses import dataclass
from pathlib import Path
from typing import Hashable, cast

import coolname
import rich
from omegaconf import OmegaConf
from rich.syntax import Syntax
import shutup
from rich.logging import RichHandler
import logging
import os
import yaml

from . import tree_export
from . import copytree, preproc_data, serialize
from .LLM_API_Key import LLM_API_Key

shutup.mute_warnings()
logging.basicConfig(
    level="WARNING", format="%(message)s", datefmt="[%X]", handlers=[RichHandler()]
)
logger = logging.getLogger("aide")
logger.setLevel(logging.WARNING)

default_max_token_limit = 4096
default_max_output_tokens = 8192
default_delay = 0
default_temperature = 0
default_top_p = 0.95
default_top_k = 64

_OPENAI = "OpenAI"
_META = "Meta"
_GOOGLE = "Google"

_llm_model = None
_llm_platform = None
_max_token_limit = None
_max_out_token_limit = None
_delay = None
_temperature = None
_top_p = None
_top_k = None
_last_API_Key = None
_LLM_API_Key = None

_CODE_FORMATTING_BINARY_EVALUATION = None
_CODE_FORMATTING_MULTICLASS_EVALUATION = None
_CODE_FORMATTING_REGRESSION_EVALUATION = None
_CODE_FORMATTING_ACC_EVALUATION = None

""" these dataclasses are just for type hinting, the actual config is in config.yaml """


@dataclass
class StageConfig:
    model: str
    temp: float


@dataclass
class SearchConfig:
    max_debug_depth: int
    debug_prob: float
    num_drafts: int


@dataclass
class AgentConfig:
    steps: int
    k_fold_validation: int
    expose_prediction: bool
    data_preview: bool

    code: StageConfig
    feedback: StageConfig

    search: SearchConfig


@dataclass
class ExecConfig:
    timeout: int
    agent_file_name: str
    format_tb_ipython: bool


@dataclass
class Config(Hashable):
    data_dir: Path
    desc_file: Path | None

    goal: str | None
    eval: str | None

    log_dir: Path
    workspace_dir: Path

    preprocess_data: bool
    copy_data: bool

    exp_name: str

    exec: ExecConfig
    generate_report: bool
    report: StageConfig
    agent: AgentConfig


def _get_next_logindex(dir: Path) -> int:
    """Get the next available index for a log directory."""
    max_index = -1
    for p in dir.iterdir():
        try:
            if current_index := int(p.name.split("-")[0]) > max_index:
                max_index = current_index
        except ValueError:
            pass
    return max_index + 1


def _load_catdb_style_config(llm_model: str = None,
                             config_path: str = "Config.yaml",
                             api_config_path: str = None,
                             rules_path: str = "Rules.yaml",
                             evaluation_acc: bool = False):
    if api_config_path is None:
        api_config_path = os.environ.get("APIKeys_File")
    global _llm_model
    global _llm_platform
    global _max_token_limit
    global _max_out_token_limit
    global _delay
    global _last_API_Key
    global _LLM_API_Key
    global _temperature
    global _top_k
    global _top_p

    with open(config_path, "r") as f:
        try:
            configs = yaml.load(f, Loader=yaml.FullLoader)
            for conf in configs:
                plt = conf.get("llm_platform")
                try:
                    if conf.get(llm_model) is not None:
                        _llm_model = llm_model
                        _llm_platform = plt

                        try:
                            _max_token_limit = int(conf.get(llm_model).get('token_limit'))
                        except:
                            _max_token_limit = default_max_token_limit

                        try:
                            _max_out_token_limit = int(conf.get(llm_model).get('max_output_tokens'))
                        except:
                            _max_out_token_limit = default_max_output_tokens

                        try:
                            _delay = int(conf.get(llm_model).get('delay'))
                        except:
                            _delay = default_delay

                        try:
                            _temperature = float(conf.get(llm_model).get('temperature'))
                        except:
                            _temperature = default_temperature

                        try:
                            _top_k = int(conf.get(llm_model).get('top_k'))
                        except:
                            _top_k = default_top_k

                        try:
                            _top_p = float(conf.get(llm_model).get('top_p'))
                        except:
                            _top_p = default_top_p

                        break
                except Exception as ex:
                    pass

        except yaml.YAMLError as ex:
            raise Exception(ex)

        if _llm_model is None:
            raise Exception(f'Error: model "{llm_model}" is not in the Config.yaml list!')

        _LLM_API_Key = LLM_API_Key(api_config_path=api_config_path)
        load_rules(rules_path=rules_path, evaluation_acc=evaluation_acc)



def load_rules(rules_path: str, evaluation_acc: bool = False):
    global _CODE_FORMATTING_BINARY_EVALUATION
    global _CODE_FORMATTING_MULTICLASS_EVALUATION
    global _CODE_FORMATTING_REGRESSION_EVALUATION
    global _CODE_FORMATTING_ACC_EVALUATION

    with (open(rules_path, "r") as f):
        try:
            configs = yaml.load(f, Loader=yaml.FullLoader)
            for conf in configs:
                plt = conf.get("Config")
                if plt == 'CodeFormat':
                    for k, v in conf.items():
                        if k == 'CODE_FORMATTING_BINARY_EVALUATION':
                            _CODE_FORMATTING_BINARY_EVALUATION = v
                        elif k == 'CODE_FORMATTING_ACC_EVALUATION':
                            _CODE_FORMATTING_ACC_EVALUATION = v
                        elif k == 'CODE_FORMATTING_MULTICLASS_EVALUATION':
                            _CODE_FORMATTING_MULTICLASS_EVALUATION = v
                        elif k == 'CODE_FORMATTING_REGRESSION_EVALUATION':
                            _CODE_FORMATTING_REGRESSION_EVALUATION = v
        except yaml.YAMLError as ex:
            raise Exception(ex)

    if evaluation_acc:
        _CODE_FORMATTING_BINARY_EVALUATION = _CODE_FORMATTING_ACC_EVALUATION
        _CODE_FORMATTING_MULTICLASS_EVALUATION = _CODE_FORMATTING_ACC_EVALUATION


def _load_cfg(
        path: Path = Path(__file__).parent / "config.yaml", use_cli_args=True
) -> Config:
    cfg = OmegaConf.load(path)
    if use_cli_args:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_cli())
    return cfg


def load_cfg(path: Path = Path(__file__).parent / "config.yaml") -> Config:
    """Load config from .yaml file and CLI args, and set up logging directory."""
    return prep_cfg(_load_cfg(path))


def prep_cfg(cfg: Config):
    if cfg.data_dir is None:
        raise ValueError("`data_dir` must be provided.")

    if cfg.desc_file is None and cfg.goal is None:
        raise ValueError(
            "You must provide either a description of the task goal (`goal=...`) or a path to a plaintext file containing the description (`desc_file=...`)."
        )

    if cfg.data_dir.startswith("example_tasks/"):
        cfg.data_dir = Path(__file__).parent.parent / cfg.data_dir
    cfg.data_dir = Path(cfg.data_dir).resolve()

    if cfg.desc_file is not None:
        cfg.desc_file = Path(cfg.desc_file).resolve()

    top_log_dir = Path(cfg.log_dir).resolve()
    top_log_dir.mkdir(parents=True, exist_ok=True)

    top_workspace_dir = Path(cfg.workspace_dir).resolve()
    top_workspace_dir.mkdir(parents=True, exist_ok=True)

    # generate experiment name and prefix with consecutive index
    ind = max(_get_next_logindex(top_log_dir), _get_next_logindex(top_workspace_dir))
    cfg.exp_name = cfg.exp_name or coolname.generate_slug(3)
    cfg.exp_name = f"{ind}-{cfg.exp_name}"

    cfg.log_dir = (top_log_dir / cfg.exp_name).resolve()
    cfg.workspace_dir = (top_workspace_dir / cfg.exp_name).resolve()

    # validate the config
    cfg_schema: Config = OmegaConf.structured(Config)
    cfg = OmegaConf.merge(cfg_schema, cfg)

    return cast(Config, cfg)


def print_cfg(cfg: Config) -> None:
    rich.print(Syntax(OmegaConf.to_yaml(cfg), "yaml", theme="paraiso-dark"))


def load_task_desc(cfg: Config):
    """Load task description from markdown file or config str."""

    # either load the task description from a file
    if cfg.desc_file is not None:
        if not (cfg.goal is None and cfg.eval is None):
            logger.warning(
                "Ignoring goal and eval args because task description file is provided."
            )

        with open(cfg.desc_file) as f:
            return f.read()

    # or generate it from the goal and eval args
    if cfg.goal is None:
        raise ValueError(
            "`goal` (and optionally `eval`) must be provided if a task description file is not provided."
        )

    task_desc = {"Task goal": cfg.goal}
    if cfg.eval is not None:
        task_desc["Task evaluation"] = cfg.eval

    return task_desc


def prep_agent_workspace(cfg: Config):
    """Setup the agent's workspace and preprocess data if necessary."""
    (cfg.workspace_dir / "input").mkdir(parents=True, exist_ok=True)
    (cfg.workspace_dir / "working").mkdir(parents=True, exist_ok=True)

    copytree(cfg.data_dir, cfg.workspace_dir / "input", use_symlinks=not cfg.copy_data)
    if cfg.preprocess_data:
        preproc_data(cfg.workspace_dir / "input")


def save_run(cfg: Config, journal):
    cfg.log_dir.mkdir(parents=True, exist_ok=True)

    # save journal
    serialize.dump_json(journal, cfg.log_dir / "journal.json")
    # save config
    # OmegaConf.save(config=cfg, f=cfg.log_dir / "config.yaml")
    # # create the tree + code visualization
    # tree_export.generate(cfg, journal, cfg.log_dir / "tree_plot.html")
    # # save the best found solution
    # best_node = journal.get_best_node(only_good=False)
    # with open(cfg.log_dir / "best_solution.py", "w") as f:
    #     f.write(best_node.code)
