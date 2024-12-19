from dataclasses import dataclass

from .agent import Agent
from .interpreter import Interpreter
from .journal import Journal
from omegaconf import OmegaConf
from rich.status import Status
from .utils.config import (
    load_task_desc,
    prep_agent_workspace,
    save_run,
    _load_cfg,
    _load_catdb_style_config,
    prep_cfg,
    Config,
    ExecConfig,
    StageConfig,
    SearchConfig,
    AgentConfig
)


@dataclass
class Solution:
    code: str
    valid_metric: float


class Experiment:


    def __init__(self, data_dir: str, goal: str, log_dir: str, workspace_dir: str, exp_name: str, iterations: int ,llm_model: str = None,  config_path: str = "Config.yaml",
    api_config_path: str = None, rules_path: str = "Rules.yaml", evaluation_acc: bool = False, eval: str | None = None):
        """Initialize a new experiment run.

        Args:
            data_dir (str): Path to the directory containing the data files.
            goal (str): Description of the goal of the task.
            eval (str | None, optional): Optional description of the preferred way for the agent to evaluate its solutions.
        """

        _load_catdb_style_config(llm_model=llm_model, config_path=config_path, api_config_path=api_config_path,
                                 rules_path=rules_path, evaluation_acc=evaluation_acc)


        from .utils.config import  _temperature, _llm_model
        ec = ExecConfig(timeout=300, format_tb_ipython=False, agent_file_name='runfile.py')
        ac = AgentConfig(steps=iterations, k_fold_validation=1, expose_prediction=False, data_preview=False,
                         code=StageConfig(model=_llm_model, temp=_temperature),
                         feedback=StageConfig(model=_llm_model, temp=_temperature),
                         search=SearchConfig(max_debug_depth=3, debug_prob=0.5, num_drafts=5))
        _cfg = Config(data_dir= data_dir, desc_file=None,eval=eval, goal=goal, log_dir=log_dir,
                      workspace_dir=workspace_dir, preprocess_data=True, copy_data=False, exp_name=exp_name,
                      exec=ec, generate_report=False, report=StageConfig(model=_llm_model, temp=_temperature),
                      agent=ac)

        self.cfg = prep_cfg(_cfg)

        self.task_desc = load_task_desc(self.cfg)

        with Status("Preparing agent workspace (copying and extracting files) ..."):
            prep_agent_workspace(self.cfg)

        self.journal = Journal()
        self.agent = Agent(
            task_desc=self.task_desc,
            cfg=self.cfg,
            journal=self.journal,
        )
        self.interpreter = Interpreter(
            self.cfg.workspace_dir, **OmegaConf.to_container(self.cfg.exec)  # type: ignore
        )

    def run(self, steps: int) -> Solution:
        for _i in range(steps):
            self.agent.step(exec_callback=self.interpreter.run)
            save_run(self.cfg, self.journal)
        self.interpreter.cleanup_session()

        best_node = self.journal.get_best_node(only_good=False)
        return Solution(code=best_node.code, valid_metric=best_node.metric.value)
