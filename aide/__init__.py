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
    prep_cfg,
    Config,
    ExecConfig,
    StageConfig,
    SearchConfig,
    AgentConfig
)
import time
from .utils.LogResults import LogResults

@dataclass
class Solution:
    code: str
    valid_metric: float


class Experiment:


    def __init__(self, data_dir: str, goal: str, log_dir: str, workspace_dir: str, exp_name: str, steps: int,
                 dataset_name: str, task_type: str, out_path:str,  eval: str | None = None):
        """Initialize a new experiment run.

        Args:
            data_dir (str): Path to the directory containing the data files.
            goal (str): Description of the goal of the task.
            eval (str | None, optional): Optional description of the preferred way for the agent to evaluate its solutions.
        """
        self.dataset_name = dataset_name
        self.task_type = task_type
        self.out_path = out_path

        from .utils.config import  _temperature, _llm_model
        ec = ExecConfig(timeout=3600, format_tb_ipython=False, agent_file_name='runfile.py')
        ac = AgentConfig(steps=steps, k_fold_validation=1, expose_prediction=False, data_preview=False,
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

    def run(self, iteration: int):
            time_start = time.time()
            self.agent.step(exec_callback=self.interpreter.run)
            time_end = time.time()
            wait_time = self.journal.nodes[0].wait_time
            total_time = time_end - time_start - wait_time
            save_run(self.cfg, self.journal)
            self.save_log(self.journal.nodes[0].term_out, iteration=iteration, total_time=total_time, execution_time=self.journal.nodes[0].exec_time, tokens=self.journal.nodes[0].total_tokens_count)
            self.interpreter.cleanup_session()

        # best_node = self.journal.get_best_node(only_good=False)
        # return Solution(code=best_node.code, valid_metric=best_node.metric.value)

    def save_log(self, results, iteration, total_time, execution_time, tokens):
        pipeline_evl = {"Train_AUC": -2,
                        "Train_AUC_OVO": -2,
                        "Train_AUC_OVR": -2,
                        "Train_Accuracy": -2,
                        "Train_F1_score": -2,
                        "Train_Log_loss": -2,
                        "Train_R_Squared": -2,
                        "Train_RMSE": -2,
                        "Test_AUC": -2,
                        "Test_AUC_OVO": -2,
                        "Test_AUC_OVR": -2,
                        "Test_Accuracy": -2,
                        "Test_F1_score": -2,
                        "Test_Log_loss": -2,
                        "Test_R_Squared": -2,
                        "Test_RMSE": -2}
        if results is not None:
            raw_results = results.splitlines()
            for rr in raw_results:
                row = rr.strip().split(":")
                if row[0] in pipeline_evl.keys():
                    pipeline_evl[row[0]] = row[1].strip()

        from .utils.config import _llm_model
        log_results = LogResults(dataset_name=self.dataset_name,
                                 config="AIDE", sub_task="", llm_model=_llm_model, classifier="Auto", task_type= self.task_type,
                                 status="True", number_iteration=iteration, number_iteration_error=0, has_description="No", time_catalog_load=0,
                                 time_pipeline_generate=0, time_total=total_time, time_execution=execution_time, train_auc=pipeline_evl["Train_AUC"],
                                 train_auc_ovo=pipeline_evl["Train_AUC_OVO"] , train_auc_ovr= pipeline_evl["Train_AUC_OVR"], train_accuracy=pipeline_evl["Train_Accuracy"],
                                 train_f1_score=pipeline_evl["Train_F1_score"], train_log_loss=pipeline_evl["Train_Log_loss"], train_r_squared=pipeline_evl["Train_R_Squared"], train_rmse=pipeline_evl["Train_RMSE"],
                                 test_auc=pipeline_evl["Test_AUC"], test_auc_ovo=pipeline_evl["Test_AUC_OVO"],  test_auc_ovr=pipeline_evl["Test_AUC_OVR"], test_accuracy=pipeline_evl["Test_Accuracy"],
                                 test_f1_score=pipeline_evl["Test_F1_score"], test_log_loss=pipeline_evl["Test_Log_loss"], test_r_squared=pipeline_evl["Test_R_Squared"], test_rmse=pipeline_evl["Test_RMSE"],
                                 prompt_token_count=tokens,all_token_count=tokens, operation="Run-Pipeline", number_of_samples=0)

        log_results.save_results(result_output_path=self.out_path)
