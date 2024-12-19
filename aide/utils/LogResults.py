import pandas as pd


class LogResults(object):
    def __init__(self,
                 dataset_name: str,
                 config: str,
                 sub_task: str,
                 llm_model: str,
                 classifier: str,
                 task_type: str,
                 status: str,
                 number_iteration: int,
                 number_iteration_error: int,
                 has_description: str,
                 time_catalog_load: float,
                 time_pipeline_generate: float,
                 time_total: float,
                 time_execution: float,
                 train_auc: float = -2,
                 train_auc_ovo: float = -2,
                 train_auc_ovr: float = -2,
                 train_accuracy: float = -2,
                 train_f1_score: float = -2,
                 train_log_loss: float = -2,
                 train_r_squared: float = -2,
                 train_rmse: float = -2,
                 test_auc: float = -2,
                 test_auc_ovo: float = -2,
                 test_auc_ovr: float = -2,
                 test_accuracy: float = -2,
                 test_f1_score: float = -2,
                 test_log_loss: float = -2,
                 test_r_squared: float = -2,
                 test_rmse: float = -2,
                 prompt_token_count: int = 0,
                 all_token_count: int = 0,
                 operation: str = None,
                 number_of_samples: int = 0
                 ):
        self.config = config
        self.sub_task = sub_task
        self.llm_model = llm_model
        self.classifier = classifier
        self.task_type = task_type
        self.status = status
        self.number_iteration = number_iteration
        self.number_iteration_error = number_iteration_error
        self.has_description = has_description
        self.dataset_name = dataset_name
        self.time_catalog_load = time_catalog_load
        self.time_pipeline_generate = time_pipeline_generate
        self.time_total = time_total
        self.time_execution = time_execution
        self.train_auc = train_auc
        self.train_auc_ovo = train_auc_ovo
        self.train_auc_ovr = train_auc_ovr
        self.train_accuracy = train_accuracy
        self.train_f1_score = train_f1_score
        self.train_log_loss = train_log_loss
        self.train_r_squared = train_r_squared
        self.train_rmse = train_rmse
        self.test_auc = test_auc
        self.test_auc_ovo = test_auc_ovo
        self.test_auc_ovr = test_auc_ovr
        self.test_accuracy = test_accuracy
        self.test_f1_score = test_f1_score
        self.test_log_loss = test_log_loss
        self.test_r_squared = test_r_squared
        self.test_rmse = test_rmse
        self.prompt_token_count = prompt_token_count
        self.all_token_count = all_token_count
        self.operation = operation
        self.number_of_samples = number_of_samples

        self.columns = ["dataset_name", "config", "sub_task", "llm_model", "classifier", "task_type", "status",
                        "number_iteration", "number_iteration_error", "has_description", "time_catalog_load",
                        "time_pipeline_generate",
                        "time_total", "time_execution", "train_auc", "train_auc_ovo", "train_auc_ovr", "train_accuracy",
                        "train_f1_score", "train_log_loss", "train_r_squared", "train_rmse", "test_auc", "test_auc_ovo",
                        "test_auc_ovr", "test_accuracy", "test_f1_score", "test_log_loss", "test_r_squared",
                        "test_rmse",
                        "prompt_token_count", "all_token_count", "operation", "number_of_samples"]

    def save_results(self, result_output_path: str):
        try:
            df_result = pd.read_csv(result_output_path)

        except Exception as err:
            df_result = pd.DataFrame(columns=self.columns)

        df_result.loc[len(df_result)] = [self.dataset_name,
                                         self.config,
                                         self.sub_task,
                                         self.llm_model,
                                         self.classifier,
                                         self.task_type,
                                         self.status,
                                         self.number_iteration,
                                         self.number_iteration_error,
                                         self.has_description,
                                         self.time_catalog_load,
                                         self.time_pipeline_generate,
                                         self.time_total,
                                         self.time_execution,
                                         self.train_auc,
                                         self.train_auc_ovo,
                                         self.train_auc_ovr,
                                         self.train_accuracy,
                                         self.train_f1_score,
                                         self.train_log_loss,
                                         self.train_r_squared,
                                         self.train_rmse,
                                         self.test_auc,
                                         self.test_auc_ovo,
                                         self.test_auc_ovr,
                                         self.test_accuracy,
                                         self.test_f1_score,
                                         self.test_log_loss,
                                         self.test_r_squared,
                                         self.test_rmse,
                                         self.prompt_token_count,
                                         self.all_token_count,
                                         self.operation,
                                         self.number_of_samples]

        df_result.to_csv(result_output_path, index=False)