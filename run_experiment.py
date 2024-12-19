
import aide

if __name__ == '__main__':
    task_type = "multiclass"
    metric = None
    if task_type == 'binary':
        metric = "roc_auc"
    elif task_type == 'multiclass':
        metric = "roc_auc_ovr"
    elif task_type == 'regression':
        metric = "r2"

    goal = f"""
           Your goal is to predict the target column `c_10`.
           Perform data analysis, data preprocessing, feature engineering, and modeling to predict the target. 
           Code formatting for multiclass classification evaluation:
           # Report evaluation based on train and test dataset
           # Calculate the model accuracy, represented by a value between 0 and 1, where 0 indicates low accuracy and 1 signifies higher accuracy. Store the accuracy value in a variable labeled as "Train_Accuracy=..." and "Test_Accuracy=...".
           # Calculate the model log loss, a lower log-loss value means better predictions. Store the  log loss value in a variable labeled as "Train_Log_loss=..." and "Test_Log_loss=...".
           # Calculate AUC_OVO (Area Under the Curve One-vs-One), represented by a value between 0 and 1.
           # Calculate AUC_OVR (Area Under the Curve One-vs-Rest), represented by a value between 0 and 1.
           # print(f"Train_AUC_OVO:{{Train_AUC_OVO}}")
           # print(f"Train_AUC_OVR:{{Train_AUC_OVR}}")
           # print(f"Train_Accuracy:{{Train_Accuracy}}")   
           # print(f"Train_Log_loss:{{Train_Log_loss}}") 
           # print(f"Test_AUC_OVO:{{Test_AUC_OVO}}")
           # print(f"Test_AUC_OVR:{{Test_AUC_OVR}}")
           # print(f"Test_Accuracy:{{Test_Accuracy}}")   
           # print(f"Test_Log_loss:{{Test_Log_loss}}")
           Do not plot or make any visualizations.\n')

           # Data dir
           training (with labels): oml_dataset_3_rnc_train.csv
           testing (with labels): oml_dataset_3_rnc_test.csv
           """
    data_dir = "/home/saeed/Documents/Github/CatDB/Experiments/data/oml_dataset_3_rnc/"
    exp = aide.Experiment(
        data_dir=data_dir,
        goal=goal,
        log_dir="/home/saeed/Downloads/AIDE/log/",
        workspace_dir="/home/saeed/Downloads/AIDE/workspace/",
        exp_name="DS1",
        iterations=1,
        llm_model="gemini-1.5-pro-latest",
        config_path= "Config.yaml",
        api_config_path="APIKeys.yaml",
        rules_path= "Rules.yaml",
        evaluation_acc = False,
        eval = metric
    )

    best_solution = exp.run(steps=1)
    #
    # print(f"Best solution has validation metric: {best_solution.valid_metric}")
    # print(f"Best solution code: {best_solution.code}")
    # end_time = time.time()
    # execution_time = end_time - start_time