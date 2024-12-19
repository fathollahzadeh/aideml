import aide

from argparse import ArgumentParser
import yaml
from .aide.utils.config import _load_catdb_style_config


def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument('--metadata-path', type=str, default=None)
    parser.add_argument('--root-data-path', type=str, default=None)
    parser.add_argument('--prompt-number-iteration', type=int, default=1)
    parser.add_argument('--output-path', type=str, default=None)
    parser.add_argument('--llm-model', type=str, default=None)
    parser.add_argument('--result-output-path', type=str, default="/tmp/results.csv")
    parser.add_argument('--system-log', type=str, default="/tmp/catdb-system-log.dat")

    args = parser.parse_args()

    if args.metadata_path is None:
        raise Exception("--metadata-path is a required parameter!")

    if args.root_data_path is None:
        raise Exception("--root-data-path is a required parameter!")

    if args.catalog_path is None:
        raise Exception("--catalog-path is a required parameter!")

    # read .yaml file and extract values:
    with open(args.metadata_path, "r") as f:
        try:
            config_data = yaml.load(f, Loader=yaml.FullLoader)
            args.dataset_name = config_data[0].get('name')
            args.target_attribute = config_data[0].get('dataset').get('target')
            args.task_type = config_data[0].get('dataset').get('type')
            args.multi_table = config_data[0].get('dataset').get('multi_table')
            args.target_table = config_data[0].get('dataset').get('target_table')
            if args.multi_table is None or args.multi_table not in {True, False}:
                args.multi_table = False

            try:
                args.data_source_path = f"{args.root_data_path}/{args.dataset_name}/{args.dataset_name}.csv"
                args.data_source_train_path = f"{args.root_data_path}/{config_data[0].get('dataset').get('train')}"
                args.data_source_test_path = f"{args.root_data_path}/{config_data[0].get('dataset').get('test')}"
                args.data_source_verify_path = f"{args.root_data_path}/{config_data[0].get('dataset').get('verify')}"

            except Exception as ex:
                raise Exception(ex)

        except yaml.YAMLError as ex:
            raise Exception(ex)

    if args.llm_model is None:
        raise Exception("--llm-model is a required parameter!")

    if args.prompt_number_iteration is None:
        args.prompt_number_iteration = 1

    return args


if __name__ == '__main__':
    args = parse_arguments()
    _load_catdb_style_config(llm_model=args.llm_model, config_path="Config.yaml", api_config_path="APIKeys.yaml",
                             rules_path="Rules.yaml", evaluation_acc=args.dataset_name == "EU-IT")

    from .aide.utils.config import (_CODE_FORMATTING_BINARY_EVALUATION, _CODE_FORMATTING_MULTICLASS_EVALUATION,
                                    _CODE_FORMATTING_REGRESSION_EVALUATION)

    code_format = ""
    metric = None
    if args.task_type == 'binary':
        metric = "roc_auc"
        code_format = _CODE_FORMATTING_BINARY_EVALUATION
    elif args.task_type == 'multiclass':
        metric = "roc_auc_ovr"
        code_format = _CODE_FORMATTING_MULTICLASS_EVALUATION
    elif args.task_type == 'regression':
        metric = "r2"
        code_format = _CODE_FORMATTING_REGRESSION_EVALUATION

    goal = f"""
           Your goal is to predict the target column `{args.target_attribute}`.
           Perform data analysis, data preprocessing, feature engineering, and modeling to predict the target. 
           ## Do not split the train_data into train and test sets. Use only the given datasets.
           ## Don't report model validation part (Only and Only report Train and Test model evaluation).
           
           ##{code_format} 
           
           ## Data dir
           training (with labels): {args.data_source_train_path}
           testing (with labels): {args.data_source_test_path}
           """
    exp = aide.Experiment(
        data_dir=args.args.root_data_path,
        goal=goal,
        log_dir=f"{args.output_path}/log/",
        workspace_dir=f"{args.output_path}/workspace/",
        exp_name="EXP",
        iterations=1,
        eval=metric,
        dataset_name=args.dataset_name,
        out_path=args.result_output_path,
        task_type=args.task_type
    )
    exp.run(steps=args.prompt_number_iteration)
