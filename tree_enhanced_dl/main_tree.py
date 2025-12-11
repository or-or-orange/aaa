"""
专门用于测试树模型的主脚本
支持LightGBM/XGBoost/CatBoost等框架的训练、评估和对比
可同时实验多种模型并保存详细结果，用于与原有算法对比
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
import sys
import logging
import copy
from datetime import datetime

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent))

# 复用项目现有模块
from utils.config_parser import ConfigParser
from utils.logger import setup_logger
from data.preprocessor import DataPreprocessor
from tree_module.tree_trainer import TreeModelTrainer
from evaluation.metrics import MetricsCalculator
from main import set_seed  # 复用随机种子设置函数


def load_data(config: dict, logger):
    """完全复用main.py的数加载/预处理逻辑"""
    logger.info("===== 开始加载并预处理数据 =====")
    # 读取数据（保持与main.py一致的分隔符/编码）
    try:
        train_df = pd.read_csv(config['data']['train_path'], sep=";", encoding="latin-1")
        val_df = pd.read_csv(config['data']['val_path'], sep=";", encoding="latin-1")
        test_df = pd.read_csv(config['data']['test_path'], sep=";", encoding="latin-1") if config['data'].get('test_path') else None
    except Exception as e:
        logger.error(f"数据读取失败: {str(e)}", exc_info=True)
        raise

    # 打印数据规模（日志增强）
    logger.info(f"数据规模 - 训练集: {len(train_df)}, 验证集: {len(val_df)}, 测试集: {len(test_df) if test_df is not None else 0}")

    # 预处理（复用DataPreprocessor）
    preprocessor = DataPreprocessor(config)
    X_train, y_train = preprocessor.fit_transform(train_df)
    X_val, y_val = preprocessor.transform(val_df)
    X_test, y_test = preprocessor.transform(test_df) if test_df is not None else (None, None)

    # 保存预处理工具（方便后续推理/对比）
    save_dir = Path(config['system']['checkpoint']['save_dir'])
    save_dir.mkdir(parents=True, exist_ok=True)
    preprocessor.save(save_dir / 'tree_preprocessor.pkl')
    logger.info(f"预处理工具已保存至: {save_dir / 'tree_preprocessor.pkl'}")

    # 转换为DataFrame（树模型需要特征名）
    feature_names = preprocessor.get_feature_names()
    X_train_df = pd.DataFrame(X_train, columns=feature_names)
    X_val_df = pd.DataFrame(X_val, columns=feature_names)
    X_test_df = pd.DataFrame(X_test, columns=feature_names) if X_test is not None else None

    return (X_train_df, y_train), (X_val_df, y_val), (X_test_df, y_test), preprocessor, save_dir


def evaluate_and_save_results(
    model: TreeModelTrainer,
    X: pd.DataFrame,
    y_true: np.ndarray,
    logger,
    save_dir: Path,
    model_name: str,
    prefix: str = "Test"
) -> dict:
    """
    评估模型并保存结果（复用MetricsCalculator）
    :param model: 训练好的树模型
    :param X: 特征数据
    :param y_true: 真实标签
    :param logger: 日志对象
    :param save_dir: 结果保存目录
    :param model_name: 模型名称（lightgbm/xgboost/catboost）
    :param prefix: 数据集类型（Validation/Test）
    :return: 评估指标字典
    """
    logger.info(f"\n===== 开始评估 {prefix} 集 - {model_name} =====")
    # 预测（复用TreeModelTrainer的预测方法）
    try:
        y_pred = model.predict(X)
        y_pred_proba = model.predict_proba(X)  # (n_samples, n_classes)
    except Exception as e:
        logger.error(f"{model_name} 预测失败: {str(e)}", exc_info=True)
        raise

    # 计算指标（复用MetricsCalculator，日志格式化输出）
    metrics = MetricsCalculator.calculate(
        y_true=y_true,
        y_pred=y_pred,
        y_pred_proba=y_pred_proba[:, 1]  # 二分类取正类概率
    )

    # 日志输出指标（标准化格式，方便对比）
    logger.info(f"\n【{model_name} - {prefix} 集指标】")
    for metric, value in sorted(metrics.items()):
        logger.info(f"{metric.upper()}: {value:.6f}")  # 保留6位小数，提升对比精度

    # 保存预测结果为CSV（含真实标签、预测类别、正负类概率）
    result_df = pd.DataFrame({
        "true_label": y_true.astype(int),
        "pred_label": y_pred.astype(int),
        "pred_prob_negative": y_pred_proba[:, 0],  # 负类概率
        "pred_prob_positive": y_pred_proba[:, 1]   # 正类概率
    })
    result_path = save_dir / f"{model_name}_{prefix.lower()}_results.csv"
    result_df.to_csv(result_path, index=False, encoding="utf-8")
    logger.info(f"{prefix} 集结果已保存: {result_path}")

    return metrics


def train_evaluate_single_model(config: dict, framework: str, data: tuple, save_dir: Path, logger):
    """训练单个树模型并完成全流程评估（复用TreeModelTrainer）"""
    # 解包数据
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = data

    # 深拷贝配置，避免多模型间参数干扰
    model_config = copy.deepcopy(config)
    model_config['tree']['framework'] = framework
    logger.info(f"\n===== 开始训练 {framework} 模型 =====")

    # 初始化并训练模型（复用TreeModelTrainer）
    try:
        tree_trainer = TreeModelTrainer(model_config)
        tree_trainer.fit(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            categorical_features=model_config['data']['categorical_features']
        )
    except Exception as e:
        logger.error(f"{framework} 训练失败: {str(e)}", exc_info=True)
        raise

    # 保存模型（复用TreeModelTrainer的save方法）
    model_save_path = save_dir / f"{framework}_model.txt"
    tree_trainer.save(model_save_path)
    logger.info(f"{framework} 模型已保存: {model_save_path}")

    # 保存特征重要性（复用TreeModelTrainer的方法）
    fi_df = tree_trainer.get_feature_importance()
    fi_path = save_dir / f"{framework}_feature_importance.csv"
    fi_df.to_csv(fi_path, index=False, encoding="utf-8")
    logger.info(f"{framework} 特征重要性已保存: {fi_path}")
    logger.info(f"\n{framework} Top 10 重要特征:\n{fi_df.head(10).to_string(index=False)}")

    # 评估验证集
    val_metrics = evaluate_and_save_results(
        model=tree_trainer,
        X=X_val,
        y_true=y_val,
        logger=logger,
        save_dir=save_dir,
        model_name=framework,
        prefix="Validation"
    )

    # 评估测试集（如有）
    test_metrics = None
    if X_test is not None and y_test is not None:
        test_metrics = evaluate_and_save_results(
            model=tree_trainer,
            X=X_test,
            y_true=y_test,
            logger=logger,
            save_dir=save_dir,
            model_name=framework,
            prefix="Test"
        )

    logger.info(f"\n===== {framework} 模型训练/评估完成 =====")
    return {
        "framework": framework,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics
    }


def main(args):
    # 1. 加载配置（修复from_args问题，适配ConfigParser实际用法）
    try:
        # 优先通过ConfigParser加载配置文件（项目标准方式）
        config_parser = ConfigParser(args.config)
        config = config_parser.config  # 若ConfigParser返回对象，取其config属性；若直接返回字典则用config_parser
        # 兼容load_config方法（备选）
        # config = ConfigParser.load_config(args.config)
    except Exception as e:
        # 兜底：直接读取yaml文件（确保配置加载成功）
        import yaml
        with open(args.config, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        logger.warning(f"ConfigParser加载失败，使用兜底方式读取配置: {str(e)}")

    # 2. 初始化日志（复用setup_logger）
    logger = setup_logger(config)
    logger.info("===== 启动树模型对比实验 =====")
    logger.info(f"实验时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"配置文件: {args.config}")

    # 3. 设置随机种子（复用main.py的set_seed）
    set_seed(config['system']['seed'])
    logger.info(f"随机种子已设置为: {config['system']['seed']}")

    # 4. 创建实验目录（按时间戳区分，避免覆盖）
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_root = Path(config['system']['checkpoint']['save_dir']) / f"tree_model_comparison_{timestamp}"
    save_root.mkdir(parents=True, exist_ok=True)
    config['system']['checkpoint']['save_dir'] = str(save_root)
    logger.info(f"实验结果根目录: {save_root}")

    # 5. 加载数据（复用load_data）
    data = load_data(config, logger)
    (X_train, y_train), (X_val, y_val), (X_test, y_test), preprocessor, save_dir = data

    # 6. 定义待测试的树模型框架
    frameworks = ["lightgbm", "xgboost", "catboost"]
    # 命令行指定单一框架（方便调试）
    if args.tree_framework and args.tree_framework in frameworks:
        frameworks = [args.tree_framework]
        logger.info(f"仅测试指定框架: {args.tree_framework}")
    else:
        logger.info(f"测试所有框架: {', '.join(frameworks)}")

    # 7. 批量训练/评估模型
    all_metrics = []
    for framework in frameworks:
        # 覆盖n_estimators（命令行参数优先）
        if args.n_estimators:
            config['tree']['n_estimators'] = args.n_estimators
            logger.info(f"覆盖{framework}的n_estimators为: {args.n_estimators}")

        # 训练评估单个模型
        model_metrics = train_evaluate_single_model(
            config=config,
            framework=framework,
            data=((X_train, y_train), (X_val, y_val), (X_test, y_test)),
            save_dir=save_dir,
            logger=logger
        )
        all_metrics.append(model_metrics)

    # 8. 汇总所有模型指标（方便与原有算法对比）
    logger.info("\n===== 所有树模型指标汇总（用于与原有算法对比） =====")
    summary_rows = []
    for metrics in all_metrics:
        fw = metrics['framework']
        # 验证集指标
        if metrics['val_metrics']:
            val_row = {
                "framework": fw,
                "dataset": "validation",
                **metrics['val_metrics']
            }
            summary_rows.append(val_row)
        # 测试集指标
        if metrics['test_metrics']:
            test_row = {
                "framework": fw,
                "dataset": "test",
                **metrics['test_metrics']
            }
            summary_rows.append(test_row)

    # 保存汇总指标（CSV格式，方便对比）
    summary_df = pd.DataFrame(summary_rows)
    summary_path = save_dir / "tree_models_metrics_summary.csv"
    summary_df.to_csv(summary_path, index=False, encoding="utf-8")
    logger.info(f"\n指标汇总表已保存: {summary_path}")
    logger.info("\n【模型指标汇总表】")
    logger.info(summary_df.to_string(index=False))

    logger.info("\n===== 树模型对比实验完成 =====")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="树模型测试脚本（用于与原有算法对比）")
    parser.add_argument("--config", type=str,
                      default="/gpool/home/wanghongyang/WangHY/WYY/tree_enhanced_dl/configs/default_config.yaml",
                      help="配置文件路径")
    parser.add_argument("--tree-framework", type=str,
                      choices=["lightgbm", "xgboost", "catboost"],
                      help="指定单一树模型框架（不指定则测试所有三种）")
    parser.add_argument("--n-estimators", type=int,
                      help="树的数量（覆盖配置文件中的设置）")

    args = parser.parse_args()
    main(args)