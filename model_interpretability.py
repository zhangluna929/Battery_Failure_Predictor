import numpy as np
import shap
import lime
import lime.lime_tabular
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix, classification_report, 
    roc_curve, auc, precision_recall_curve
)

class BatteryModelInterpreter:
    def __init__(self, model, feature_names):
        """
        初始化模型解释器
        :param model: 训练好的模型
        :param feature_names: 特征名称列表
        """
        self.model = model
        self.feature_names = feature_names
        
    def explain_with_shap(self, X_test, sample_idx=None):
        """
        使用SHAP解释模型预测
        :param X_test: 测试数据
        :param sample_idx: 要解释的样本索引，如果为None则解释所有样本
        """
        # 创建SHAP解释器
        explainer = shap.KernelExplainer(self.model.predict, X_test)
        
        if sample_idx is not None:
            # 解释单个样本
            shap_values = explainer.shap_values(X_test[sample_idx:sample_idx+1])
            plt.figure()
            shap.force_plot(
                explainer.expected_value[0],
                shap_values[0][0],
                X_test[sample_idx],
                feature_names=self.feature_names,
                show=False
            )
            plt.title("SHAP解释 - 单个样本")
            plt.show()
        else:
            # 解释所有样本
            shap_values = explainer.shap_values(X_test)
            plt.figure()
            shap.summary_plot(
                shap_values[0],
                X_test,
                feature_names=self.feature_names,
                show=False
            )
            plt.title("SHAP特征重要性总结")
            plt.tight_layout()
            plt.show()
            
    def explain_with_lime(self, X_train, X_test, sample_idx):
        """
        使用LIME解释模型预测
        :param X_train: 训练数据
        :param X_test: 测试数据
        :param sample_idx: 要解释的样本索引
        """
        # 创建LIME解释器
        explainer = lime.lime_tabular.LimeTabularExplainer(
            X_train,
            feature_names=self.feature_names,
            class_names=['Normal', 'Capacity Fade', 'Resistance Increase', 'Overheating'],
            mode='classification'
        )
        
        # 解释预测
        exp = explainer.explain_instance(
            X_test[sample_idx],
            self.model.predict,
            num_features=len(self.feature_names)
        )
        
        # 显示解释
        plt.figure(figsize=(10, 6))
        exp.as_pyplot_figure()
        plt.title("LIME解释 - 局部特征重要性")
        plt.tight_layout()
        plt.show()

class BatteryModelEvaluator:
    def __init__(self, model):
        """
        初始化模型评估器
        :param model: 训练好的模型
        """
        self.model = model
        
    def evaluate_classification_metrics(self, X_test, y_test):
        """
        评估分类指标
        :param X_test: 测试数据
        :param y_test: 测试标签
        """
        # 获取预测结果
        y_pred = self.model.predict(X_test)
        y_pred_classes = np.argmax(y_pred[0], axis=1)
        
        # 计算混淆矩阵
        cm = confusion_matrix(y_test, y_pred_classes)
        
        # 打印分类报告
        print("\n分类报告:")
        print(classification_report(y_test, y_pred_classes, target_names=['Normal', 'Capacity Fade', 'Resistance Increase', 'Overheating']))
        
        # 绘制混淆矩阵
        plt.figure(figsize=(8, 6))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('混淆矩阵')
        plt.colorbar()
        plt.xlabel('预测标签')
        plt.ylabel('真实标签')
        plt.tight_layout()
        plt.show()
        
    def plot_roc_curve(self, X_test, y_test):
        """
        绘制ROC曲线
        :param X_test: 测试数据
        :param y_test: 测试标签
        """
        y_pred_proba = self.model.predict(X_test)[0][:, 1]
        
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2,
                 label=f'ROC曲线 (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('假阳性率')
        plt.ylabel('真阳性率')
        plt.title('接收者操作特征(ROC)曲线')
        plt.legend(loc="lower right")
        plt.show()
        
    def plot_precision_recall_curve(self, X_test, y_test):
        """
        绘制精确率-召回率曲线
        :param X_test: 测试数据
        :param y_test: 测试标签
        """
        y_pred_proba = self.model.predict(X_test)[0][:, 1]
        
        precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
        pr_auc = auc(recall, precision)
        
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, color='blue', lw=2,
                 label=f'PR曲线 (AUC = {pr_auc:.2f})')
        plt.xlabel('召回率')
        plt.ylabel('精确率')
        plt.title('精确率-召回率曲线')
        plt.legend(loc="lower left")
        plt.show()
        
    def evaluate_soc_estimation(self, X_test, y_test_soc):
        """
        评估SOC估算性能
        :param X_test: 测试数据
        :param y_test_soc: 真实SOC值
        """
        y_pred_soc = self.model.predict(X_test)[1]
        
        # 计算MAE和RMSE
        mae = np.mean(np.abs(y_pred_soc - y_test_soc))
        rmse = np.sqrt(np.mean((y_pred_soc - y_test_soc) ** 2))
        
        print(f"\nSOC估算评估结果:")
        print(f"平均绝对误差 (MAE): {mae:.4f}")
        print(f"均方根误差 (RMSE): {rmse:.4f}")
        
        # 绘制预测值vs真实值散点图
        plt.figure(figsize=(8, 6))
        plt.scatter(y_test_soc, y_pred_soc, alpha=0.5)
        plt.plot([0, 1], [0, 1], 'r--')
        plt.xlabel('真实SOC值')
        plt.ylabel('预测SOC值')
        plt.title('SOC估算性能')
        plt.tight_layout()
        plt.show() 