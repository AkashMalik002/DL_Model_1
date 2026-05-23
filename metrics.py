"""
Evaluation Metrics for model performance assessment
"""

import numpy as np


class EvaluationMetrics:
    """Calculate classification and regression metrics"""
    
    @staticmethod
    def accuracy(y_true, y_pred):
        """Accuracy = (TP + TN) / (TP + TN + FP + FN)"""
        return np.mean(y_true == y_pred)
    
    @staticmethod
    def precision(y_true, y_pred, pos_label=1):
        """Precision = TP / (TP + FP)"""
        tp = np.sum((y_pred == pos_label) & (y_true == pos_label))
        fp = np.sum((y_pred == pos_label) & (y_true != pos_label))
        return tp / (tp + fp) if (tp + fp) > 0 else 0.0
    
    @staticmethod
    def recall(y_true, y_pred, pos_label=1):
        """Recall = TP / (TP + FN)"""
        tp = np.sum((y_pred == pos_label) & (y_true == pos_label))
        fn = np.sum((y_pred != pos_label) & (y_true == pos_label))
        return tp / (tp + fn) if (tp + fn) > 0 else 0.0
    
    @staticmethod
    def f1_score(y_true, y_pred, pos_label=1):
        """F1 = 2 * (Precision * Recall) / (Precision + Recall)"""
        precision = EvaluationMetrics.precision(y_true, y_pred, pos_label)
        recall = EvaluationMetrics.recall(y_true, y_pred, pos_label)
        if precision + recall == 0:
            return 0.0
        return 2 * (precision * recall) / (precision + recall)
    
    @staticmethod
    def rmse(y_true, y_pred):
        """Root Mean Squared Error = √(1/m * Σ(y_true - y_pred)²)"""
        return np.sqrt(np.mean((y_true - y_pred) ** 2))
    
    @staticmethod
    def mae(y_true, y_pred):
        """Mean Absolute Error = 1/m * Σ|y_true - y_pred|"""
        return np.mean(np.abs(y_true - y_pred))
    
    @staticmethod
    def confusion_matrix(y_true, y_pred, num_classes=2):
        """Compute confusion matrix"""
        cm = np.zeros((num_classes, num_classes))
        for i in range(len(y_true)):
            cm[y_true[i], y_pred[i]] += 1
        return cm
