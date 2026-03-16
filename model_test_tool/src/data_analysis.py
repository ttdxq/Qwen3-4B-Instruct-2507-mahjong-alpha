#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模型测试结果数据分析程序
计算每个模型的样本全对率和样本0分率
"""

import json
import os
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Any
import argparse


class ModelDataAnalyzer:
    """分析模型测试结果的类"""
    
    def __init__(self, result_dir: str = "result"):
        """
        初始化分析器
        
        Args:
            result_dir: 结果文件目录
        """
        self.result_dir = Path(result_dir)
        self.model_stats = defaultdict(lambda: {
            'total_samples': 0,
            'perfect_samples': 0,  # 全对样本数
            'zero_samples': 0,     # 0分样本数
            'sample_details': defaultdict(lambda: {
                'scores': [],
                'dataset': '',
                'correct_answer': ''
            })
        })
    
    def load_result_files(self) -> List[Path]:
        """
        加载所有结果文件
        
        Returns:
            结果文件路径列表
        """
        result_files = []
        
        # 检查路径是否存在
        if not self.result_dir.exists():
            return result_files
        
        # 如果路径是文件，直接返回
        if self.result_dir.is_file():
            result_files.append(self.result_dir)
            return result_files
        
        # 如果是目录，先检查是否直接包含结果文件（任务目录）
        direct_result_file = self.result_dir / "complete_test_results.json"
        if direct_result_file.exists():
            result_files.append(direct_result_file)
            return result_files
        
        # 否则作为结果根目录，遍历所有任务子目录
        for task_dir in self.result_dir.iterdir():
            if task_dir.is_dir():
                result_file = task_dir / "complete_test_results.json"
                if result_file.exists():
                    result_files.append(result_file)
        
        return sorted(result_files)
    
    def analyze_file(self, file_path: Path) -> None:
        """
        分析单个结果文件
        
        Args:
            file_path: 结果文件路径
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 遍历每个模型的结果
            for model_name, samples in data.items():
                model_data = self.model_stats[model_name]
                
                # 遍历每个样本
                for sample in samples:
                    sample_id = sample['sample_id']
                    dataset = sample['dataset']
                    correct_answer = sample['correct_answer']
                    scores = sample['scores']
                    
                    # 存储样本详细信息
                    sample_key = f"{dataset}_{sample_id}"
                    if not model_data['sample_details'][sample_key]['scores']:
                        model_data['sample_details'][sample_key]['dataset'] = dataset
                        model_data['sample_details'][sample_key]['correct_answer'] = correct_answer
                    
                    model_data['sample_details'][sample_key]['scores'].extend(scores)
            
        except Exception as e:
            # 分析文件时出错，静默处理
            pass
    
    def calculate_statistics(self) -> Dict[str, Dict[str, float]]:
        """
        计算统计信息
        
        Returns:
            模型统计信息字典
        """
        results = {}
        
        for model_name, model_data in self.model_stats.items():
            # 计算统计
            total_samples = len(model_data['sample_details'])
            perfect_samples = 0
            zero_samples = 0
            
            for sample_key, sample_info in model_data['sample_details'].items():
                scores = sample_info['scores']
                
                if not scores:
                    continue
                
                # 检查是否全对（所有分数都是1.0）
                if all(score == 1.0 for score in scores):
                    perfect_samples += 1
                
                # 检查是否0分（所有分数都是0.0）
                elif all(score == 0.0 for score in scores):
                    zero_samples += 1
            
            # 计算比例
            perfect_rate = (perfect_samples / total_samples * 100) if total_samples > 0 else 0
            zero_rate = (zero_samples / total_samples * 100) if total_samples > 0 else 0
            
            results[model_name] = {
                'total_samples': total_samples,
                'perfect_samples': perfect_samples,
                'zero_samples': zero_samples,
                'perfect_rate': perfect_rate,
                'zero_rate': zero_rate
            }
        
        return results
    
    def print_statistics(self, stats: Dict[str, Dict[str, float]]) -> None:
        """
        打印统计结果
        
        Args:
            stats: 统计信息字典
        """
        # 模型测试结果统计分析（静默输出）
        
        # 按模型名称排序
        sorted_models = sorted(stats.items())
        
        # 统计结果已保存到文件，不再打印到控制台
    
    def save_statistics(self, stats: Dict[str, Dict[str, float]], output_file: str = "model_statistics.json") -> None:
        """
        保存统计结果到文件
        
        Args:
            stats: 统计信息字典
            output_file: 输出文件路径
        """
        try:
            # 准备保存的数据
            save_data = {
                'summary': stats,
                'details': {}
            }
            
            # 添加详细样本信息
            for model_name, model_data in self.model_stats.items():
                save_data['details'][model_name] = {
                    'perfect_samples': [],
                    'zero_samples': [],
                    'mixed_samples': []
                }
                
                for sample_key, sample_info in model_data['sample_details'].items():
                    scores = sample_info['scores']
                    sample_data = {
                        'sample_key': sample_key,
                        'dataset': sample_info['dataset'],
                        'correct_answer': sample_info['correct_answer'],
                        'scores': scores,
                        'test_count': len(scores)
                    }
                    
                    if all(score == 1.0 for score in scores):
                        save_data['details'][model_name]['perfect_samples'].append(sample_data)
                    elif all(score == 0.0 for score in scores):
                        save_data['details'][model_name]['zero_samples'].append(sample_data)
                    else:
                        save_data['details'][model_name]['mixed_samples'].append(sample_data)
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(save_data, f, ensure_ascii=False, indent=2)
            
            # 详细统计信息已保存到文件
            
        except Exception as e:
            # 保存统计信息时出错，静默处理
            pass
    
    def run(self, output_file: str = None) -> Dict[str, Dict[str, float]]:
        """
        运行完整的分析流程
        
        Args:
            output_file: 输出文件路径，如果为None则自动确定
            
        Returns:
            统计信息字典
        """
        # 加载并分析所有结果文件
        result_files = self.load_result_files()
        
        if not result_files:
            return {}
        
        for result_file in result_files:
            self.analyze_file(result_file)
        
        # 计算统计信息
        stats = self.calculate_statistics()
        
        # 打印统计结果
        self.print_statistics(stats)
        
        # 保存统计结果
        if output_file is not None:
            # 使用指定的输出路径
            self.save_statistics(stats, output_file)
        else:
            # 自动确定输出路径：保存在源结果文件夹中
            if len(result_files) == 1:
                # 如果是单个结果文件，保存到该文件所在目录
                output_path = result_files[0].parent / "model_statistics.json"
            else:
                # 如果是多个结果文件，保存到结果根目录
                output_path = self.result_dir / "model_statistics.json"
            self.save_statistics(stats, str(output_path))
        
        return stats


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='分析模型测试结果的统计信息')
    parser.add_argument('--result-dir', default='result', help='结果文件目录')
    parser.add_argument('--output', help='输出文件路径（默认保存在源结果文件夹中）')
    parser.add_argument('--no-save', action='store_true', help='不保存结果到文件')
    
    args = parser.parse_args()
    
    # 创建分析器并运行
    analyzer = ModelDataAnalyzer(args.result_dir)
    output_file = None if args.no_save else args.output
    analyzer.run(output_file)


if __name__ == '__main__':
    main()