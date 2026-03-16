import json
import random
import time
import os
import threading
from datetime import datetime
from openai import OpenAI
from typing import Dict, Any, Optional, List
from concurrent.futures import CancelledError, ThreadPoolExecutor, as_completed


class ModelEvaluator:
    def __init__(self):
        # 添加请求时间统计
        self.model_request_times = {}  # {model_name: [request_times_list]}
        self.first_request_time = None  # 记录第一个请求的时间
        self.last_request_time = None  # 记录最后一个请求的时间
        # 为每个模型单独记录时间统计
        self.model_first_request_times = {}  # {model_name: first_request_time}
        self.model_last_request_times = {}  # {model_name: last_request_time}
        # 添加token消耗统计
        self.model_token_usage = {}  # {model_name: {"prompt_tokens": [], "completion_tokens": [], "total_tokens": []}}
        # 添加停止标志
        self.should_stop = False
        self.stop_lock = threading.Lock()
        # 添加活动请求计数器，用于等待已发出请求完成
        self.active_requests = 0
        self.active_requests_lock = threading.Lock()
        # 未完成请求记录
        self.unfinished_requests = []  # list of {model,dataset,sample_id,eval_id,reason}
        self.unfinished_lock = threading.Lock()
        self._unfinished_request_map = {}  # {(model,dataset_path,sample_id,eval_id): entry}

    def set_stop_flag(self, stop=True):
        """设置停止标志"""
        with self.stop_lock:
            self.should_stop = stop

    def check_stop_flag(self):
        """检查停止标志"""
        with self.stop_lock:
            return self.should_stop

    def increment_active_requests(self):
        """增加活动请求计数"""
        with self.active_requests_lock:
            self.active_requests += 1

    def decrement_active_requests(self):
        """减少活动请求计数"""
        with self.active_requests_lock:
            self.active_requests -= 1
            if self.active_requests < 0:
                self.active_requests = 0

    def add_unfinished_request(
        self, model_name, dataset_path, sample_id, eval_id, reason="unfinished"
    ):
        """记录未完成的请求"""
        with self.unfinished_lock:
            dataset_key = os.path.normpath(dataset_path) if dataset_path else ""
            key = (model_name, dataset_key, sample_id, eval_id)
            existing = self._unfinished_request_map.get(key)
            if existing is None:
                entry = {
                    "model": model_name,
                    "dataset": os.path.basename(dataset_path) if dataset_path else "",
                    "dataset_path": dataset_path,
                    "sample_id": sample_id,
                    "eval_id": eval_id,
                    "reason": reason,
                }
                self._unfinished_request_map[key] = entry
                self.unfinished_requests.append(entry)
                return

            # 去重：如果已有记录，尽量保留“更具体”的原因与更完整的路径信息
            priority = {"unfinished": 0, "not_completed": 1, "empty_result": 2}
            if priority.get(reason, 0) > priority.get(existing.get("reason"), 0):
                existing["reason"] = reason
            if (not existing.get("dataset_path")) and dataset_path:
                existing["dataset_path"] = dataset_path
                existing["dataset"] = os.path.basename(dataset_path)

    def get_active_requests_count(self):
        """获取活动请求计数"""
        with self.active_requests_lock:
            return self.active_requests

    def wait_for_active_requests(self, timeout=300):
        """等待所有活动请求完成"""
        start_time = time.time()
        while time.time() - start_time < timeout:
            with self.active_requests_lock:
                if self.active_requests <= 0:
                    return True
            time.sleep(0.1)  # 等待0.1秒后再次检查
        return False  # 超时未完成

    def set_stop_flag(self, stop=True):
        """设置停止标志"""
        with self.stop_lock:
            self.should_stop = stop

    def get_stop_flag(self):
        """获取停止标志"""
        with self.stop_lock:
            return self.should_stop

    @staticmethod
    def _sanitize_task_name(task_name):
        """将任务名转换为安全的文件夹名（与 save_results 保持一致）"""
        if not task_name:
            return None
        import re

        return re.sub(r'[<>:"/\\\\|?*]', "_", str(task_name))

    def _get_task_dir(self, output_dir, task_name):
        safe_task_name = self._sanitize_task_name(task_name)
        if not safe_task_name or not output_dir:
            return None
        return os.path.join(output_dir, safe_task_name)

    def _build_pending_requests_from_disk(
        self,
        task_dir,
        models,
        datasets,
        sample_count,
        eval_times,
        preloaded_samples=None,
        filter_empty_results=False,
    ):
        """
        从任务目录中读取历史进度，计算剩余待执行的请求 (sample_id, eval_id)。

        返回:
            (pending_requests, completed_total, completed_per_model)
            - pending_requests: {model_name: {dataset_path: [(sample_id, eval_id), ...]}}
              若无法读取历史进度，则返回 (None, None, None)
            - completed_total: 历史已完成（有效）请求总数（去重后）；若无历史进度则为 None
            - completed_per_model: {model_name: completed_count}（去重后）；若无历史进度则为 None
        """
        if not task_dir or not os.path.exists(task_dir):
            return None, None, None

        # 快速判定：没有任何进度文件时不计算 pending（避免首跑生成超大结构）
        import glob

        temp_dir = os.path.join(task_dir, "temp")
        has_progress = False
        if os.path.exists(temp_dir) and glob.glob(
            os.path.join(temp_dir, "temp_progress_*.json")
        ):
            has_progress = True
        if os.path.exists(os.path.join(task_dir, "complete_test_results.json")):
            has_progress = True
        if not has_progress:
            return None, None, None

        merged = self.merge_results(
            task_dir,
            {"raw_data": {}, "scores": {}, "unfinished_requests": []},
            is_final=False,
        )
        raw_data = merged.get("raw_data", {}) or {}

        # dataset basename -> [dataset_path,...] 映射（用于兼容旧 raw_data 只存 basename 的情况）
        basename_to_paths = {}
        for dp in datasets:
            basename_to_paths.setdefault(os.path.basename(dp), []).append(dp)

        completed_keys = set()
        completed_ids = {}  # model -> dataset_path -> set((sample_id, eval_id))

        if isinstance(raw_data, dict):
            for model_name, items in raw_data.items():
                if not isinstance(items, list):
                    continue
                for item in items:
                    if not isinstance(item, dict):
                        continue
                    dp = item.get("dataset_path")
                    if not dp:
                        base = item.get("dataset")
                        paths = basename_to_paths.get(base, [])
                        if len(paths) == 1:
                            dp = paths[0]
                        elif paths:
                            dp = paths[0]
                    if not dp:
                        continue

                    dp_norm = os.path.normpath(dp)
                    # 尽量与 datasets 列表里的路径统一（同一文件不同分隔符会导致匹配失败）
                    for cand in datasets:
                        if os.path.normpath(cand) == dp_norm:
                            dp = cand
                            break

                    sample_id = item.get("sample_id")
                    if sample_id is None:
                        continue
                    try:
                        sid = int(sample_id)
                    except Exception:
                        continue

                    extracted_answers = item.get("extracted_answers", [])
                    if not isinstance(extracted_answers, list):
                        continue
                    eval_ids = item.get("eval_ids")
                    if not isinstance(eval_ids, list) or len(eval_ids) != len(
                        extracted_answers
                    ):
                        eval_ids = list(range(len(extracted_answers)))

                    for ev, ans in zip(eval_ids, extracted_answers):
                        try:
                            eid = int(ev)
                        except Exception:
                            continue
                        if filter_empty_results and (
                            ans is None or not str(ans).strip()
                        ):
                            continue
                        key = (model_name, os.path.normpath(dp), sid, eid)
                        if key in completed_keys:
                            continue
                        completed_keys.add(key)
                        completed_ids.setdefault(model_name, {}).setdefault(
                            dp, set()
                        ).add((sid, eid))

        completed_total = len(completed_keys)
        if completed_total <= 0:
            return None, 0, {}

        completed_per_model = {}
        for model_name, ds_map in completed_ids.items():
            if not isinstance(ds_map, dict):
                continue
            model_count = 0
            for id_set in ds_map.values():
                if isinstance(id_set, set):
                    model_count += len(id_set)
            completed_per_model[model_name] = model_count

        pending = {}
        for model_name in models:
            model_pending = {}
            for dp in datasets:
                done = completed_ids.get(model_name, {}).get(dp, set())
                # 使用预加载样本长度（若存在）确保 sample_id 范围正确
                sample_len = sample_count
                if (
                    preloaded_samples
                    and isinstance(preloaded_samples, dict)
                    and dp in preloaded_samples
                    and isinstance(preloaded_samples[dp], list)
                ):
                    sample_len = len(preloaded_samples[dp])

                pending_pairs = []
                for sid in range(sample_len):
                    for eid in range(eval_times):
                        if (sid, eid) not in done:
                            pending_pairs.append((sid, eid))
                if pending_pairs:
                    model_pending[dp] = pending_pairs

            pending[model_name] = model_pending

        return pending, completed_total, completed_per_model

    def save_partial_results(
        self,
        results,
        output_dir,
        config=None,
        task_name=None,
        filter_empty_results=False,
    ):
        """
        保存部分结果到临时文件

        Args:
            results: 评测结果字典
            output_dir: 输出目录
            config: 配置字典
            task_name: 任务名称
            filter_empty_results: 是否过滤空结果
        """
        import os
        import json
        from datetime import datetime

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[
            :-3
        ]  # Include milliseconds

        # 如果提供了任务名称，则使用任务专用文件夹
        safe_task_name = task_name if task_name else None
        if safe_task_name:
            # 替换可能导致路径问题的字符
            import re

            safe_task_name = re.sub(r'[<>:"/\\|?*]', "_", safe_task_name)

        if safe_task_name:
            task_dir = os.path.join(output_dir, safe_task_name)
            os.makedirs(task_dir, exist_ok=True)

            # 创建temp子文件夹用于存放临时数据
            temp_dir = os.path.join(task_dir, "temp")
            os.makedirs(temp_dir, exist_ok=True)

            # 保存到任务的temp文件夹
            temp_results_file = os.path.join(
                temp_dir, f"partial_results_{timestamp}.json"
            )
            temp_scores_file = os.path.join(temp_dir, f"partial_scores_{timestamp}.txt")
        else:
            task_dir = output_dir
            temp_results_file = os.path.join(
                output_dir, f"partial_results_{timestamp}.json"
            )
            temp_scores_file = os.path.join(
                output_dir, f"partial_scores_{timestamp}.txt"
            )

        try:
            # 如果需要过滤空结果，处理raw_data
            filtered_results = results.copy()
            if filter_empty_results and "raw_data" in results:
                filtered_raw_data = {}
                for model_name, model_results in results["raw_data"].items():
                    filtered_model_results = []
                    for result_item in model_results:
                        # 检查提取的答案是否为空
                        extracted_answers = result_item.get("extracted_answers", [])
                        # 如果所有提取的答案都为空，则跳过此结果
                        if extracted_answers and all(
                            not ans.strip() for ans in extracted_answers
                        ):
                            continue  # 跳过空结果
                        filtered_model_results.append(result_item)
                    filtered_raw_data[model_name] = filtered_model_results
                filtered_results["raw_data"] = filtered_raw_data

            # 保存原始数据
            with open(temp_results_file, "w", encoding="utf-8") as f:
                json.dump(filtered_results, f, ensure_ascii=False, indent=2)

            # 保存分数（如果有的话）
            if "scores" in filtered_results:
                with open(temp_scores_file, "w", encoding="utf-8") as f:
                    f.write("模型评测分数（临时）\n")
                    f.write("=" * 30 + "\n\n")
                    f.write("模型分数:\n")
                    for model_name, score in filtered_results["scores"].items():
                        f.write(f" {model_name}: {score}\n")

            return {
                "temp_results_file": temp_results_file,
                "temp_scores_file": temp_scores_file,
            }
        except Exception as e:
            print(f"保存临时结果失败: {e}")
            return None

    def validate_turn_ratios(self, turn_ratios):
        """
        验证巡目比例配置的合法性
        早巡1-6: early
        中巡7-12: mid
        晚巡>=13: late
        """
        if not isinstance(turn_ratios, dict):
            return False

        required_keys = {"early", "mid", "late"}
        if set(turn_ratios.keys()) != required_keys:
            return False

        total = 0.0
        for key, value in turn_ratios.items():
            if not isinstance(value, (int, float)) or value < 0 or value > 1:
                return False
            total += value

        # 检查总和是否接近1.0（允许小的浮点数误差）
        return abs(total - 1.0) < 0.001

    @staticmethod
    def _report_progress(progress_callback, current, total):
        """Report progress to either a Qt signal or a plain callable."""
        if not progress_callback:
            return
        try:
            if hasattr(progress_callback, "emit"):
                progress_callback.emit(current, total)
            elif callable(progress_callback):
                progress_callback(current, total)
        except Exception:
            # Progress reporting must never break evaluation.
            pass

    def evaluate_with_progress_tracking(
        self,
        datasets,
        models,
        config,
        progress_callback=None,
        result_callback=None,
        max_requests=None,
        model_max_requests=None,
        total_task_requests=None,
        previous_task_completed=0,
        pending_requests=None,
    ):
        """
        Evaluate models on datasets with progress tracking and ability to stop at specified request count
        max_requests: 总的请求数限制
        model_max_requests: 每个模型的请求数限制字典 {model_name: max_requests_for_model}
        total_task_requests: 整个任务的总请求数（用于判断是否最终完成）
        previous_task_completed: 之前已完成的请求数
        """
        # 初始化模型请求时间统计（如果不存在）
        if not hasattr(self, "model_request_times") or self.model_request_times is None:
            self.model_request_times = {}

        results: Dict[str, Any] = {
            "scores": {},
            "details": {},
            "raw_data": {},  # 添加原始数据存储
        }

        sample_count = config.get("sample_count", 10)
        eval_times = config.get("eval_times", 1)
        correct_score = config.get("correct_score", 1.0)
        wrong_score = config.get("wrong_score", 0.0)
        score_agg = config.get("score_agg", "sum")
        output_dir = config.get("output_dir", ".")
        concurrent_models = config.get("concurrent_models", True)  # 获取并发模型设置
        filter_empty_results = config.get("filter_empty_results", False)

        # 获取巡目比例配置
        turn_ratios = config.get(
            "turn_ratios", {"early": 0.33, "mid": 0.33, "late": 0.34}
        )
        # 验证巡目比例配置的合法性
        if not self.validate_turn_ratios(turn_ratios):
            raise ValueError("巡目比例配置不合法：比例总和必须为1.0")

        # 预加载样本（如果配置中提供），用于保持样本ID一致
        preloaded_samples = config.get("preloaded_samples", {})

        # 自动从任务目录推导 pending_requests（用于中断续跑/避免重复执行）
        task_name = config.get("task_name", None)
        try:
            previous_total_completed = (
                int(previous_task_completed)
                if isinstance(previous_task_completed, (int, float))
                else 0
            )
        except Exception:
            previous_total_completed = 0
        previous_model_completed = {}
        if pending_requests is None and task_name:
            task_dir = self._get_task_dir(output_dir, task_name)
            auto_pending, completed_total, completed_per_model = (
                self._build_pending_requests_from_disk(
                    task_dir,
                    models,
                    datasets,
                    sample_count,
                    eval_times,
                    preloaded_samples=preloaded_samples,
                    filter_empty_results=filter_empty_results,
                )
            )
            if completed_total is not None:
                previous_total_completed = completed_total
            if completed_per_model is not None and isinstance(
                completed_per_model, dict
            ):
                previous_model_completed = completed_per_model
            if auto_pending is not None:
                pending_requests = auto_pending

        # 计算总任务数
        if pending_requests is not None and isinstance(pending_requests, dict):
            total_tasks = 0
            for model_name in models:
                model_pending = pending_requests.get(model_name, {})
                if not isinstance(model_pending, dict):
                    continue
                for dataset_path in datasets:
                    dataset_pending = model_pending.get(dataset_path, [])
                    if isinstance(dataset_pending, list):
                        total_tasks += len(dataset_pending)
        else:
            total_tasks = len(models) * len(datasets) * sample_count * eval_times
        completed_tasks = 0
        completed_requests = 0  # 跟踪已完成的请求数
        completed_requests_per_model = {}  # 跟踪每个模型已完成的请求数
        attempted_requests = 0  # 跟踪本次运行实际发出的请求数（包含空回答等失败情况）
        attempted_requests_per_model = {}  # 每个模型本次运行实际发出的请求数

        stop_requested_at_entry = self.check_stop_flag()
        # 重置未完成请求记录
        self.unfinished_requests = []
        self._unfinished_request_map = {}

        # 初始化每个模型的请求数计数器
        for model_name in models:
            completed_requests_per_model[model_name] = 0
            attempted_requests_per_model[model_name] = 0

        # 生成一个本次评测的随机种子，确保所有模型使用相同的数据集，但每次评测运行使用不同的数据
        evaluation_seed = (
            int(time.time() * 10000) % 100000
        )  # Use microseconds as seed for this evaluation run

        # 预先加载所有数据集的样本，确保所有模型使用相同的数据集
        dataset_samples = {}
        for dataset_path in datasets:
            if self.check_stop_flag():
                break
            if preloaded_samples and dataset_path in preloaded_samples:
                dataset_samples[dataset_path] = preloaded_samples[dataset_path]
            else:
                dataset_samples[dataset_path] = self.read_dataset_with_fixed_seed(
                    dataset_path,
                    sample_count,
                    seed=evaluation_seed,
                    turn_ratios=turn_ratios,
                )

        # 确保 model_max_requests 是字典类型
        if model_max_requests is not None and not isinstance(model_max_requests, dict):
            # 如果不是字典，可能是传递了错误的参数，将其设为 None
            model_max_requests = None

        if stop_requested_at_entry:
            concurrent_models = False

        if concurrent_models and len(models) > 1 and not self.check_stop_flag():
            # Use concurrent model evaluation - 需要修改并发模型评估以支持请求数限制
            model_results = self.evaluate_models_concurrent_with_limits(
                models,
                datasets,
                config,
                sample_count,
                eval_times,
                score_agg,
                progress_callback,
                result_callback,
                total_tasks,
                completed_tasks,
                evaluation_seed,
                max_requests,
                model_max_requests,
                pending_requests=pending_requests,
                preloaded_samples=dataset_samples,
            )
            # Merge results and update completed requests per model
            for model_name, model_result in model_results.items():
                # 确保model_result是字典类型
                if not isinstance(model_result, dict):
                    continue

                results["scores"][model_name] = model_result.get("score", 0.0)
                results["details"][model_name] = model_result.get("details", [])
                results["raw_data"][model_name] = model_result.get("raw_data", [])
                # 更新每个模型的完成请求数 - 从model_result中获取实际完成的请求数
                if "completed_requests" in model_result:
                    actual_model_requests = model_result["completed_requests"]
                    completed_requests_per_model[model_name] += actual_model_requests
                    completed_requests += actual_model_requests  # 累加到总请求数
                else:
                    # 如果model_result中没有completed_requests，根据模型处理的数据量估算
                    expected_requests = len(datasets) * sample_count * eval_times
                    actual_model_requests = expected_requests
                    completed_requests_per_model[model_name] += actual_model_requests
                    completed_requests += actual_model_requests  # 累加到总请求数
        else:
            # Process each model sequentially
            for model_name in models:
                # 检查停止标志
                if self.check_stop_flag():
                    break

                model_total_score = 0.0
                model_details = []
                model_raw_data = []  # 存储原始数据

                # Get model config to check concurrent settings
                model_config_data = config.get("model_configs", {}).get(model_name)
                if model_config_data and isinstance(model_config_data, dict):
                    from core.model_config import ModelConfig

                    model_config = ModelConfig.from_dict(model_config_data)
                    enable_concurrent = model_config.enable_concurrent
                    max_concurrent_requests = model_config.max_concurrent_requests
                else:
                    enable_concurrent = False
                    max_concurrent_requests = 10

                # 获取当前模型的请求数限制
                model_request_limit = None
                if (
                    model_max_requests
                    and isinstance(model_max_requests, dict)
                    and model_name in model_max_requests
                ):
                    model_request_limit = model_max_requests[model_name]

                # Process each dataset
                for dataset_path in datasets:
                    # 检查停止标志
                    if self.check_stop_flag():
                        break

                    # 若存在 pending_requests，仅执行剩余待补齐的 (sample_id, eval_id)
                    allowed_eval_ids = None
                    if pending_requests is not None and isinstance(
                        pending_requests, dict
                    ):
                        model_pending = pending_requests.get(model_name, {})
                        pending_pairs = (
                            model_pending.get(dataset_path)
                            if isinstance(model_pending, dict)
                            else None
                        )
                        if not pending_pairs:
                            continue
                        allowed_eval_ids = {}
                        for sid, eid in pending_pairs:
                            allowed_eval_ids.setdefault(sid, set()).add(eid)

                    # Use pre-loaded samples to ensure all models use the same dataset
                    samples = dataset_samples[dataset_path]
                    if enable_concurrent:
                        # Use concurrent processing for samples with request limits
                        sample_results = self.process_samples_concurrent_with_limits(
                            model_name,
                            samples,
                            eval_times,
                            config,
                            max_concurrent_requests,
                            progress_callback,
                            total_tasks,
                            completed_tasks,
                            max_requests,
                            model_max_requests,
                            model_request_limit,
                            completed_requests,
                            completed_requests_per_model[model_name],
                            dataset_path=dataset_path,
                            allowed_eval_ids=allowed_eval_ids,
                        )
                        # 更新计数器 - 统计实际的请求数（样本数 × 评测次数）
                        actual_processed_requests = 0
                        for sample_result in sample_results:
                            if sample_result is not None:
                                # Count the actual number of requests (number of scores in sample_scores)
                                actual_processed_requests += len(
                                    sample_result.get("sample_scores", [])
                                )
                        completed_tasks += actual_processed_requests
                        completed_requests += actual_processed_requests
                        completed_requests_per_model[model_name] += (
                            actual_processed_requests
                        )

                        # 检查是否达到总请求数限制
                        if max_requests and completed_requests >= max_requests:
                            break
                    else:
                        # Use optimized processing for samples - batch process to reduce gaps
                        sample_results = []
                        # Collect all requests first, then process them in batches to minimize gaps
                        all_requests = []
                        for i, sample in enumerate(samples):
                            if allowed_eval_ids and i not in allowed_eval_ids:
                                continue
                            correct_answer = self.extract_correct_answer(sample)
                            original_prompt = sample.get("text", "")

                            eval_loop = (
                                sorted(allowed_eval_ids[i])
                                if allowed_eval_ids and i in allowed_eval_ids
                                else list(range(eval_times))
                            )
                            for j in eval_loop:
                                # 检查是否达到总的请求数限制（按“发出请求数”计数）
                                if (
                                    max_requests
                                    and (attempted_requests + len(all_requests))
                                    >= max_requests
                                ):
                                    break
                                # 检查是否达到当前模型的请求数限制（按“发出请求数”计数）
                                if (
                                    model_request_limit
                                    and (
                                        attempted_requests_per_model[model_name]
                                        + len(all_requests)
                                    )
                                    >= model_request_limit
                                ):
                                    break
                                # 检查停止标志
                                if self.check_stop_flag():
                                    break

                                all_requests.append(
                                    {
                                        "sample_id": i,
                                        "sample": sample,
                                        "eval_id": j,
                                        "correct_answer": correct_answer,
                                        "original_prompt": original_prompt,
                                    }
                                )
                            # 达到限制或停止后，不再继续收集更多样本
                            if self.check_stop_flag():
                                break
                            if (
                                max_requests
                                and (attempted_requests + len(all_requests))
                                >= max_requests
                            ):
                                break
                            if (
                                model_request_limit
                                and (
                                    attempted_requests_per_model[model_name]
                                    + len(all_requests)
                                )
                                >= model_request_limit
                            ):
                                break

                        # Process all requests sequentially but with minimal gaps
                        current_sample_results = {}
                        planned_ids = {
                            (r.get("sample_id"), r.get("eval_id")) for r in all_requests
                        }
                        completed_ids = set()
                        empty_ids = set()
                        for request in all_requests:
                            # 检查是否达到限制
                            if max_requests and attempted_requests >= max_requests:
                                break
                            if (
                                model_request_limit
                                and attempted_requests_per_model[model_name]
                                >= model_request_limit
                            ):
                                break
                            # 检查停止标志
                            if self.check_stop_flag():
                                break

                            sample_id = request["sample_id"]
                            sample = request["sample"]
                            correct_answer = request["correct_answer"]
                            original_prompt = request["original_prompt"]
                            eval_id = request["eval_id"]

                            # Call model API
                            if self.check_stop_flag():
                                break
                            attempted_requests += 1
                            attempted_requests_per_model[model_name] += 1
                            model_answer = self.call_model_api(
                                model_name, sample, config
                            )
                            extracted_answer = self.extract_answer(model_answer)
                            is_empty = (extracted_answer is None) or (
                                isinstance(extracted_answer, str)
                                and not extracted_answer.strip()
                            )
                            if filter_empty_results and is_empty:
                                # 空回答视为未完成：不计入完成数、不写入结果
                                self.add_unfinished_request(
                                    model_name,
                                    dataset_path,
                                    sample_id,
                                    eval_id,
                                    "empty_result",
                                )
                                empty_ids.add((sample_id, eval_id))
                                continue
                            score = self.calculate_score(
                                model_answer, correct_answer, correct_score, wrong_score
                            )

                            # Store results temporarily grouped by sample_id
                            if sample_id not in current_sample_results:
                                current_sample_results[sample_id] = {
                                    "correct_answer": correct_answer,
                                    "original_prompt": original_prompt,
                                    "sample_scores": [],
                                    "model_answers": [],
                                    "extracted_answers": [],
                                    "eval_ids": [],
                                }

                            current_sample_results[sample_id]["sample_scores"].append(
                                score
                            )
                            current_sample_results[sample_id]["model_answers"].append(
                                model_answer
                            )
                            current_sample_results[sample_id][
                                "extracted_answers"
                            ].append(extracted_answer)
                            current_sample_results[sample_id]["eval_ids"].append(
                                eval_id
                            )
                            completed_ids.add((sample_id, eval_id))

                            # Report progress
                            completed_tasks += 1
                            completed_requests += 1
                            completed_requests_per_model[model_name] += 1

                            if progress_callback:
                                progress_callback.emit(completed_tasks, total_tasks)

                        # 记录未完成的请求（未获得有效结果或未被执行）
                        missing_ids = planned_ids - completed_ids
                        for sid, eid in missing_ids:
                            if (sid, eid) in empty_ids:
                                continue
                            self.add_unfinished_request(
                                model_name, dataset_path, sid, eid, "not_completed"
                            )

                        # Convert to final sample results format
                        for sample_id, data in current_sample_results.items():
                            sample_results.append(
                                {
                                    "sample_id": sample_id,
                                    "correct_answer": data["correct_answer"],
                                    "original_prompt": data["original_prompt"],
                                    "sample_scores": data["sample_scores"],
                                    "model_answers": data["model_answers"],
                                    "extracted_answers": data["extracted_answers"],
                                    "eval_ids": data.get("eval_ids", []),
                                }
                            )

                    # Process sample results
                    for sample_result in sample_results:
                        if sample_result is None:  # 如果是空结果（因为达到限制），跳过
                            continue
                        # 检查停止标志
                        if self.check_stop_flag():
                            break

                        i = sample_result["sample_id"]
                        correct_answer = sample_result["correct_answer"]
                        original_prompt = sample_result["original_prompt"]
                        sample_scores = sample_result["sample_scores"]
                        model_answers = sample_result["model_answers"]
                        extracted_answers = sample_result.get("extracted_answers", [])

                        # Aggregate scores for this sample
                        if score_agg == "sum":
                            final_sample_score = sum(sample_scores)
                        else:  # avg
                            final_sample_score = sum(sample_scores) / len(sample_scores)

                        model_total_score += final_sample_score

                        # Store details
                        detail = {
                            "dataset": os.path.basename(dataset_path),
                            "sample_id": i,
                            "correct_answer": correct_answer,
                            "model_answers": sample_scores,
                            "final_score": final_sample_score,
                        }
                        model_details.append(detail)

                        # 提取巡目信息 - 使用 original_prompt 创建临时样本对象
                        temp_sample = {"text": original_prompt}
                        turn_number = self.extract_turn_number(temp_sample)
                        turn_category = self.get_turn_category(turn_number)

                        # Store raw data including original prompt, model answers, and extracted answers
                        raw_data_entry = {
                            "dataset": os.path.basename(dataset_path),
                            "dataset_path": dataset_path,
                            "sample_id": i,
                            "original_prompt": original_prompt,
                            "correct_answer": correct_answer,
                            "model_name": model_name,
                            "model_answers": model_answers,
                            "extracted_answers": extracted_answers,  # 添加提取的答案
                            "eval_ids": sample_result.get(
                                "eval_ids", list(range(len(extracted_answers)))
                            ),
                            "scores": sample_scores,
                            "final_score": final_sample_score,
                            "turn_number": turn_number,  # 巡目数字
                            "turn_category": turn_category,  # 巡目类别
                        }
                        model_raw_data.append(raw_data_entry)

                        # Report result
                        if result_callback:
                            result_callback.emit(
                                {"model": model_name, "score": model_total_score}
                            )

                results["scores"][model_name] = model_total_score
                results["details"][model_name] = model_details
                results["raw_data"][model_name] = model_raw_data  # 保存原始数据

        # 更新评估器的当前结果（用于在停止时保存部分结果）
        self.current_scores = results.get("scores", {})
        self.current_details = results.get("details", {})
        self.current_raw_data = results.get("raw_data", {})
        # 依据过滤后的raw_data重新计算有效完成数，避免空结果计入
        effective_model_completed = {}
        total_effective_completed = 0
        for m_name, entries in self.current_raw_data.items():
            if not isinstance(entries, list):
                continue
            count = 0
            for entry in entries:
                if not isinstance(entry, dict):
                    continue
                if filter_empty_results:
                    extracted = entry.get("extracted_answers", [])
                    if isinstance(extracted, list):
                        count += sum(
                            1
                            for ans in extracted
                            if ans is not None and str(ans).strip()
                        )
                    else:
                        eval_ids = entry.get("eval_ids", [])
                        if isinstance(eval_ids, list):
                            count += len(eval_ids)
                else:
                    scores = entry.get("scores", [])
                    if isinstance(scores, list):
                        count += len(scores)
                    else:
                        model_answers = entry.get("model_answers", [])
                        if isinstance(model_answers, list):
                            count += len(model_answers)
            effective_model_completed[m_name] = count
            total_effective_completed += count
        # 如果重新计算得到的有效值大于原值，使用原值；否则使用有效值
        if total_effective_completed <= completed_requests:
            completed_requests = total_effective_completed
            completed_requests_per_model = effective_model_completed
        # self.completed_requests 用于写入 temp_progress：这里存储“累计完成数”（历史 + 本次增量）
        self.completed_requests = previous_total_completed + completed_requests
        self.model_completed_requests = completed_requests_per_model
        # 为 UI 刷新/续跑持久化：按模型累计完成数（历史 + 本次增量）
        cumulative_per_model = (
            dict(previous_model_completed)
            if isinstance(previous_model_completed, dict)
            else {}
        )
        if isinstance(completed_requests_per_model, dict):
            for m_name, inc in completed_requests_per_model.items():
                try:
                    inc_n = int(inc)
                except Exception:
                    inc_n = 0
                try:
                    prev_n = int(cumulative_per_model.get(m_name, 0) or 0)
                except Exception:
                    prev_n = 0
                cumulative_per_model[m_name] = prev_n + inc_n
        self.total_model_completed_requests = cumulative_per_model

        # 获取任务名称
        task_name = config.get("task_name", None)

        # 判断任务是否完成（所有请求都已完成）
        # 先基于计数判断，再结合未完成列表修正
        if total_task_requests is not None and total_task_requests > 0:
            total_completed_now = previous_total_completed + completed_requests
            is_task_completed = total_completed_now >= total_task_requests
        else:
            total_expected_requests = (
                len(models) * len(datasets) * sample_count * eval_times
            )
            is_task_completed = (
                previous_total_completed + completed_requests
            ) >= total_expected_requests

        # 只要存在未完成请求（空回答/中断/未执行等），都视为未完成
        if getattr(self, "unfinished_requests", []):
            is_task_completed = False

        # 最终填充计数结果，供保存临时文件/最终文件使用
        results["completed_requests"] = completed_requests
        results["completed_tasks"] = completed_tasks
        results["model_completed_requests"] = (
            completed_requests_per_model  # 添加每个模型的完成请求数
        )
        results["is_completed"] = is_task_completed  # 添加任务完成状态
        results["unfinished_requests"] = getattr(self, "unfinished_requests", [])
        saved_files = self.save_results(
            results,
            output_dir,
            config,
            task_name=task_name,
            is_final=is_task_completed,
            filter_empty_results=filter_empty_results,
        )

        if is_task_completed:
            # 任务完成，保存最终结果
            if saved_files:
                results["scores_file"] = saved_files.get("scores_file")
                results["details_file"] = saved_files.get("details_file")
                results["raw_data_file"] = saved_files.get("raw_data_file")
        else:
            # 任务进行中，保存临时文件
            if saved_files and saved_files.get("temp_file"):
                results["temp_file"] = saved_files["temp_file"]

        # 新增：如果任务完成，从合并结果中恢复完整的统计数据到实例变量
        if is_task_completed and task_name:
            # 调用merge_results获取完整的历史数据
            task_dir = self._get_task_dir(output_dir, task_name) or os.path.join(
                output_dir, str(task_name)
            )
            merged_results = self.merge_results(task_dir, results, is_final=True)

            # 恢复请求时间统计数据
            if "model_request_times" in merged_results:
                for model_name, request_times in merged_results[
                    "model_request_times"
                ].items():
                    if not hasattr(self, "model_request_times"):
                        self.model_request_times = {}
                    self.model_request_times[model_name] = request_times

            # 恢复token使用统计数据
            if "model_token_usage" in merged_results:
                for model_name, token_data in merged_results[
                    "model_token_usage"
                ].items():
                    if not hasattr(self, "model_token_usage"):
                        self.model_token_usage = {}
                    self.model_token_usage[model_name] = token_data

            # 恢复未完成请求列表，确保中断信息不丢失
            if "unfinished_requests" in merged_results:
                self.unfinished_requests = merged_results.get("unfinished_requests", [])
                results["unfinished_requests"] = self.unfinished_requests

        results["completed_requests"] = completed_requests
        results["completed_tasks"] = completed_tasks
        results["model_completed_requests"] = (
            completed_requests_per_model  # 添加每个模型的完成请求数
        )
        results["is_completed"] = is_task_completed  # 添加任务完成状态
        results["unfinished_requests"] = getattr(self, "unfinished_requests", [])

        return results

    def save_partial_results(
        self,
        results,
        output_dir,
        config=None,
        task_name=None,
        filter_empty_results=False,
    ):
        """
        保存部分结果到临时文件
        """
        import os
        import json
        from datetime import datetime

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[
            :-3
        ]  # Include milliseconds

        # 如果提供了任务名称，则使用任务专用文件夹
        safe_task_name = task_name if task_name else None
        if safe_task_name:
            # 替换可能导致路径问题的字符
            import re

            safe_task_name = re.sub(r'[<>:"/\\|?*]', "_", safe_task_name)

        if safe_task_name:
            task_dir = os.path.join(output_dir, safe_task_name)
            os.makedirs(task_dir, exist_ok=True)

            # 创建temp子文件夹用于存放临时数据
            temp_dir = os.path.join(task_dir, "temp")
            os.makedirs(temp_dir, exist_ok=True)

            # 保存到任务的temp文件夹
            temp_results_file = os.path.join(
                temp_dir, f"partial_results_{timestamp}.json"
            )
            temp_scores_file = os.path.join(temp_dir, f"partial_scores_{timestamp}.txt")
        else:
            task_dir = output_dir
            temp_results_file = os.path.join(
                output_dir, f"partial_results_{timestamp}.json"
            )
            temp_scores_file = os.path.join(
                output_dir, f"partial_scores_{timestamp}.txt"
            )

        try:
            # 如果需要过滤空结果，处理raw_data
            filtered_results = results.copy()
            if filter_empty_results and "raw_data" in results:
                filtered_raw_data = {}
                for model_name, model_results in results["raw_data"].items():
                    filtered_model_results = []
                    for result_item in model_results:
                        # 检查提取的答案是否为空
                        extracted_answers = result_item.get("extracted_answers", [])
                        # 如果所有提取的答案都为空，则跳过此结果
                        if extracted_answers and all(
                            not ans.strip() for ans in extracted_answers
                        ):
                            continue  # 跳过空结果
                        filtered_model_results.append(result_item)
                    filtered_raw_data[model_name] = filtered_model_results
                filtered_results["raw_data"] = filtered_raw_data

            # 保存原始数据
            with open(temp_results_file, "w", encoding="utf-8") as f:
                json.dump(filtered_results, f, ensure_ascii=False, indent=2)

            # 保存分数（如果有的话）
            if "scores" in results:
                with open(temp_scores_file, "w", encoding="utf-8") as f:
                    f.write("模型评测分数（临时）\n")
                    f.write("=" * 30 + "\n\n")
                    f.write("模型分数:\n")
                    for model_name, score in results["scores"].items():
                        f.write(f" {model_name}: {score}\n")

            return {
                "temp_results_file": temp_results_file,
                "temp_scores_file": temp_scores_file,
            }
        except Exception as e:
            print(f"保存临时结果失败: {e}")
            return None

    def evaluate(
        self, datasets, models, config, progress_callback=None, result_callback=None
    ):
        """
        Evaluate models on datasets
        """
        # 初始化模型请求时间统计（如果不存在）
        if not hasattr(self, "model_request_times") or self.model_request_times is None:
            self.model_request_times = {}

        results: Dict[str, Any] = {
            "scores": {},
            "details": {},
            "raw_data": {},  # 添加原始数据存储
        }

        sample_count = config.get("sample_count", 10)
        eval_times = config.get("eval_times", 1)
        correct_score = config.get("correct_score", 1.0)
        wrong_score = config.get("wrong_score", 0.0)
        score_agg = config.get("score_agg", "sum")
        output_dir = config.get("output_dir", ".")
        concurrent_models = config.get("concurrent_models", True)  # 获取并发模型设置

        total_tasks = len(models) * len(datasets) * sample_count * eval_times
        completed_tasks = 0

        # 生成一个本次评测的随机种子，确保所有模型使用相同的数据集，但每次评测运行使用不同的数据
        evaluation_seed = (
            int(time.time() * 100000) % 100000
        )  # Use microseconds as seed for this evaluation run

        # 预先加载所有数据集的样本，确保所有模型使用相同的数据集
        dataset_samples = {}

        # 获取巡目比例配置
        turn_ratios = config.get(
            "turn_ratios", {"early": 0.33, "mid": 0.33, "late": 0.34}
        )
        # 验证巡目比例配置的合法性
        if not self.validate_turn_ratios(turn_ratios):
            raise ValueError("巡目比例配置不合法：比例总和必须为1.0")

        for dataset_path in datasets:
            dataset_samples[dataset_path] = self.read_dataset_with_fixed_seed(
                dataset_path,
                sample_count,
                seed=evaluation_seed,
                turn_ratios=turn_ratios,
            )

        if concurrent_models and len(models) > 1:
            # Use concurrent model evaluation
            model_results = self.evaluate_models_concurrent(
                models,
                datasets,
                config,
                sample_count,
                eval_times,
                score_agg,
                progress_callback,
                result_callback,
                total_tasks,
                completed_tasks,
                evaluation_seed,
            )
            # Merge results
            for model_name, model_result in model_results.items():
                # 确保model_result是字典类型
                if not isinstance(model_result, dict):
                    print(
                        f"Error: model_result for {model_name} is not a dict: {type(model_result)} - {model_result}"
                    )
                    continue

                results["scores"][model_name] = model_result.get("score", 0.0)
                results["details"][model_name] = model_result.get("details", [])
                results["raw_data"][model_name] = model_result.get("raw_data", [])
        else:
            # Process each model sequentially
            for model_name in models:
                model_total_score = 0.0
                model_details = []
                model_raw_data = []  # 存储原始数据

                # Get model config to check concurrent settings
                model_config_data = config.get("model_configs", {}).get(model_name)
                if model_config_data and isinstance(model_config_data, dict):
                    from core.model_config import ModelConfig

                    model_config = ModelConfig.from_dict(model_config_data)
                    enable_concurrent = model_config.enable_concurrent
                    max_concurrent_requests = model_config.max_concurrent_requests
                else:
                    enable_concurrent = False
                    max_concurrent_requests = 10

                # Process each dataset
                for dataset_path in datasets:
                    # Use pre-loaded samples to ensure all models use the same dataset
                    samples = dataset_samples[dataset_path]
                    if enable_concurrent:
                        # Use concurrent processing for samples
                        sample_results = self.process_samples_concurrent(
                            model_name,
                            samples,
                            eval_times,
                            config,
                            max_concurrent_requests,
                            progress_callback,
                            total_tasks,
                            completed_tasks,
                        )
                        completed_tasks += len(samples) * eval_times
                    else:
                        # Use optimized processing for samples - batch process to reduce gaps
                        sample_results = []
                        # Collect all requests first, then process them in batches to minimize gaps
                        all_requests = []
                        for i, sample in enumerate(samples):
                            correct_answer = self.extract_correct_answer(sample)
                            original_prompt = sample.get("text", "")

                            for j in range(eval_times):
                                all_requests.append(
                                    {
                                        "sample_id": i,
                                        "sample": sample,
                                        "eval_id": j,
                                        "correct_answer": correct_answer,
                                        "original_prompt": original_prompt,
                                    }
                                )

                        # Process all requests sequentially but with minimal gaps
                        current_sample_results = {}
                        for request in all_requests:
                            sample_id = request["sample_id"]
                            sample = request["sample"]
                            correct_answer = request["correct_answer"]
                            original_prompt = request["original_prompt"]
                            eval_id = request["eval_id"]

                            # Call model API
                            model_answer = self.call_model_api(
                                model_name, sample, config
                            )
                            extracted_answer = self.extract_answer(model_answer)
                            score = self.calculate_score(
                                model_answer, correct_answer, correct_score, wrong_score
                            )

                            # Store results temporarily grouped by sample_id
                            if sample_id not in current_sample_results:
                                current_sample_results[sample_id] = {
                                    "correct_answer": correct_answer,
                                    "original_prompt": original_prompt,
                                    "sample_scores": [],
                                    "model_answers": [],
                                    "extracted_answers": [],
                                }

                            current_sample_results[sample_id]["sample_scores"].append(
                                score
                            )
                            current_sample_results[sample_id]["model_answers"].append(
                                model_answer
                            )
                            current_sample_results[sample_id][
                                "extracted_answers"
                            ].append(extracted_answer)

                            # Report progress
                            completed_tasks += 1
                            if progress_callback:
                                progress_callback.emit(completed_tasks, total_tasks)

                        # Convert to final sample results format
                        for sample_id, data in current_sample_results.items():
                            sample_results.append(
                                {
                                    "sample_id": sample_id,
                                    "correct_answer": data["correct_answer"],
                                    "original_prompt": data["original_prompt"],
                                    "sample_scores": data["sample_scores"],
                                    "model_answers": data["model_answers"],
                                    "extracted_answers": data["extracted_answers"],
                                }
                            )

                    # Process sample results
                    for sample_result in sample_results:
                        i = sample_result["sample_id"]
                        correct_answer = sample_result["correct_answer"]
                        original_prompt = sample_result["original_prompt"]
                        sample_scores = sample_result["sample_scores"]
                        model_answers = sample_result["model_answers"]
                        extracted_answers = sample_result.get("extracted_answers", [])

                        # Aggregate scores for this sample
                        if score_agg == "sum":
                            final_sample_score = sum(sample_scores)
                        else:  # avg
                            final_sample_score = sum(sample_scores) / len(sample_scores)

                        model_total_score += final_sample_score

                        # Store details
                        detail = {
                            "dataset": os.path.basename(dataset_path),
                            "sample_id": i,
                            "correct_answer": correct_answer,
                            "model_answers": sample_scores,
                            "final_score": final_sample_score,
                        }
                        model_details.append(detail)

                        # 提取巡目信息 - 使用 original_prompt 创建临时样本对象
                        temp_sample = {"text": original_prompt}
                        turn_number = self.extract_turn_number(temp_sample)
                        turn_category = self.get_turn_category(turn_number)

                        # Store raw data including original prompt, model answers, and extracted answers
                        raw_data_entry = {
                            "dataset": os.path.basename(dataset_path),
                            "sample_id": i,
                            "original_prompt": original_prompt,
                            "correct_answer": correct_answer,
                            "model_name": model_name,
                            "model_answers": model_answers,
                            "extracted_answers": extracted_answers,  # 添加提取的答案
                            "scores": sample_scores,
                            "final_score": final_sample_score,
                            "turn_number": turn_number,  # 巡目数字
                            "turn_category": turn_category,  # 巡目类别
                        }
                        model_raw_data.append(raw_data_entry)

                        # Report result
                        if result_callback:
                            result_callback.emit(
                                {"model": model_name, "score": model_total_score}
                            )

                results["scores"][model_name] = model_total_score
                results["details"][model_name] = model_details
                results["raw_data"][model_name] = model_raw_data  # 保存原始数据

        # 获取任务名称（evaluate方法不使用任务名称，总是生成最终结果）
        task_name = config.get("task_name", None)

        # Save results to files (总是作为最终结果)
        saved_files = self.save_results(
            results, output_dir, config, task_name=task_name, is_final=True
        )
        results["scores_file"] = saved_files["scores_file"]
        results["details_file"] = saved_files["details_file"]
        results["raw_data_file"] = saved_files["raw_data_file"]

        # 新增：如果任务完成，从合并结果中恢复完整的统计数据到实例变量
        if task_name:
            # 调用merge_results获取完整的历史数据
            task_dir = os.path.join(output_dir, task_name)
            merged_results = self.merge_results(task_dir, results, is_final=True)

            # 恢复请求时间统计数据
            if "model_request_times" in merged_results:
                for model_name, request_times in merged_results[
                    "model_request_times"
                ].items():
                    if not hasattr(self, "model_request_times"):
                        self.model_request_times = {}
                    self.model_request_times[model_name] = request_times

            # 恢复token使用统计数据
            if "model_token_usage" in merged_results:
                for model_name, token_data in merged_results[
                    "model_token_usage"
                ].items():
                    if not hasattr(self, "model_token_usage"):
                        self.model_token_usage = {}
                    self.model_token_usage[model_name] = token_data

        return results

    def read_dataset_with_fixed_seed(
        self, dataset_path, sample_count, seed=None, turn_ratios=None
    ):
        """
        Read samples from dataset file with random seed to ensure consistent sampling across models in same evaluation
        If seed is None, use current time-based seed for different samples each evaluation run
        Implements strict stratified sampling based on turn ratios by reading sufficient data
        """
        # Default turn ratios if not provided
        turn_ratios = turn_ratios or {"early": 0.33, "mid": 0.33, "late": 0.34}

        # Set random seed for consistent sampling
        if seed is None:
            import time

            seed = int(time.time() * 100000) % 10000  # Use microseconds as seed

        random.seed(seed)

        # Classify all samples into turn categories by reading through the file
        categorized_samples = {"early": [], "mid": [], "late": []}

        with open(dataset_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        data = json.loads(line)
                        turn_number = self.extract_turn_number(data)
                        turn_category = self.get_turn_category(turn_number)
                        if turn_category in categorized_samples:
                            categorized_samples[turn_category].append(data)
                    except json.JSONDecodeError:
                        continue

        # If no samples found, return empty list
        total_available = sum(len(samples) for samples in categorized_samples.values())
        if total_available == 0:
            return []

        # Calculate exact number of samples to take from each category
        # Ensure at least 1 sample from each category and total matches sample_count
        early_count = max(1, int(sample_count * turn_ratios["early"]))
        mid_count = max(1, int(sample_count * turn_ratios["mid"]))
        late_count = max(1, int(sample_count * turn_ratios["late"]))

        # Adjust counts to match exact sample_count
        total_calculated = early_count + mid_count + late_count
        excess = total_calculated - sample_count

        # Reduce excess from categories with the most samples available
        if excess > 0:
            categories = sorted(
                ["early", "mid", "late"],
                key=lambda x: len(categorized_samples[x]),
                reverse=True,
            )
            for i in range(excess):
                cat = categories[i % len(categories)]
                if eval(f"{cat}_count") > 1:  # Ensure at least 1 sample per category
                    exec(f"{cat}_count -= 1")

        # Handle case where total is still less than sample_count (rare)
        if total_calculated < sample_count:
            deficit = sample_count - total_calculated
            categories = sorted(
                ["early", "mid", "late"],
                key=lambda x: len(categorized_samples[x]),
                reverse=True,
            )
            for i in range(deficit):
                cat = categories[i % len(categories)]
                if len(categorized_samples[cat]) > eval(f"{cat}_count"):
                    exec(f"{cat}_count += 1")

        samples_per_category = {
            "early": early_count,
            "mid": mid_count,
            "late": late_count,
        }

        # Ensure we have enough samples in each category
        for category, count in samples_per_category.items():
            if len(categorized_samples[category]) < count:
                # If not enough samples in a category, use all available
                # This ensures we maintain as close to the desired ratios as possible
                samples_per_category[category] = len(categorized_samples[category])

        # Adjust total to match sum of available samples
        new_total = sum(samples_per_category.values())
        if new_total < sample_count:
            # Add remaining samples to the category with the most available samples
            remaining = sample_count - new_total
            most_available_category = max(
                categorized_samples,
                key=lambda x: len(categorized_samples[x]) - samples_per_category[x],
            )
            samples_per_category[most_available_category] += remaining

        # Sample from each category
        selected_samples = []

        for category, count in samples_per_category.items():
            if count <= 0:
                continue

            category_samples = categorized_samples[category]
            if len(category_samples) >= count:
                # Randomly sample from this category
                selected_samples.extend(random.sample(category_samples, count))
            else:
                # Take all available if not enough
                selected_samples.extend(category_samples)

        # Ensure exactly sample_count samples (should only happen if adjustments above are off)
        if len(selected_samples) > sample_count:
            selected_samples = random.sample(selected_samples, sample_count)
        elif len(selected_samples) < sample_count:
            # Last resort: add random samples from all available
            all_samples = [
                sample
                for cat_samples in categorized_samples.values()
                for sample in cat_samples
            ]
            if len(all_samples) >= sample_count:
                selected_samples = random.sample(all_samples, sample_count)

        return selected_samples

    def read_dataset(self, dataset_path, sample_count):
        """
        Read samples from dataset file (maintaining original method for backward compatibility)
        """
        return self.read_dataset_with_fixed_seed(dataset_path, sample_count)

    def extract_correct_answer(self, sample):
        """
        Extract correct answer from sample
        """
        text = sample.get("text", "")
        lines = text.split("\n")

        # Find the answer line (last non-empty line that doesn't start with [)
        for line in reversed(lines):
            stripped_line = line.strip()
            if stripped_line and not stripped_line.startswith("["):
                return stripped_line

        return ""

    def extract_turn_number(self, sample):
        """
        从样本中提取巡目信息
        格式通常是：(第X巡，牌墙余Y张)
        返回巡目数字，如果未找到则返回None
        """
        text = sample.get("text", "")
        import re

        # 匹配模式：(第X巡，牌墙余Y张) 或 (第X巡)
        pattern = r"第(\d+)巡"
        match = re.search(pattern, text)
        if match:
            return int(match.group(1))
        return None

    def get_turn_category(self, turn_number):
        """
        根据巡目数字返回巡目类别
        早巡: 1-6
        中巡: 7-12
        晚巡: >=13
        """
        if turn_number is None:
            return "unknown"
        elif 1 <= turn_number <= 6:
            return "early"
        elif 7 <= turn_number <= 12:
            return "mid"
        elif turn_number >= 13:
            return "late"
        else:
            return "unknown"

    def clean_prompt_for_model(self, original_prompt):
        """
        Remove the answer from the prompt to prevent model from seeing it
        """
        lines = original_prompt.split("\n")
        cleaned_lines = []
        task_section_found = False
        task_description_found = False  # 标记是否已找到任务描述

        for line in lines:
            if line.startswith("[任务]"):
                task_section_found = True
                task_description_found = False  # 重置任务描述标记
                cleaned_lines.append(line)
                continue

            if task_section_found:
                if not line.strip():  # 空行，添加到结果中
                    cleaned_lines.append(line)
                elif not line.startswith("[") and not task_description_found:
                    # 这是任务描述（如"根据当前情景，选择一张最应该打出的手牌。"），保留它
                    cleaned_lines.append(line)
                    task_description_found = True  # 标记已找到任务描述
                elif not line.startswith("[") and task_description_found:
                    # 这是答案行（在任务描述之后的非空行），跳过
                    continue
                else:
                    # 遇到下一个部分标记，重置标志
                    task_section_found = False
                    task_description_found = False
                    cleaned_lines.append(line)
            else:
                cleaned_lines.append(line)

        return "\n".join(cleaned_lines)

    def call_model_api(self, model_name, sample, config):
        """
        Call model API to get answer
        Returns only the content string (not the tuple with token info)
        """
        # Get model config
        model_config_data = config.get("model_configs", {}).get(model_name)
        if not model_config_data or not isinstance(model_config_data, dict):
            # Use default config with the correct model name
            from core.model_config import ModelConfig

            model_config = ModelConfig(model_name)
        else:
            # Convert dict to ModelConfig object
            from core.model_config import ModelConfig

            model_config = ModelConfig.from_dict(model_config_data)
            # Ensure the model name is correct even when loaded from config
            model_config.model_name = model_name

        # Check if concurrent requests are enabled
        if model_config.enable_concurrent:
            # Use concurrent request handling (returns string)
            result = self.call_model_api_concurrent(model_config, sample, config)
        else:
            # Use normal request handling (returns tuple)
            api_result = self.call_model_api_normal(model_config, sample, config)
            # Extract only the content string from the tuple
            if isinstance(api_result, tuple) and len(api_result) >= 1:
                result = api_result[0]  # First element is the content
            else:
                result = api_result

        return result

    def call_model_api_normal(
        self, model_config, sample, config=None, record_time=True
    ):
        """
        Call model API with normal (non-concurrent) requests
        Returns content and token usage info
        """
        if self.check_stop_flag():
            return ("", 0, 0, 0)

        # 增加活动请求计数
        self.increment_active_requests()

        # Prepare prompt and clean it
        original_prompt = sample.get("text", "")
        cleaned_prompt = self.clean_prompt_for_model(original_prompt)

        try:
            # 记录请求开始时间
            start_time = time.time()

            # 从模型配置获取超时时间
            timeout_value = getattr(model_config, "timeout", 60.0)

            if self.check_stop_flag():
                return ("", 0, 0, 0)

            # Create OpenAI client with only supported parameters
            client_args = {
                "base_url": model_config.api_base,
                "api_key": model_config.api_key
                or "sk-no-key-required",  # Provide default key if empty
                "timeout": timeout_value,
            }

            # Only add proxies if it's a supported parameter in this version
            # Remove any unsupported parameters to avoid the "unexpected keyword argument" error
            client = OpenAI(**client_args)

            # Prepare messages with optional system message
            messages = []
            if model_config.system_message:
                messages.append(
                    {"role": "system", "content": model_config.system_message}
                )
            messages.append({"role": "user", "content": cleaned_prompt})

            if self.check_stop_flag():
                return ("", 0, 0, 0)

            # Call API with the configured request model
            response = client.chat.completions.create(
                model=model_config.request_model,  # 使用配置的请求模型名称
                messages=messages,
                temperature=model_config.temperature,
                max_tokens=model_config.max_tokens,
                top_p=model_config.top_p,
                frequency_penalty=model_config.frequency_penalty,
                presence_penalty=model_config.presence_penalty,
                timeout=timeout_value,
            )

            content = response.choices[0].message.content

            # Debug输出：API返回的原始内容
            print(
                f"[DEBUG] API返回原始内容: model={model_config.model_name}, content={repr(content[:100] if content else 'None')}"
            )

            # 获取token使用情况
            prompt_tokens = 0
            completion_tokens = 0
            total_tokens = 0
            if hasattr(response, "usage") and response.usage:
                prompt_tokens = response.usage.prompt_tokens
                completion_tokens = response.usage.completion_tokens
                total_tokens = response.usage.total_tokens

            # 记录请求结束时间并计算耗时
            end_time = time.time()
            request_time = end_time - start_time

            # 如果需要记录时间（在非并发模式下调用时记录）
            if record_time:
                # 使用线程锁确保时间统计的线程安全性
                with threading.Lock():
                    # 记录模型请求时间
                    model_name = model_config.model_name
                    if not hasattr(self, "model_request_times"):
                        self.model_request_times = {}
                    if model_name not in self.model_request_times:
                        self.model_request_times[model_name] = []
                    self.model_request_times[model_name].append(request_time)

                    # 记录token消耗
                    if not hasattr(self, "model_token_usage"):
                        self.model_token_usage = {}
                    if model_name not in self.model_token_usage:
                        self.model_token_usage[model_name] = {
                            "prompt_tokens": [],
                            "completion_tokens": [],
                            "total_tokens": [],
                        }
                    self.model_token_usage[model_name]["prompt_tokens"].append(
                        prompt_tokens
                    )
                    self.model_token_usage[model_name]["completion_tokens"].append(
                        completion_tokens
                    )
                    self.model_token_usage[model_name]["total_tokens"].append(
                        total_tokens
                    )

                    # 记录第一个请求时间
                    if self.first_request_time is None:
                        self.first_request_time = start_time
                    # 更新最后一个请求时间
                    self.last_request_time = end_time

                    # 记录每个模型的第一个和最后一个请求时间
                    if model_name not in self.model_first_request_times:
                        self.model_first_request_times[model_name] = start_time
                    else:
                        # 更新模型的最早请求时间（如果当前时间更早）
                        if start_time < self.model_first_request_times[model_name]:
                            self.model_first_request_times[model_name] = start_time

                    # 更新模型的最后请求时间（如果当前时间更晚）
                    if model_name not in self.model_last_request_times:
                        self.model_last_request_times[model_name] = end_time
                    else:
                        if end_time > self.model_last_request_times[model_name]:
                            self.model_last_request_times[model_name] = end_time

            # 返回内容和token信息
            return (
                content.strip() if content else "",
                prompt_tokens,
                completion_tokens,
                total_tokens,
            )

        except Exception as e:
            print(f"Error calling model API: {e}")
            # 即使出错也要记录时间
            end_time = time.time()
            request_time = end_time - start_time if "start_time" in locals() else 0
            # 如果需要记录时间（在非并发模式下调用时记录）
            if record_time:
                # 使用线程锁确保时间统计的线程安全性
                with threading.Lock():
                    model_name = model_config.model_name
                    if model_name not in self.model_request_times:
                        self.model_request_times[model_name] = []
                    self.model_request_times[model_name].append(request_time)

                    # 记录token消耗（出错时为0）
                    if model_name not in self.model_token_usage:
                        self.model_token_usage[model_name] = {
                            "prompt_tokens": [],
                            "completion_tokens": [],
                            "total_tokens": [],
                        }
                    self.model_token_usage[model_name]["prompt_tokens"].append(0)
                    self.model_token_usage[model_name]["completion_tokens"].append(0)
                    self.model_token_usage[model_name]["total_tokens"].append(0)

            # 返回空内容和0 token
            return ("", 0, 0, 0)
        finally:
            # 确保在请求完成时减少活动请求计数
            self.decrement_active_requests()

    def _call_model_api_with_timing(self, model_name, sample, config):
        """
        Call model API and return both answer and timing information
        Used for concurrent processing to properly measure request times
        """
        # Debug输出：发出请求
        # active_requests is tracked inside call_model_api_normal; avoid double counting here.
        print(
            f"[DEBUG] 发出请求: model={model_name}, 当前未完成请求数={self.get_active_requests_count()}"
        )

        try:
            # Get model config
            model_config_data = config.get("model_configs", {}).get(model_name)
            if not model_config_data or not isinstance(model_config_data, dict):
                # Use default config with the correct model name
                from core.model_config import ModelConfig

                model_config = ModelConfig(model_name)
            else:
                # Convert dict to ModelConfig object
                from core.model_config import ModelConfig

                model_config = ModelConfig.from_dict(model_config_data)
                # Ensure the model name is correct even when loaded from config
                model_config.model_name = model_name

            # Call the API and measure time (don't record in the call itself to avoid double recording)
            start_time = time.time()
            try:
                result = self.call_model_api_normal(
                    model_config, sample, record_time=False
                )
            except Exception as e:
                print(f"Error in timed API call: {e}")
                result = ""
            end_time = time.time()
            request_time = end_time - start_time

            # Debug输出：收到回复
            result_content = (
                result[0] if isinstance(result, tuple) and len(result) >= 1 else result
            )
            print(
                f"[DEBUG] 收到回复: model={model_name}, 长度={len(result_content) if result_content else 0}, 当前未完成请求数={self.get_active_requests_count()}"
            )
            if result_content and len(result_content) >= 10:
                print(f"[DEBUG] 回复内容前10个字符: {result_content[:10]}")

            # 在并发模式下也需要记录第一个和最后一个请求时间以及每个模型的时间
            with threading.Lock():
                # 记录第一个请求时间
                if self.first_request_time is None:
                    self.first_request_time = start_time
                # 更新最后一个请求时间
                if self.last_request_time is None or end_time > self.last_request_time:
                    self.last_request_time = end_time

                # 记录每个模型的第一个和最后一个请求时间
                if model_name not in self.model_first_request_times:
                    self.model_first_request_times[model_name] = start_time
                else:
                    # 更新模型的最早请求时间（如果当前时间更早）
                    if start_time < self.model_first_request_times[model_name]:
                        self.model_first_request_times[model_name] = start_time

                # 更新模型的最后请求时间（如果当前时间更晚）
                if model_name not in self.model_last_request_times:
                    self.model_last_request_times[model_name] = end_time
                else:
                    if end_time > self.model_last_request_times[model_name]:
                        self.model_last_request_times[model_name] = end_time

            # 在并发模式下也需要记录模型请求时间，因为call_model_api_normal中的record_time=False
            if not hasattr(self, "model_request_times"):
                self.model_request_times = {}
            if model_name not in self.model_request_times:
                self.model_request_times[model_name] = []
            self.model_request_times[model_name].append(request_time)

            # 同样记录token使用情况
            if hasattr(result, "__len__") and len(result) >= 4:
                prompt_tokens = result[1] if result[1] is not None else 0
                completion_tokens = result[2] if result[2] is not None else 0
                total_tokens = result[3] if result[3] is not None else 0
            else:
                # 如果result不包含token信息，使用0
                prompt_tokens = 0
                completion_tokens = 0
                total_tokens = 0

            if not hasattr(self, "model_token_usage"):
                self.model_token_usage = {}
            if model_name not in self.model_token_usage:
                self.model_token_usage[model_name] = {
                    "prompt_tokens": [],
                    "completion_tokens": [],
                    "total_tokens": [],
                }
            self.model_token_usage[model_name]["prompt_tokens"].append(prompt_tokens)
            self.model_token_usage[model_name]["completion_tokens"].append(
                completion_tokens
            )
            self.model_token_usage[model_name]["total_tokens"].append(total_tokens)

            return result, request_time  # Return both answer and timing info
        finally:
            # call_model_api_normal handles active_requests accounting.
            pass

    def call_model_api_concurrent(self, model_config, sample, config=None):
        """
        Call model API with concurrent request handling
        Returns content only
        """
        # For concurrent requests, we call the normal method but don't record time
        # Time recording will be handled separately in the concurrent processing methods
        api_result = self.call_model_api_normal(
            model_config, sample, config, record_time=False
        )
        # 处理返回值，确保只返回内容字符串
        if isinstance(api_result, tuple):
            content, _, _, _ = api_result
            return content
        else:
            return api_result

    def process_samples_concurrent(
        self,
        model_name,
        samples,
        eval_times,
        config,
        max_concurrent_requests,
        progress_callback,
        total_tasks,
        completed_tasks_start,
    ):
        """
        Process samples with concurrent requests (每分钟最大并发)
        """
        sample_results = []
        planned_ids = {(i, j) for i in range(len(samples)) for j in range(eval_times)}
        completed_ids = set()
        filter_empty_results = config.get("filter_empty_results", False)
        # 使用线程锁确保时间统计的线程安全
        time_lock = threading.Lock()
        # 请求时间记录，用于控制每分钟的请求数
        request_times = []
        request_times_lock = threading.Lock()

        # 获取同时并发总数限制（从模型配置读取）
        # 从模型配置中读取max_concurrent_total
        model_name = model_name if isinstance(model_name, str) else ""
        model_config_data = config.get("model_configs", {}).get(model_name, {})
        max_concurrent_total = model_config_data.get("max_concurrent_total", -1)

        # Use ThreadPoolExecutor for concurrent processing
        with ThreadPoolExecutor(max_workers=max_concurrent_requests) as executor:
            # Create futures for all sample evaluations
            futures = []
            in_flight = set()  # submitted but not completed futures
            future_to_sample = {}
            stop_submission = False

            for i, sample in enumerate(samples):
                # 【关键修复】在提交任何请求前，先检查停止标志
                if self.check_stop_flag():
                    stop_submission = True
                    break

                for j in range(eval_times):
                    # 等待直到在每分钟限制内
                    with request_times_lock:
                        current_time = time.time()
                        # 清理一分钟前的请求时间记录
                        request_times[:] = [
                            t for t in request_times if current_time - t <= 60
                        ]

                        # 如果当前分钟内请求数已达到限制，则等待
                        while len(request_times) >= max_concurrent_requests:
                            # 【关键修复】在等待时检查停止标志
                            if self.check_stop_flag():
                                stop_submission = True
                                break
                            time.sleep(0.1)  # 等待0.1秒后重新检查
                            current_time = time.time()
                            request_times[:] = [
                                t for t in request_times if current_time - t <= 60
                            ]

                        # 【关键修复】如果停止标志已设置，跳出循环
                        if self.check_stop_flag():
                            stop_submission = True
                            break

                        # 记录当前请求时间
                        request_times.append(current_time)

                    # 【关键修复】检查是否需要继续
                    if self.check_stop_flag():
                        stop_submission = True
                        break

                    # 等待直到同时并发总数限制内（考虑已提交但未完成的futures）
                    if max_concurrent_total > 0:
                        while True:
                            if self.check_stop_flag():
                                stop_submission = True
                                break
                            # Only count unfinished submitted tasks; futures list is cumulative.
                            in_flight = {f for f in in_flight if not f.done()}
                            if len(in_flight) < max_concurrent_total:
                                break
                            time.sleep(0.05)  # 等待0.05秒后重新检查

                    # 【关键修复】在提交任务前再次检查停止标志
                    if self.check_stop_flag():
                        stop_submission = True
                        break

                    # Submit with timing wrapper to record request times in concurrent mode
                    future = executor.submit(
                        self._call_model_api_with_timing, model_name, sample, config
                    )
                    futures.append(future)
                    in_flight.add(future)
                    future_to_sample[future] = (i, j, sample)

                if stop_submission:
                    break

            if stop_submission or self.check_stop_flag():
                for future in futures:
                    if not future.running() and not future.done():
                        future.cancel()

            # Process completed futures
            completed_count = 0
            sample_evaluations = {}  # {sample_id: {eval_id: (answer, score)}}

            for future in as_completed(futures):
                # 先获取sample信息，确保即使发生异常也能访问sample
                sample_id, eval_id, sample = future_to_sample[future]
                model_answer = ""

                try:
                    # Get result which includes both answer and timing info
                    result = future.result()
                except CancelledError:
                    continue
                except Exception as e:
                    print(f"Error in concurrent request: {e}")
                    # model_answer 已经初始化为空字符串
                    result = ""

                try:
                    if isinstance(result, tuple) and len(result) == 2:
                        model_answer, request_time = result
                        # Record timing information for concurrent requests
                        model_config_data = config.get("model_configs", {}).get(
                            model_name
                        )
                        if model_config_data and isinstance(model_config_data, dict):
                            from core.model_config import ModelConfig

                            model_config = ModelConfig.from_dict(model_config_data)
                            model_name_key = model_config.model_name
                        else:
                            model_name_key = model_name

                        # Use thread-safe timing recording
                        with time_lock:
                            if not hasattr(self, "model_request_times"):
                                self.model_request_times = {}
                            if model_name_key not in self.model_request_times:
                                self.model_request_times[model_name_key] = []
                            self.model_request_times[model_name_key].append(
                                request_time
                            )

                            # Also record token usage in concurrent processing
                            if not hasattr(self, "model_token_usage"):
                                self.model_token_usage = {}
                            if model_name_key not in self.model_token_usage:
                                self.model_token_usage[model_name_key] = {
                                    "prompt_tokens": [],
                                    "completion_tokens": [],
                                    "total_tokens": [],
                                }
                            # For concurrent processing, we don't have token info from the result, so use 0
                            self.model_token_usage[model_name_key][
                                "prompt_tokens"
                            ].append(0)
                            self.model_token_usage[model_name_key][
                                "completion_tokens"
                            ].append(0)
                            self.model_token_usage[model_name_key][
                                "total_tokens"
                            ].append(0)
                    else:
                        model_answer = result
                except Exception as e:
                    print(f"Error in concurrent result processing: {e}")

                # Get correct answer and calculate score
                correct_answer = self.extract_correct_answer(sample)
                correct_score = config.get("correct_score", 1.0)
                wrong_score = config.get("wrong_score", 0.0)
                score = self.calculate_score(
                    model_answer, correct_answer, correct_score, wrong_score
                )

                # Extract answer from model response
                extracted_answer = self.extract_answer(model_answer)

                # 若需要过滤空结果且答案为空，则标记未完成并跳过计数/保存
                is_empty = (extracted_answer is None) or (
                    isinstance(extracted_answer, str) and not extracted_answer.strip()
                )
                if filter_empty_results and is_empty:
                    self.add_unfinished_request(
                        model_name, None, sample_id, eval_id, "empty_result"
                    )
                    continue

                # Store evaluation results
                if sample_id not in sample_evaluations:
                    sample_evaluations[sample_id] = {
                        "correct_answer": correct_answer,
                        "original_prompt": sample.get("text", ""),
                        "sample_scores": [None] * eval_times,
                        "model_answers": [None] * eval_times,
                        "extracted_answers": [None] * eval_times,  # 添加提取答案的存储
                    }

                sample_evaluations[sample_id]["sample_scores"][eval_id] = score
                sample_evaluations[sample_id]["model_answers"][eval_id] = model_answer
                sample_evaluations[sample_id]["extracted_answers"][eval_id] = (
                    extracted_answer  # 存储提取的答案
                )
                completed_ids.add((sample_id, eval_id))

                # Update progress
                completed_count += 1
                if progress_callback and hasattr(progress_callback, "emit"):
                    # Use atomic update to prevent progress from jumping back
                    current_progress = completed_tasks_start + completed_count
                    try:
                        progress_callback.emit(
                            min(current_progress, total_tasks), total_tasks
                        )
                    except:
                        # If emit fails (e.g., due to threading issues), skip it
                        pass

            # Convert to sample results format
            for sample_id, eval_data in sample_evaluations.items():
                # Ensure all evaluations are completed
                sample_scores = eval_data["sample_scores"]
                model_answers = eval_data["model_answers"]
                extracted_answers = eval_data["extracted_answers"]  # 获取提取的答案

                # Filter out any None values (in case of errors)
                sample_scores = [s for s in sample_scores if s is not None]
                model_answers = [a for a in model_answers if a is not None]
                extracted_answers = [
                    a for a in extracted_answers if a is not None
                ]  # 过滤提取的答案

                sample_results.append(
                    {
                        "sample_id": sample_id,
                        "correct_answer": eval_data["correct_answer"],
                        "original_prompt": eval_data["original_prompt"],
                        "sample_scores": sample_scores,
                        "model_answers": model_answers,
                        "extracted_answers": extracted_answers,  # 添加提取的答案
                    }
                )

        # 统计未完成的请求（未获得有效结果）
        missing_ids = planned_ids - completed_ids
        for sample_id, eval_id in missing_ids:
            self.add_unfinished_request(
                model_name, dataset_path, sample_id, eval_id, "not_completed"
            )

        return sample_results

    def process_samples_concurrent_with_limits(
        self,
        model_name,
        samples,
        eval_times,
        config,
        max_concurrent_requests,
        progress_callback,
        total_tasks,
        completed_tasks_start,
        max_requests,
        model_max_requests,
        model_request_limit,
        current_total_requests,
        current_model_requests,
        dataset_path=None,
        allowed_eval_ids=None,
    ):
        """
        Process samples with concurrent requests and request limits (每分钟最大并发)
        Parameters:
            current_total_requests: 当前总的已完成请求数（整数）
            current_model_requests: 当前模型的已完成请求数（整数）
        """
        sample_results = []
        filter_empty_results = config.get("filter_empty_results", False)
        # 使用线程锁确保时间统计的线程安全
        time_lock = threading.Lock()
        # 本地计数器
        local_completed_requests = 0
        local_completed_requests_per_model = 0
        # 请求时间记录，用于控制每分钟的请求数
        request_times = []
        request_times_lock = threading.Lock()
        if allowed_eval_ids:
            planned_ids = set()
            for sid, evals in allowed_eval_ids.items():
                for ev in evals:
                    planned_ids.add((sid, ev))
        else:
            planned_ids = {
                (i, j) for i in range(len(samples)) for j in range(eval_times)
            }
        completed_ids = set()

        # 获取同时并发总数限制（从模型配置读取）
        # 从模型配置中读取max_concurrent_total
        model_name = model_name if isinstance(model_name, str) else ""
        model_config_data = config.get("model_configs", {}).get(model_name, {})
        max_concurrent_total = model_config_data.get("max_concurrent_total", -1)

        # Use ThreadPoolExecutor for concurrent processing
        with ThreadPoolExecutor(max_workers=max_concurrent_requests) as executor:
            # Create futures for sample evaluations WITH LIMITS
            futures = []
            in_flight = set()  # submitted but not completed futures
            future_to_sample = {}
            stop_submission = False

            # 只提交在限制范围内的任务
            for i, sample in enumerate(samples):
                # 【关键修复】在提交任何请求前，先检查停止标志
                if self.check_stop_flag():
                    stop_submission = True
                    break

                if allowed_eval_ids and i not in allowed_eval_ids:
                    continue
                eval_id_list = (
                    sorted(allowed_eval_ids[i])
                    if allowed_eval_ids and i in allowed_eval_ids
                    else list(range(eval_times))
                )
                for j in eval_id_list:
                    # 【关键修复】在提交任何请求前，先检查停止标志
                    if self.check_stop_flag():
                        stop_submission = True
                        break

                    # 等待直到在每分钟限制内
                    with request_times_lock:
                        current_time = time.time()
                        # 清理一分钟前的请求时间记录
                        request_times[:] = [
                            t for t in request_times if current_time - t <= 60
                        ]

                        # 如果当前分钟内请求数已达到限制，则等待
                        while len(request_times) >= max_concurrent_requests:
                            # 【关键修复】在等待时检查停止标志
                            if self.check_stop_flag():
                                stop_submission = True
                                break
                            time.sleep(0.1)  # 等待0.1秒后重新检查
                            current_time = time.time()
                            request_times[:] = [
                                t for t in request_times if current_time - t <= 60
                            ]

                        # 【关键修复】如果停止标志已设置，跳出循环
                        if self.check_stop_flag():
                            stop_submission = True
                            break

                        # 记录当前请求时间
                        request_times.append(current_time)

                    # 【关键修复】检查是否需要继续
                    if self.check_stop_flag():
                        stop_submission = True
                        break

                    # 等待直到同时并发总数限制内（考虑已提交但未完成的futures）
                    if max_concurrent_total > 0:
                        while True:
                            if self.check_stop_flag():
                                stop_submission = True
                                break
                            # Only count unfinished submitted tasks; futures list is cumulative.
                            in_flight = {f for f in in_flight if not f.done()}
                            if len(in_flight) < max_concurrent_total:
                                break
                            time.sleep(0.05)  # 等待0.05秒后重新检查

                    # 【关键修复】等待循环结束后，再次检查停止标志
                    # 这是为了防止在future完成后，active_requests减少导致循环立即结束
                    # 从而在停止标志已设置的情况下继续提交新请求
                    if self.check_stop_flag():
                        stop_submission = True
                        break

                    # 在提交任务前检查是否达到限制
                    total_submitted = len(futures)
                    current_total = current_total_requests + local_completed_requests
                    current_model = (
                        current_model_requests + local_completed_requests_per_model
                    )

                    # 检查总请求数限制
                    if max_requests and current_total + total_submitted >= max_requests:
                        break
                    # 检查模型请求数限制
                    if (
                        model_request_limit
                        and current_model + total_submitted >= model_request_limit
                    ):
                        break

                    # Submit with timing wrapper to record request times in concurrent mode
                    future = executor.submit(
                        self._call_model_api_with_timing, model_name, sample, config
                    )
                    futures.append(future)
                    in_flight.add(future)
                    future_to_sample[future] = (i, j, sample)

                # 【关键修复】检查外层循环是否需要继续
                if self.check_stop_flag():
                    stop_submission = True
                    break

                # 如果已经达到限制,不再处理更多样本
                if (
                    max_requests
                    and current_total_requests + local_completed_requests + len(futures)
                    >= max_requests
                ):
                    break
                if (
                    model_request_limit
                    and current_model_requests
                    + local_completed_requests_per_model
                    + len(futures)
                    >= model_request_limit
                ):
                    break

                if stop_submission:
                    break

            if stop_submission or self.check_stop_flag():
                for future in futures:
                    if not future.running() and not future.done():
                        future.cancel()

            # Process completed futures
            completed_count = 0
            sample_evaluations = {}  # {sample_id: {eval_id: (answer, score)}}

            for future in as_completed(futures):
                sample_id, eval_id, sample = future_to_sample[future]
                try:
                    # Get result which includes both answer and timing info
                    result = future.result()
                except CancelledError:
                    continue
                except Exception as e:
                    print(f"Error in concurrent request: {e}")
                    model_answer = ""
                    result = ""

                try:
                    if isinstance(result, tuple) and len(result) == 2:
                        model_answer, request_time = result
                        # Record timing information for concurrent requests
                        model_config_data = config.get("model_configs", {}).get(
                            model_name
                        )
                        if model_config_data and isinstance(model_config_data, dict):
                            from core.model_config import ModelConfig

                            model_config = ModelConfig.from_dict(model_config_data)
                            model_name_key = model_config.model_name
                        else:
                            model_name_key = model_name

                        # Use thread-safe timing recording
                        with time_lock:
                            if not hasattr(self, "model_request_times"):
                                self.model_request_times = {}
                            if model_name_key not in self.model_request_times:
                                self.model_request_times[model_name_key] = []
                            self.model_request_times[model_name_key].append(
                                request_time
                            )

                            # Also record token usage in concurrent processing
                            if not hasattr(self, "model_token_usage"):
                                self.model_token_usage = {}
                            if model_name_key not in self.model_token_usage:
                                self.model_token_usage[model_name_key] = {
                                    "prompt_tokens": [],
                                    "completion_tokens": [],
                                    "total_tokens": [],
                                }
                            # For concurrent processing, we don't have token info from the result, so use 0
                            self.model_token_usage[model_name_key][
                                "prompt_tokens"
                            ].append(0)
                            self.model_token_usage[model_name_key][
                                "completion_tokens"
                            ].append(0)
                            self.model_token_usage[model_name_key][
                                "total_tokens"
                            ].append(0)
                    else:
                        model_answer = result
                except Exception as e:
                    print(f"Error in concurrent result processing: {e}")
                    model_answer = ""

                # Get correct answer and calculate score
                correct_answer = self.extract_correct_answer(sample)
                correct_score = config.get("correct_score", 1.0)
                wrong_score = config.get("wrong_score", 0.0)
                score = self.calculate_score(
                    model_answer, correct_answer, correct_score, wrong_score
                )

                # Extract answer from model response
                extracted_answer = self.extract_answer(model_answer)

                # 如果需要过滤空结果且答案为空，则记录未完成，不计入结果
                is_empty = (extracted_answer is None) or (
                    isinstance(extracted_answer, str) and not extracted_answer.strip()
                )
                if filter_empty_results and is_empty:
                    self.add_unfinished_request(
                        model_name, dataset_path, sample_id, eval_id, "empty_result"
                    )
                else:
                    # Store evaluation results
                    if sample_id not in sample_evaluations:
                        sample_evaluations[sample_id] = {
                            "correct_answer": correct_answer,
                            "original_prompt": sample.get("text", ""),
                            "sample_scores": [None] * eval_times,
                            "model_answers": [None] * eval_times,
                            "extracted_answers": [None]
                            * eval_times,  # 添加提取答案的存储
                        }

                    sample_evaluations[sample_id]["sample_scores"][eval_id] = score
                    sample_evaluations[sample_id]["model_answers"][eval_id] = (
                        model_answer
                    )
                    sample_evaluations[sample_id]["extracted_answers"][eval_id] = (
                        extracted_answer  # 存储提取的答案
                    )
                    completed_ids.add((sample_id, eval_id))

                    # Update local counters
                    local_completed_requests += 1
                    local_completed_requests_per_model += 1

                    # Update progress
                    completed_count += 1
                    # Use atomic update to prevent progress from jumping back
                    current_progress = completed_tasks_start + completed_count
                    self._report_progress(
                        progress_callback,
                        min(current_progress, total_tasks),
                        total_tasks,
                    )

            # Convert to sample results format
            for sample_id, eval_data in sample_evaluations.items():
                # Ensure all evaluations are completed
                sample_scores = eval_data["sample_scores"]
                model_answers = eval_data["model_answers"]
                extracted_answers = eval_data["extracted_answers"]  # 获取提取的答案

                # Filter out any None values (in case of errors)
                sample_scores = [s for s in sample_scores if s is not None]
                model_answers = [a for a in model_answers if a is not None]
                extracted_answers = [
                    a for a in extracted_answers if a is not None
                ]  # 过滤提取的答案

                sample_results.append(
                    {
                        "sample_id": sample_id,
                        "correct_answer": eval_data["correct_answer"],
                        "original_prompt": eval_data["original_prompt"],
                        "sample_scores": sample_scores,
                        "model_answers": model_answers,
                        "extracted_answers": extracted_answers,  # 添加提取的答案
                    }
                )

        # 统计未完成的请求（未获得有效结果）
        missing_ids = planned_ids - completed_ids
        for sample_id, eval_id in missing_ids:
            self.add_unfinished_request(
                model_name, None, sample_id, eval_id, "not_completed"
            )

        return sample_results

    def evaluate_models_concurrent_with_limits(
        self,
        models,
        datasets,
        config,
        sample_count,
        eval_times,
        score_agg,
        progress_callback,
        result_callback,
        total_tasks,
        completed_tasks_start,
        evaluation_seed=None,
        max_requests=None,
        model_max_requests=None,
        pending_requests=None,
        preloaded_samples=None,
    ):
        """
        Evaluate multiple models concurrently with proper progress tracking and request limits
        """
        model_results = {}
        # Use a lock to ensure thread-safe progress updates
        import threading

        progress_lock = threading.Lock()
        # Track completed tasks for each model separately
        model_completed_tasks = {model_name: 0 for model_name in models}
        if pending_requests is not None and isinstance(pending_requests, dict):
            total_model_tasks = {}
            for model_name in models:
                count = 0
                model_pending = pending_requests.get(model_name, {})
                for dataset_path in datasets:
                    dataset_pending = model_pending.get(dataset_path, [])
                    if isinstance(dataset_pending, list):
                        count += len(dataset_pending)
                total_model_tasks[model_name] = count
        else:
            total_model_tasks = {
                model_name: len(datasets) * sample_count * eval_times
                for model_name in models
            }
        global_completed_tasks = [
            completed_tasks_start
        ]  # Use list for thread-safe updates
        # Track completed requests for each model and total
        completed_requests = [0]  # Use list for thread-safe updates
        completed_requests_per_model = {model_name: 0 for model_name in models}
        request_count_lock = threading.Lock()
        # Track models that have reached their limits
        models_reached_limit = set()
        models_reached_limit_lock = threading.Lock()

        # Use ThreadPoolExecutor for concurrent model evaluation
        with ThreadPoolExecutor(max_workers=len(models)) as executor:
            # Create futures for model evaluations WITH LIMITS
            futures = {}

            # 只为在限制范围内的模型创建任务
            for model_name in models:
                if self.check_stop_flag():
                    break
                # 检查模型是否已达到限制
                if (
                    model_max_requests
                    and isinstance(model_max_requests, dict)
                    and model_name in model_max_requests
                ):
                    model_limit = model_max_requests[model_name]
                    if model_limit <= 0:
                        # 如果模型限制为0,跳过该模型
                        continue

                # Create a thread-safe progress callback for each model
                def create_model_progress_callback(
                    model_name_local,
                    progress_lock,
                    global_completed_tasks,
                    total_tasks,
                    progress_callback,
                    model_completed_tasks,
                    total_model_tasks,
                ):
                    def model_progress_callback(current, total_per_model):
                        with progress_lock:
                            # Update this model's progress
                            model_completed_tasks[model_name_local] = current
                            # Calculate total completed tasks across all models
                            total_completed = (
                                sum(model_completed_tasks.values())
                                + completed_tasks_start
                            )
                            global_completed_tasks[0] = min(
                                total_completed, total_tasks
                            )
                            if progress_callback and hasattr(progress_callback, "emit"):
                                # Check if the callback is a Qt signal before calling emit
                                if hasattr(progress_callback, "emit"):
                                    try:
                                        progress_callback.emit(
                                            global_completed_tasks[0], total_tasks
                                        )
                                    except:
                                        # If emit fails (e.g., due to threading issues), skip it
                                        pass

                    return model_progress_callback

                # Pass the thread-safe callback to evaluate_single_model_with_limits
                model_progress_callback = create_model_progress_callback(
                    model_name,
                    progress_lock,
                    global_completed_tasks,
                    total_tasks,
                    progress_callback,
                    model_completed_tasks,
                    total_model_tasks,
                )

                # Don't pass result_callback to concurrent evaluation to avoid threading issues
                future = executor.submit(
                    self.evaluate_single_model_with_limits,
                    model_name,
                    datasets,
                    config,
                    sample_count,
                    eval_times,
                    score_agg,
                    model_progress_callback,
                    None,
                    total_model_tasks[model_name],
                    0,
                    evaluation_seed,
                    max_requests,
                    model_max_requests,
                    completed_requests,
                    completed_requests_per_model,
                    request_count_lock,
                    models_reached_limit,
                    models_reached_limit_lock,
                    pending_requests=pending_requests,
                    preloaded_samples=preloaded_samples,
                )  # Add limit parameters
                futures[future] = model_name

            # Process completed futures
            completed_futures = 0
            for future in as_completed(futures):
                completed_futures += 1
                model_name = None

                # 先获取模型名称，确保即使发生异常也能访问
                for f, name in futures.items():
                    if f == future:
                        model_name = name
                        break

                if model_name is None:
                    # If we can't get model name, skip this future
                    continue

                try:
                    model_result = future.result()
                    model_results[model_name] = model_result
                except Exception as e:
                    print(f"Error evaluating model {model_name}: {e}")
                    # 为失败的模型创建基本结果结构，确保completed_requests字段存在
                    model_results[model_name] = {
                        "score": 0.0,
                        "details": [],
                        "raw_data": [],
                        "completed_requests": 0,  # 添加完成请求数，确保字段存在
                    }
                    # Still count this model as completed for progress purposes
                    with progress_lock:
                        model_completed_tasks[model_name] = total_model_tasks[
                            model_name
                        ]
                        total_completed = (
                            sum(model_completed_tasks.values()) + completed_tasks_start
                        )
                        global_completed_tasks[0] = min(total_completed, total_tasks)
                        if progress_callback and hasattr(progress_callback, "emit"):
                            try:
                                progress_callback.emit(
                                    global_completed_tasks[0], total_tasks
                                )
                            except:
                                # If emit fails (e.g., due to threading issues), skip it
                                pass
        return model_results

    def evaluate_single_model_with_limits(
        self,
        model_name,
        datasets,
        config,
        sample_count,
        eval_times,
        score_agg,
        progress_callback,
        result_callback,
        total_tasks,
        completed_tasks_start,
        evaluation_seed=None,
        max_requests=None,
        model_max_requests=None,
        completed_requests=None,
        completed_requests_per_model=None,
        request_count_lock=None,
        models_reached_limit=None,
        models_reached_limit_lock=None,
        pending_requests=None,
        preloaded_samples=None,
    ):
        """
        Evaluate a single model with request limits (used for concurrent model evaluation with limits)
        """
        model_total_score = 0.0
        model_details = []
        model_raw_data = []
        completed_tasks = 0  # Local counter for this model only
        local_completed_requests = 0  # Local counter for this model's requests
        local_completed_requests_per_model = (
            0  # Local counter for this model's requests (for model-specific limit)
        )
        filter_empty_results = config.get("filter_empty_results", False)
        model_request_limit = None
        # 确保 model_max_requests 是字典类型
        if model_max_requests is not None and not isinstance(model_max_requests, dict):
            model_max_requests = None
        if (
            model_max_requests
            and isinstance(model_max_requests, dict)
            and model_name in model_max_requests
        ):
            model_request_limit = model_max_requests[model_name]

        # Check if this model has reached its limit
        if models_reached_limit and model_name in models_reached_limit:
            return {
                "score": model_total_score,
                "details": model_details,
                "raw_data": model_raw_data,
                "completed_requests": 0,  # 已达到限制，返回0
            }

        # Get model config to check concurrent settings
        model_config_data = config.get("model_configs", {}).get(model_name)
        if model_config_data and isinstance(model_config_data, dict):
            from core.model_config import ModelConfig

            model_config = ModelConfig.from_dict(model_config_data)
            enable_concurrent = model_config.enable_concurrent
            max_concurrent_requests = model_config.max_concurrent_requests
        else:
            enable_concurrent = False
            max_concurrent_requests = 10

        # Calculate total tasks for this model to report progress correctly
        model_total_tasks = len(datasets) * sample_count * eval_times

        # Process each dataset
        for dataset_path in datasets:
            if self.check_stop_flag():
                break
            # 筛选待执行请求
            allowed_eval_ids = None
            if pending_requests is not None and isinstance(pending_requests, dict):
                model_pending = pending_requests.get(model_name, {})
                pending_pairs = (
                    model_pending.get(dataset_path)
                    if isinstance(model_pending, dict)
                    else None
                )
                if not pending_pairs:
                    continue
                allowed_eval_ids = {}
                for sid, eid in pending_pairs:
                    allowed_eval_ids.setdefault(sid, set()).add(eid)

            # Check if we've reached limits before processing dataset
            if models_reached_limit and model_name in models_reached_limit:
                break
            if (
                max_requests
                and completed_requests
                and completed_requests[0] + local_completed_requests >= max_requests
            ):
                break
            if (
                model_request_limit
                and completed_requests_per_model
                and completed_requests_per_model[model_name]
                + local_completed_requests_per_model
                >= model_request_limit
            ):
                break
            # 检查停止标志
            if self.check_stop_flag():
                break

            # Read dataset with fixed seed to ensure all models use the same samples
            if preloaded_samples and dataset_path in preloaded_samples:
                samples = preloaded_samples[dataset_path]
            else:
                samples = self.read_dataset_with_fixed_seed(
                    dataset_path, sample_count, seed=evaluation_seed
                )
            if allowed_eval_ids:
                planned_ids = set()
                for sid, evals in allowed_eval_ids.items():
                    for ev in evals:
                        planned_ids.add((sid, ev))
            else:
                planned_ids = {
                    (i, j) for i in range(len(samples)) for j in range(eval_times)
                }
            completed_ids = set()
            if enable_concurrent:
                # Use concurrent processing for samples with request limits
                # Calculate current total and model-specific request counts
                current_total_requests = (
                    completed_requests[0] + local_completed_requests
                    if completed_requests
                    else local_completed_requests
                )
                current_model_requests = (
                    completed_requests_per_model[model_name]
                    + local_completed_requests_per_model
                    if completed_requests_per_model
                    else local_completed_requests_per_model
                )

                sample_results = self.process_samples_concurrent_with_limits(
                    model_name,
                    samples,
                    eval_times,
                    config,
                    max_concurrent_requests,
                    progress_callback,
                    model_total_tasks,
                    completed_tasks,
                    max_requests,
                    model_max_requests,
                    model_request_limit,
                    current_total_requests,
                    current_model_requests,
                    dataset_path=dataset_path,
                    allowed_eval_ids=allowed_eval_ids,
                )
                # Update counters based on actual processed samples
                # Each sample result contains multiple evaluations (eval_times)
                actual_processed_samples = len(
                    [r for r in sample_results if r is not None]
                )
                actual_processed_requests = 0
                for sample_result in sample_results:
                    if sample_result is not None:
                        # Count the actual number of requests (number of scores in sample_scores)
                        actual_processed_requests += len(
                            sample_result.get("sample_scores", [])
                        )
                # 只统计非空的已完成请求，未完成不计入
                completed_tasks += actual_processed_requests
                local_completed_requests += actual_processed_requests
                local_completed_requests_per_model += actual_processed_requests
            else:
                # Use normal processing for samples with request limits
                sample_results = []
                for i, sample in enumerate(samples):
                    if self.check_stop_flag():
                        break
                    if allowed_eval_ids and i not in allowed_eval_ids:
                        continue
                    if models_reached_limit and model_name in models_reached_limit:
                        break
                    if (
                        max_requests
                        and completed_requests
                        and completed_requests[0] + local_completed_requests
                        >= max_requests
                    ):
                        break
                    if (
                        model_request_limit
                        and completed_requests_per_model
                        and completed_requests_per_model[model_name]
                        + local_completed_requests_per_model
                        >= model_request_limit
                    ):
                        break

                    # Get correct answer
                    correct_answer = self.extract_correct_answer(sample)
                    # Get original prompt
                    original_prompt = sample.get("text", "")

                    # Evaluate multiple times
                    sample_scores = []
                    model_answers = []  # 存储每次的模型回答
                    extracted_answers = []  # 存储每次提取的答案
                    eval_loop = (
                        sorted(allowed_eval_ids[i])
                        if allowed_eval_ids and i in allowed_eval_ids
                        else list(range(eval_times))
                    )
                    for j in eval_loop:
                        # Check limits before making request
                        if (
                            max_requests
                            and completed_requests
                            and completed_requests[0] + local_completed_requests
                            >= max_requests
                        ):
                            break
                        if (
                            model_request_limit
                            and completed_requests_per_model
                            and completed_requests_per_model[model_name]
                            + local_completed_requests_per_model
                            >= model_request_limit
                        ):
                            break
                        # 检查停止标志
                        if self.check_stop_flag():
                            break

                        # Call model API
                        if self.check_stop_flag():
                            break
                        model_answer = self.call_model_api(model_name, sample, config)
                        model_answers.append(model_answer)  # 保存模型回答

                        # Extract answer from model response
                        extracted_answer = self.extract_answer(model_answer)
                        is_empty = (extracted_answer is None) or (
                            isinstance(extracted_answer, str)
                            and not extracted_answer.strip()
                        )
                        if filter_empty_results and is_empty:
                            # 标记未完成，不计入结果
                            self.add_unfinished_request(
                                model_name, dataset_path, i, j, "empty_result"
                            )
                        else:
                            extracted_answers.append(extracted_answer)  # 保存提取的答案
                            completed_ids.add((i, j))

                            # Calculate score
                            correct_score = config.get("correct_score", 1.0)
                            wrong_score = config.get("wrong_score", 0.0)
                            score = self.calculate_score(
                                model_answer, correct_answer, correct_score, wrong_score
                            )
                            sample_scores.append(score)

                            # Update local counters
                            local_completed_requests += 1
                            local_completed_requests_per_model += 1
                            completed_ids.add((i, j))

                            # Report progress for this model only
                            completed_tasks += 1
                            self._report_progress(
                                progress_callback, completed_tasks, model_total_tasks
                            )

                    sample_results.append(
                        {
                            "sample_id": i,
                            "correct_answer": correct_answer,
                            "original_prompt": original_prompt,
                            "sample_scores": sample_scores,
                            "model_answers": model_answers,
                            "extracted_answers": extracted_answers,
                        }
                    )

            # 标记未完成的请求（未获得有效结果）
            missing_ids = planned_ids - completed_ids
            for sample_id, eval_id in missing_ids:
                self.add_unfinished_request(
                    model_name, dataset_path, sample_id, eval_id, "not_completed"
                )

            # Process sample results
            for sample_result in sample_results:
                if sample_result is None:  # 如果是空结果（因为达到限制），跳过
                    continue
                i = sample_result["sample_id"]
                correct_answer = sample_result["correct_answer"]
                original_prompt = sample_result["original_prompt"]
                sample_scores = sample_result["sample_scores"]
                model_answers = sample_result["model_answers"]
                extracted_answers = sample_result.get("extracted_answers", [])

                # Aggregate scores for this sample
                if score_agg == "sum":
                    final_sample_score = sum(sample_scores)
                else:  # avg
                    final_sample_score = sum(sample_scores) / len(sample_scores)

                model_total_score += final_sample_score

                # Store details
                detail = {
                    "dataset": os.path.basename(dataset_path),
                    "sample_id": i,
                    "correct_answer": correct_answer,
                    "model_answers": sample_scores,
                    "final_score": final_sample_score,
                }
                model_details.append(detail)

                # Store raw data including original prompt, model answers, and extracted answers
                # Extract turn information from original_prompt
                temp_sample = {"text": original_prompt}
                turn_number = self.extract_turn_number(temp_sample)
                turn_category = self.get_turn_category(turn_number)

                raw_data_entry = {
                    "dataset": os.path.basename(dataset_path),
                    "sample_id": i,
                    "original_prompt": original_prompt,
                    "correct_answer": correct_answer,
                    "model_name": model_name,
                    "model_answers": model_answers,
                    "extracted_answers": extracted_answers,  # 添加提取的答案
                    "scores": sample_scores,
                    "final_score": final_sample_score,
                    "turn_number": turn_number,  # 巡目数字
                    "turn_category": turn_category,  # 巡目类别
                }
                model_raw_data.append(raw_data_entry)

                # Report result - only if result_callback is not None and not a threading issue
                if result_callback and hasattr(result_callback, "emit"):
                    try:
                        result_callback.emit(
                            {"model": model_name, "score": model_total_score}
                        )
                    except:
                        # If emit fails (e.g., due to threading issues), skip it
                        pass

        # Update global counters if provided
        if completed_requests and request_count_lock:
            with request_count_lock:
                completed_requests[0] += local_completed_requests
        if completed_requests_per_model and request_count_lock:
            with request_count_lock:
                completed_requests_per_model[model_name] += (
                    local_completed_requests_per_model
                )
        if models_reached_limit and models_reached_limit_lock:
            if (
                max_requests
                and completed_requests
                and completed_requests[0] >= max_requests
            ):
                with models_reached_limit_lock:
                    models_reached_limit.add(model_name)
            if (
                model_request_limit
                and completed_requests_per_model
                and completed_requests_per_model[model_name] >= model_request_limit
            ):
                with models_reached_limit_lock:
                    models_reached_limit.add(model_name)

        return {
            "score": model_total_score,
            "details": model_details,
            "raw_data": model_raw_data,
            "completed_requests": len(completed_ids),  # 使用实际完成的非空请求数
        }

    def extract_answer(self, model_answer):
        """
        Extract answer from model response
        Support both direct answer format and analysis+answer format
        For multi-line responses with analysis, extract the last non-empty line as the answer
        Returns the extracted answer, or the original answer if no extraction is needed
        """
        if not model_answer:
            return model_answer

        # 检查 model_answer 是否为字符串类型
        if not isinstance(model_answer, str):
            # 如果是元组，尝试提取第一个元素
            if isinstance(model_answer, tuple):
                model_answer = model_answer[0] if len(model_answer) > 0 else ""
            else:
                # 转换为字符串
                model_answer = str(model_answer)

        # Split the model answer into lines
        lines = model_answer.split("\n")
        # Remove empty lines from the end
        lines = [line.strip() for line in lines if line.strip()]
        if not lines:
            return model_answer.strip()

        # Check if model answer contains "答案" keyword
        if "答案：" in model_answer:
            try:
                # Split by "答案：" and take the part after it
                answer_parts = model_answer.split("答案：", 1)
                if len(answer_parts) > 1:
                    extracted_answer = answer_parts[1].strip()
                    # Split by newline and take the first non-empty line after "答案："
                    answer_lines = [
                        line.strip()
                        for line in extracted_answer.split("\n")
                        if line.strip()
                    ]
                    if answer_lines:
                        actual_answer = answer_lines[
                            0
                        ]  # Take the first line after "答案："

                        # Remove any trailing punctuation or text that might be explanations
                        if "." in actual_answer and not actual_answer.startswith("."):
                            # Split by period and take the first part (in case of "6筒。在本例中...")
                            actual_answer = actual_answer.split(".")[0].strip()
                        if "。" in actual_answer and not actual_answer.startswith("。"):
                            # Split by Chinese period and take the first part
                            actual_answer = actual_answer.split("。")[0].strip()
                        if "，" in actual_answer and not actual_answer.startswith("，"):
                            # Split by Chinese comma and take the first part (if it's a complete sentence)
                            parts = actual_answer.split("，")
                            # Only take the first part if it's short (likely the actual answer)
                            if (
                                len(parts[0].strip()) <= 10
                            ):  # Adjust threshold as needed
                                actual_answer = parts[0].strip()

                        # Remove formatting characters like **, __, etc.
                        # Remove leading/trailing formatting markers
                        import re

                        # Remove markdown-style formatting like **text**, __text__, etc.
                        actual_answer = re.sub(r"^[*_]+|[*_]+$", "", actual_answer)
                        # Remove any remaining formatting characters at the end
                        actual_answer = actual_answer.rstrip("*_")

                        return actual_answer
                    else:
                        # If no lines after "答案：", return original
                        return model_answer.strip()
                else:
                    # If "答案：" exists but can't split properly, return original
                    return model_answer.strip()
            except:
                # If extraction fails, return original answer
                return model_answer.strip()
        else:
            # For responses without "答案：" keyword, take the last non-empty line as the answer
            # This handles cases where there's analysis followed by the answer on the last line
            last_line = lines[-1]  # Take the last non-empty line

            # Remove any trailing punctuation or text that might be explanations
            if "." in last_line and not last_line.startswith("."):
                # Split by period and take the first part (in case of "6筒。在本例中...")
                last_line = last_line.split(".")[0].strip()
            if "。" in last_line and not last_line.startswith("。"):
                # Split by Chinese period and take the first part
                last_line = last_line.split("。")[0].strip()
            if "，" in last_line and not last_line.startswith("，"):
                # Split by Chinese comma and take the first part (if it's a complete sentence)
                parts = last_line.split("，")
                # Only take the first part if it's short (likely the actual answer)
                if len(parts[0].strip()) <= 10:  # Adjust threshold as needed
                    last_line = parts[0].strip()

            # Remove formatting characters like **, __, etc.
            import re

            # Remove markdown-style formatting like **text**, __text__, etc.
            last_line = re.sub(r"^[*_]+|[*_]+$", "", last_line)
            # Remove any remaining formatting characters at the end
            last_line = last_line.rstrip("*_")

            return last_line

    def calculate_score(self, model_answer, correct_answer, correct_score, wrong_score):
        """
        Calculate score based on model answer and correct answer
        Support both direct answer format and analysis+answer format
        """
        if not model_answer or not correct_answer:
            return wrong_score

        # Extract answer from model response
        extracted_answer = self.extract_answer(model_answer)

        # Compare the extracted answer with correct answer
        if extracted_answer == correct_answer.strip():
            return correct_score
        else:
            return wrong_score

    def calculate_sample_accuracy(self, raw_data, config=None):
        """
        Calculate sample-based accuracy metrics for each model
        :param raw_data: Dictionary of {model_name: list of sample results}
        :param config: Evaluation config containing correct_score and wrong_score
        :return: Dictionary with metrics for each model
        """
        metrics = {}

        # Default scores if not provided in config
        correct_score = config.get("correct_score", 1.0) if config else 1.0
        wrong_score = config.get("wrong_score", 0.0) if config else 0.0

        for model_name, model_results in raw_data.items():
            # Group results by sample_id
            sample_groups = {}
            for result in model_results:
                sample_key = (result["dataset"], result["sample_id"])
                if sample_key not in sample_groups:
                    sample_groups[sample_key] = []
                sample_groups[sample_key].append(result)

            total_samples = len(sample_groups)
            full_correct_count = 0
            full_wrong_count = 0

            for sample_key, results in sample_groups.items():
                # Get all scores for this sample
                all_scores = []
                for result in results:
                    all_scores.extend(result["scores"])

                # Check if all scores are correct
                if all(score == correct_score for score in all_scores):
                    full_correct_count += 1
                # Check if all scores are wrong
                elif all(score == wrong_score for score in all_scores):
                    full_wrong_count += 1

            # Calculate rates
            full_correct_rate = (
                full_correct_count / total_samples if total_samples > 0 else 0.0
            )
            full_wrong_rate = (
                full_wrong_count / total_samples if total_samples > 0 else 0.0
            )

            metrics[model_name] = {
                "sample_count": total_samples,
                "full_correct_count": full_correct_count,
                "full_correct_rate": full_correct_rate,
                "full_wrong_count": full_wrong_count,
                "full_wrong_rate": full_wrong_rate,
            }

        return metrics

    def evaluate_single_model_concurrent(
        self,
        model_name,
        datasets,
        config,
        sample_count,
        eval_times,
        score_agg,
        progress_callback,
        result_callback,
        total_tasks,
        completed_tasks_start,
        evaluation_seed=None,
    ):
        """
        Evaluate a single model with thread-safe progress updates (used for concurrent model evaluation) (每分钟最大并发)
        """
        model_total_score = 0.0
        model_details = []
        model_raw_data = []
        completed_tasks = 0  # Local counter for this model only

        # Get model config to check concurrent settings
        model_config_data = config.get("model_configs", {}).get(model_name)
        if model_config_data and isinstance(model_config_data, dict):
            from core.model_config import ModelConfig

            model_config = ModelConfig.from_dict(model_config_data)
            enable_concurrent = model_config.enable_concurrent
            max_concurrent_requests = model_config.max_concurrent_requests
        else:
            enable_concurrent = False
            max_concurrent_requests = 10

        # Calculate total tasks for this model to report progress correctly
        model_total_tasks = len(datasets) * sample_count * eval_times

        # Process each dataset
        for dataset_path in datasets:
            # Read dataset with fixed seed to ensure all models use the same samples
            samples = self.read_dataset_with_fixed_seed(
                dataset_path, sample_count, seed=evaluation_seed
            )
            planned_ids = {
                (i, j) for i in range(len(samples)) for j in range(eval_times)
            }
            completed_ids = set()
            if enable_concurrent:
                # Use concurrent processing for samples
                sample_results = self.process_samples_concurrent(
                    model_name,
                    samples,
                    eval_times,
                    config,
                    max_concurrent_requests,
                    progress_callback,
                    model_total_tasks,
                    completed_tasks,
                )
                completed_tasks += len(samples) * eval_times
                # 根据返回结果标记完成
                for sr in sample_results:
                    sid = sr.get("sample_id")
                    scores = sr.get("sample_scores", [])
                    if sid is not None:
                        for idx in range(len(scores)):
                            completed_ids.add((sid, idx))
            else:
                # Use normal processing for samples
                sample_results = []
                for i, sample in enumerate(samples):
                    # Get correct answer
                    correct_answer = self.extract_correct_answer(sample)
                    # Get original prompt
                    original_prompt = sample.get("text", "")

                    # Evaluate multiple times
                    sample_scores = []
                    model_answers = []  # 存储每次的模型回答
                    extracted_answers = []  # 存储每次提取的答案
                    for j in range(eval_times):
                        # Call model API
                        model_answer = self.call_model_api(model_name, sample, config)
                        model_answers.append(model_answer)  # 保存模型回答

                        # Extract answer from model response
                        extracted_answer = self.extract_answer(model_answer)
                        is_empty = (extracted_answer is None) or (
                            isinstance(extracted_answer, str)
                            and not extracted_answer.strip()
                        )
                        if filter_empty_results and is_empty:
                            self.add_unfinished_request(
                                model_name, dataset_path, i, j, "empty_result"
                            )
                        else:
                            extracted_answers.append(extracted_answer)  # 保存提取的答案

                        # Calculate score（空结果时不计分）
                        if not (filter_empty_results and is_empty):
                            correct_score = config.get("correct_score", 1.0)
                            wrong_score = config.get("wrong_score", 0.0)
                            score = self.calculate_score(
                                model_answer, correct_answer, correct_score, wrong_score
                            )
                            sample_scores.append(score)

                        # Report progress for this model only
                        completed_tasks += 1
                        if progress_callback and hasattr(progress_callback, "emit"):
                            try:
                                progress_callback.emit(
                                    completed_tasks, model_total_tasks
                                )
                            except:
                                # If emit fails (e.g., due to threading issues), skip it
                                pass

                        if not (filter_empty_results and is_empty):
                            completed_ids.add((i, j))

                sample_results.append(
                    {
                        "sample_id": i,
                        "correct_answer": correct_answer,
                        "original_prompt": original_prompt,
                        "sample_scores": sample_scores,
                        "model_answers": model_answers,
                        "extracted_answers": extracted_answers,
                    }
                )

            # 标记未完成
            missing_ids = planned_ids - completed_ids
            for sample_id, eval_id in missing_ids:
                self.add_unfinished_request(
                    model_name, dataset_path, sample_id, eval_id, "not_completed"
                )

            # Process sample results
            for sample_result in sample_results:
                i = sample_result["sample_id"]
                correct_answer = sample_result["correct_answer"]
                original_prompt = sample_result["original_prompt"]
                sample_scores = sample_result["sample_scores"]
                model_answers = sample_result["model_answers"]
                extracted_answers = sample_result.get("extracted_answers", [])

                # Aggregate scores for this sample
                if score_agg == "sum":
                    final_sample_score = sum(sample_scores)
                else:  # avg
                    final_sample_score = sum(sample_scores) / len(sample_scores)

                model_total_score += final_sample_score

                # Store details
                detail = {
                    "dataset": os.path.basename(dataset_path),
                    "sample_id": i,
                    "correct_answer": correct_answer,
                    "model_answers": sample_scores,
                    "final_score": final_sample_score,
                }
                model_details.append(detail)

                # Store raw data including original prompt, model answers, and extracted answers
                # Extract turn information from original_prompt
                temp_sample = {"text": original_prompt}
                turn_number = self.extract_turn_number(temp_sample)
                turn_category = self.get_turn_category(turn_number)

                raw_data_entry = {
                    "dataset": os.path.basename(dataset_path),
                    "sample_id": i,
                    "original_prompt": original_prompt,
                    "correct_answer": correct_answer,
                    "model_name": model_name,
                    "model_answers": model_answers,
                    "extracted_answers": extracted_answers,  # 添加提取的答案
                    "scores": sample_scores,
                    "final_score": final_sample_score,
                    "turn_number": turn_number,  # 巡目数字
                    "turn_category": turn_category,  # 巡目类别
                }
                model_raw_data.append(raw_data_entry)

                # Report result - only if result_callback is not None and not a threading issue
                if result_callback and hasattr(result_callback, "emit"):
                    try:
                        result_callback.emit(
                            {"model": model_name, "score": model_total_score}
                        )
                    except:
                        # If emit fails (e.g., due to threading issues), skip it
                        pass

        return {
            "score": model_total_score,
            "details": model_details,
            "raw_data": model_raw_data,
        }

    def evaluate_single_model(
        self,
        model_name,
        datasets,
        config,
        sample_count,
        eval_times,
        score_agg,
        progress_callback,
        result_callback,
        total_tasks,
        completed_tasks_start,
        evaluation_seed=None,
    ):
        """
        Evaluate a single model (used for concurrent model evaluation) (每分钟最大并发)
        """
        model_total_score = 0.0
        model_details = []
        model_raw_data = []
        completed_tasks = completed_tasks_start

        # Get model config to check concurrent settings
        model_config_data = config.get("model_configs", {}).get(model_name)
        if model_config_data:
            from core.model_config import ModelConfig

            model_config = ModelConfig.from_dict(model_config_data)
            enable_concurrent = model_config.enable_concurrent
            max_concurrent_requests = model_config.max_concurrent_requests
        else:
            enable_concurrent = False
            max_concurrent_requests = 10

        # Process each dataset
        for dataset_path in datasets:
            # Read dataset with fixed seed to ensure all models use the same samples
            samples = self.read_dataset_with_fixed_seed(
                dataset_path, sample_count, seed=evaluation_seed
            )
            if enable_concurrent:
                # Use concurrent processing for samples
                sample_results = self.process_samples_concurrent(
                    model_name,
                    samples,
                    eval_times,
                    config,
                    max_concurrent_requests,
                    progress_callback,
                    total_tasks,
                    completed_tasks,
                )
                completed_tasks += len(samples) * eval_times
            else:
                # Use normal processing for samples
                sample_results = []
                for i, sample in enumerate(samples):
                    # Get correct answer
                    correct_answer = self.extract_correct_answer(sample)
                    # Get original prompt
                    original_prompt = sample.get("text", "")

                    # Evaluate multiple times
                    sample_scores = []
                    model_answers = []  # 存储每次的模型回答
                    extracted_answers = []  # 存储每次提取的答案
                    for j in range(eval_times):
                        # Call model API
                        model_answer = self.call_model_api(model_name, sample, config)
                        model_answers.append(model_answer)  # 保存模型回答

                        # Extract answer from model response
                        extracted_answer = self.extract_answer(model_answer)
                        extracted_answers.append(extracted_answer)  # 保存提取的答案

                        # Calculate score
                        correct_score = config.get("correct_score", 1.0)
                        wrong_score = config.get("wrong_score", 0.0)
                        score = self.calculate_score(
                            model_answer, correct_answer, correct_score, wrong_score
                        )
                        sample_scores.append(score)

                        # Report progress
                        completed_tasks += 1
                        if progress_callback:
                            progress_callback.emit(completed_tasks, total_tasks)

                    sample_results.append(
                        {
                            "sample_id": i,
                            "correct_answer": correct_answer,
                            "original_prompt": original_prompt,
                            "sample_scores": sample_scores,
                            "model_answers": model_answers,
                            "extracted_answers": extracted_answers,
                        }
                    )

            # Process sample results
            for sample_result in sample_results:
                i = sample_result["sample_id"]
                correct_answer = sample_result["correct_answer"]
                original_prompt = sample_result["original_prompt"]
                sample_scores = sample_result["sample_scores"]
                model_answers = sample_result["model_answers"]
                extracted_answers = sample_result.get("extracted_answers", [])

                # Aggregate scores for this sample
                if score_agg == "sum":
                    final_sample_score = sum(sample_scores)
                else:  # avg
                    final_sample_score = sum(sample_scores) / len(sample_scores)

                model_total_score += final_sample_score

                # Store details
                detail = {
                    "dataset": os.path.basename(dataset_path),
                    "sample_id": i,
                    "correct_answer": correct_answer,
                    "model_answers": sample_scores,
                    "final_score": final_sample_score,
                }
                model_details.append(detail)

                # Store raw data including original prompt, model answers, and extracted answers
                # Extract turn information from original_prompt
                temp_sample = {"text": original_prompt}
                turn_number = self.extract_turn_number(temp_sample)
                turn_category = self.get_turn_category(turn_number)

                raw_data_entry = {
                    "dataset": os.path.basename(dataset_path),
                    "sample_id": i,
                    "original_prompt": original_prompt,
                    "correct_answer": correct_answer,
                    "model_name": model_name,
                    "model_answers": model_answers,
                    "extracted_answers": extracted_answers,  # 添加提取的答案
                    "scores": sample_scores,
                    "final_score": final_sample_score,
                    "turn_number": turn_number,  # 巡目数字
                    "turn_category": turn_category,  # 巡目类别
                }
                model_raw_data.append(raw_data_entry)

                # Report result
                if result_callback:
                    result_callback.emit(
                        {"model": model_name, "score": model_total_score}
                    )
        return {
            "score": model_total_score,
            "details": model_details,
            "raw_data": model_raw_data,
        }

    def merge_results(self, task_dir, current_results, is_final=False):
        """
        Merge current results with existing results in the task directory
        """
        import glob
        import json

        # Find all existing complete_test_results files
        existing_result_files = glob.glob(
            os.path.join(task_dir, "complete_test_results_*.json")
        )
        existing_score_files = glob.glob(os.path.join(task_dir, "model_scores_*.txt"))
        # Also look for the main complete_test_results.json file
        main_result_file = os.path.join(task_dir, "complete_test_results.json")
        main_score_file = os.path.join(task_dir, "model_scores.txt")

        merged_raw_data = {}
        merged_scores = {}
        merged_request_time_stats = {}
        merged_model_request_times = {}  # 新增：合并原始请求时间列表
        merged_model_token_usage = {}  # 新增：合并原始token使用数据
        merged_unfinished = []  # 新增：合并未完成请求

        # Load existing results first from timestamped files
        for result_file in existing_result_files:
            try:
                with open(result_file, "r", encoding="utf-8") as f:
                    existing_results = json.load(f)
                    # Merge raw data - 支持两种格式：按模型分组的字典或直接的列表
                    if isinstance(existing_results, dict):
                        # 按模型分组的格式
                        for model_name, model_results in existing_results.items():
                            if model_name not in merged_raw_data:
                                merged_raw_data[model_name] = []
                            # Ensure model_results is a list
                            if isinstance(model_results, list):
                                merged_raw_data[model_name].extend(model_results)
                            else:
                                # If it's not a list, wrap it in a list
                                merged_raw_data[model_name].append(model_results)
                    elif isinstance(existing_results, list):
                        # 直接的列表格式 - 需要从每个条目中提取模型信息
                        for item in existing_results:
                            if isinstance(item, dict) and "model_name" in item:
                                model_name = item["model_name"]
                                if model_name not in merged_raw_data:
                                    merged_raw_data[model_name] = []
                                merged_raw_data[model_name].append(item)
            except Exception as e:
                print(f"Error loading existing result file {result_file}: {e}")

        # Load main result file if it exists
        if os.path.exists(main_result_file):
            try:
                with open(main_result_file, "r", encoding="utf-8") as f:
                    existing_results = json.load(f)
                    # Merge raw data from main file - 支持两种格式
                    if isinstance(existing_results, dict):
                        # 按模型分组的格式
                        for model_name, model_results in existing_results.items():
                            if model_name not in merged_raw_data:
                                merged_raw_data[model_name] = []
                            # Ensure model_results is a list
                            if isinstance(model_results, list):
                                merged_raw_data[model_name].extend(model_results)
                            else:
                                # If it's not a list, wrap it in a list
                                merged_raw_data[model_name].append(model_results)
                    elif isinstance(existing_results, list):
                        # 直接的列表格式
                        for item in existing_results:
                            if isinstance(item, dict) and "model_name" in item:
                                model_name = item["model_name"]
                                if model_name not in merged_raw_data:
                                    merged_raw_data[model_name] = []
                                merged_raw_data[model_name].append(item)
            except Exception as e:
                print(f"Error loading main result file {main_result_file}: {e}")

        # 新增：从temp文件夹加载临时文件中的统计数据
        temp_dir = os.path.join(task_dir, "temp")
        if os.path.exists(temp_dir):
            temp_files = glob.glob(os.path.join(temp_dir, "temp_progress_*.json"))
            for temp_file in temp_files:
                try:
                    with open(temp_file, "r", encoding="utf-8") as f:
                        temp_data = json.load(f)

                        # [新增修复 1] 从临时文件合并 raw_data
                        if "raw_data" in temp_data:
                            temp_raw = temp_data["raw_data"]
                            if isinstance(temp_raw, dict):
                                for m_name, m_data in temp_raw.items():
                                    if m_name not in merged_raw_data:
                                        merged_raw_data[m_name] = []
                                    if isinstance(m_data, list):
                                        merged_raw_data[m_name].extend(m_data)
                                    else:
                                        merged_raw_data[m_name].append(m_data)

                        # [新增修复 2] 从临时文件合并 scores
                        if "scores" in temp_data:
                            temp_scores = temp_data["scores"]
                            for m_name, m_score in temp_scores.items():
                                if m_name not in merged_scores:
                                    merged_scores[m_name] = 0.0
                                # 如果是不同的模型，直接赋值或累加均可；如果是同一模型分批跑，应该是累加
                                # 这里假设临时文件中存储的是该次运行的得分
                                merged_scores[m_name] += m_score

                        # 合并原始请求时间数据
                        if "model_request_times" in temp_data:
                            for model_name, request_times in temp_data[
                                "model_request_times"
                            ].items():
                                if model_name not in merged_model_request_times:
                                    merged_model_request_times[model_name] = []
                                if isinstance(request_times, list):
                                    merged_model_request_times[model_name].extend(
                                        request_times
                                    )

                        # 合并原始token使用数据
                        if "model_token_usage" in temp_data:
                            for model_name, token_data in temp_data[
                                "model_token_usage"
                            ].items():
                                if model_name not in merged_model_token_usage:
                                    merged_model_token_usage[model_name] = {
                                        "prompt_tokens": [],
                                        "completion_tokens": [],
                                        "total_tokens": [],
                                    }
                                if isinstance(token_data, dict):
                                    merged_model_token_usage[model_name][
                                        "prompt_tokens"
                                    ].extend(token_data.get("prompt_tokens", []))
                                    merged_model_token_usage[model_name][
                                        "completion_tokens"
                                    ].extend(token_data.get("completion_tokens", []))
                                    merged_model_token_usage[model_name][
                                        "total_tokens"
                                    ].extend(token_data.get("total_tokens", []))

                        # 合并未完成请求
                        if "unfinished_requests" in temp_data and isinstance(
                            temp_data["unfinished_requests"], list
                        ):
                            merged_unfinished.extend(temp_data["unfinished_requests"])
                except Exception as e:
                    # 静默处理错误
                    pass

        # Load existing scores from timestamped score files
        for score_file in existing_score_files:
            try:
                with open(score_file, "r", encoding="utf-8") as f:
                    content = f.read()
                    # Extract scores from the score file
                    lines = content.split("\n")
                    in_scores_section = False
                    for line in lines:
                        if "模型分数:" in line:
                            in_scores_section = True
                            continue
                        elif (
                            in_scores_section
                            and line.strip()
                            and not line.startswith("模型请求时间统计")
                        ):
                            # Parse score line: "  model_name: score_value"
                            if ":" in line and not line.startswith("="):
                                parts = line.strip().split(":", 1)
                                if len(parts) == 2:
                                    model_name = parts[0].strip()
                                    # Remove leading spaces
                                    model_name = model_name.lstrip()
                                    try:
                                        score = float(parts[1].strip())
                                        if model_name not in merged_scores:
                                            merged_scores[model_name] = 0.0
                                        merged_scores[model_name] += (
                                            score  # Add to existing score
                                        )
                                    except ValueError:
                                        continue
                        elif in_scores_section and (
                            line.startswith("=") or "模型请求时间统计" in line
                        ):
                            # End of scores section or start of time stats section
                            in_scores_section = False
            except Exception as e:
                # 静默处理错误
                pass

        # Load main score file if it exists
        if os.path.exists(main_score_file):
            try:
                with open(main_score_file, "r", encoding="utf-8") as f:
                    content = f.read()
                    # Extract scores from the main score file
                    lines = content.split("\n")
                    in_scores_section = False
                    for line in lines:
                        if "模型分数:" in line:
                            in_scores_section = True
                            continue
                        elif (
                            in_scores_section
                            and line.strip()
                            and not line.startswith("模型请求时间统计")
                        ):
                            # Parse score line: "  model_name: score_value"
                            if ":" in line and not line.startswith("="):
                                parts = line.strip().split(":", 1)
                                if len(parts) == 2:
                                    model_name = parts[0].strip()
                                    # Remove leading spaces
                                    model_name = model_name.lstrip()
                                    try:
                                        score = float(parts[1].strip())
                                        if model_name not in merged_scores:
                                            merged_scores[model_name] = 0.0
                                        merged_scores[model_name] += (
                                            score  # Add to existing score
                                        )
                                    except ValueError:
                                        continue
                        elif in_scores_section and (
                            line.startswith("=") or "模型请求时间统计" in line
                        ):
                            # End of scores section or start of time stats section
                            in_scores_section = False
            except Exception as e:
                print(f"Error loading main score file {main_score_file}: {e}")

        # Now add current results
        current_raw_data = current_results.get("raw_data", {})
        # 处理当前结果 - 支持两种格式
        if isinstance(current_raw_data, dict):
            # 按模型分组的格式
            for model_name, model_results in current_raw_data.items():
                if model_name not in merged_raw_data:
                    merged_raw_data[model_name] = []
                # Ensure model_results is a list
                if isinstance(model_results, list):
                    merged_raw_data[model_name].extend(model_results)
                else:
                    # If it's not a list, wrap it in a list
                    merged_raw_data[model_name].append(model_results)

                # Add current scores
                current_score = current_results.get("scores", {}).get(model_name, 0.0)
                if model_name not in merged_scores:
                    merged_scores[model_name] = 0.0
                merged_scores[model_name] += current_score
        else:
            # 直接的列表格式 - 从results中提取模型信息
            scores = current_results.get("scores", {})
            if isinstance(scores, dict):
                for model_name, score in scores.items():
                    if model_name not in merged_scores:
                        merged_scores[model_name] = 0.0
                    merged_scores[model_name] += score

            # 如果current_raw_data是列表，需要从中提取模型信息
            if isinstance(current_raw_data, list):
                for item in current_raw_data:
                    if isinstance(item, dict) and "model_name" in item:
                        model_name = item["model_name"]
                        if model_name not in merged_raw_data:
                            merged_raw_data[model_name] = []
                        merged_raw_data[model_name].append(item)

        # 新增：合并当前运行的原始请求时间数据
        if hasattr(self, "model_request_times") and self.model_request_times:
            for model_name, request_times in self.model_request_times.items():
                if model_name not in merged_model_request_times:
                    merged_model_request_times[model_name] = []
                if isinstance(request_times, list):
                    merged_model_request_times[model_name].extend(request_times)

        # 新增：合并当前运行的原始token使用数据
        if hasattr(self, "model_token_usage") and self.model_token_usage:
            for model_name, token_data in self.model_token_usage.items():
                if model_name not in merged_model_token_usage:
                    merged_model_token_usage[model_name] = {
                        "prompt_tokens": [],
                        "completion_tokens": [],
                        "total_tokens": [],
                    }
                if isinstance(token_data, dict):
                    merged_model_token_usage[model_name]["prompt_tokens"].extend(
                        token_data.get("prompt_tokens", [])
                    )
                    merged_model_token_usage[model_name]["completion_tokens"].extend(
                        token_data.get("completion_tokens", [])
                    )
                    merged_model_token_usage[model_name]["total_tokens"].extend(
                        token_data.get("total_tokens", [])
                    )

        # Merge request time stats from current results
        if hasattr(self, "model_request_times") and self.model_request_times:
            for model_name, request_times in self.model_request_times.items():
                total_time = sum(request_times)
                avg_time = total_time / len(request_times) if request_times else 0
                request_count = len(request_times)

                if model_name not in merged_request_time_stats:
                    merged_request_time_stats[model_name] = {
                        "total_time": 0,
                        "avg_time": 0,
                        "request_count": 0,
                        "individual_times": [],
                    }

                # Add current stats to existing stats
                merged_request_time_stats[model_name]["total_time"] += total_time
                merged_request_time_stats[model_name]["request_count"] += request_count
                # Recalculate average time
                if merged_request_time_stats[model_name]["request_count"] > 0:
                    merged_request_time_stats[model_name]["avg_time"] = (
                        merged_request_time_stats[model_name]["total_time"]
                        / merged_request_time_stats[model_name]["request_count"]
                    )

        # 合并当前运行的未完成请求
        current_unfinished = current_results.get("unfinished_requests", [])
        if isinstance(current_unfinished, list):
            merged_unfinished.extend(current_unfinished)

        # Return merged results
        return {
            "raw_data": merged_raw_data,
            "scores": merged_scores,
            "request_time_stats": merged_request_time_stats,
            "model_request_times": merged_model_request_times,  # 新增：返回合并后的原始请求时间
            "model_token_usage": merged_model_token_usage,  # 新增：返回合并后的原始token使用数据
            "unfinished_requests": merged_unfinished,  # 新增：返回合并后的未完成请求
        }

    def save_results(
        self,
        results,
        output_dir,
        config=None,
        task_name=None,
        is_final=False,
        filter_empty_results=True,
    ):
        """
        Save evaluation results to separate files

        Args:
            results: 评测结果字典
            output_dir: 输出目录
            config: 配置字典
            task_name: 任务名称
            is_final: 是否为最终结果（任务完成时为True）
            filter_empty_results: 是否过滤空结果
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        def _filter_raw_data(raw_data):
            """按需过滤提取结果全空的记录"""
            if not filter_empty_results or not isinstance(raw_data, dict):
                return raw_data
            filtered = {}
            for model_name, items in raw_data.items():
                if not isinstance(items, list):
                    continue
                kept = []
                for item in items:
                    extracted = item.get("extracted_answers", [])
                    # 若无提取结果或全部为空白，视为无效，跳过
                    if not extracted or all(
                        (ans is None or not str(ans).strip()) for ans in extracted
                    ):
                        continue
                    kept.append(item)
                filtered[model_name] = kept
            return filtered

        # 如果提供了任务名称，则使用任务专用文件夹
        # 处理任务名称中可能包含的路径分隔符，确保安全的文件夹名称
        safe_task_name = task_name if task_name else None
        if safe_task_name:
            # 替换可能导致路径问题的字符
            import re

            safe_task_name = re.sub(
                r'[<>:"/\\|?*]', "_", safe_task_name
            )  # 替换Windows非法字符

        if safe_task_name:
            task_dir = os.path.join(output_dir, safe_task_name)
            os.makedirs(task_dir, exist_ok=True)

            # 创建temp子文件夹用于存放临时数据
            temp_dir = os.path.join(task_dir, "temp")
            os.makedirs(temp_dir, exist_ok=True)
        else:
            task_dir = output_dir
            temp_dir = output_dir

        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)
        if safe_task_name:
            os.makedirs(task_dir, exist_ok=True)
            os.makedirs(temp_dir, exist_ok=True)

        # 准备时间统计信息
        request_time_stats = {}
        if hasattr(self, "model_request_times") and self.model_request_times:
            for model_name, request_times in self.model_request_times.items():
                total_time = sum(request_times)
                avg_time = total_time / len(request_times) if request_times else 0
                request_time_stats[model_name] = {
                    "total_time": total_time,
                    "avg_time": avg_time,
                    "request_count": len(request_times),
                    "individual_times": request_times,
                }

        # 获取巡目比例配置并保存到结果中
        turn_ratios = {"early": 0.33, "mid": 0.33, "late": 0.34}  # 默认值
        if config is not None:
            turn_ratios = config.get("turn_ratios", turn_ratios)

        # 准备token使用统计信息
        token_usage_stats = {}
        if hasattr(self, "model_token_usage") and self.model_token_usage:
            for model_name, token_data in self.model_token_usage.items():
                if token_data["prompt_tokens"]:  # 检查是否有数据
                    avg_prompt_tokens = sum(token_data["prompt_tokens"]) / len(
                        token_data["prompt_tokens"]
                    )
                    avg_completion_tokens = sum(token_data["completion_tokens"]) / len(
                        token_data["completion_tokens"]
                    )
                    avg_total_tokens = sum(token_data["total_tokens"]) / len(
                        token_data["total_tokens"]
                    )
                    total_prompt_tokens = sum(token_data["prompt_tokens"])
                    total_completion_tokens = sum(token_data["completion_tokens"])
                    total_tokens = sum(token_data["total_tokens"])
                    request_count = len(token_data["prompt_tokens"])

                    token_usage_stats[model_name] = {
                        "avg_prompt_tokens": avg_prompt_tokens,
                        "avg_completion_tokens": avg_completion_tokens,
                        "avg_total_tokens": avg_total_tokens,
                        "total_prompt_tokens": total_prompt_tokens,
                        "total_completion_tokens": total_completion_tokens,
                        "total_tokens": total_tokens,
                        "request_count": request_count,
                    }

        if is_final:
            # 任务完成时，生成最终结果文件
            try:
                # 如果是最终结果，先合并所有之前的结果
                merged_results = self.merge_results(task_dir, results, is_final=True)
                final_raw_data = _filter_raw_data(merged_results.get("raw_data", {}))
                final_scores = merged_results["scores"]
                final_request_time_stats = merged_results["request_time_stats"]

                # 新增：从合并结果中获取完整的统计数据
                if "model_request_times" in merged_results:
                    # 更新self.model_request_times，确保包含所有历史数据
                    for model_name, request_times in merged_results[
                        "model_request_times"
                    ].items():
                        if not hasattr(self, "model_request_times"):
                            self.model_request_times = {}
                        self.model_request_times[model_name] = request_times

                if "model_token_usage" in merged_results:
                    # 更新self.model_token_usage，确保包含所有历史数据
                    for model_name, token_data in merged_results[
                        "model_token_usage"
                    ].items():
                        if not hasattr(self, "model_token_usage"):
                            self.model_token_usage = {}
                        self.model_token_usage[model_name] = token_data

                # 重新计算token_usage_stats，使用更新后的完整数据
                token_usage_stats = {}
                if hasattr(self, "model_token_usage") and self.model_token_usage:
                    for model_name, token_data in self.model_token_usage.items():
                        if token_data["prompt_tokens"]:  # 检查是否有数据
                            avg_prompt_tokens = sum(token_data["prompt_tokens"]) / len(
                                token_data["prompt_tokens"]
                            )
                            avg_completion_tokens = sum(
                                token_data["completion_tokens"]
                            ) / len(token_data["completion_tokens"])
                            avg_total_tokens = sum(token_data["total_tokens"]) / len(
                                token_data["total_tokens"]
                            )
                            total_prompt_tokens = sum(token_data["prompt_tokens"])
                            total_completion_tokens = sum(
                                token_data["completion_tokens"]
                            )
                            total_tokens = sum(token_data["total_tokens"])
                            request_count = len(token_data["prompt_tokens"])

                            token_usage_stats[model_name] = {
                                "avg_prompt_tokens": avg_prompt_tokens,
                                "avg_completion_tokens": avg_completion_tokens,
                                "avg_total_tokens": avg_total_tokens,
                                "total_prompt_tokens": total_prompt_tokens,
                                "total_completion_tokens": total_completion_tokens,
                                "total_tokens": total_tokens,
                                "request_count": request_count,
                            }

                # 重新计算request_time_stats，使用更新后的完整数据
                request_time_stats = {}
                if hasattr(self, "model_request_times") and self.model_request_times:
                    for model_name, request_times in self.model_request_times.items():
                        total_time = sum(request_times)
                        avg_time = (
                            total_time / len(request_times) if request_times else 0
                        )
                        request_time_stats[model_name] = {
                            "total_time": total_time,
                            "avg_time": avg_time,
                            "request_count": len(request_times),
                            "individual_times": request_times,
                        }

                # 1. 生成 complete_test_results.json - 完整详细记录（按模型分组，综合所有结果）
                raw_data_filename = f"complete_test_results.json"
                raw_data_filepath = os.path.join(task_dir, raw_data_filename)

                # 直接使用raw_data作为根结构，按模型名称分组
                complete_results = final_raw_data

                with open(raw_data_filepath, "w", encoding="utf-8") as f:
                    json.dump(complete_results, f, ensure_ascii=False, indent=2)
                print(f"已生成综合结果文件: {raw_data_filepath}")  # 调试信息

                # 2. 生成 model_scores.txt - 模型分数 + 请求时间统计（综合所有结果）
                scores_filename = f"model_scores.txt"
                scores_filepath = os.path.join(task_dir, scores_filename)

                with open(scores_filepath, "w", encoding="utf-8") as f:
                    f.write("模型评测分数\n")
                    f.write("=" * 30 + "\n\n")

                    # 写入模型分数
                    f.write("模型分数:\n")
                    for model_name, score in final_scores.items():
                        f.write(f" {model_name}: {score}\n")

                    # 写入模型请求时间统计
                    f.write(f"\n模型请求时间统计:\n")
                    # 确保所有模型都有请求时间统计，即使为0
                    for model_name in final_scores.keys():
                        if (
                            model_name in final_request_time_stats
                            and final_request_time_stats[model_name]["request_count"]
                            > 0
                        ):
                            stats = final_request_time_stats[model_name]
                            f.write(
                                f"  {model_name}: 总耗时 {stats['total_time']:.2f}秒, "
                                f"平均耗时 {stats['avg_time']:.2f}秒, "
                                f"请求次数 {stats['request_count']}\n"
                            )
                        else:
                            # 检查是否在合并的原始请求时间数据中存在
                            if (
                                hasattr(self, "model_request_times")
                                and model_name in self.model_request_times
                            ):
                                request_times = self.model_request_times[model_name]
                                total_time = sum(request_times)
                                avg_time = (
                                    total_time / len(request_times)
                                    if request_times
                                    else 0
                                )
                                request_count = len(request_times)
                                f.write(
                                    f"  {model_name}: 总耗时 {total_time:.2f}秒, "
                                    f"平均耗时 {avg_time:.2f}秒, "
                                    f"请求次数 {request_count}\n"
                                )
                            else:
                                # 即使没有请求时间数据，也要显示该模型的信息
                                f.write(
                                    f"  {model_name}: 总耗时 0.00秒, 平均耗时 0.00秒, 请求次数 0\n"
                                )

                    # 写入token消耗统计
                    f.write(f"\nToken消耗统计:\n")
                    for model_name in final_scores.keys():
                        if (
                            model_name in token_usage_stats
                            and token_usage_stats[model_name]["request_count"] > 0
                        ):
                            stats = token_usage_stats[model_name]
                            f.write(f"  {model_name}:\n")
                            f.write(
                                f"    平均输入Tokens: {stats['avg_prompt_tokens']:.0f}\n"
                            )
                            f.write(
                                f"    平均输出Tokens: {stats['avg_completion_tokens']:.0f}\n"
                            )
                            f.write(
                                f"    平均总Tokens: {stats['avg_total_tokens']:.0f}\n"
                            )
                            f.write(
                                f"    总输入Tokens: {stats['total_prompt_tokens']:.0f}\n"
                            )
                            f.write(
                                f"    总输出Tokens: {stats['total_completion_tokens']:.0f}\n"
                            )
                            f.write(f"    总Tokens: {stats['total_tokens']:.0f}\n")
                            f.write(f"    请求次数: {stats['request_count']}\n")
                        else:
                            # 检查是否在合并的原始token数据中存在
                            if (
                                hasattr(self, "model_token_usage")
                                and model_name in self.model_token_usage
                            ):
                                token_data = self.model_token_usage[model_name]
                                if token_data["prompt_tokens"]:
                                    avg_prompt_tokens = sum(
                                        token_data["prompt_tokens"]
                                    ) / len(token_data["prompt_tokens"])
                                    avg_completion_tokens = sum(
                                        token_data["completion_tokens"]
                                    ) / len(token_data["completion_tokens"])
                                    avg_total_tokens = sum(
                                        token_data["total_tokens"]
                                    ) / len(token_data["total_tokens"])
                                    total_prompt_tokens = sum(
                                        token_data["prompt_tokens"]
                                    )
                                    total_completion_tokens = sum(
                                        token_data["completion_tokens"]
                                    )
                                    total_tokens = sum(token_data["total_tokens"])
                                    request_count = len(token_data["prompt_tokens"])

                                    f.write(f"  {model_name}:\n")
                                    f.write(
                                        f"    平均输入Tokens: {avg_prompt_tokens:.0f}\n"
                                    )
                                    f.write(
                                        f"    平均输出Tokens: {avg_completion_tokens:.0f}\n"
                                    )
                                    f.write(
                                        f"    平均总Tokens: {avg_total_tokens:.0f}\n"
                                    )
                                    f.write(
                                        f"    总输入Tokens: {total_prompt_tokens:.0f}\n"
                                    )
                                    f.write(
                                        f"    总输出Tokens: {total_completion_tokens:.0f}\n"
                                    )
                                    f.write(f"    总Tokens: {total_tokens:.0f}\n")
                                    f.write(f"    请求次数: {request_count}\n")
                                else:
                                    # 即使没有token数据，也要显示该模型的信息
                                    f.write(f"  {model_name}: 暂无Token消耗数据\n")
                            else:
                                # 即使没有token数据，也要显示该模型的信息
                                f.write(f"  {model_name}: 暂无Token消耗数据\n")

                    # 写入整体请求时间统计（从第一个请求到最后一个请求）
                    if (
                        self.first_request_time is not None
                        and self.last_request_time is not None
                    ):
                        total_duration = (
                            self.last_request_time - self.first_request_time
                        )
                        f.write(f"\n整体请求时间统计:\n")
                        f.write(
                            f"  总耗时: {total_duration * 1000:.0f}毫秒 ({total_duration:.3f}秒)\n"
                        )
                        f.write(
                            f"  开始时间: {datetime.fromtimestamp(self.first_request_time).strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}\n"
                        )
                        f.write(
                            f"  结束时间: {datetime.fromtimestamp(self.last_request_time).strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}\n"
                        )

                    # 写入巡目比例配置
                    f.write(f"\n巡目比例配置:\n")
                    f.write(f"  早巡比例: {turn_ratios['early']:.2f}\n")
                    f.write(f"  中巡比例: {turn_ratios['mid']:.2f}\n")
                    f.write(f"  晚巡比例: {turn_ratios['late']:.2f}\n")

                    # 写入样本全对率和样本0分率
                    f.write(f"\n样本准确率统计:\n")
                    f.write(f"  说明: 样本全对率=同一条测试数据每次都回答正确的比例\n")
                    f.write(f"        样本0分率=同一条测试数据每次都回答错误的比例\n")

                    # Calculate sample accuracy metrics
                    sample_metrics = self.calculate_sample_accuracy(
                        final_raw_data, config
                    )
                    for model_name, metrics in sample_metrics.items():
                        full_correct_rate = metrics["full_correct_rate"] * 100
                        full_wrong_rate = metrics["full_wrong_rate"] * 100
                        f.write(f"  {model_name}:\n")
                        f.write(f"    样本全对率: {full_correct_rate:.1f}%\n")
                        f.write(f"    样本0分率: {full_wrong_rate:.1f}%\n")

                    # 写入每个模型的请求时间统计
                    if hasattr(self, "model_first_request_times") and hasattr(
                        self, "model_last_request_times"
                    ):
                        f.write(f"\n各模型请求时间统计:\n")
                        for model_name in self.model_first_request_times.keys():
                            if model_name in self.model_last_request_times:
                                model_duration = (
                                    self.model_last_request_times[model_name]
                                    - self.model_first_request_times[model_name]
                                )
                                f.write(f"  {model_name}:\n")
                                f.write(
                                    f"    总耗时: {model_duration * 1000:.0f}毫秒 ({model_duration:.3f}秒)\n"
                                )
                                f.write(
                                    f"    开始时间: {datetime.fromtimestamp(self.model_first_request_times[model_name]).strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}\n"
                                )
                                f.write(
                                    f"    结束时间: {datetime.fromtimestamp(self.model_last_request_times[model_name]).strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}\n"
                                )
                print(f"已生成综合分数文件: {scores_filepath}")  # 调试信息

                # 3. 删除所有时间戳命名的临时结果文件，保留综合结果文件
                import glob

                timestamp_result_files = glob.glob(
                    os.path.join(task_dir, "complete_test_results_*.json")
                )
                timestamp_score_files = glob.glob(
                    os.path.join(task_dir, "model_scores_*.txt")
                )
                for file_path in timestamp_result_files + timestamp_score_files:
                    try:
                        os.remove(file_path)
                        print(f"已删除临时结果文件: {file_path}")  # 调试信息
                    except Exception as e:
                        print(f"警告: 删除临时文件失败 {file_path}: {e}")

                # 4. 只删除temp文件夹及其内容，保留evaluation_task文件和任务文件夹
                # 注意：现在只在最终完成时才删除临时文件，而不是每次保存中间结果时删除
                if is_final and task_name and os.path.exists(temp_dir):
                    import shutil

                    try:
                        shutil.rmtree(temp_dir)
                        print(f"已删除temp文件夹: {temp_dir}")  # 调试信息
                    except Exception as e:
                        print(f"警告: 删除临时文件夹失败: {e}")

                # 返回最终文件路径
                return {
                    "scores_file": scores_filepath,
                    "details_file": None,  # 不再生成archive文件
                    "raw_data_file": raw_data_filepath,
                }
            except Exception as e:
                print(f"错误: 生成结果文件失败: {e}")  # 错误信息
                import traceback

                traceback.print_exc()  # 打印详细错误信息
                # 如果出现异常，返回None表示文件未成功生成
                return {
                    "scores_file": None,
                    "details_file": None,
                    "raw_data_file": None,
                }
        else:
            # 任务进行中，保存临时数据到temp文件夹
            if task_name:
                temp_data_filename = f"temp_progress_{timestamp}.json"
                temp_data_filepath = os.path.join(temp_dir, temp_data_filename)

                # 保存临时进度数据 - 包含完整的原始统计数据
                temp_data = {
                    "timestamp": timestamp,
                    "task_name": task_name,
                    "scores": results.get("scores", {}),
                    "details": results.get("details", {}),
                    "raw_data": _filter_raw_data(results.get("raw_data", {})),
                    "request_time_stats": request_time_stats,
                    "model_request_times": self.model_request_times
                    if hasattr(self, "model_request_times")
                    else {},  # 保存原始请求时间列表
                    "model_token_usage": self.model_token_usage
                    if hasattr(self, "model_token_usage")
                    else {},  # 保存原始token使用数据
                    # 统一在临时文件中存储已完成请求数，优先使用实例变量保证准确
                    "completed_requests": getattr(
                        self, "completed_requests", results.get("completed_requests", 0)
                    ),
                    # 按模型累计完成数（用于 UI 刷新时显示每个模型的剩余请求数）
                    "model_completed_requests": getattr(
                        self,
                        "total_model_completed_requests",
                        getattr(
                            self,
                            "model_completed_requests",
                            results.get("model_completed_requests", {}),
                        ),
                    ),
                    "total_tasks": results.get("total_tasks", 0),
                    "unfinished_requests": getattr(
                        self, "unfinished_requests", []
                    ),  # 保存未完成请求，便于中断续跑
                }

                with open(temp_data_filepath, "w", encoding="utf-8") as f:
                    json.dump(temp_data, f, ensure_ascii=False, indent=2)

                return {
                    "temp_file": temp_data_filepath,
                    "scores_file": None,
                    "details_file": None,
                    "raw_data_file": None,
                }
            else:
                # 如果没有任务名称，使用旧的保存方式
                scores_filename = f"model_scores_{timestamp}.txt"
                scores_filepath = os.path.join(task_dir, scores_filename)

                with open(scores_filepath, "w", encoding="utf-8") as f:
                    f.write("模型评测分数\n")
                    f.write("=" * 30 + "\n\n")
                    f.write("模型分数:\n")
                    for model_name, score in results["scores"].items():
                        f.write(f" {model_name}: {score}\n")

                    if request_time_stats:
                        f.write(f"\n模型请求时间统计:\n")
                        for model_name, stats in request_time_stats.items():
                            f.write(
                                f"  {model_name}: 总耗时 {stats['total_time']:.2f}秒, "
                                f"平均耗时 {stats['avg_time']:.2f}秒, "
                                f"请求次数 {stats['request_count']}\n"
                            )

                    # 写入整体请求时间统计（从第一个请求到最后一个请求）
                    if (
                        self.first_request_time is not None
                        and self.last_request_time is not None
                    ):
                        total_duration = (
                            self.last_request_time - self.first_request_time
                        )
                        f.write(f"\n整体请求时间统计:\n")
                        f.write(
                            f"  总耗时: {total_duration * 1000:.0f}毫秒 ({total_duration:.3f}秒)\n"
                        )
                        f.write(
                            f"  开始时间: {datetime.fromtimestamp(self.first_request_time).strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}\n"
                        )
                        f.write(
                            f"  结束时间: {datetime.fromtimestamp(self.last_request_time).strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}\n"
                        )
                return {
                    "scores_file": scores_filepath,
                    "details_file": None,
                    "raw_data_file": None,
                }
